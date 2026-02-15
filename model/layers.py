import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Literal, Tuple
from .modelargs import ModelArgs

from kernel import act_quant, fp8_gemm, fp8_index

world_size = 1
rank = 0
block_size = 128

class ParallelEmbedding(nn.Module):
    """
    Not Required, Since i am using a single GPU for training. 
    But this is for future reference when i want to scale to multiple GPUs.
    
    Embedding layer with parallelism support across distributed processes.
    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.
        Args:
            x (torch.Tensor): Input tensor containing token indices.
        Returns:
            torch.Tensor: Embedded representations.
        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0

        y = F.embedding(x, self.weight)
        
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y
    

def linear(x:torch.Tensor, weight:torch.Tensor, 
           bias:Optional[torch.Tensor] = None,
           scale_fmt:Optional[str] = None)-> torch.Tensor:
    
    ''' 
    Applies a linear transformation to the incoming data: y = xA^T + b.
    '''

    if weight.dtype != torch.float8_e4m3fn:
        return F.linear(x, weight)
    else:
        x, scale = act_quant(x,block_size,scale_fmt)
        return fp8_gemm(x, scale, weight,weight.shape)
    

class Linear(nn.Module):
    '''
    A linear layer that supports both standard and FP8 quantized weights.
    '''
    
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, 
                in_features: int, 
                out_features: int, 
                bias: bool = False, 
                dtype = None):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias, self.scale_fmt)    


    def reset_parameters(self):
        if self.weight.element_size() != 1:
            # Normal (bf16 / fp16 / fp32)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            return

        # ---------- FP8 PATH ----------

        # 1️⃣ Initialize a FP32 master weight
        w_fp32 = torch.empty(
            self.out_features,
            self.in_features,
            device=self.weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(w_fp32, a=math.sqrt(5))

        # 2️⃣ Compute block-wise scale
        B = block_size
        max_fp8 = 240.0  # depends on FP8 format (e4m3)

        scale = torch.empty_like(self.scale)

        for i in range(scale.shape[0]):
            for j in range(scale.shape[1]):
                block = w_fp32[
                    i*B : (i+1)*B,
                    j*B : (j+1)*B,
                ]
                scale[i, j] = block.abs().max() / max_fp8 + 1e-8

        self.scale.data.copy_(scale)

        # 3️⃣ Quantize FP32 → FP8
        w_fp8 = torch.clamp(w_fp32 / scale.repeat_interleave(B, 0)
                                       .repeat_interleave(B, 1),
                            -max_fp8, max_fp8)

        self.weight.data.copy_(w_fp8.to(self.weight.dtype))

        # 4️⃣ Bias (if used)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias, self.scale_fmt)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, reduce_output = True, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        self.reduce_output = reduce_output
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight, None, self.scale_fmt)
        if self.reduce_output and world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)
    

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.
    Args:
        dim (int): Dimension of the input features.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor,residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype

        if residual is None:
            x = x.float()
            var  = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        
        else:
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)
        
class LayerNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.empty(dim))

    def forward(self,x:torch.Tensor):
        return F.layer_norm(x.float(),(self.dim,),self.weight,self.bias,self.eps).type_as(x)
    
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)    


def weight_dequant(weight, scale):
    shape = weight.shape
    assert weight.dim() == 2
    weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size).transpose(1, 2).contiguous().view(-1, block_size * block_size)
    weight = (weight.float() * scale.view(-1, 1).float()).to(torch.get_default_dtype()).view(shape[0] // block_size, shape[1] // block_size, block_size, block_size).transpose(1, 2).contiguous().view(shape)
    return weight
    

class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Layer.
    Implementation adapted for the existing architecture, replacing MLA.
    """
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.head_dim = args.dim // args.n_heads
        self.rope_head_dim = args.qk_rope_head_dim # Use the configured rope dim
        
        # Differential attention uses 2x heads for Q, K, V internally
        # We project to 2 * head_dim per head
        self.wq = ColumnParallelLinear(self.dim, 2 * self.n_heads * self.head_dim, bias=False)
        self.wk = ColumnParallelLinear(self.dim, 2 * self.n_heads * self.head_dim, bias=False)
        self.wv = ColumnParallelLinear(self.dim, 2 * self.n_heads * self.head_dim, bias=False)
        self.wo = RowParallelLinear(2 * self.n_heads * self.head_dim, self.dim, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(self.n_local_heads, self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.n_local_heads, self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.n_local_heads, self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.n_local_heads, self.head_dim))
        
        # Calculate lambda_init based on layer_id
        # formula: 0.8 - 0.6 * exp(-0.3 * (l - 1))
        # layer_id is 0-indexed in our code, so we use layer_id directly to match l-1 if l is 1-indexed
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_id)

        self.rms_scale = nn.Parameter(torch.ones(self.n_local_heads, 2 * self.head_dim))
        self.eps = 1e-5
        
        # Cache for generation
        # K, V will store the full projected key/values (2 * head_dim)
        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, 2 * self.head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, 2 * self.head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Project
        q = self.wq(x) # (bsz, seqlen, 2 * n_local_heads * head_dim)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (bsz, seqlen, n_local_heads, 2 * head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, 2 * self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, 2 * self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, 2 * self.head_dim)

        # Split Q, K into (Q1, Q2), (K1, K2) for Differential Attention
        # Shape: (bsz, seqlen, n_local_heads, head_dim)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # Apply RoPE
        # We apply RoPE to the first rope_head_dim of q1, q2, k1, k2
        if self.rope_head_dim > 0:
            q1_rope, q1_nope = q1.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            q2_rope, q2_nope = q2.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            k1_rope, k1_nope = k1.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            k2_rope, k2_nope = k2.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

            q1_rope = apply_rotary_emb(q1_rope, freqs_cis)
            q2_rope = apply_rotary_emb(q2_rope, freqs_cis)
            k1_rope = apply_rotary_emb(k1_rope, freqs_cis)
            k2_rope = apply_rotary_emb(k2_rope, freqs_cis)

            q1 = torch.cat([q1_rope, q1_nope], dim=-1)
            q2 = torch.cat([q2_rope, q2_nope], dim=-1)
            k1 = torch.cat([k1_rope, k1_nope], dim=-1)
            k2 = torch.cat([k2_rope, k2_nope], dim=-1)

        # Update Cache
        # We reconstruct k from k1, k2 to store in standard cache format (or we could just store k before split)
        # Actually k was modified by RoPE, so reconstruct it
        k = torch.cat([k1, k2], dim=-1)
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        
        # Read from cache for attention computation
        k_cached = self.k_cache[:bsz, :end_pos] # (bsz, total_seq, heads, 2*head_dim)
        v_cached = self.v_cache[:bsz, :end_pos]

        # Split cached K
        k1_cached, k2_cached = k_cached.chunk(2, dim=-1)

        # Transpose for attention: (bsz, heads, seqlen, head_dim)
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1_T = k1_cached.permute(0, 2, 3, 1) # (bsz, heads, head_dim, total_seq)
        k2_T = k2_cached.permute(0, 2, 3, 1)
        
        # Differential Attention Calculation
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        lambda_val = lambda_val.view(1, self.n_local_heads, 1, 1).type_as(q1)

        scaling = self.head_dim ** -0.5
        
        a1 = torch.matmul(q1, k1_T) * scaling
        a2 = torch.matmul(q2, k2_T) * scaling

        if mask is not None:
             # Mask should be (bsz, 1, seqlen, total_seq) or broadcastable
             # Existing mask in Transformer.forward is (seqlen, seqlen) triu
             # Adjusted for generation or training
             # In generation, mask is usually None or handled via cache logic?
             # existing code: scores += index_mask.unsqueeze(2) ... 
             # existing Transformer.forward passes mask as (seqlen, seqlen) for the Current chunk. 
             # If start_pos > 0, we are generating. mask should cover previous pos?
             # Standard implementation: For generation, if processing 1 token, mask is None usually (causal implicit)
             # But if training/prefill, mask is full causal mask.
             # We need to broadcast mask to previous positions.
             # If mask is provided, it's (seqlen, seqlen). 
             # We are attending to equal or larger memory.
             # Let's align with standard:
             # If generating, we attend to all previous. No mask needed for past, just causal for current (which is identity if len 1).
             # If prefill, mask is (S, S).
             
             if mask.size(-1) != a1.size(-1):
                 # This happens if we are generating (seqlen 1) but attending to (end_pos)
                 # We usually don't need mask for past tokens in standard causal LM unless using window
                 # Or if existing mask passed was only for q-seqlen.
                 # Let's assume mask is handled correctly by caller if dims match, otherwise ignore if causal implied?
                 # Actually, for step-generation, mask is usually None.
                 pass
             else:
                 a1 = a1 + mask
                 a2 = a2 + mask

        attn1 = F.softmax(a1, dim=-1)
        attn2 = F.softmax(a2, dim=-1)
        attn = attn1 - lambda_val * attn2
        
        v_cached = v_cached.transpose(1, 2) # (bsz, heads, total_seq, 2*head_dim)
        
        o = torch.matmul(attn, v_cached) # (bsz, heads, seqlen, 2*head_dim)
        
        # RMSNorm per head
        # o shape: (bsz, heads, seqlen, 2*head_dim) -> flatten to norm -> (...)
        o_flat = o.transpose(1, 2).reshape(bsz * seqlen * self.n_local_heads, 2 * self.head_dim)
        rms = torch.sqrt(o_flat.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        o_norm = (o_flat / rms) * self.rms_scale.view(1, -1) # Broadcasting rms_scale over batch*seq
        # rms_scale is (n_local_heads, 2*head_dim). Need to match flatten order
        # flatten is (bsz, seq, heads, ...) ? No.
        # o was (bsz, heads, seqlen, ...). transpose(1,2) -> (bsz, seqlen, heads, ...)
        # reshape -> (bsz*seq*heads, ...)
        # rms_scale needs to be repeated for (bsz*seq)
        rms_scale_expanded = self.rms_scale.unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1, -1).reshape(-1, 2 * self.head_dim)
        o_norm = (o_flat / rms) * rms_scale_expanded

        o_norm = o_norm.view(bsz, seqlen, self.n_local_heads, 2 * self.head_dim)
        o_norm = o_norm * (1 - self.lambda_init)

        # Reshape for output projection
        o_norm = o_norm.reshape(bsz, seqlen, self.n_local_heads * 2 * self.head_dim)
        
        out = self.wo(o_norm)
        return out


    
class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, reduce_output: bool = True):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim, reduce_output=reduce_output)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))

    
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """

        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices
    
class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))
    
class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim, reduce_output=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices, minlength=self.n_routed_experts).tolist()
        
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        y += self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return y.type_as(x).view(shape)
    
class Block(nn.Module):
    """
    Transformer block combining Differential Attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MultiHeadDifferentialAttention).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MultiHeadDifferentialAttention(args, layer_id)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        x = self.attn(x, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        return x, residual
    


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        Linear.scale_fmt = args.scale_fmt
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.float32)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1) if seqlen > 1 else None
        h, residual = self.embed(tokens), None
        
        for layer in self.layers:
            h, residual = layer(h, residual, start_pos, freqs_cis, mask)
            
        h = self.norm(h + residual) if residual is not None else self.norm(h)
        logits = self.head(h)
        return logits