import torch
import torch.nn.functional as F
from typing import List, Optional
import tiktoken
from model.modelargs import ModelArgs
from model.layers import Transformer

class Generator:
    def __init__(self, model: Transformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load Args
        if 'args' in checkpoint:
            args = checkpoint['args']
        else:
            print("Warning: ModelArgs not found in checkpoint. Loading from config.yaml")
            from model.modelargs import load_model_args_from_yaml
            args = load_model_args_from_yaml("config.yaml")

        model = Transformer(args)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        # Tokenizer
        tokenizer = tiktoken.get_encoding("gpt-4") 

        return cls(model, tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8, # slightly higher for creativity
        top_p: float = 0.9,
    ) -> List[str]:
        bsz = len(prompts)
        
        # Encode with EOS if needed? Usually prompt is start.
        prompt_tokens_list = [self.tokenizer.encode(p, allowed_special={'<|endoftext|>'}) for p in prompts]
        
        min_prompt_len = min(len(t) for t in prompt_tokens_list)
        max_prompt_len = max(len(t) for t in prompt_tokens_list)
        
        total_len = min(self.model.max_seq_len, max_gen_len + max_prompt_len)
        
        pad_id = 0 
        eos_id = self.tokenizer.eot_token
        
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=next(self.model.parameters()).device)
        
        for k, t in enumerate(prompt_tokens_list):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=tokens.device)
            
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=tokens.device)
        
        # --- Prefill Phase ---
        # We process the common prefix first if optimizing, but for simplicity:
        # We can just start loop from 0 or handle chunks.
        # But 'MultiHeadDifferentialAttention' expects 'start_pos' for cache update.
        # Tokens already populated up to prompt length.
        
        # We can feed the prompts. If prompt lengths vary, we need to be careful with attention mask?
        # Our model forward builds causal mask based on input shape.
        # If we feed (bsz, L), it attends causally. 
        # But if prompt L is different, padding at end of prompt might be attended to?
        # Pad token = 0. Embedding of 0 exists. Attention mask should ideally mask padding.
        # Our simple `layers.py` mask is triangular causal (masking future).
        # It doesn't explicitly mask padding token (0) in the past if it was part of input.
        # However, we only care about `tokens` that are valid.
        # For variable length prompts in a batch, standard practice is left-padding or masking.
        # Here we right-padded with 0.
        # If we just run forward, position embeddings will correspond to 0, 1, 2... including pads.
        # This is bad for incorrect positions.
        
        # Right approach for batch generation with variable prompt length without sophisticated masking:
        # Process one by one (batch size 1 loop) or implement attention mask for padding.
        # `model/layers.py` `forward` accepts `mask`.
        # Currently `forward` creates a causal mask internally:
        # mask = torch.full((seqlen, seqlen), float("-inf"), ...).triu_(1)
        # This only handles causality.
        # If we want to mask padding, we need to pass `mask`.
        
        # Let's assume user passes batch_size=1 for now or same length prompts, 
        # OR we rely on the fact that we overwrite the pad tokens as we go?
        # No, prefill attention will see pads.
        
        # For now, let's process token-by-token from pos 0 for simplicity and correctness (slow prefill).
        
        for cur_pos in range(0, total_len - 1):
            
            # Identify which sequences are still in prompt phase
            # If cur_pos < len(prompt), we feed true token.
            # If cur_pos >= len(prompt), we generate.
            
            # Actually, to generate output at `cur_pos+1`, we need logits from `cur_pos`.
            # So we feed `tokens[:, cur_pos:cur_pos+1]`.
            
            # Input for this step
            x = tokens[:, cur_pos:cur_pos+1]
            
            logits = self.model(x, start_pos=cur_pos)
            next_token_logits = logits[:, -1, :] # (bsz, vocab)
            
            # Sample
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            next_token = next_token.reshape(-1)
            
            # Determine actual next token to keep
            # If (cur_pos + 1) is within prompt, use ground truth from tokens
            # If (cur_pos + 1) is new, use next_token
            
            # We are writing to tokens[:, cur_pos+1]
            target_pos = cur_pos + 1
            if target_pos >= total_len: break

            force_prompt_mask = torch.tensor([target_pos < len(t) for t in prompt_tokens_list], device=tokens.device)
            
            # If we force prompt, we don't update tokens (it already has prompt).
            # If we don't force, we update tokens with prediction.
            
            # Apply to tokens tensor
            # We must be careful: tokens already has prompt. 
            # We only overwrite if NOT in prompt.
            
            # If not in prompt, update
            tokens[:, target_pos] = torch.where(force_prompt_mask, tokens[:, target_pos], next_token)
            
            # Check EOS
            # Only if NOT in prompt
            is_eos = (next_token == eos_id) & (~force_prompt_mask)
            eos_reached |= is_eos
            
            if eos_reached.all():
                break
                
        # Decode
        out_prompts = []
        for i, t in enumerate(tokens.tolist()):
            # Cut at EOS
            clean_t = []
            for token_id in t:
                if token_id == eos_id:
                    break
                clean_t.append(token_id)
            out_prompts.append(self.tokenizer.decode(clean_t))
            
        return out_prompts

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        try:
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token_idx = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token_idx)
            return next_token
        except:
             return probs_idx[:, :1] # Fallback to greedy if sampling fails (e.g. all masked)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max_len", type=int, default=100, help="Max generation length")
    args = parser.parse_args()
    
    if args.checkpoint == "test":
        print("Test mode: No checkpoint loaded.")
    else:
        try:
            gen = Generator.from_checkpoint(args.checkpoint)
            output = gen.generate([args.prompt], max_gen_len=args.max_len)
            print("Output:", output[0])
        except Exception as e:
            print(f"Error loading checkpoint or generating: {e}")
