# Differential Transformer MoE

Hey there! Welcome to my journey of building a **Differential Transformer with Mixture of Experts (MoE)** from scratch. This is a learning project where I'm implementing a modern, efficient language model architecture designed to run on AMD MI300X GPUs.

## What's This About?

I'm building a ~18-22B parameter sparse MoE model that combines some really cool architectural innovations:

- **Differential Attention**: A novel attention mechanism that improves focus and reduces noise
- **Mixture of Experts (MoE)**: Sparse activation with 64 routed experts + 2 shared experts
- **Multi-head Latent Attention (MLA)**: Memory-efficient attention with LoRA-style compression
- **FP8 Quantization**: Custom CUDA kernels for efficient training on AMD MI300X
- **Extended Context**: Support for up to 8K tokens using YaRN-style RoPE scaling

## 🏗️ Architecture Highlights

### Model Configuration
- **Parameters**: ~18-22B (sparse, ~6B active per token)
- **Layers**: 48 transformer blocks (8 dense + 40 MoE)
- **Hidden Dimension**: 6144
- **Attention Heads**: 48 heads with 128-dim each
- **Context Length**: 8192 tokens (extendable)
- **Experts**: 64 routed + 2 shared, activating 4 per token

### Key Features
- **FP8 Training**: Optimized for AMD MI300X with custom kernels
- **Efficient Attention**: Multi-head latent attention with LoRA compression
- **Smart Routing**: Group-limited expert routing with load balancing
- **Extended Context**: YaRN-based RoPE scaling for longer sequences

## 📁 Project Structure

```
diff_llm/
├── config.yaml          # Model hyperparameters and training config
└── model/
    ├── __init__.py      # Package initialization
    ├── modelargs.py     # Configuration dataclass and YAML loader
    ├── layers.py        # Core neural network layers (attention, MLP, MoE)
    └── kernel.py        # Custom FP8 CUDA kernels using TileLang
```

## 🔧 Current Implementation Status

### ✅ Completed
- [x] Model configuration system with YAML support
- [x] Custom FP8 quantization kernels (act_quant, fp8_gemm, fp8_index)
- [x] Core layer implementations:
  - [x] RMSNorm and LayerNorm
  - [x] Linear layers with FP8 support
  - [x] Parallel embedding layers
  - [x] Multi-head latent attention (MLA)
  - [x] Differential attention mechanism
  - [x] MoE feed-forward layers with routing
- [x] YaRN RoPE positional encoding

### 🚧 In Progress / TODO
- [ ] Complete transformer block assembly
- [ ] Model initialization and weight loading
- [ ] Training loop and optimizer setup
- [ ] Data pipeline and tokenization
- [ ] Distributed training support
- [ ] Evaluation and benchmarking
- [ ] Inference optimization

## 🛠️ Technical Deep Dive

### FP8 Quantization
I've implemented custom CUDA kernels using TileLang for efficient FP8 operations:
- **Block-wise quantization**: Groups activations into 128-element blocks
- **Per-tensor scaling**: Safer for training stability
- **Fused operations**: Combined quantization + GEMM for speed

### Differential Attention
The attention mechanism uses a differential approach to reduce noise:
- Computes two separate attention maps
- Subtracts them to cancel out common patterns
- Helps the model focus on what's truly important

### MoE Routing
Smart expert selection with:
- **Group-limited routing**: Experts organized into 8 groups
- **Top-4 activation**: Each token routed to 4 experts
- **Load balancing**: Ensures even expert utilization

## 🎓 Learning Goals

This project is all about learning by doing. I'm exploring:
- Modern transformer architectures
- Efficient training techniques (FP8, MoE)
- CUDA kernel programming
- Large-scale model design
- AMD GPU optimization

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.10+
# PyTorch 2.0+ with ROCm support
# TileLang for custom kernels
# PyYAML for config loading
```

### Installation
```bash
# Clone the repo
git clone https://github.com/ramprasathk07/Differential-MOE.git
cd diff_llm

# Install dependencies (coming soon)
pip install -r requirements.txt  # TODO: Create this file
```

### Configuration
Edit `config.yaml` to customize the model architecture. Key parameters:
- `dim`: Model dimension (default: 6144)
- `n_layers`: Number of transformer layers (default: 48)
- `n_routed_experts`: Number of MoE experts (default: 64)
- `max_seq_len`: Maximum sequence length (default: 8192)

## 📚 References & Inspiration

This project draws inspiration from several cutting-edge papers and implementations:
- **Differential Transformers** (Microsoft Research)
- **DeepSeek-V2/V3** (MLA and MoE architecture)
- **YaRN** (Extended context via RoPE scaling)
- **Mixtral** (MoE routing strategies)

## 🤝 Contributing

This is a personal learning project, but I'm open to suggestions, bug reports, and discussions! Feel free to open an issue if you spot something interesting or have ideas to share.

## 📝 License

This project is open source and available under the MIT License (or specify your preferred license).

## 🙏 Acknowledgments

Huge thanks to the open-source ML community for sharing knowledge, papers, and implementations that make projects like this possible!

---

**Status**: 🏗️ Early Development - Training from scratch in progress!

*Last updated: February 2026*
