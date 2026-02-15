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
├── train.py             # Main training script
├── config.yaml          # Model hyperparameters and training config
├── model/
│   ├── __init__.py      # Package initialization
│   ├── modelargs.py     # Configuration dataclass and YAML loader
│   ├── layers.py        # Core neural network layers (attention, MLP, MoE)
│   └── kernel.py        # Custom FP8 CUDA kernels using TileLang
├── training/
│   ├── trainer.py       # Full-featured training loop (WandB, Tensorboard, Checkpointing)
│   ├── data.py          # Streaming dataset loader for HuggingFace
│   └── utils.py         # Helper utilities (Logging, Metrics)
└── tests/               # Unit tests
```

## 🏋️ Training Guide

### 1. Setup Environment
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training
To start training with default settings (HuggingFace FineWeb-Edu dataset):
```bash
python train.py --run_name "differential-moe-v1" --batch_size 8 --accumulate_grad_batches 4 --use_wandb
```

### 3. Monitoring
- **WandB**: If enabled, logs will be sent to your project dashboard.
- **Tensorboard**: Run `tensorboard --logdir checkpoints/` to view metrics locally.

### Key Arguments
- `--config`: Custom config file (default: `config.yaml`)
- `--dataset`: HuggingFace dataset name (default `HuggingFaceFW/fineweb-edu`)
- `--dtype`: `bf16` or `fp32` (model supports internal `fp8` logic if enabled in config)
- `--resume_from`: Path to checkpoint to resume training
- `--use_amp`: Enable automatic mixed precision

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
- [x] Full training loop with production features
- [x] Streaming data pipeline

### 🚧 In Progress / TODO
- [ ] Distributed training support (multi-gpu)
- [ ] Evaluation and benchmarking suite
- [ ] Inference optimization

## 🛠️ Technical Deep Dive

### FP8 Quantization
I've implemented custom CUDA kernels using TileLang for efficient FP8 operations:
- **Block-wise quantization**: Groups activations into 128-element blocks
- **Per-tensor scaling**: Safer for training stability
- **Fused operations**: Combined quantization + GEMM for speed

### Differential Attention
The attention mechanism uses a differential approach to reduce noise by computing two separate attention maps and subtracting them.

### MoE Routing
Smart expert selection with Group-limited routing ensuring load balancing and top-4 activation.

## 🎓 Learning Goals

This project is all about learning by doing. I'm exploring modern transformer architectures, efficient training techniques (FP8, MoE), CUDA kernel programming, and large-scale model design.

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/ramprasathk07/Differential-MOE.git
cd diff_llm
pip install -r requirements.txt
```

### Configuration
Edit `config.yaml` to customize the model architecture. Key parameters:
- `dim`: Model dimension
- `n_layers`: Number of transformer layers
- `n_routed_experts`: Number of MoE experts

## 📚 References & Inspiration
- **Differential Transformers** (Microsoft Research)
- **DeepSeek-V2/V3** (MLA and MoE architecture)
- **YaRN** (Extended context via RoPE scaling)
- **Mixtral** (MoE routing strategies)

## 🤝 Contributing
This is a personal learning project, but I'm open to suggestions!

## 📝 License
MIT License

## 🙏 Acknowledgments
Huge thanks to the open-source ML community!

---

**Status**: 🏗️ Early Development - Training from scratch in progress!

*Last updated: February 2026*
