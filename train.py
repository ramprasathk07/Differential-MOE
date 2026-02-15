import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
from model.modelargs import ModelArgs
from model.layers import Transformer
from training.trainer import Trainer
from training.data import get_dataloader
from training.utils import set_seed, setup_logging

def get_args():
    parser = argparse.ArgumentParser(description="Train Differential Transformer MoE")
    
    # Config
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to model config")
    
    # Data
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Overridden by config if not set")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp8"], help="Data type")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="differential-moe", help="WandB project name")
    parser.add_argument("--run_name", type=str, default="run-001", help="WandB run name")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency (steps)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()

def configure_optimizers(model, args):
    """
    Separate parameters for weight decay.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.RMSNorm) # Adjust if using custom norms
    
    # Loop over modules and params to separate
    # Simplified approach: Loop named parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    for pn, p in model.named_parameters():
        if p.dim() < 2: # Biases, Norms, etc usually 1D
            no_decay.add(pn)
        elif 'weight' in pn and 'norm' not in pn and 'embed' not in pn:
             decay.add(pn)
        else:
             no_decay.add(pn)

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    optimizer = optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))
    return optimizer

def main():
    args = get_args()
    set_seed(args.seed)
    
    # Setup Logger
    logger = setup_logging(os.path.join(args.output_dir, args.run_name))
    logger.info(f"Arguments: {args}")
    
    # Load Config
    if os.path.exists(args.config):
        # Override args defaults with config if necessary, or just use config for model
        # Here we load ModelArgs from file
        logger.info(f"Loading model config from {args.config}")
        from model.modelargs import load_model_args_from_yaml
        model_args = load_model_args_from_yaml(args.config)
        # Override basic training params if needed
        model_args.dtype = args.dtype
    else:
        logger.warning("Config file not found, using default ModelArgs options")
        model_args = ModelArgs()

    # Initialize Model
    logger.info("Initializing Transformer Model...")
    model = Transformer(model_args)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    if args.resume_from:
         logger.info(f"Resuming from {args.resume_from}")
         # Checkpoint loading is handled by Trainer, but we need model struct first
         pass

    # Data Loaders
    logger.info(f"Loading dataset: {args.dataset}")
    train_loader, vocab_size = get_dataloader(
        args.dataset, 
        args.split, 
        model_args.max_seq_len, 
        args.batch_size, 
        num_workers=args.num_workers
    )
    # Val loader (optional, use validation split if available)
    val_loader = None
    try:
        val_loader, _ = get_dataloader(
            args.dataset, 
            "validation", # Assuming standard split name
            model_args.max_seq_len, 
            args.batch_size
        )
    except:
        logger.warning("No validation split found/loaded. Skipping validation.")

    # Optimizer & Scheduler
    optimizer = configure_optimizers(model, args)
    
    # Scheduler: Cosine with Warmup (handled by Torch/Transformers or custom)
    # Simple CosineAnnealingWarmRestarts for now
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)

    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Start Training
    trainer.train(start_epoch=trainer.start_epoch, epochs=args.epochs)

if __name__ == "__main__":
    main()
