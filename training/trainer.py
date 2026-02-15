import os
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from .utils import AverageMeter, get_lr, save_checkpoint, setup_logging

class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader, 
                 optimizer, 
                 scheduler, 
                 args, 
                 device='cuda'):
        
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.device = device
        
        # Output Directories
        self.output_dir = os.path.join(args.output_dir, args.run_name)
        self.logger = setup_logging(self.output_dir)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))
        
        # WandB
        if args.use_wandb:
            wandb.init(project=args.project_name, name=args.run_name, config=vars(args))
            wandb.watch(self.model, log='all', log_freq=100)
            
        # Mixed Precision
        self.scaler = GradScaler(enabled=(args.dtype == 'bf16' or args.use_amp))
        # Note: If model uses internal FP8 casting, we generally keep AMP enabled for BF16/FP32 
        # wrapper ops, but ensure model handles its specific FP8 logic.
        
        self.global_step = 0
        self.start_epoch = 0
        
        # Metrics
        self.best_val_loss = float('inf')

    def load_checkpoint(self, path):
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint['global_step']
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.args.use_wandb:
            # Restore wandb run if id is saved? usually handled by wandb internally or just new run.
            pass

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args
        }
        filename = f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint(state, is_best, self.output_dir, filename)
        self.logger.info(f"Saved checkpoint: {filename}")

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        batch_times = AverageMeter()
        start_time = time.time()
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", disable=None)
        
        # Zero grad at start of epoch
        self.optimizer.zero_grad()
        
        accumulation_steps = self.args.accumulate_grad_batches
        
        for step, batch in enumerate(pbar):
            step_start = time.time()
            
            # Data to device
            input_ids = batch[:, :-1].to(self.device, non_blocking=True)
            labels = batch[:, 1:].to(self.device, non_blocking=True)
            
            # Forward
            with autocast(enabled=(self.args.dtype == 'bf16' or self.args.use_amp), dtype=torch.bfloat16):
                logits = self.model(input_ids)
                # Ensure logits are float32 for loss
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
            
            # Normalize Loss for Gradient Accumulation
            loss = loss / accumulation_steps
                
            # Backward
            self.scaler.scale(loss).backward()
            
            # Accumulate and Step
            if (step + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update Metrics (multiply loss back for logging)
            losses.update(loss.item() * accumulation_steps, input_ids.size(0))
            batch_time = time.time() - step_start
            batch_times.update(batch_time)
            
            # Logging (on step/update)
            # Log every log_freq updates (global_step)
            # Or every log_freq * accumulation_steps iterations
            if self.global_step > 0 and (step + 1) % accumulation_steps == 0 and self.global_step % self.args.log_freq == 0:
                cur_lr = get_lr(self.optimizer)
                try:
                    perplexity = math.exp(losses.avg)
                except OverflowError:
                    perplexity = float('inf')
                
                # Approx tokens/sec
                tokens_per_sec = (input_ids.numel() * accumulation_steps) / (batch_times.avg * accumulation_steps)
                
                logs = {
                    'step': self.global_step,
                    'train_loss': losses.avg,
                    'perplexity': perplexity,
                    'lr': cur_lr,
                    'tokens_per_sec': tokens_per_sec
                }
                
                self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                self.writer.add_scalar('Train/Perplexity', perplexity, self.global_step)
                self.writer.add_scalar('Train/LR', cur_lr, self.global_step)
                
                if self.args.use_wandb:
                    wandb.log(logs)
                    
                pbar.set_postfix({'loss': f"{losses.avg:.4f}", 'ppl': f"{perplexity:.2f}"})

        return losses.avg

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        
        if not self.val_dataloader:
            return 0.0

        pbar = tqdm(self.val_dataloader, desc=f"Eval Epoch {epoch}")
        
        for batch in pbar:
            input_ids = batch[:, :-1].to(self.device, non_blocking=True)
            labels = batch[:, 1:].to(self.device, non_blocking=True)
            
            with autocast(enabled=(self.args.dtype == 'bf16' or self.args.use_amp), dtype=torch.bfloat16):
                logits = self.model(input_ids)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
            
            losses.update(loss.item(), input_ids.size(0))
        
        perplexity = math.exp(losses.avg)
        
        self.writer.add_scalar('Val/Loss', losses.avg, epoch)
        self.writer.add_scalar('Val/Perplexity', perplexity, epoch)
        
        if self.args.use_wandb:
            wandb.log({'val_loss': losses.avg, 'val_perplexity': perplexity, 'epoch': epoch})
            
        self.logger.info(f"Validation Epoch {epoch}: Loss {losses.avg:.4f} | PPL {perplexity:.2f}")
        return losses.avg

    def train(self, start_epoch=0, epochs=1):
        self.logger.info("Starting training...")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(epoch, is_best)
            
        self.logger.info("Training complete.")
        self.writer.close()
