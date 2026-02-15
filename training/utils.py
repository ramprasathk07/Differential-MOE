import logging
import os
import torch
import torch.nn as nn
import time
from packaging import version
import random
import numpy as np

def setup_logging(output_dir):
    """
    Sets up logging to console and file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'training.log'))
        ]
    )
    return logging.getLogger('DiffMoE')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(state, os.path.join(output_dir, filename))
    if is_best:
        torch.save(state, os.path.join(output_dir, 'model_best.pth'))
