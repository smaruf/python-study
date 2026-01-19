"""
Deep Learning Utilities

This module provides common utilities used across different levels
of the deep learning learning path.

Author: Python Study Repository
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Timer:
    """Simple timer context manager for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.4f} seconds")


def plot_images(images: np.ndarray, labels: np.ndarray = None, 
                predictions: np.ndarray = None, n_cols: int = 5,
                figsize: Tuple[int, int] = (15, 6), title: str = None):
    """
    Plot a grid of images with optional labels and predictions.
    
    Args:
        images: Array of images
        labels: True labels (optional)
        predictions: Predicted labels (optional)
        n_cols: Number of columns in grid
        figsize: Figure size
        title: Overall title
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_images > 1 else [axes]
    
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
        
        if labels is not None and predictions is not None:
            color = 'green' if labels[i] == predictions[i] else 'red'
            axes[i].set_title(f'True: {labels[i]}, Pred: {predictions[i]}', color=color)
        elif labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
        
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer = None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint loaded from {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: How many epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: True labels
        
    Returns:
        Accuracy as percentage
    """
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, ...)
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 70)
