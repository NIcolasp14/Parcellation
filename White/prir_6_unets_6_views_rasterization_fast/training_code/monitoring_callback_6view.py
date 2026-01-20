#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoring Callback for 6-View Training (One Model Per View)

Similar to camera-conditioned monitoring, but adapted for single-view models.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from datetime import datetime
import json
from pathlib import Path


class TrainingMonitorSingleView(Callback):
    """
    Monitoring callback for single-view training (6 separate models).
    
    Similar to camera-conditioned monitor, but simpler since each model
    only sees one view.
    """
    
    def __init__(self, monitoring_dir, view_name, log_every_n_epochs=50, save_samples=True):
        super().__init__()
        self.monitoring_dir = Path(monitoring_dir)
        self.view_name = view_name
        self.log_every_n_epochs = log_every_n_epochs
        self.save_samples = save_samples
        
        # Create directory structure
        self.setup_directories()
        
        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        
        # Open log file
        self.log_file = open(self.monitoring_dir / "training_log.txt", "w", buffering=1)
        self.write_log("="*80)
        self.write_log(f"Training Monitor Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_log(f"View: {self.view_name}")
        self.write_log("="*80)
        
    def setup_directories(self):
        """Create organized directory structure for monitoring."""
        dirs = [
            self.monitoring_dir,
            self.monitoring_dir / "loss_curves",
            self.monitoring_dir / "sample_predictions",
            self.monitoring_dir / "input_visualizations",
            self.monitoring_dir / "metrics"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def write_log(self, message):
        """Write message to log file with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}\n"
        self.log_file.write(log_msg)
        print(f"[{self.view_name}] {message}")
    
    def on_train_start(self, trainer, pl_module):
        """Log training start."""
        self.write_log(f"Training started on device: {pl_module.device}")
        self.write_log(f"Total parameters: {sum(p.numel() for p in pl_module.parameters()) / 1e6:.2f}M")
        self.write_log(f"Trainable parameters: {sum(p.numel() for p in pl_module.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        # Save model architecture
        with open(self.monitoring_dir / "model_architecture.txt", "w") as f:
            f.write(str(pl_module))
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Log epoch start."""
        if trainer.current_epoch % 10 == 0:
            self.write_log(f"Epoch {trainer.current_epoch}/{trainer.max_epochs} started")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log and visualize at epoch end."""
        current_epoch = trainer.current_epoch
        
        # Get metrics
        train_loss = trainer.callback_metrics.get('training_loss_epoch', None)
        val_loss = trainer.callback_metrics.get('validation_loss', None)
        
        # Track losses
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
            self.epochs.append(current_epoch)
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
        
        # Track learning rate
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            self.learning_rates.append(current_lr)
        
        # Log every N epochs
        if current_epoch % self.log_every_n_epochs == 0:
            self.write_log(f"\n{'='*80}")
            self.write_log(f"EPOCH {current_epoch} SUMMARY")
            self.write_log(f"{'='*80}")
            
            if train_loss is not None:
                self.write_log(f"Training Loss: {train_loss:.6f}")
            if val_loss is not None:
                self.write_log(f"Validation Loss: {val_loss:.6f}")
            if self.learning_rates:
                self.write_log(f"Learning Rate: {self.learning_rates[-1]:.8f}")
            
            # GPU memory
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                self.write_log(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
            # Create visualizations
            self.plot_loss_curves(current_epoch)
            self.plot_learning_rate(current_epoch)
            
            # Save metrics JSON
            self.save_metrics(current_epoch)
            
            self.write_log(f"{'='*80}\n")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Save sample predictions during validation."""
        if not self.save_samples:
            return
        
        current_epoch = trainer.current_epoch
        
        # Save samples every N epochs and only first batch
        if current_epoch % self.log_every_n_epochs == 0 and batch_idx == 0:
            try:
                data, labels = batch
                
                # Get predictions
                with torch.no_grad():
                    outputs = pl_module(data)
                    predictions = torch.argmax(outputs, dim=1)
                    labels_gt = torch.argmax(labels, dim=1)
                
                # Save first 4 samples
                num_samples = min(4, data.shape[0])
                self.visualize_samples(
                    data[:num_samples],
                    labels_gt[:num_samples],
                    predictions[:num_samples],
                    current_epoch
                )
            except Exception as e:
                self.write_log(f"WARNING: Failed to save sample predictions: {str(e)}")
    
    def plot_loss_curves(self, epoch):
        """Plot training and validation loss curves."""
        if len(self.train_losses) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(self.train_losses) > 0:
            ax.plot(self.epochs, self.train_losses, label='Training Loss', marker='o', markersize=3)
        if len(self.val_losses) > 0:
            ax.plot(self.epochs, self.val_losses, label='Validation Loss', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.view_name} - Loss Curves (Epoch {epoch})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.monitoring_dir / "loss_curves" / f"loss_epoch_{epoch:04d}.png", dpi=150)
        plt.close()
    
    def plot_learning_rate(self, epoch):
        """Plot learning rate schedule."""
        if len(self.learning_rates) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.learning_rates, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{self.view_name} - Learning Rate Schedule (Epoch {epoch})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.monitoring_dir / "metrics" / f"lr_epoch_{epoch:04d}.png", dpi=150)
        plt.close()
    
    def visualize_samples(self, images, labels_gt, predictions, epoch):
        """Visualize input images, ground truth, and predictions."""
        num_samples = images.shape[0]
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Input image (first channel only for visualization)
            img = images[i, 0].cpu().numpy()
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Input: {self.view_name}')
            axes[i, 0].axis('off')
            
            # Ground truth
            gt = labels_gt[i].cpu().numpy()
            axes[i, 1].imshow(gt, cmap='tab20')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            pred = predictions[i].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='tab20')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{self.view_name} - Sample Predictions (Epoch {epoch})', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.monitoring_dir / "sample_predictions" / f"samples_epoch_{epoch:04d}.png", dpi=150)
        plt.close()
    
    def save_metrics(self, epoch):
        """Save metrics to JSON file."""
        metrics = {
            'view': self.view_name,
            'epoch': epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs,
            'learning_rates': self.learning_rates,
            'latest_train_loss': self.train_losses[-1] if self.train_losses else None,
            'latest_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
        }
        
        with open(self.monitoring_dir / "metrics" / f"metrics_epoch_{epoch:04d}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also save latest as current metrics
        with open(self.monitoring_dir / "current_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def on_train_end(self, trainer, pl_module):
        """Final logging and summary."""
        self.write_log("\n" + "="*80)
        self.write_log("TRAINING COMPLETED")
        self.write_log("="*80)
        
        if self.train_losses:
            self.write_log(f"Final Training Loss: {self.train_losses[-1]:.6f}")
            self.write_log(f"Best Training Loss: {min(self.train_losses):.6f}")
        
        if self.val_losses:
            self.write_log(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
            self.write_log(f"Best Validation Loss: {min(self.val_losses):.6f}")
        
        self.write_log(f"Total Epochs: {trainer.current_epoch + 1}")
        self.write_log(f"Monitoring directory: {self.monitoring_dir}")
        
        # Create final summary plots
        if len(self.train_losses) > 0:
            self.plot_loss_curves(trainer.current_epoch)
            self.plot_learning_rate(trainer.current_epoch)
        
        self.write_log("="*80)
        self.log_file.close()
    
    def on_exception(self, trainer, pl_module, exception):
        """Log exceptions."""
        self.write_log(f"\n{'!'*80}")
        self.write_log(f"EXCEPTION OCCURRED: {type(exception).__name__}")
        self.write_log(f"Message: {str(exception)}")
        self.write_log(f"{'!'*80}\n")











