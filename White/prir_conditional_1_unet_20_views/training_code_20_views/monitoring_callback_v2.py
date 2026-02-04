"""
Monitoring Callback for View-Conditioned Model (V2)
====================================================
Saves sample predictions, metrics per view type, and loss curves
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from dataset_generator_conditioned_v2 import VIEW_NAMES


class MonitoringCallback(Callback):
    """Callback to save sample predictions and per-view metrics"""
    
    def __init__(self, save_dir, num_samples=5, save_every_n_epochs=5):
        super().__init__()
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.save_every_n_epochs = save_every_n_epochs  # Save images every N epochs to reduce memory
        
        # Create subdirectories
        self.pred_dir = os.path.join(save_dir, 'validation_predictions')
        self.metrics_dir = os.path.join(save_dir, 'metrics')
        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Track per-view metrics
        self.per_view_metrics = {view: [] for view in VIEW_NAMES}
        
        # Track label distribution for debugging
        self.label_stats = []
        
        # Track loss curves
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Save sample predictions and compute per-view metrics"""
        
        # Only process first few batches for visualization, and only every N epochs
        epoch = trainer.current_epoch
        if batch_idx == 0 and epoch % self.save_every_n_epochs == 0:
            self._save_predictions(outputs, epoch)
            self._log_detailed_metrics(outputs, epoch)
        
        # Accumulate per-view metrics
        self._accumulate_metrics(outputs)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Summarize per-view metrics at end of epoch and plot loss curves"""
        epoch = trainer.current_epoch
        
        # Compute average metrics per view
        print(f"\n=== Epoch {epoch} Per-View Metrics ===")
        
        for view_name in VIEW_NAMES:
            if self.per_view_metrics[view_name]:
                avg_acc = np.mean(self.per_view_metrics[view_name])
                print(f"  {view_name:10s}: {avg_acc:.4f}")
        
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', 0.0)
        train_acc = metrics.get('train_acc', 0.0)
        val_loss = metrics.get('validation_loss', 0.0)
        val_acc = metrics.get('validation_acc', 0.0)
        
        # Store for plotting
        self.epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.train_accs.append(float(train_acc))
        self.val_accs.append(float(val_acc))
        
        # Plot loss curves every 10 epochs
        if epoch % 10 == 0 and len(self.epochs) > 1:
            self._plot_metrics()
        
        # Reset for next epoch
        self.per_view_metrics = {view: [] for view in VIEW_NAMES}
    
    def _plot_metrics(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(self.epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f'training_curves.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Loss curves saved to: {plot_path}")
    
    def _save_predictions(self, outputs, epoch):
        """Save visualization of predictions"""
        images = outputs['images'].cpu()
        labels = outputs['labels'].cpu()
        preds = outputs['preds'].cpu()
        view_names = outputs['view_names']
        subjects = outputs['subjects']
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).numpy()  # (H, W, 3)
            gt = labels[i].numpy()
            pred = preds[i].numpy()
            view_name = view_names[i]
            subject = subjects[i]
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input image
            axes[0].imshow(img)
            axes[0].set_title(f'Input\n{subject}\n{view_name}')
            axes[0].axis('off')
            
            # Ground truth
            gt_vis = gt.copy()
            gt_vis[gt == -1] = 0  # Background
            axes[1].imshow(gt_vis, cmap='tab20', vmin=0, vmax=36)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            pred_vis = pred.copy()
            pred_vis[gt == -1] = 0  # Mask background
            axes[2].imshow(pred_vis, cmap='tab20', vmin=0, vmax=36)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save
            filename = f'epoch_{epoch:04d}_{subject}_{view_name}.png'
            filepath = os.path.join(self.pred_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
    
    def _log_detailed_metrics(self, outputs, epoch):
        """Log detailed 2D metrics for debugging"""
        labels = outputs['labels'].cpu()
        preds = outputs['preds'].cpu()
        view_names = outputs['view_names']
        
        # Convert to numpy
        labels_np = labels.numpy()
        preds_np = preds.numpy()
        
        # Count valid pixels
        valid_mask = labels_np >= 0
        total_valid = np.sum(valid_mask)
        
        if total_valid == 0:
            print(f"⚠️  WARNING: No valid pixels in batch! All labels are -1 (background)")
            return
        
        # Compute overall metrics
        correct = np.sum(preds_np[valid_mask] == labels_np[valid_mask])
        accuracy = correct / total_valid
        
        # Count unique classes
        unique_gt = np.unique(labels_np[valid_mask])
        unique_pred = np.unique(preds_np[valid_mask])
        
        # Check for background collapse
        bg_ratio_gt = np.sum(labels_np == -1) / labels_np.size
        bg_ratio_pred = np.sum(preds_np == -1) / preds_np.size
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} - 2D VALIDATION METRICS (FIRST BATCH)")
        print(f"{'='*80}")
        print(f"Pixel Accuracy: {accuracy:.4f} ({correct}/{total_valid})")
        print(f"Unique classes in GT: {len(unique_gt)} - {unique_gt[:10]}")
        print(f"Unique classes in Pred: {len(unique_pred)} - {unique_pred[:10]}")
        print(f"Background ratio GT: {bg_ratio_gt:.3f}, Pred: {bg_ratio_pred:.3f}")
        
        # Per-view breakdown
        print(f"\nPer-View Breakdown:")
        batch_size = labels.shape[0]
        for i in range(min(batch_size, 5)):
            gt = labels_np[i]
            pred = preds_np[i]
            view = view_names[i]
            
            valid = gt >= 0
            if valid.sum() > 0:
                acc = np.sum(pred[valid] == gt[valid]) / valid.sum()
                n_classes_gt = len(np.unique(gt[valid]))
                n_classes_pred = len(np.unique(pred[valid]))
                print(f"  {view:10s}: acc={acc:.3f}, GT_classes={n_classes_gt}, Pred_classes={n_classes_pred}")
            else:
                print(f"  {view:10s}: NO VALID PIXELS!")
        
        print(f"{'='*80}\n")
        
        # Store for later analysis
        self.label_stats.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'n_classes_gt': len(unique_gt),
            'n_classes_pred': len(unique_pred),
            'bg_ratio_pred': bg_ratio_pred
        })
    
    def _accumulate_metrics(self, outputs):
        """Accumulate per-view accuracy metrics"""
        labels = outputs['labels'].cpu()
        preds = outputs['preds'].cpu()
        view_names = outputs['view_names']
        
        batch_size = labels.shape[0]
        
        for i in range(batch_size):
            gt = labels[i]
            pred = preds[i]
            view_name = view_names[i]
            
            # Compute accuracy on valid pixels
            valid_mask = gt >= 0
            if valid_mask.sum() > 0:
                accuracy = (pred[valid_mask] == gt[valid_mask]).float().mean().item()
                self.per_view_metrics[view_name].append(accuracy)


class MetricsLogger(Callback):
    """Log detailed metrics to file"""
    
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        
        # Create file with header
        with open(self.log_file, 'w') as f:
            f.write('Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_MeanClassAcc,LR\n')
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics at end of each validation epoch"""
        epoch = trainer.current_epoch
        
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        
        train_loss = metrics.get('train_loss_epoch', 0.0)
        train_acc = metrics.get('train_acc', 0.0)
        val_loss = metrics.get('validation_loss', 0.0)
        val_acc = metrics.get('validation_acc', 0.0)
        val_mean_class_acc = metrics.get('validation_mean_class_acc', 0.0)
        
        # Get learning rate
        lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},'
                   f'{val_acc:.6f},{val_mean_class_acc:.6f},{lr:.8f}\n')
        
        # Print summary every epoch
        print(f"\n=== Epoch {epoch} Summary ===")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, mean_class_acc={val_mean_class_acc:.4f}")
        print(f"  LR:    {lr:.8f}")


if __name__ == '__main__':
    print("Monitoring callbacks created successfully")
    print(f"View names: {VIEW_NAMES}")

