"""
PyTorch Lightning Model for View-Conditioned U-Net (V2 - Arbitrary Views)
===========================================================================
Improved version with continuous camera pose conditioning
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from unet_conditioned_v2 import ViewConditionedUNet
from gpu_raycasting_pytorch3d_fast import clear_mesh_cache, get_cache_info


class ViewConditionedLightningModel(pl.LightningModule):
    """Lightning wrapper for view-conditioned U-Net with arbitrary view support"""
    
    def __init__(self, num_classes=37, pose_dim=3, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ViewConditionedUNet(
            in_channels=3,
            num_classes=num_classes,
            pose_dim=pose_dim
        )
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.pose_dim = pose_dim
        
        # Class weights (ignore background=-1)
        self.register_buffer('class_weights', torch.ones(num_classes))
        
    def forward(self, x, camera_pose):
        return self.model(x, camera_pose)
    
    def training_step(self, batch, batch_idx):
        images = batch['images']  # (B, 3, H, W)
        labels = batch['labels']  # (B, H, W)
        camera_poses = batch['camera_poses']  # (B, 3)
        
        # CRITICAL: Clear cache every 200 batches to prevent OOM
        if batch_idx > 0 and batch_idx % 200 == 0 and torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            num_cleared = clear_mesh_cache()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            freed = allocated_before - allocated_after
            print(f"\n[MID-EPOCH CACHE CLEAR @ Batch {batch_idx}] Cleared {num_cleared} meshes, freed {freed:.2f}GB\n")
        
        # Memory monitoring every 100 batches
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            cache_info = get_cache_info()
            print(f"[Batch {batch_idx}] GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {cache_info['num_meshes']} meshes cached")
        
        # Forward pass
        logits = self.forward(images, camera_poses)  # (B, num_classes, H, W)
        
        # Create mask for valid pixels (labels >= 0)
        valid_mask = labels >= 0
        
        # Compute loss only on valid pixels
        loss = F.cross_entropy(
            logits, 
            labels,
            weight=self.class_weights,
            ignore_index=-1  # Ignore background
        )
        
        # Compute accuracy on valid pixels
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            valid_preds = preds[valid_mask]
            valid_labels = labels[valid_mask]
            accuracy = (valid_preds == valid_labels).float().mean()
        
        # Log metrics (step-level for progress bar only, epoch-level for CSV logger)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Print GPU memory stats at start of epoch"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            cache_info = get_cache_info()
            print(f"\n[EPOCH {self.current_epoch} START] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {cache_info['num_meshes']} meshes cached")
    
    def on_train_epoch_end(self):
        """Clear GPU cache and mesh cache at end of training epoch to prevent OOM"""
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            num_cleared = clear_mesh_cache()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            freed = allocated_before - allocated_after
            print(f"[EPOCH {self.current_epoch} END] Cleared {num_cleared} cached meshes, freed {freed:.2f}GB GPU memory")
    
    def on_validation_epoch_end(self):
        """Clear GPU cache at end of validation epoch to prevent OOM"""
        torch.cuda.empty_cache()
    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        camera_poses = batch['camera_poses']
        
        # Forward pass
        logits = self.forward(images, camera_poses)
        
        # Create mask for valid pixels
        valid_mask = labels >= 0
        
        # Compute loss
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            ignore_index=-1
        )
        
        # Compute metrics on valid pixels
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            valid_preds = preds[valid_mask]
            valid_labels = labels[valid_mask]
            accuracy = (valid_preds == valid_labels).float().mean()
            
            # Per-class metrics
            per_class_acc = []
            for class_id in range(self.num_classes):
                class_mask = valid_labels == class_id
                if class_mask.sum() > 0:
                    class_acc = (valid_preds[class_mask] == valid_labels[class_mask]).float().mean()
                    per_class_acc.append(class_acc.item())
            
            mean_class_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
        
        # Log metrics
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_mean_class_acc', mean_class_acc, on_step=False, on_epoch=True)
        
        # Return for monitoring callback
        return {
            'loss': loss,
            'images': images,
            'labels': labels,
            'preds': preds,
            'view_names': batch['view_names'],
            'subjects': batch['subjects']
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'validation_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_single_view(self, image, view_name=None, camera_pose=None):
        """
        Predict on a single image with view name or camera pose
        
        Args:
            image: (3, H, W) tensor
            view_name: str, one of predefined view names (optional)
            camera_pose: (3,) tensor or tuple of (rx, ry, rz) in radians (optional)
        
        Returns:
            pred: (H, W) prediction
            probs: (num_classes, H, W) probabilities
        """
        from dataset_generator_conditioned_v2 import DEFAULT_VIEW_ROTATIONS, normalize_rotation_angles
        
        self.eval()
        with torch.no_grad():
            # Convert view_name to camera_pose if provided
            if camera_pose is None:
                if view_name is None:
                    raise ValueError("Must provide either view_name or camera_pose")
                if view_name not in DEFAULT_VIEW_ROTATIONS:
                    raise ValueError(f"Unknown view_name: {view_name}")
                rotation = DEFAULT_VIEW_ROTATIONS[view_name]
                camera_pose = normalize_rotation_angles(*rotation)
                camera_pose = torch.from_numpy(camera_pose).float()
            
            # Ensure camera_pose is on correct device
            if isinstance(camera_pose, (tuple, list)):
                camera_pose = torch.tensor(camera_pose, dtype=torch.float32)
            camera_pose = camera_pose.to(image.device).unsqueeze(0)  # (1, 3)
            
            # Add batch dimension
            image_batch = image.unsqueeze(0)  # (1, 3, H, W)
            
            # Forward pass
            logits = self.forward(image_batch, camera_pose)  # (1, num_classes, H, W)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            
            return pred.squeeze(0), probs.squeeze(0)


if __name__ == '__main__':
    # Test the model
    model = ViewConditionedLightningModel(num_classes=37, pose_dim=3)
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 800, 800)
    camera_poses = torch.randn(batch_size, 3) * 0.5  # Random poses
    
    logits = model(images, camera_poses)
    print(f"Output logits shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


