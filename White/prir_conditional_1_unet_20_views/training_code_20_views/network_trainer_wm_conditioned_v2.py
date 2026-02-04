"""
Training Script for View-Conditioned U-Net (V2)
================================================
Improved version with one-hot view conditioning and configurable random views
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset_generator_conditioned_v2 import ViewConditionedDataset, collate_fn
from lightning_model_conditioned_v2 import ViewConditionedLightningModel
from monitoring_callback_v2 import MonitoringCallback, MetricsLogger


def main():
    parser = argparse.ArgumentParser(description='Train view-conditioned U-Net (V2)')
    
    # Data
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to HCP aligned data')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save model checkpoints')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_random_views', type=int, default=4,
                       help='Number of random augmented views per mesh per epoch (1-20)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='Number of epochs')
    
    # Image parameters
    parser.add_argument('--img_width', type=int, default=800,
                       help='Image width')
    parser.add_argument('--img_height', type=int, default=800,
                       help='Image height')
    
    # Mesh processing
    parser.add_argument('--recompute_normals', action='store_true',
                       help='Recompute mesh normals (recommended)')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 for GPU raycasting in main process)')
    
    args = parser.parse_args()
    
    # Validate num_random_views
    if not 1 <= args.num_random_views <= 20:
        raise ValueError(f"num_random_views must be 1-20, got {args.num_random_views}")
    
    # Print configuration
    print("=" * 80)
    print("VIEW-CONDITIONED U-NET TRAINING (V2)")
    print("=" * 80)
    print(f"Data: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random views per mesh: {args.num_random_views}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Image size: {args.img_width}x{args.img_height}")
    print(f"Recompute normals: {args.recompute_normals}")
    print(f"Data workers: {args.num_workers}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ViewConditionedDataset(
        data_dir=args.input_path,
        split='train',
        img_width=args.img_width,
        img_height=args.img_height,
        num_random_views=args.num_random_views,
        recompute_normals=args.recompute_normals
    )
    
    val_dataset = ViewConditionedDataset(
        data_dir=args.input_path,
        split='val',
        img_width=args.img_width,
        img_height=args.img_height,
        num_random_views=args.num_random_views,
        recompute_normals=args.recompute_normals
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.num_workers > 0 else False
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Create model
    print("\nCreating model...")
    model = ViewConditionedLightningModel(
        num_classes=37,
        num_views=20,
        learning_rate=args.learning_rate
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path,
        filename='conditioned_v2-epoch={epoch:04d}-validation_loss={validation_loss:.2f}',
        save_top_k=3,
        monitor='validation_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    monitoring_callback = MonitoringCallback(
        save_dir=os.path.join(args.output_path, 'monitoring'),
        num_samples=5
    )
    
    metrics_logger = MetricsLogger(
        save_dir=args.output_path
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            monitoring_callback,
            metrics_logger
        ],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        deterministic=False,  # For performance
        precision='16-mixed'  # Mixed precision for speed
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {args.output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
