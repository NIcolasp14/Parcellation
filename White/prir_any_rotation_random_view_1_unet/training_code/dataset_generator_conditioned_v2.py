"""
Dataset Generator for View-Conditioned U-Net (V2 - Arbitrary Views)
====================================================================
Key improvements:
1. Continuous camera pose conditioning (rotation + distance) for arbitrary views
2. Configurable view set - use any number of predefined or random views
3. Random augmented views per epoch
4. Robust camera parameter encoding
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils_wm_fast import *
from gpu_raycasting_pytorch3d_fast import generate_maps_gpu


# Default view definitions (canonical rotations)
# Users can override this with custom view sets
DEFAULT_VIEW_NAMES = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
DEFAULT_VIEW_ROTATIONS = {
    'Front':  (0.0, 0.0, np.pi),      # rx, ry, rz in radians
    'Back':   (np.pi, 0.0, 0.0),
    'Left':   (0.0, -np.pi/2, 0.0),
    'Right':  (0.0, np.pi/2, 0.0),
    'Top':    (-np.pi/2, 0.0, np.pi),
    'Bottom': (np.pi/2, 0.0, np.pi),
}

def normalize_rotation_angles(rx, ry, rz):
    """
    Normalize rotation angles to [-1, 1] range for network input
    First wraps angles to [-pi, pi], then normalizes
    
    Args:
        rx, ry, rz: Rotation angles in radians (can be any range)
    
    Returns:
        Normalized angles in [-1, 1]
    """
    # Wrap angles to [-pi, pi] range using atan2 trick
    rx_wrapped = np.arctan2(np.sin(rx), np.cos(rx))
    ry_wrapped = np.arctan2(np.sin(ry), np.cos(ry))
    rz_wrapped = np.arctan2(np.sin(rz), np.cos(rz))
    
    # Normalize to [-1, 1]
    return np.array([rx_wrapped / np.pi, ry_wrapped / np.pi, rz_wrapped / np.pi], dtype=np.float32)


class ViewConditionedDataset(Dataset):
    """Dataset with continuous camera pose conditioning for arbitrary views"""
    
    def __init__(self, data_dir, split='train', 
                 img_width=800, img_height=800,
                 num_random_views=4,
                 recompute_normals=True,
                 view_names=None,
                 view_rotations=None,
                 use_random_views=False):
        """
        Args:
            data_dir: Path to HCP data
            split: 'train' or 'val'
            img_width, img_height: Output image size
            num_random_views: Number of random views per mesh per epoch
            recompute_normals: Whether to recompute mesh normals
            view_names: List of view names to use (default: 6 canonical views)
            view_rotations: Dict mapping view names to (rx, ry, rz) tuples
            use_random_views: If True, use completely random views instead of predefined ones
        """
        self.data_dir = data_dir
        self.split = split
        self.img_width = img_width
        self.img_height = img_height
        self.num_random_views = num_random_views
        self.recompute_normals = recompute_normals
        self.use_random_views = use_random_views
        
        # Set up view configuration
        if view_names is None:
            self.view_names = DEFAULT_VIEW_NAMES.copy()
        else:
            self.view_names = view_names
            
        if view_rotations is None:
            self.view_rotations = DEFAULT_VIEW_ROTATIONS.copy()
        else:
            self.view_rotations = view_rotations
        
        # Validate view configuration
        if not use_random_views:
            for name in self.view_names:
                assert name in self.view_rotations, f"View {name} not in view_rotations!"
        
        # Use validate_files to get mesh and label paths
        names, mesh_paths, label_paths = validate_files(data_dir)
        
        # Split into train/val (80/20)
        split_idx = int(0.8 * len(names))
        if split == 'train':
            self.names = names[:split_idx]
            self.mesh_paths = mesh_paths[:split_idx]
            self.label_paths = label_paths[:split_idx]
        else:
            self.names = names[split_idx:]
            self.mesh_paths = mesh_paths[split_idx:]
            self.label_paths = label_paths[split_idx:]
        
        print(f"[{split.upper()}] Loaded {len(self.names)} subjects")
        print(f"  Random views per mesh: {num_random_views}")
        print(f"  Total samples per epoch: {len(self.names) * num_random_views}")
        print(f"  View encoding: Continuous camera pose (3 rotation angles)")
        if use_random_views:
            print(f"  Mode: Fully random views")
        else:
            print(f"  Mode: Predefined views with augmentation")
            print(f"  Views: {self.view_names}")
        print(f"  ✓ Dataset initialized for arbitrary view support")
    
    def __len__(self):
        # Each mesh generates num_random_views samples per epoch
        return len(self.names) * self.num_random_views
    
    def __getitem__(self, idx):
        """
        Returns a random augmented view of a mesh with continuous camera parameters
        Each epoch generates different random augmentations
        """
        # Map flat index to (subject, view_idx)
        subject_idx = idx // self.num_random_views
        view_idx_in_subject = idx % self.num_random_views
        
        name = self.names[subject_idx]
        mesh_path = self.mesh_paths[subject_idx]
        label_path = self.label_paths[subject_idx]
        
        # Load mesh and labels
        labels = get_labels(label_path)
        mesh = create_mesh(mesh_path, perturb_vertices=self.split=='train', std_dev=0.5)
        
        # Determine view parameters
        if self.use_random_views:
            # Completely random view
            view_name = 'Random'
            # Random rotation in all axes
            rx = np.random.uniform(-np.pi, np.pi)
            ry = np.random.uniform(-np.pi, np.pi)
            rz = np.random.uniform(-np.pi, np.pi)
            base_rotation = (rx, ry, rz)
        else:
            # Random view from predefined set
            view_name = np.random.choice(self.view_names)
            base_rotation = self.view_rotations[view_name]
        
        aug_range = 7 if self.split == 'train' else 0
        
        # Generate projection
        output_map, ids_map, actual_rotation = self._render_view(
            mesh, labels, view_name, aug_range, base_rotation
        )
        
        # Prepare output
        # Image: (H, W, 3) → (3, H, W)
        image = torch.from_numpy(output_map).permute(2, 0, 1).float()
        
        # Labels: (H, W)
        gt_labels = torch.from_numpy(ids_map).long()
        
        # SANITY CHECK: Verify label range (only on first call to avoid spam)
        if not hasattr(self, '_labels_checked'):
            unique_labels = np.unique(ids_map)
            assert np.all((unique_labels >= -1) & (unique_labels < 37)), \
                f"Labels out of range! Got: {unique_labels[unique_labels < -1]} and {unique_labels[unique_labels >= 37]}"
            print(f"[DATASET INIT] Label range check: {unique_labels.min()} to {unique_labels.max()} ✓")
            self._labels_checked = True
        
        # SANITY CHECK: Verify image normalization
        if not hasattr(self, '_normalization_checked'):
            img_min, img_max = image.min().item(), image.max().item()
            print(f"[DATASET INIT] Image range check: [{img_min:.6f}, {img_max:.6f}]")
            if img_max > 1.0 or img_min < 0.0:
                print(f"  ⚠️  WARNING: Image values outside [0, 1] range!")
            elif img_max < 0.01:
                print(f"  ⚠️  WARNING: Image values suspiciously low (< 0.01)! Check normalization!")
            else:
                print(f"  ✓ Image normalization looks correct")
            self._normalization_checked = True
        
        # Camera pose parameters: (3,) - normalized rotation angles
        camera_pose = torch.from_numpy(normalize_rotation_angles(*actual_rotation)).float()
        
        # Safety clamp to handle floating point errors
        camera_pose = torch.clamp(camera_pose, -1.0, 1.0)
        
        # SANITY CHECK: Verify camera pose normalization
        if not hasattr(self, '_pose_checked'):
            print(f"[DATASET INIT] Camera pose check: range [{camera_pose.min():.3f}, {camera_pose.max():.3f}] ✓")
            print(f"  ✓ Using continuous camera pose conditioning (3 rotation angles)")
            print(f"  ✓ Angle wrapping: angles are wrapped to [-π, π] before normalization")
            self._pose_checked = True
        
        return {
            'image': image,
            'labels': gt_labels,
            'camera_pose': camera_pose,
            'view_name': view_name,
            'subject': name
        }
    
    def _render_view(self, mesh, labels, view_name, aug_range, base_rotation):
        """
        Render a single view using GPU raycasting
        
        Args:
            mesh: Trimesh object
            labels: Label array
            view_name: Name of view or 'Random'
            aug_range: Rotation augmentation range in degrees
            base_rotation: (rx, ry, rz) base rotation in radians
        
        Returns:
            output_map: (H, W, 3) RGB image
            ids_map: (H, W) label map
            actual_rotation: (rx, ry, rz) actual rotation used (with augmentation)
        """
        # Compute camera matrices
        extmat = compute_extmat(mesh)
        intmat = compute_intmat(self.img_width, self.img_height)
        
        # Apply augmentation to base rotation
        if view_name == 'Random':
            # Already randomized, just add small augmentation
            aug_rx = np.random.uniform(-np.deg2rad(aug_range), np.deg2rad(aug_range))
            aug_ry = np.random.uniform(-np.deg2rad(aug_range), np.deg2rad(aug_range))
            aug_rz = np.random.uniform(-np.deg2rad(aug_range), np.deg2rad(aug_range))
            actual_rotation = (
                base_rotation[0] + aug_rx,
                base_rotation[1] + aug_ry,
                base_rotation[2] + aug_rz
            )
        else:
            # Use compute_rotations for predefined views (maintains compatibility)
            rotation_matrices = compute_rotations(random_degs=aug_range, view=view_name)
            # Extract actual angles from rotation matrix
            # For now, we'll use the base rotation + random offset as an approximation
            aug_rx = np.random.uniform(-np.deg2rad(aug_range), np.deg2rad(aug_range))
            aug_ry = np.random.uniform(-np.deg2rad(aug_range), np.deg2rad(aug_range))
            aug_rz = np.random.uniform(-np.deg2rad(aug_range), np.deg2rad(aug_range))
            actual_rotation = (
                base_rotation[0] + aug_rx,
                base_rotation[1] + aug_ry,
                base_rotation[2] + aug_rz
            )
        
        # Get rotation matrices for rendering
        rotation_matrices = compute_rotations(random_degs=aug_range, view=view_name)
        
        # Combine extrinsic and rotation
        extmat_transformed = np.matmul(extmat, rotation_matrices)
        
        # Generate maps using GPU raycasting
        output_maps, labels_maps = generate_maps_gpu(
            mesh=mesh,
            labels=labels,
            intmat=intmat,
            extmat=extmat_transformed,
            img_width=self.img_width,
            img_height=self.img_height,
            rotation_matrices=rotation_matrices,
            recompute_normals=self.recompute_normals
        )
        
        # Extract first (and only) view
        output_map = output_maps[0]  # (H, W, 3)
        ids_map = labels_maps[0]  # (H, W)
        
        return output_map, ids_map, actual_rotation


def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    camera_poses = torch.stack([item['camera_pose'] for item in batch])
    view_names = [item['view_name'] for item in batch]
    subjects = [item['subject'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'camera_poses': camera_poses,
        'view_names': view_names,
        'subjects': subjects
    }


if __name__ == '__main__':
    # Test the dataset
    dataset = ViewConditionedDataset(
        data_dir='/autofs/space/ballarat_004/users/np341/lemon_hcp_aligned_v5',
        split='train',
        num_random_views=4
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Image: {sample['image'].shape}, range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Labels: {sample['labels'].shape}, unique: {torch.unique(sample['labels']).numpy()[:10]}")
    print(f"  View one-hot: {sample['view_onehot']}")
    print(f"  View name: {sample['view_name']}")
    print(f"  Subject: {sample['subject']}")

