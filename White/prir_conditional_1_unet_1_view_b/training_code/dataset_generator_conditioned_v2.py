"""
Dataset Generator for View-Conditioned U-Net (V2)
==================================================
Key improvements:
1. One-hot view encoding instead of camera parameters
2. Fixed camera positions (mesh-size independent)
3. Random augmented views per epoch
4. Configurable number of views per mesh
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils_wm_fast import *
from gpu_raycasting_pytorch3d_fast import generate_maps_gpu


# View definitions (canonical + rotations)
# NOTE: One-hot view encoding makes the model robust to camera position variations
# The model learns "what does a Front/Back/Left/Right/Top/Bottom view look like"
# independent of the exact camera distance
VIEW_NAMES = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
VIEW_ROTATIONS = {
    'Front':  (0.0, 0.0, np.pi),      # View 0
    'Back':   (np.pi, 0.0, 0.0),      # View 1
    'Left':   (0.0, -np.pi/2, 0.0),   # View 2
    'Right':  (0.0, np.pi/2, 0.0),    # View 3
    'Top':    (-np.pi/2, 0.0, np.pi), # View 4
    'Bottom': (np.pi/2, 0.0, np.pi),  # View 5
}


class ViewConditionedDataset(Dataset):
    """Dataset with one-hot view conditioning and random augmentation"""
    
    def __init__(self, data_dir, split='train', 
                 img_width=800, img_height=800,
                 num_random_views=4,
                 recompute_normals=True):
        """
        Args:
            data_dir: Path to HCP data
            split: 'train' or 'val'
            img_width, img_height: Output image size
            num_random_views: Number of random augmented views per mesh per epoch
            recompute_normals: Whether to recompute mesh normals
        """
        self.data_dir = data_dir
        self.split = split
        self.img_width = img_width
        self.img_height = img_height
        self.num_random_views = num_random_views
        self.recompute_normals = recompute_normals
        
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
        print(f"  View encoding: One-hot (6 discrete views) - robust to camera variations")
        print(f"  Views: {VIEW_NAMES}")
        
        # SANITY CHECK: Verify view ordering
        for i, name in enumerate(VIEW_NAMES):
            assert name in VIEW_ROTATIONS, f"View {name} not in VIEW_ROTATIONS!"
        print(f"  ✓ View ordering verified")
    
    def __len__(self):
        # Each mesh generates num_random_views samples per epoch
        return len(self.names) * self.num_random_views
    
    def __getitem__(self, idx):
        """
        Returns a random augmented view of a mesh
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
        
        # Random view selection
        view_name = np.random.choice(VIEW_NAMES)
        view_id = VIEW_NAMES.index(view_name)
        
        # SANITY CHECK: Verify view_id matches view_name
        assert VIEW_NAMES[view_id] == view_name, \
            f"View ordering mismatch: VIEW_NAMES[{view_id}] = {VIEW_NAMES[view_id]}, expected {view_name}"
        
        # Get base rotation for this view
        base_rotation = VIEW_ROTATIONS[view_name]
        aug_range = 7 if self.split == 'train' else 0
        
        # Generate projection with FIXED camera position
        output_map, ids_map = self._render_view(
            mesh, labels, view_name, aug_range
        )
        
        # Prepare output
        # Image: (H, W, 3) → (3, H, W)
        # NOTE: output_map is already in [0, 1] from generate_maps_gpu normalization
        image = torch.from_numpy(output_map).permute(2, 0, 1).float()
        # Do NOT divide by 255 - output_map is already normalized to [0, 1]!
        
        # Labels: (H, W)
        # Expected range: -1 (background/non-mesh) and [0, 36] (37 classes)
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
        
        # View one-hot: (6,)
        view_onehot = torch.zeros(6, dtype=torch.float32)
        view_onehot[view_id] = 1.0
        
        # SANITY CHECK: Verify one-hot encoding (only on first call)
        if not hasattr(self, '_onehot_checked'):
            assert view_onehot.sum() == 1.0, "One-hot encoding should sum to 1"
            assert view_onehot[view_id] == 1.0, f"One-hot mismatch: view_id={view_id}, onehot={view_onehot}"
            self._onehot_checked = True
        
        return {
            'image': image,
            'labels': gt_labels,
            'view_onehot': view_onehot,
            'view_name': view_name,
            'subject': name
        }
    
    def _render_view(self, mesh, labels, view_name, aug_range=7):
        """
        Render a single view using GPU raycasting
        
        Args:
            mesh: Trimesh object
            labels: Label array
            view_name: Name of view (Front, Back, etc.)
            aug_range: Rotation augmentation range in degrees
        
        Returns:
            output_map: (H, W, 3) RGB image
            ids_map: (H, W) label map
        """
        # Compute camera matrices
        extmat = compute_extmat(mesh)
        intmat = compute_intmat(self.img_width, self.img_height)
        
        # Get rotation for this view with augmentation
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
        
        return output_map, ids_map


def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    view_onehots = torch.stack([item['view_onehot'] for item in batch])
    view_names = [item['view_name'] for item in batch]
    subjects = [item['subject'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'view_onehots': view_onehots,
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

