"""
Test Script for View-Conditioned Model V2 (Arbitrary Views)
============================================================
Uses continuous camera pose encoding for arbitrary views
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add directories to path - ORDER MATTERS! (most specific first)
sys.path.insert(0, '/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/training_code')
sys.path.insert(0, '/autofs/space/ballarat_004/users/np341/PRIR_Code/Predict')

from lightning_model_conditioned_v2 import ViewConditionedLightningModel
from dataset_generator_conditioned_v2 import DEFAULT_VIEW_NAMES, DEFAULT_VIEW_ROTATIONS, normalize_rotation_angles
from utils_wm_fast import create_mesh, get_labels
from compute_output_maps_pytorch3d import compute_output_maps_pytorch3d
from collections import defaultdict
import nibabel as nib
import scipy.io


# NOTE: Camera position is computed per mesh (mesh-size dependent)
# We use continuous camera pose parameters, which are normalized and fed to the network


def build_adjacency_from_faces(faces):
    """Build vertex adjacency list from mesh faces"""
    adjacency = defaultdict(set)
    for face in faces:
        v0, v1, v2 = face
        adjacency[v0].update([v1, v2])
        adjacency[v1].update([v0, v2])
        adjacency[v2].update([v0, v1])
    return adjacency


def smooth_vertex_probabilities(P, adjacency, n_iter=20, alpha=0.75):
    """Iterative neighborhood smoothing"""
    NL, NV = P.shape
    P_smooth = P.copy()
    
    for iteration in range(n_iter):
        if iteration % 5 == 0:
            print(f"    Smoothing iteration {iteration}/{n_iter}...", flush=True)
        P_new = P_smooth.copy()
        
        for v in range(NV):
            if v not in adjacency or len(adjacency[v]) == 0:
                continue
            
            neighbors = list(adjacency[v])
            n_neighbors = len(neighbors)
            neighbor_sum = np.sum(P_smooth[:, neighbors], axis=1)
            P_new[:, v] = alpha * P_smooth[:, v] + (1 - alpha) * neighbor_sum / n_neighbors
        
        P_smooth = P_new
    
    # Normalize AFTER all smoothing iterations
    P_normalized = P_smooth / np.sum(P_smooth, axis=0, keepdims=True)
    
    return P_normalized


def apply_mrf_simple(P_normalized, adjacency, beta=0.1, n_iterations=5):
    """Apply simple iterative MRF refinement (ICM)"""
    NL, NV = P_normalized.shape
    labels = np.argmax(P_normalized, axis=0)
    
    # Data term: log probabilities
    eps = 1e-10
    P_normalized = np.nan_to_num(P_normalized, nan=eps, posinf=1.0, neginf=eps)
    P_normalized = np.clip(P_normalized, eps, 1.0)
    log_prob = np.log(P_normalized + eps)
    
    # Iterative refinement
    for iteration in range(n_iterations):
        print(f"    MRF iteration {iteration+1}/{n_iterations}...", flush=True)
        changed = 0
        
        for v in range(NV):
            if v not in adjacency or len(adjacency[v]) == 0:
                continue
            
            neighbors = list(adjacency[v])
            neighbor_labels = labels[neighbors]
            
            best_label = labels[v]
            best_energy = -np.inf
            
            for label in range(NL):
                data_term = log_prob[label, v]
                smoothness_term = beta * np.sum(neighbor_labels == label) / len(neighbors)
                energy = data_term + smoothness_term
                
                if energy > best_energy:
                    best_energy = energy
                    best_label = label
            
            if labels[v] != best_label:
                labels[v] = best_label
                changed += 1
        
        if changed == 0:
            break
    
    return labels


def compute_dice_score(pred, gt):
    """Compute per-label Dice scores"""
    unique_labels = np.unique(np.concatenate([pred, gt]))
    dice_scores = {}
    
    print(f"    Dice calculation debug:")
    print(f"      Pred labels: min={pred.min()}, max={pred.max()}, unique={len(np.unique(pred))}")
    print(f"      GT labels:   min={gt.min()}, max={gt.max()}, unique={len(np.unique(gt))}")
    
    for label in unique_labels:
        if label < 0:
            continue
        
        pred_mask = (pred == label)
        gt_mask = (gt == label)
        
        intersection = np.sum(pred_mask & gt_mask)
        union_sum = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union_sum == 0:
            dice_scores[label] = np.nan
        else:
            dice_scores[label] = (2.0 * intersection) / union_sum
            
    # Print a few sample dice scores
    valid_scores = [s for s in dice_scores.values() if not np.isnan(s)]
    if valid_scores:
        print(f"      Computed {len(valid_scores)} valid scores. Mean: {np.mean(valid_scores):.4f}")
    else:
        print(f"      No valid dice scores computed!")
            
    return dice_scores


def generate_predictions_v2(mesh_path, model, device='cuda:0', view_names=None):
    """
    Generate predictions using V2 model with continuous camera pose encoding
    
    Args:
        mesh_path: Path to mesh file
        model: Trained ViewConditionedLightningModel
        device: Device to use
        view_names: List of view names to use (default: all 6 canonical views)
    
    Returns:
        P: (num_classes, num_vertices) probability matrix
    """
    if view_names is None:
        view_names = DEFAULT_VIEW_NAMES
    
    # Load mesh using the same function as training
    mesh = create_mesh(mesh_path, perturb_vertices=False, std_dev=0.0)
    num_vertices = len(mesh.vertex.positions.numpy())
    num_classes = 37
    
    # Initialize probability accumulator
    P = np.zeros((num_classes, num_vertices), dtype=np.float32)
    
    num_faces = len(mesh.triangle.indices.numpy())
    print(f"  Mesh: {num_vertices} vertices, {num_faces} faces")
    print(f"  Rendering {len(view_names)} views with PyTorch3D...")
    
    # Render all 6 views at once using PyTorch3D (matches training pipeline)
    output_maps, ids_maps, weights_maps = compute_output_maps_pytorch3d(
        mesh, img_width=800, img_height=800
    )
    
    print(f"  Generating predictions for {len(view_names)} views (continuous camera pose)...")
    
    # NOTE: compute_output_maps_pytorch3d returns views in order:
    # ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']
    pytorch3d_view_order = ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']
    
    for view_idx, view_name in enumerate(pytorch3d_view_order):
        # Skip if this view is not in our requested view_names
        if view_name not in view_names:
            print(f"    ⊘ {view_name}: skipped (not in view subset)")
            continue
        
        # Get rotation for this view
        if view_name not in DEFAULT_VIEW_ROTATIONS:
            print(f"    ⊘ {view_name}: unknown view, skipping")
            continue
        
        rotation = DEFAULT_VIEW_ROTATIONS[view_name]
        
        # Get rendered data for this view
        output_map = output_maps[view_idx]  # (H, W, 3)
        ids_map = ids_maps[view_idx]  # (H, W)
        weights_map = weights_maps[view_idx]  # (H, W, 3)
        
        # Convert output_map to tensor (already in [0,1] range from compute_output_maps)
        image = torch.from_numpy(output_map).permute(2, 0, 1).float()  # (3, H, W)
        image = image.unsqueeze(0).to(device)  # (1, 3, H, W)
        
        # Create camera pose encoding (continuous)
        camera_pose_np = normalize_rotation_angles(*rotation)
        camera_pose = torch.from_numpy(camera_pose_np).float().to(device)
        camera_pose = camera_pose.unsqueeze(0)  # (1, 3)
        
        # Predict with model
        with torch.no_grad():
            logits = model.forward(image, camera_pose)  # (1, num_classes, H, W)
            probs = torch.softmax(logits, dim=1)[0]  # (num_classes, H, W)
        
        probs_np = probs.cpu().numpy()  # (num_classes, H, W)
        
        print(f"    ✓ {view_name} (pose={rotation}): shape {probs_np.shape}")
        
        # Aggregate probabilities to vertices using barycentric coordinates
        faces_np = mesh.triangle.indices.numpy()
        
        # Accumulate from this view using ids_map and weights_map
        valid_mask = ids_map >= 0
        
        # DEBUG PRINTS
        n_valid = valid_mask.sum()
        print(f"      Valid pixels: {n_valid}/{valid_mask.size} ({n_valid/valid_mask.size*100:.1f}%)")
        print(f"      Probs stats: min={probs_np.min():.4f}, max={probs_np.max():.4f}, mean={probs_np.mean():.4f}")
        
        if valid_mask.any():
            valid_face_ids = ids_map[valid_mask]
            valid_weights = weights_map[valid_mask]  # (N, 3)
            
            # Check face IDs
            if len(valid_face_ids) > 0:
                print(f"      Face IDs: min={valid_face_ids.min()}, max={valid_face_ids.max()}, num_faces={len(faces_np)}")
            
            # Correctly index the spatial dimensions - reshape and use flattened mask
            H, W = ids_map.shape
            probs_flat = probs_np.reshape(num_classes, H * W)  # (num_classes, H*W)
            valid_mask_flat = valid_mask.reshape(-1)  # (H*W,)
            valid_probs = probs_flat[:, valid_mask_flat]  # (num_classes, N)
            
            print(f"      Valid probs shape: {valid_probs.shape}")

            # For each valid pixel, add its probability to the 3 vertices of the triangle
            for face_id, weights, prob in zip(valid_face_ids, valid_weights, valid_probs.T):
                if 0 <= face_id < len(faces_np):
                    v_ids = faces_np[face_id]
                    for j in range(3):
                        P[:, v_ids[j]] += prob * weights[j]
    
    # Normalize probabilities
    vertex_sums = P.sum(axis=0, keepdims=True)
    vertex_sums = np.maximum(vertex_sums, 1e-10)  # Avoid division by zero
    P = P / vertex_sums
    
    print(f"    P stats: min={P.min():.4f}, max={P.max():.4f}, mean={P.mean():.4f}")
    print(f"    Non-zero vertices: {(P.sum(axis=0) > 1e-5).sum()}/{num_vertices}")
    
    print(f"  ✓ Aggregation complete")
    
    return P


def render_view_fixed_camera(vertices, faces, rotation_degs):
    """
    Render a view using standard camera positioning
    
    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) mesh faces
        rotation_degs: (rx, ry, rz) rotation in radians
    
    Returns:
        output_map: (H, W, 3) rendered image
        ids_map: (H, W) face IDs
        depth_map: (H, W) depth values
    """
    # Compute camera position (mesh-dependent, same as training)
    mesh_center = np.mean(vertices, axis=0)
    corner1 = np.min(vertices, axis=0)
    corner2 = np.max(vertices, axis=0)
    
    cam_x = mesh_center[0]
    cam_y = mesh_center[1]
    cam_z = -7.5 * corner1[2]  # Same formula as compute_extmat
    
    # Create camera position
    camera_position = np.array([cam_x, cam_y, cam_z])
    
    # Create rotation matrix
    rx, ry, rz = rotation_degs
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx
    
    # Create extrinsic matrix
    extmat = np.eye(4)
    extmat[:3, :3] = rotation_matrix
    extmat[:3, 3] = -rotation_matrix @ camera_position
    
    # Create dummy labels (not used for prediction)
    labels = np.zeros(vertices.shape[0], dtype=np.int32)
    
    # Render
    output_map, ids_map, depth_map = compute_output_maps_pytorch3d_gpu(
        vertices=vertices,
        faces=faces,
        labels=labels,
        extmat=extmat,
        img_width=800,
        img_height=800,
        recompute_normals=True,
        bin_size=None
    )
    
    return output_map, ids_map, depth_map


def test_subject(subject_dir, model, device='cuda:0', view_names=None):
    """Test a single subject"""
    subject_name = os.path.basename(subject_dir)
    
    print(f"\n{'='*80}")
    print(f"Subject: {subject_name}")
    print('='*80)
    
    # Paths (FreeSurfer format)
    mesh_path = os.path.join(subject_dir, 'surf', 'lh.white')
    gt_path = os.path.join(subject_dir, 'label', 'lh.aparc.annot')
    output_dir = os.path.join(subject_dir, 'pred_conditioned_v2_pytorch3d')
    
    # Check files exist
    if not os.path.exists(mesh_path):
        print(f"  ✗ Mesh not found: {mesh_path}")
        return None
    if not os.path.exists(gt_path):
        print(f"  ✗ Ground truth not found: {gt_path}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 0: Generate predictions
    print(f"  Step 0: Generating predictions with V2 model (arbitrary views)...")
    P = generate_predictions_v2(mesh_path, model, device, view_names)
    
    # Get mesh data for smoothing/MRF
    mesh = create_mesh(mesh_path, perturb_vertices=False, std_dev=0.0)
    vertices = mesh.vertex.positions.numpy()
    faces = mesh.triangle.indices.numpy()
    
    # Get initial prediction
    # CRITICAL FIX: Based on diagnostic test, model uses Identity mapping!
    # FreeSurfer labels ARE the class indices (no offset needed)
    pred_raw = np.argmax(P, axis=0)  # Keep as-is: FS labels = class indices
    
    print(f"  Predicted Labels Distribution:")
    print(f"    Unique pred_raw labels: {np.unique(pred_raw)}")
    print(f"    Pred label counts: {np.bincount(pred_raw + 1)[:10]}")  # First 10 classes
    
    # Save P matrix
    P_file = os.path.join(output_dir, 'P.npy')
    np.save(P_file, P)
    
    # Load label tables from ground truth
    gt_annot = nib.freesurfer.io.read_annot(gt_path)
    gt_labels, ctab, names = gt_annot
    
    print(f"  Ground Truth Stats:")
    print(f"    Labels shape: {gt_labels.shape}")
    print(f"    Unique labels: {np.unique(gt_labels)}")
    print(f"    CTab len: {len(ctab) if ctab is not None else 'None'}")
    
    # Save raw prediction
    pred_raw_file = os.path.join(output_dir, 'lh_aligned.aparc.annot')
    nib.freesurfer.io.write_annot(pred_raw_file, pred_raw, ctab, names, fill_ctab=True)
    print(f"  ✓ Raw prediction saved to: {output_dir}")
    
    # Build adjacency for smoothing/MRF
    print(f"  Building vertex adjacency...")
    adjacency = build_adjacency_from_faces(faces)
    
    # Step 1: Smoothing
    print(f"  Step 1: Smoothing probabilities (20 iter, α=0.75)...")
    P_smooth = smooth_vertex_probabilities(P, adjacency, n_iter=20, alpha=0.75)
    pred_smooth = np.argmax(P_smooth, axis=0)  # FIXED: no -1 offset
    pred_smooth_file = os.path.join(output_dir, 'lh_aligned.aparc.smooth.annot')
    nib.freesurfer.io.write_annot(pred_smooth_file, pred_smooth, ctab, names, fill_ctab=True)
    print(f"  ✓ Smoothing complete")
    
    # Step 2: MRF
    print(f"  Step 2: MRF refinement (β=0.1)...")
    labels_mrf = apply_mrf_simple(P_smooth, adjacency, beta=0.1, n_iterations=5)
    pred_mrf = labels_mrf  # FIXED: no -1 offset (Identity mapping)
    pred_mrf_file = os.path.join(output_dir, 'lh_aligned.aparc.mrf.annot')
    nib.freesurfer.io.write_annot(pred_mrf_file, pred_mrf, ctab, names, fill_ctab=True)
    print(f"  ✓ MRF refinement complete")
    
    # Compute Dice scores (GT already loaded)
    if os.path.exists(gt_path):
        gt_labels = gt_annot[0]
        
        dice_raw = compute_dice_score(pred_raw, gt_labels)
        dice_smooth = compute_dice_score(pred_smooth, gt_labels)
        dice_mrf = compute_dice_score(pred_mrf, gt_labels)
        
        # Calculate summary statistics
        valid_dice_raw = [d for d in dice_raw.values() if not np.isnan(d)]
        valid_dice_smooth = [d for d in dice_smooth.values() if not np.isnan(d)]
        valid_dice_mrf = [d for d in dice_mrf.values() if not np.isnan(d)]
        
        top10_raw = sorted(valid_dice_raw, reverse=True)[:10]
        top10_smooth = sorted(valid_dice_smooth, reverse=True)[:10]
        top10_mrf = sorted(valid_dice_mrf, reverse=True)[:10]
        
        # Compute all 4 metrics: median(top-10), mean(top-10), median(all), mean(all)
        median_top10_raw = np.median(top10_raw)
        median_top10_smooth = np.median(top10_smooth)
        median_top10_mrf = np.median(top10_mrf)
        
        mean_top10_raw = np.mean(top10_raw)
        mean_top10_smooth = np.mean(top10_smooth)
        mean_top10_mrf = np.mean(top10_mrf)
        
        median_all_raw = np.median(valid_dice_raw)
        median_all_smooth = np.median(valid_dice_smooth)
        median_all_mrf = np.median(valid_dice_mrf)
        
        mean_all_raw = np.mean(valid_dice_raw)
        mean_all_smooth = np.mean(valid_dice_smooth)
        mean_all_mrf = np.mean(valid_dice_mrf)
        
        improvement_smooth = median_top10_smooth - median_top10_raw
        improvement_mrf = median_top10_mrf - median_top10_smooth
        total_improvement = median_top10_mrf - median_top10_raw
        
        print(f"\n  RESULTS:")
        print(f"  RAW:         median(top-10)={median_top10_raw:.4f}, mean(top-10)={mean_top10_raw:.4f}, median(all)={median_all_raw:.4f}, mean(all)={mean_all_raw:.4f}")
        print(f"  + SMOOTHING: median(top-10)={median_top10_smooth:.4f}, mean(top-10)={mean_top10_smooth:.4f}, median(all)={median_all_smooth:.4f}, mean(all)={mean_all_smooth:.4f} [{improvement_smooth:+.4f}]")
        print(f"  + MRF:       median(top-10)={median_top10_mrf:.4f}, mean(top-10)={mean_top10_mrf:.4f}, median(all)={median_all_mrf:.4f}, mean(all)={mean_all_mrf:.4f} [{improvement_mrf:+.4f}]")
        print(f"  TOTAL IMPROVEMENT (median top-10): {total_improvement:+.4f}")
        
        # Return dict with all metrics for aggregation
        return {
            'raw_median_top10': median_top10_raw,
            'raw_mean_top10': mean_top10_raw,
            'raw_median_all': median_all_raw,
            'raw_mean_all': mean_all_raw,
            'smooth_median_top10': median_top10_smooth,
            'smooth_mean_top10': mean_top10_smooth,
            'smooth_median_all': median_all_smooth,
            'smooth_mean_all': mean_all_smooth,
            'mrf_median_top10': median_top10_mrf,
            'mrf_mean_top10': mean_top10_mrf,
            'mrf_median_all': median_all_mrf,
            'mrf_mean_all': mean_all_mrf,
            'n_labels': len(valid_dice_mrf)
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Test view-conditioned model V2 (Arbitrary Views)')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Specific checkpoint to use (otherwise use latest)')
    parser.add_argument('--test_data', type=str,
                       default='/autofs/space/ballarat_004/users/np341/mindboggle2',
                       help='Path to test data')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--view_subset', type=str, nargs='+', default=None,
                       help='Subset of views to use for prediction (e.g., Front Back Left)')
    
    args = parser.parse_args()
    
    # Determine which views to use
    if args.view_subset is not None:
        view_names = args.view_subset
        print(f"Using custom view subset: {view_names}")
    else:
        view_names = None  # Will use all default views
    
    # Find checkpoint
    if args.checkpoint_path is None:
        # Try last.ckpt first (most recent checkpoint)
        last_ckpt = os.path.join(args.model_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            checkpoint_path = last_ckpt
            print(f"Auto-detected latest checkpoint: last.ckpt")
        else:
            # Find best checkpoint by validation loss
            checkpoints = [f for f in os.listdir(args.model_dir) 
                          if f.endswith('.ckpt') and f.startswith('conditioned_v2')]
            if not checkpoints:
                print(f"ERROR: No checkpoints found in {args.model_dir}")
                sys.exit(1)
            
            # Sort by filename (contains validation loss)
            checkpoints.sort()
            checkpoint_path = os.path.join(args.model_dir, checkpoints[-1])
            print(f"Auto-detected best checkpoint: {checkpoints[-1]}")
    else:
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint does not exist: {checkpoint_path}")
            sys.exit(1)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = ViewConditionedLightningModel.load_from_checkpoint(checkpoint_path)
    model = model.to(args.device)
    # IMPORTANT: Use train() mode because BatchNorm running_var is corrupted (all zeros)
    # In train mode, BN uses batch statistics instead of broken running statistics
    model.train()
    print("✓ Model loaded (using TRAIN mode due to corrupted BN running stats)")
    
    # Print test configuration
    print("\n" + "="*80)
    print("Testing VIEW-CONDITIONED model V2 (Arbitrary Views) with PyTorch3D pipeline")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {args.test_data}")
    if view_names:
        print(f"View subset: {view_names}")
    else:
        print(f"Using all default views: {DEFAULT_VIEW_NAMES}")
    print("1. Generate predictions using V2 model (continuous camera pose)")
    print("2. Iterative smoothing (α=0.75, 20 iter)")
    print("3. MRF with ICM (β=0.1)")
    print("="*80)
    
    # Get test subjects
    test_subjects = sorted([d for d in os.listdir(args.test_data)
                          if os.path.isdir(os.path.join(args.test_data, d))])
    
    print(f"\nFound {len(test_subjects)} test subjects")
    
    # Test all subjects
    results = []
    for subject in test_subjects:
        subject_dir = os.path.join(args.test_data, subject)
        result = test_subject(subject_dir, model, args.device, view_names)
        if result is not None:
            results.append(result)
    
    # Aggregate Summary
    if results:
        print(f"\n{'='*80}")
        print("AGGREGATE RESULTS ACROSS ALL TEST SUBJECTS")
        print('='*80)
        print(f"Number of subjects: {len(results)}\n")
        
        # Extract all metrics
        raw_median_top10 = [r['raw_median_top10'] for r in results]
        raw_mean_top10 = [r['raw_mean_top10'] for r in results]
        raw_median_all = [r['raw_median_all'] for r in results]
        raw_mean_all = [r['raw_mean_all'] for r in results]
        
        smooth_median_top10 = [r['smooth_median_top10'] for r in results]
        smooth_mean_top10 = [r['smooth_mean_top10'] for r in results]
        smooth_median_all = [r['smooth_median_all'] for r in results]
        smooth_mean_all = [r['smooth_mean_all'] for r in results]
        
        mrf_median_top10 = [r['mrf_median_top10'] for r in results]
        mrf_mean_top10 = [r['mrf_mean_top10'] for r in results]
        mrf_median_all = [r['mrf_median_all'] for r in results]
        mrf_mean_all = [r['mrf_mean_all'] for r in results]
        
        # Print per-stage metrics (all 4 metrics: median/mean × top-10/all)
        print("RAW PREDICTIONS:")
        print(f"  Median(top-10):  {np.mean(raw_median_top10):.4f} ± {np.std(raw_median_top10):.4f}  (min: {np.min(raw_median_top10):.4f}, max: {np.max(raw_median_top10):.4f})")
        print(f"  Mean(top-10):    {np.mean(raw_mean_top10):.4f} ± {np.std(raw_mean_top10):.4f}  (min: {np.min(raw_mean_top10):.4f}, max: {np.max(raw_mean_top10):.4f})")
        print(f"  Median(all):     {np.mean(raw_median_all):.4f} ± {np.std(raw_median_all):.4f}  (min: {np.min(raw_median_all):.4f}, max: {np.max(raw_median_all):.4f})")
        print(f"  Mean(all):       {np.mean(raw_mean_all):.4f} ± {np.std(raw_mean_all):.4f}  (min: {np.min(raw_mean_all):.4f}, max: {np.max(raw_mean_all):.4f})")
        
        print("\n+ SMOOTHING (α=0.75, 20 iter):")
        print(f"  Median(top-10):  {np.mean(smooth_median_top10):.4f} ± {np.std(smooth_median_top10):.4f}  (min: {np.min(smooth_median_top10):.4f}, max: {np.max(smooth_median_top10):.4f})")
        print(f"  Mean(top-10):    {np.mean(smooth_mean_top10):.4f} ± {np.std(smooth_mean_top10):.4f}  (min: {np.min(smooth_mean_top10):.4f}, max: {np.max(smooth_mean_top10):.4f})")
        print(f"  Median(all):     {np.mean(smooth_median_all):.4f} ± {np.std(smooth_median_all):.4f}  (min: {np.min(smooth_median_all):.4f}, max: {np.max(smooth_median_all):.4f})")
        print(f"  Mean(all):       {np.mean(smooth_mean_all):.4f} ± {np.std(smooth_mean_all):.4f}  (min: {np.min(smooth_mean_all):.4f}, max: {np.max(smooth_mean_all):.4f})")
        print(f"  Improvement:     median(top-10) {np.mean(smooth_median_top10) - np.mean(raw_median_top10):+.4f}, mean(all) {np.mean(smooth_mean_all) - np.mean(raw_mean_all):+.4f}")
        
        print("\n+ MRF (β=0.1, 5 iter):")
        print(f"  Median(top-10):  {np.mean(mrf_median_top10):.4f} ± {np.std(mrf_median_top10):.4f}  (min: {np.min(mrf_median_top10):.4f}, max: {np.max(mrf_median_top10):.4f})")
        print(f"  Mean(top-10):    {np.mean(mrf_mean_top10):.4f} ± {np.std(mrf_mean_top10):.4f}  (min: {np.min(mrf_mean_top10):.4f}, max: {np.max(mrf_mean_top10):.4f})")
        print(f"  Median(all):     {np.mean(mrf_median_all):.4f} ± {np.std(mrf_median_all):.4f}  (min: {np.min(mrf_median_all):.4f}, max: {np.max(mrf_median_all):.4f})")
        print(f"  Mean(all):       {np.mean(mrf_mean_all):.4f} ± {np.std(mrf_mean_all):.4f}  (min: {np.min(mrf_mean_all):.4f}, max: {np.max(mrf_mean_all):.4f})")
        print(f"  Improvement:     median(top-10) {np.mean(mrf_median_top10) - np.mean(smooth_median_top10):+.4f}, mean(all) {np.mean(mrf_mean_all) - np.mean(smooth_mean_all):+.4f}")
        
        print("\nTOTAL IMPROVEMENT (RAW → MRF):")
        print(f"  Median(top-10):  {np.mean(mrf_median_top10) - np.mean(raw_median_top10):+.4f}")
        print(f"  Mean(top-10):    {np.mean(mrf_mean_top10) - np.mean(raw_mean_top10):+.4f}")
        print(f"  Median(all):     {np.mean(mrf_median_all) - np.mean(raw_median_all):+.4f}")
        print(f"  Mean(all):       {np.mean(mrf_mean_all) - np.mean(raw_mean_all):+.4f}")
        
        print('='*80)


if __name__ == '__main__':
    main()

