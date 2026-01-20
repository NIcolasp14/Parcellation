#!/usr/bin/env python
"""
Test newly trained pytorch3d models and compute Dice scores with full post-processing.

This script:
1. Generates predictions on MindBoggle test subjects using NEW trained models
2. Computes surface-based Dice scores with FULL post-processing:
   - Iterative probability smoothing (α=0.75, 20 iterations)
   - MRF with ICM (β=0.1)

Usage:
    python test_p1.py --checkpoints_path /path/to/Models_WM_NEW_rec
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import scipy.io
import open3d as o3d
from collections import defaultdict

# Import PRIR prediction utilities
sys.path.insert(0, '/autofs/space/ballarat_004/users/np341/PRIR_Code/Predict')
from parcelation_utils import get_labels, get_models, get_probs, predict_segmentation

# CRITICAL: Use PyTorch3D raycasting (same as training!)
from compute_output_maps_pytorch3d import compute_output_maps_pytorch3d

# Also need training utilities for camera setup
sys.path.insert(0, '/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White')

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
    """
    Iterative neighborhood smoothing on UNNORMALIZED probability matrix
    Following paper Section 2.4: p'_i ← α*p'_i + (1-α)/|N_i| * Σ p'_{i'}
    
    P: (NL, NV) - UNNORMALIZED probabilities for each label at each vertex
    adjacency: dict mapping vertex -> set of neighbor vertices
    n_iter: number of iterations (paper uses 20)
    alpha: smoothing factor (paper uses 0.75, higher = less smoothing)
    """
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
            
            # Sum of neighbor probabilities
            neighbor_sum = np.sum(P_smooth[:, neighbors], axis=1)
            
            # Paper formula: p'_i ← α*p'_i + (1-α)/|N_i| * Σ_{i'∈N_i} p'_{i'}
            P_new[:, v] = alpha * P_smooth[:, v] + (1 - alpha) * neighbor_sum / n_neighbors
        
        P_smooth = P_new
    
    # Normalize AFTER all smoothing iterations
    P_normalized = P_smooth / np.sum(P_smooth, axis=0, keepdims=True)
    
    return P_normalized

def apply_mrf_simple(P_normalized, adjacency, beta=0.1, n_iterations=5):
    """
    Apply simple iterative MRF refinement (ICM - Iterated Conditional Modes)
    Following paper Equation 1: max_S Σ log(p_{s_i}) + β Σ δ(s_i = s_{i'})
    
    P_normalized: (NL, NV) - normalized probabilities  
    adjacency: dict mapping vertex -> neighbor vertices
    beta: MRF strength parameter (paper uses 0.1)
    n_iterations: number of ICM iterations
    
    Returns:
        labels: (NV,) - refined labeling
    """
    NL, NV = P_normalized.shape
    
    # Start with MAP estimate
    labels = np.argmax(P_normalized, axis=0)
    
    # Data term: log probabilities
    # Ensure P_normalized is valid (no zeros, nans, or infs)
    eps = 1e-10
    P_normalized = np.nan_to_num(P_normalized, nan=eps, posinf=1.0, neginf=eps)
    P_normalized = np.clip(P_normalized, eps, 1.0)
    log_prob = np.log(P_normalized + eps)
    
    # Iterative refinement
    changed_total = 0
    for iteration in range(n_iterations):
        print(f"    MRF iteration {iteration+1}/{n_iterations}...", flush=True)
        changed = 0
        
        for v in range(NV):
            if v not in adjacency or len(adjacency[v]) == 0:
                continue
            
            neighbors = list(adjacency[v])
            neighbor_labels = labels[neighbors]
            
            # Compute energy for each possible label
            best_label = labels[v]
            best_energy = -np.inf
            
            for label in range(NL):
                # Data term: log p(label | data)
                data_term = log_prob[label, v]
                
                # Smoothness term: β * sum of matching neighbors
                smoothness_term = beta * np.sum(neighbor_labels == label) / len(neighbors)
                
                energy = data_term + smoothness_term
                
                if energy > best_energy:
                    best_energy = energy
                    best_label = label
            
            if labels[v] != best_label:
                labels[v] = best_label
                changed += 1
        
        changed_total += changed
        if changed == 0:
            break
    
    return labels

def generate_prediction(mesh_path, checkpoints_path, output_dir):
    """
    Generate prediction for a single subject using the trained models
    
    Returns:
        pred_annot_path, prob_mat_path (or None, None if failed)
    """
    try:
        # Load reference annotation for color table
        DEFAULT_L_PATH = '/autofs/space/calico_004/users/pblasco/project/aligned_closed_meshes/100206/label/lh.annot'
        annot_data = nib.freesurfer.io.read_annot(DEFAULT_L_PATH)
        
        # Setup output paths
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(mesh_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        pred_annot_path = os.path.join(output_dir, f"{file_name_without_extension}_pred.annot")
        prob_mat_path = os.path.join(output_dir, f"{file_name_without_extension}_prob.mat")
        
        # Load mesh
        img_width = 800
        img_height = 800
        
        vertices, faces, info = nib.freesurfer.read_geometry(mesh_path, read_metadata=True)
        
        # Center vertices
        vertices = vertices - np.mean(vertices, axis=0)
        vertices = vertices[:, [2, 0, 1]]
        
        # Create Open3D mesh
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.float32(vertices)),
            o3d.core.Tensor(np.int64(faces))
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # Compute output maps using PyTorch3D (matches training exactly!)
        print(f"  Using PyTorch3D GPU raycasting (matching training pipeline)...")
        output_maps, ids_maps_o, weights_maps_full_o = compute_output_maps_pytorch3d(mesh, img_width, img_height)
        print(f"  ✓ Got {len(output_maps)} output maps, shape: {output_maps.shape}")
        
        # Load models and predict
        print(f"  Loading trained models...")
        models = get_models(checkpoints_path)
        print(f"  Running U-Net predictions on {len(output_maps)} views...")
        view_predictions = get_probs(output_maps, models)
        print(f"  ✓ Got predictions from {len(view_predictions)} views")
        print(f"  Aggregating predictions to vertices...")
        Lhat, P = predict_segmentation(vertices, faces, output_maps, ids_maps_o, weights_maps_full_o, view_predictions)
        print(f"  ✓ Aggregation complete")
        
        # Save predictions
        nib.freesurfer.io.write_annot(pred_annot_path, Lhat, annot_data[1], annot_data[2], fill_ctab=True)
        scipy.io.savemat(prob_mat_path, {'probabilities': P})
        
        return pred_annot_path, prob_mat_path, faces
        
    except Exception as e:
        print(f"  ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def compute_dice_score(pred, gt):
    """Compute per-label Dice scores"""
    unique_labels = np.unique(np.concatenate([pred, gt]))
    dice_scores = {}
    
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
    
    return dice_scores

def find_latest_slurm_checkpoint_dir(base_dir="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White"):
    """Find the most recent Models_WM_SLURM_* directory"""
    import glob
    pattern = os.path.join(base_dir, "Models_WM_SLURM_*")
    dirs = glob.glob(pattern)
    
    if not dirs:
        return None
    
    # Sort by modification time (most recent first)
    dirs_with_mtime = [(d, os.path.getmtime(d)) for d in dirs]
    dirs_with_mtime.sort(key=lambda x: x[1], reverse=True)
    
    return dirs_with_mtime[0][0]

def find_slurm_checkpoint_dir_by_id(slurm_id, base_dir="/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White"):
    """Find Models_WM_SLURM_<slurm_id> directory"""
    checkpoint_dir = os.path.join(base_dir, f"Models_WM_SLURM_{slurm_id}")
    
    if os.path.exists(checkpoint_dir):
        return checkpoint_dir
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Test new models and compute Dice scores")
    parser.add_argument("--checkpoints_path", type=str, 
                       default=None,
                       help="Path to trained model checkpoints (overrides --slurm_id and auto-detection)")
    parser.add_argument("--slurm_id", type=str,
                       default=None,
                       help="SLURM job ID to test (e.g., 7615773). Will use Models_WM_SLURM_<slurm_id>")
    parser.add_argument("--test_data_path", type=str,
                       default="/autofs/space/ballarat_004/users/np341/mindboggle2",
                       help="Path to test subjects (MindBoggle)")
    parser.add_argument("--output_suffix", type=str,
                       default="_new_pytorch3d",
                       help="Suffix for output directory (pred{suffix})")
    args = parser.parse_args()
    
    mindboggle_path = args.test_data_path
    
    # Determine checkpoint path with priority: explicit path > slurm_id > auto-detect latest
    if args.checkpoints_path:
        checkpoints_path = args.checkpoints_path
        print(f"Using explicitly provided checkpoint path: {checkpoints_path}")
    elif args.slurm_id:
        checkpoints_path = find_slurm_checkpoint_dir_by_id(args.slurm_id)
        if checkpoints_path is None:
            print(f"ERROR: Could not find checkpoint directory for SLURM job {args.slurm_id}")
            print(f"Expected: Models_WM_SLURM_{args.slurm_id}")
            return
        print(f"Using SLURM job {args.slurm_id}: {checkpoints_path}")
    else:
        checkpoints_path = find_latest_slurm_checkpoint_dir()
        if checkpoints_path is None:
            print("ERROR: Could not auto-detect any Models_WM_SLURM_* directories")
            print("Please specify --checkpoints_path or --slurm_id")
            return
        slurm_id = os.path.basename(checkpoints_path).replace("Models_WM_SLURM_", "")
        print(f"Auto-detected latest SLURM run: {slurm_id}")
        print(f"Using checkpoint path: {checkpoints_path}")
    
    # Verify checkpoint directory exists
    if not os.path.exists(checkpoints_path):
        print(f"ERROR: Checkpoint directory does not exist: {checkpoints_path}")
        return
    
    subjects = sorted([d for d in os.listdir(mindboggle_path) 
                      if os.path.isdir(os.path.join(mindboggle_path, d))])
    
    results_raw = []
    
    print("=" * 80)
    print(f"Testing NEW pytorch3d models with FULL post-processing pipeline")
    print(f"Checkpoints: {checkpoints_path}")
    print(f"Test data: {mindboggle_path}")
    print(f"1. Generate predictions using NEW models")
    print(f"2. Iterative smoothing (α=0.75, 20 iter)")
    print(f"3. MRF with ICM (β=0.1)")
    print("=" * 80)
    print()
    
    for subject in subjects:
        # Paths
        gt_path = os.path.join(mindboggle_path, subject, "label", "lh.DKTaparc.2016-03-20.annot")
        
        # Try white or pial surface
        surf_candidates = [
            os.path.join(mindboggle_path, subject, "surf", "lh.white"),
            os.path.join(mindboggle_path, subject, "surf", "lh.pial"),
        ]
        mesh_path = next((s for s in surf_candidates if os.path.exists(s)), None)
        
        if not os.path.exists(gt_path) or mesh_path is None:
            print(f"Skipping {subject:25s} - missing files")
            continue
        
        print(f"\n{'='*80}")
        print(f"Subject: {subject}")
        print(f"{'='*80}")
        
        # Output directory for this subject's predictions
        pred_dir = os.path.join(mindboggle_path, subject, f"pred{args.output_suffix}")
        
        # Step 0: Generate prediction with NEW models
        print(f"  Step 0: Generating predictions with NEW models...")
        pred_annot_path, prob_mat_path, faces = generate_prediction(mesh_path, checkpoints_path, pred_dir)
        
        if pred_annot_path is None:
            print(f"  Prediction failed for {subject}")
            continue
        
        print(f"  ✓ Prediction saved to: {pred_dir}")
        
        # Load ground truth
        try:
            gt_labels = get_labels(gt_path)
            gt_labels = np.asarray(gt_labels, dtype=np.int32)
        except Exception as e:
            print(f"  Error loading GT: {e}")
            continue
        
        # Load raw predictions
        pred_labels_raw = nib.freesurfer.io.read_annot(pred_annot_path)[0]
        pred_labels_raw = np.asarray(pred_labels_raw, dtype=np.int32)
        
        # Load probability matrix
        prob_data = scipy.io.loadmat(prob_mat_path)
        P = prob_data['probabilities']
        print(f"  Loaded P matrix: {P.shape}, vertices: {len(gt_labels)}")
        
        # Build adjacency
        adjacency = build_adjacency_from_faces(faces)
        print(f"  Mesh: {len(gt_labels)} vertices, {len(faces)} faces")
        
        # Compute Dice BEFORE post-processing
        dice_raw = compute_dice_score(pred_labels_raw, gt_labels)
        valid_raw = [d for d in dice_raw.values() if not np.isnan(d)]
        top10_raw = sorted(valid_raw, reverse=True)[:min(10, len(valid_raw))] if valid_raw else []
        
        # Step 1: Iterative smoothing
        print(f"  Step 1: Smoothing probabilities (20 iter, α=0.75)...", flush=True)
        P_smoothed = smooth_vertex_probabilities(P, adjacency, n_iter=20, alpha=0.75)
        print(f"  ✓ Smoothing complete", flush=True)
        
        pred_labels_smoothed = np.argmax(P_smoothed, axis=0) - 1
        dice_smoothed = compute_dice_score(pred_labels_smoothed, gt_labels)
        valid_smoothed = [d for d in dice_smoothed.values() if not np.isnan(d)]
        top10_smoothed = sorted(valid_smoothed, reverse=True)[:min(10, len(valid_smoothed))] if valid_smoothed else []
        
        # Step 2: MRF refinement
        print(f"  Step 2: MRF refinement (β=0.1)...", flush=True)
        pred_labels_mrf = apply_mrf_simple(P_smoothed, adjacency, beta=0.1, n_iterations=5)
        print(f"  ✓ MRF refinement complete", flush=True)
        pred_labels_mrf = pred_labels_mrf - 1  # Adjust for label indexing
        
        dice_mrf = compute_dice_score(pred_labels_mrf, gt_labels)
        valid_mrf = [d for d in dice_mrf.values() if not np.isnan(d)]
        top10_mrf = sorted(valid_mrf, reverse=True)[:min(10, len(valid_mrf))] if valid_mrf else []
        
        if valid_mrf and top10_mrf:
            # Compute all 4 metrics for each stage
            result = {
                'raw_median_top10': np.median(top10_raw),
                'raw_mean_top10': np.mean(top10_raw),
                'raw_median_all': np.median(valid_raw),
                'raw_mean_all': np.mean(valid_raw),
                'smooth_median_top10': np.median(top10_smoothed),
                'smooth_mean_top10': np.mean(top10_smoothed),
                'smooth_median_all': np.median(valid_smoothed),
                'smooth_mean_all': np.mean(valid_smoothed),
                'mrf_median_top10': np.median(top10_mrf),
                'mrf_mean_top10': np.mean(top10_mrf),
                'mrf_median_all': np.median(valid_mrf),
                'mrf_mean_all': np.mean(valid_mrf),
                'n_labels': len(valid_mrf)
            }
            results_raw.append(result)
            
            improvement1 = result['smooth_median_top10'] - result['raw_median_top10']
            improvement2 = result['mrf_median_top10'] - result['smooth_median_top10']
            improvement_total = result['mrf_median_top10'] - result['raw_median_top10']
            
            print(f"\n  RESULTS:")
            print(f"  RAW:         median(top-10)={result['raw_median_top10']:.4f}, mean(top-10)={result['raw_mean_top10']:.4f}, median(all)={result['raw_median_all']:.4f}, mean(all)={result['raw_mean_all']:.4f}")
            print(f"  + SMOOTHING: median(top-10)={result['smooth_median_top10']:.4f}, mean(top-10)={result['smooth_mean_top10']:.4f}, median(all)={result['smooth_median_all']:.4f}, mean(all)={result['smooth_mean_all']:.4f} [{improvement1:+.4f}]")
            print(f"  + MRF:       median(top-10)={result['mrf_median_top10']:.4f}, mean(top-10)={result['mrf_mean_top10']:.4f}, median(all)={result['mrf_median_all']:.4f}, mean(all)={result['mrf_mean_all']:.4f} [{improvement2:+.4f}]")
            print(f"  TOTAL IMPROVEMENT (median top-10): {improvement_total:+.4f}")
        else:
            print(f"  No valid dice scores")
    
    # Summary
    print("\n" + "=" * 80)
    if len(results_raw) > 0:
        print(f"AGGREGATE RESULTS ACROSS ALL TEST SUBJECTS")
        print('='*80)
        print(f"Number of subjects: {len(results_raw)}\n")
        
        # Extract all metrics
        raw_median_top10 = [r['raw_median_top10'] for r in results_raw]
        raw_mean_top10 = [r['raw_mean_top10'] for r in results_raw]
        raw_median_all = [r['raw_median_all'] for r in results_raw]
        raw_mean_all = [r['raw_mean_all'] for r in results_raw]
        
        smooth_median_top10 = [r['smooth_median_top10'] for r in results_raw]
        smooth_mean_top10 = [r['smooth_mean_top10'] for r in results_raw]
        smooth_median_all = [r['smooth_median_all'] for r in results_raw]
        smooth_mean_all = [r['smooth_mean_all'] for r in results_raw]
        
        mrf_median_top10 = [r['mrf_median_top10'] for r in results_raw]
        mrf_mean_top10 = [r['mrf_mean_top10'] for r in results_raw]
        mrf_median_all = [r['mrf_median_all'] for r in results_raw]
        mrf_mean_all = [r['mrf_mean_all'] for r in results_raw]
        
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
        
        print("\n" + "=" * 80)
        print("*** PAPER TARGET: ~0.90 (median across 10 largest parcels) ***")
        print("*** FULL PIPELINE: NEW pytorch3d models + Smoothing + MRF ***")
        print(f"*** Predictions saved in: pred{args.output_suffix}/ directories ***")
        print("=" * 80)
    else:
        print("\nNo results computed!")

if __name__ == "__main__":
    main()

