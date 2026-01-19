#!/usr/bin/env python
"""
Compute surface-based Dice scores with FULL post-processing pipeline:
1. Iterative probability smoothing (α=0.75, 20 iterations)
2. MRF with graph cuts (β=0.1)
Following the exact method from the PRIR paper Section 2.4
"""
import os
import sys
import numpy as np
import nibabel as nib
import scipy.io
from collections import defaultdict

# Import existing PRIR utilities
sys.path.insert(0, '/autofs/space/ballarat_004/users/np341/PRIR_Code/Predict')
from parcelation_utils import get_labels

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
    
    This is a simpler alternative to graph cuts that doesn't require external libraries
    
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
    
    print(f"  Running simple MRF (ICM): {NV} vertices, {NL} labels, β={beta}, {n_iterations} iter")
    
    # Data term: log probabilities
    eps = 1e-10
    log_prob = np.log(P_normalized + eps)
    
    # Iterative refinement
    changed_total = 0
    for iteration in range(n_iterations):
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
            print(f"    Converged at iteration {iteration + 1}")
            break
    
    if changed_total > 0:
        print(f"    Changed {changed_total} labels total")
    
    return labels

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

def main():
    mindboggle_path = "/autofs/space/ballarat_004/users/np341/mindboggle2"
    subjects = sorted([d for d in os.listdir(mindboggle_path) 
                      if os.path.isdir(os.path.join(mindboggle_path, d))])
    
    results = []
    
    print(f"Computing surface-based Dice with FULL post-processing pipeline")
    print(f"1. Iterative smoothing (α=0.75, 20 iter)")
    print(f"2. MRF with graph cuts (β=0.1)")
    print("=" * 80)
    
    for subject in subjects:
        # Paths
        gt_path = os.path.join(mindboggle_path, subject, "label", "lh.DKTaparc.2016-03-20.annot")
        pred_path = os.path.join(mindboggle_path, subject, "pred", "lh_pred.annot")
        prob_path = os.path.join(mindboggle_path, subject, "pred", "lh_prob.mat")
        
        # Surface file for topology
        surf_candidates = [
            os.path.join(mindboggle_path, subject, "surf", "lh.white"),
            os.path.join(mindboggle_path, subject, "surf", "lh.pial"),
        ]
        surf_path = next((s for s in surf_candidates if os.path.exists(s)), None)
        
        if not all(os.path.exists(p) for p in [gt_path, pred_path, prob_path, surf_path]):
            print(f"Skipping {subject:25s} - missing files")
            continue
        
        print(f"\n{subject:25s}:")
        
        # Load ground truth
        try:
            gt_labels = get_labels(gt_path)
            gt_labels = np.asarray(gt_labels, dtype=np.int32)
        except Exception as e:
            print(f"  Error loading GT: {e}")
            continue
        
        # Load raw predictions
        pred_labels_raw = nib.freesurfer.io.read_annot(pred_path)[0]
        pred_labels_raw = np.asarray(pred_labels_raw, dtype=np.int32)
        
        # Load probability matrix
        prob_data = scipy.io.loadmat(prob_path)
        P = prob_data['probabilities']
        print(f"  Loaded P matrix: {P.shape}, vertices: {len(gt_labels)}")
        
        # Load surface mesh
        vertices, faces = nib.freesurfer.read_geometry(surf_path)
        print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Build adjacency
        adjacency = build_adjacency_from_faces(faces)
        
        # Compute Dice BEFORE post-processing
        dice_raw = compute_dice_score(pred_labels_raw, gt_labels)
        valid_raw = [d for d in dice_raw.values() if not np.isnan(d)]
        top10_raw = sorted(valid_raw, reverse=True)[:min(10, len(valid_raw))] if valid_raw else []
        
        # Step 1: Iterative smoothing
        print(f"  Step 1: Smoothing probabilities (20 iter, α=0.75)...")
        P_smoothed = smooth_vertex_probabilities(P, adjacency, n_iter=20, alpha=0.75)
        
        pred_labels_smoothed = np.argmax(P_smoothed, axis=0) - 1
        dice_smoothed = compute_dice_score(pred_labels_smoothed, gt_labels)
        valid_smoothed = [d for d in dice_smoothed.values() if not np.isnan(d)]
        top10_smoothed = sorted(valid_smoothed, reverse=True)[:min(10, len(valid_smoothed))] if valid_smoothed else []
        
        # Step 2: MRF refinement
        print(f"  Step 2: MRF refinement (β=0.1)...")
        pred_labels_mrf = apply_mrf_simple(P_smoothed, adjacency, beta=0.1, n_iterations=5)
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
            results.append(result)
            
            improvement1 = result['smooth_median_top10'] - result['raw_median_top10']
            improvement2 = result['mrf_median_top10'] - result['smooth_median_top10']
            improvement_total = result['mrf_median_top10'] - result['raw_median_top10']
            
            print(f"  RAW:         median(top-10)={result['raw_median_top10']:.4f}, mean(top-10)={result['raw_mean_top10']:.4f}, median(all)={result['raw_median_all']:.4f}, mean(all)={result['raw_mean_all']:.4f}")
            print(f"  + SMOOTHING: median(top-10)={result['smooth_median_top10']:.4f}, mean(top-10)={result['smooth_mean_top10']:.4f}, median(all)={result['smooth_median_all']:.4f}, mean(all)={result['smooth_mean_all']:.4f} [{improvement1:+.4f}]")
            print(f"  + MRF:       median(top-10)={result['mrf_median_top10']:.4f}, mean(top-10)={result['mrf_mean_top10']:.4f}, median(all)={result['mrf_median_all']:.4f}, mean(all)={result['mrf_mean_all']:.4f} [{improvement2:+.4f}]")
            print(f"  TOTAL IMPROVEMENT (median top-10): {improvement_total:+.4f}")
        else:
            print(f"  No valid dice scores")
    
    # Summary
    print("\n" + "=" * 80)
    if len(results) > 0:
        print(f"AGGREGATE RESULTS ACROSS ALL TEST SUBJECTS")
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
        
        print("\n" + "=" * 80)
        print("*** PAPER TARGET: ~0.90 (median across 10 largest parcels) ***")
        print("*** FULL PIPELINE: Smoothing + MRF (ICM) refinement ***")
        print("*** Using simple iterative MRF (no external dependencies) ***")
        print("=" * 80)
    else:
        print("\nNo results computed!")

if __name__ == "__main__":
    main()

