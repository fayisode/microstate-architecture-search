"""
Metrics Utilities for Fair VAE vs Baseline Comparison
======================================================

This module provides utilities for:
1. Polarity invariance checking using SSIM
2. Fair metrics computation in consistent feature spaces
3. GEV computation for VAE models
4. Unified comparison framework

Key insight: EEG microstates are polarity-invariant, meaning a map and its
inverted version (multiplied by -1) represent the SAME brain state.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from config.config import config as cfg

EPSILON = 1e-8


# =============================================================================
# PYCROSTATES-STYLE METRICS (Correlation-based + Polarity Aligned)
# =============================================================================

def correlation_distance_matrix(X):
    """
    Compute distance matrix using pycrostates formula: |1/correlation| - 1.

    This matches pycrostates' _distance_matrix function exactly:
        distances = np.abs(1 / np.corrcoef(X, Y)) - 1

    The formula gives:
    - correlation=1.0  → distance=0.0 (identical)
    - correlation=0.5  → distance=1.0
    - correlation=0.2  → distance=4.0 (very far)
    - correlation→0    → distance→∞ (handled as large finite value)

    This is more aggressive than `1 - |correlation|` because low correlations
    produce much higher distances, creating sharper cluster boundaries.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)

    Returns
    -------
    distances : np.ndarray
        Distance matrix of shape (n_samples, n_samples)
        Distance = |1/correlation| - 1 (pycrostates formula)
    """
    # Use np.corrcoef which computes Pearson correlation (expects rows as samples)
    correlation = np.corrcoef(X)

    # Pycrostates formula: |1/correlation| - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        distances = np.abs(1.0 / correlation) - 1

    # Handle inf/nan (when correlation is 0 or very small)
    # Use exact pycrostates values: nan=10e300, posinf=1e300, neginf=-1e300
    distances = np.nan_to_num(distances, copy=False, nan=10e300, posinf=1e300, neginf=-1e300)

    # Ensure diagonal is exactly 0 and matrix is symmetric
    np.fill_diagonal(distances, 0)
    distances = (distances + distances.T) / 2

    return distances


def align_polarities(X, labels, centroids):
    """
    Align sample polarities to match their assigned centroids.

    This is critical for sklearn metrics which use Euclidean distance.
    Without alignment, opposite-polarity samples of the same microstate
    appear far apart.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)
    centroids : np.ndarray
        Cluster centers of shape (n_clusters, n_features)

    Returns
    -------
    X_aligned : np.ndarray
        Polarity-aligned data matrix
    """
    centroid_for_each_sample = centroids[labels]

    # Compute sign of dot product (indicates polarity match)
    dot_products = np.sum(X * centroid_for_each_sample, axis=1)
    signs = np.sign(dot_products)
    signs[signs == 0] = 1  # Handle edge case

    # Flip samples to match centroid polarity
    X_aligned = X * signs[:, np.newaxis]

    return X_aligned


def pycrostates_silhouette_score(X, labels):
    """
    Compute Silhouette score using correlation-based distance (pycrostates-style).

    This uses metric="precomputed" with a correlation distance matrix,
    matching pycrostates' implementation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)

    Returns
    -------
    score : float
        Silhouette score (-1 to 1, higher is better)
    """
    # Filter out zero-norm samples
    norms = np.linalg.norm(X, axis=1)
    keep = norms != 0
    X_filtered = X[keep]
    labels_filtered = labels[keep]

    if len(np.unique(labels_filtered)) < 2:
        return -1.0

    # Compute correlation-based distance matrix
    distances = correlation_distance_matrix(X_filtered)

    # Use sklearn with precomputed distances
    return float(silhouette_score(distances, labels_filtered, metric="precomputed"))


def pycrostates_calinski_harabasz_score(X, labels, centroids=None):
    """
    Compute Calinski-Harabasz score with polarity alignment (pycrostates-style).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)
    centroids : np.ndarray, optional
        Cluster centers for polarity alignment. If None, computed from data.

    Returns
    -------
    score : float
        Calinski-Harabasz score (higher is better)
    """
    # Filter out zero-norm samples
    norms = np.linalg.norm(X, axis=1)
    keep = norms != 0
    X_filtered = X[keep]
    labels_filtered = labels[keep]

    if len(np.unique(labels_filtered)) < 2:
        return 0.0

    # Compute centroids if not provided
    if centroids is None:
        unique_labels = np.unique(labels_filtered)
        centroids = np.array([X_filtered[labels_filtered == k].mean(axis=0)
                             for k in unique_labels])
        # Remap labels to 0, 1, 2, ...
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels_filtered = np.array([label_map[l] for l in labels_filtered])

    # Align polarities
    X_aligned = align_polarities(X_filtered, labels_filtered, centroids)

    return float(calinski_harabasz_score(X_aligned, labels_filtered))


def pycrostates_davies_bouldin_score(X, labels, centroids=None):
    """
    Compute Davies-Bouldin score using correlation-based distance (pycrostates-style).

    This implementation uses spatial correlation for distance computation
    instead of Euclidean distance, matching pycrostates' approach.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)
    centroids : np.ndarray, optional
        Cluster centers. If None, computed from data.

    Returns
    -------
    score : float
        Davies-Bouldin score (lower is better)
    """
    from sklearn.preprocessing import LabelEncoder

    # Filter out zero-norm samples
    norms = np.linalg.norm(X, axis=1)
    keep = norms != 0
    X_filtered = X[keep]
    labels_filtered = labels[keep]

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_filtered)
    n_labels = len(le.classes_)

    if n_labels < 2:
        return 0.0

    # Align polarities if centroids provided
    if centroids is not None:
        X_filtered = align_polarities(X_filtered, labels_encoded, centroids)

    # Compute cluster centroids and intra-cluster distances
    intra_dists = np.zeros(n_labels)
    cluster_centroids = np.zeros((n_labels, X_filtered.shape[1]))

    for k in range(n_labels):
        cluster_k = X_filtered[labels_encoded == k]
        centroid = cluster_k.mean(axis=0)
        cluster_centroids[k] = centroid

        # Intra-cluster distance using vectorized point-to-centroid correlation
        if len(cluster_k) > 1:
            x = cluster_k - cluster_k.mean(axis=1, keepdims=True)
            y = centroid - centroid.mean()
            x_norms = np.linalg.norm(x, axis=1)
            y_norm = np.linalg.norm(y)
            corrs = (x @ y) / (x_norms * y_norm + 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                dists_to_centroid = np.abs(1.0 / corrs) - 1
            dists_to_centroid = np.nan_to_num(dists_to_centroid, nan=10e300, posinf=1e300)
            intra_dists[k] = np.mean(dists_to_centroid)
        else:
            intra_dists[k] = 0

    # Compute inter-cluster distances (between centroids)
    centroid_distances = correlation_distance_matrix(cluster_centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    # Avoid division by zero (exact pycrostates behavior)
    centroid_distances[centroid_distances == 0] = np.inf

    # Davies-Bouldin formula
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)

    return float(np.mean(scores))


def _delta_fast(ck, cl, distances):
    """
    Min distance between two clusters (exact pycrostates helper).

    Parameters
    ----------
    ck : np.ndarray
        Boolean mask for cluster k
    cl : np.ndarray
        Boolean mask for cluster l
    distances : np.ndarray
        Precomputed distance matrix

    Returns
    -------
    float
        Minimum distance between clusters
    """
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]
    return np.min(values) if len(values) > 0 else 1000000


def _big_delta_fast(ci, distances):
    """
    Max distance within a cluster (exact pycrostates helper).

    Parameters
    ----------
    ci : np.ndarray
        Boolean mask for cluster i
    distances : np.ndarray
        Precomputed distance matrix

    Returns
    -------
    float
        Maximum distance within cluster (diameter)
    """
    values = distances[np.where(ci)][:, np.where(ci)]
    return np.max(values)


def pycrostates_dunn_score(X, labels):
    """
    Compute Dunn index using correlation-based distance (exact pycrostates algorithm).

    This implementation matches pycrostates' dunn_score() exactly, using
    _delta_fast and _big_delta_fast helper functions.

    Dunn = min(inter-cluster distance) / max(intra-cluster diameter)
    Higher is better.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)

    Returns
    -------
    score : float
        Dunn index (higher is better)
    """
    # Filter out zero-norm samples (matches pycrostates: keep = np.linalg.norm(data.T, axis=1) != 0)
    norms = np.linalg.norm(X, axis=1)
    keep = norms != 0
    X_filtered = X[keep]
    labels_filtered = labels[keep]

    # Compute full distance matrix
    distances = correlation_distance_matrix(X_filtered)

    ks = np.sort(np.unique(labels_filtered))
    n_clusters = len(ks)

    if n_clusters < 2:
        return 0.0

    deltas = np.ones([n_clusters, n_clusters]) * 1000000
    big_deltas = np.zeros([n_clusters, 1])

    for i, ks_i in enumerate(ks):
        for j, ks_j in enumerate(ks):
            if i == j:
                continue
            deltas[i, j] = _delta_fast((labels_filtered == ks_i), (labels_filtered == ks_j), distances)
        big_deltas[i] = _big_delta_fast((labels_filtered == ks_i), distances)

    max_big_delta = np.max(big_deltas)
    if max_big_delta == 0:
        return 0.0
    di = np.min(deltas) / max_big_delta
    return float(di)


def correlation_distance_matrix_with_mask(X, labels):
    """
    Compute correlation distance matrix with zero-norm filtering.

    Returns the precomputed distance matrix and filtered labels so callers
    can share a single corrcoef computation across silhouette + dunn.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)

    Returns
    -------
    distances : np.ndarray or None
        Precomputed distance matrix (n_filtered, n_filtered), or None if <2 clusters
    labels_filtered : np.ndarray
        Filtered cluster labels (zero-norm samples removed)
    """
    norms = np.linalg.norm(X, axis=1)
    keep = norms != 0
    X_filtered = X[keep]
    labels_filtered = labels[keep]

    if len(np.unique(labels_filtered)) < 2:
        return None, labels_filtered

    distances = correlation_distance_matrix(X_filtered)
    return distances, labels_filtered


def pycrostates_silhouette_score_precomputed(distances, labels):
    """Silhouette score from precomputed correlation distance matrix."""
    if distances is None or len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(distances, labels, metric="precomputed"))


def pycrostates_dunn_score_precomputed(distances, labels):
    """Dunn index from precomputed correlation distance matrix."""
    if distances is None:
        return 0.0

    ks = np.sort(np.unique(labels))
    n_clusters = len(ks)
    if n_clusters < 2:
        return 0.0

    deltas = np.ones([n_clusters, n_clusters]) * 1000000
    big_deltas = np.zeros([n_clusters, 1])

    for i, ks_i in enumerate(ks):
        for j, ks_j in enumerate(ks):
            if i == j:
                continue
            deltas[i, j] = _delta_fast((labels == ks_i), (labels == ks_j), distances)
        big_deltas[i] = _big_delta_fast((labels == ks_i), distances)

    max_big_delta = np.max(big_deltas)
    if max_big_delta == 0:
        return 0.0
    return float(np.min(deltas) / max_big_delta)


def dunn_score_euclidean(X, labels):
    """
    Compute Dunn index using Euclidean distance.

    Dunn = min(inter-cluster distance) / max(intra-cluster diameter)
    Higher is better.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels of shape (n_samples,)

    Returns
    -------
    score : float
        Dunn index (higher is better)
    """
    from scipy.spatial.distance import cdist

    ks = np.sort(np.unique(labels))
    n_clusters = len(ks)

    if n_clusters < 2:
        return 0.0

    # Compute full Euclidean distance matrix
    distances = cdist(X, X, metric="euclidean")

    deltas = np.ones([n_clusters, n_clusters]) * 1e10
    big_deltas = np.zeros([n_clusters, 1])

    for i, ks_i in enumerate(ks):
        for j, ks_j in enumerate(ks):
            if i == j:
                continue
            deltas[i, j] = _delta_fast((labels == ks_i), (labels == ks_j), distances)
        big_deltas[i] = _big_delta_fast((labels == ks_i), distances)

    if np.max(big_deltas) == 0:
        return 0.0

    return float(np.min(deltas) / np.max(big_deltas))


class PolarityInvarianceChecker:
    """
    Check if two maps are inverted versions of each other using SSIM.

    EEG microstates have the property that Map A and -Map A (inverted)
    represent the same underlying brain state. This class helps detect
    when two maps are actually the same state with opposite polarity.

    Usage:
    ------
    checker = PolarityInvarianceChecker()
    result = checker.check_map_pair(map_a, map_b)
    # result contains SSIM scores for original and inverted comparisons
    """

    def __init__(self, ssim_threshold: Optional[float] = None, output_dir: Optional[str] = None):
        """
        Args:
            ssim_threshold: SSIM value above which maps are considered "same"
                           (defaults to config.toml clustering.ssim_threshold)
            output_dir: Directory to save analysis results
        """
        # Use config threshold if not explicitly provided
        thresholds = cfg.get_merge_thresholds()
        self.ssim_threshold = ssim_threshold if ssim_threshold is not None else thresholds["ssim_threshold"]
        self.output_dir = Path(output_dir) if output_dir else Path("./polarity_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_log = []

    def compute_ssim(self, map_a: np.ndarray, map_b: np.ndarray) -> float:
        """Compute SSIM between two maps."""
        # Ensure maps are 2D
        if map_a.ndim > 2:
            map_a = map_a.squeeze()
        if map_b.ndim > 2:
            map_b = map_b.squeeze()

        # Normalize to [0, 1] for SSIM
        map_a_norm = (map_a - map_a.min()) / (map_a.max() - map_a.min() + EPSILON)
        map_b_norm = (map_b - map_b.min()) / (map_b.max() - map_b.min() + EPSILON)

        return ssim(map_a_norm, map_b_norm, data_range=1.0)

    def check_map_pair(
        self,
        map_a: np.ndarray,
        map_b: np.ndarray,
        map_a_name: str = "Map_A",
        map_b_name: str = "Map_B"
    ) -> Dict:
        """
        Check if two maps are inverted versions of each other.

        Compares:
        1. Map A vs Map B (original)
        2. Map A vs -Map B (inverted B)

        Returns dict with SSIM scores and whether maps are likely inverted.
        """
        # Original comparison
        ssim_original = self.compute_ssim(map_a, map_b)

        # Inverted comparison (invert map B)
        map_b_inverted = -map_b
        ssim_inverted = self.compute_ssim(map_a, map_b_inverted)

        # Determine relationship
        is_same_original = ssim_original >= self.ssim_threshold
        is_same_inverted = ssim_inverted >= self.ssim_threshold

        if is_same_inverted and not is_same_original:
            relationship = "INVERTED"
            best_ssim = ssim_inverted
        elif is_same_original:
            relationship = "SAME"
            best_ssim = ssim_original
        elif ssim_inverted > ssim_original:
            relationship = "POSSIBLY_INVERTED"
            best_ssim = ssim_inverted
        else:
            relationship = "DIFFERENT"
            best_ssim = max(ssim_original, ssim_inverted)

        result = {
            "map_a_name": map_a_name,
            "map_b_name": map_b_name,
            "ssim_original": float(ssim_original),
            "ssim_inverted": float(ssim_inverted),
            "relationship": relationship,
            "best_ssim": float(best_ssim),
            "threshold": self.ssim_threshold,
            "is_polarity_inverted": ssim_inverted > ssim_original,
            "timestamp": datetime.now().isoformat()
        }

        self.results_log.append(result)
        return result

    def check_all_cluster_pairs(
        self,
        vae_centroids: np.ndarray,
        baseline_centroids: np.ndarray
    ) -> Dict:
        """
        Compare all VAE centroids against all baseline centroids.

        This helps identify which VAE clusters correspond to which baseline clusters,
        accounting for polarity inversion.

        Args:
            vae_centroids: Shape (n_clusters, H, W) or (n_clusters, 1, H, W)
            baseline_centroids: Shape (n_clusters, H, W) or (n_clusters, n_features)

        Returns:
            Dict with matching analysis
        """
        # Reshape if needed
        if vae_centroids.ndim == 4:
            vae_centroids = vae_centroids.squeeze(1)
        if baseline_centroids.ndim == 1 or (baseline_centroids.ndim == 2 and baseline_centroids.shape[1] == 1600):
            # Reshape flat centroids to 40x40
            n_clusters = baseline_centroids.shape[0]
            baseline_centroids = baseline_centroids.reshape(n_clusters, 40, 40)

        n_vae = vae_centroids.shape[0]
        n_baseline = baseline_centroids.shape[0]

        # Compute all pairwise comparisons
        comparison_matrix_original = np.zeros((n_vae, n_baseline))
        comparison_matrix_inverted = np.zeros((n_vae, n_baseline))

        all_pairs = []

        for i in range(n_vae):
            for j in range(n_baseline):
                result = self.check_map_pair(
                    vae_centroids[i],
                    baseline_centroids[j],
                    f"VAE_Cluster_{i}",
                    f"Baseline_Cluster_{j}"
                )
                comparison_matrix_original[i, j] = result["ssim_original"]
                comparison_matrix_inverted[i, j] = result["ssim_inverted"]
                all_pairs.append(result)

        # Best match for each VAE cluster (accounting for polarity)
        best_matches = []
        for i in range(n_vae):
            max_orig = comparison_matrix_original[i].max()
            max_inv = comparison_matrix_inverted[i].max()

            if max_inv > max_orig:
                best_j = comparison_matrix_inverted[i].argmax()
                best_ssim = max_inv
                is_inverted = True
            else:
                best_j = comparison_matrix_original[i].argmax()
                best_ssim = max_orig
                is_inverted = False

            best_matches.append({
                "vae_cluster": i,
                "baseline_cluster": int(best_j),
                "ssim": float(best_ssim),
                "is_inverted": is_inverted
            })

        summary = {
            "n_vae_clusters": n_vae,
            "n_baseline_clusters": n_baseline,
            "comparison_matrix_original": comparison_matrix_original.tolist(),
            "comparison_matrix_inverted": comparison_matrix_inverted.tolist(),
            "best_matches": best_matches,
            "all_pair_results": all_pairs,
            "n_inverted_matches": sum(1 for m in best_matches if m["is_inverted"]),
            "avg_best_ssim": np.mean([m["ssim"] for m in best_matches])
        }

        return summary

    def check_intra_cluster_redundancy(
        self,
        centroids: np.ndarray,
        cluster_type: str = "VAE"
    ) -> Dict:
        """
        Compare all clusters within the same set to detect redundant/inverted clusters.

        This checks if any cluster is a polarity-inverted version of another cluster,
        which would indicate the model is NOT learning polarity-invariant representations.

        For example, if Cluster 1 ≈ -Cluster 5 (high SSIM when inverted),
        they represent the same microstate and the model has redundancy.

        Args:
            centroids: Shape (n_clusters, H, W) or (n_clusters, 1, H, W)
            cluster_type: Name for logging (e.g., "VAE", "Baseline")

        Returns:
            Dict with redundancy analysis including:
            - similarity_matrix: SSIM between all cluster pairs
            - redundant_pairs: Pairs that are likely the same microstate (inverted)
            - unique_clusters: Estimated number of truly unique microstates
        """
        # Reshape if needed
        if centroids.ndim == 4:
            centroids = centroids.squeeze(1)
        if centroids.ndim == 2 and centroids.shape[1] == 1600:
            n_clusters = centroids.shape[0]
            centroids = centroids.reshape(n_clusters, 40, 40)

        n_clusters = centroids.shape[0]

        # Compute all pairwise comparisons (only upper triangle, i < j)
        similarity_matrix_original = np.zeros((n_clusters, n_clusters))
        similarity_matrix_inverted = np.zeros((n_clusters, n_clusters))

        redundant_pairs = []
        all_pairs = []

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):  # Only compare i < j (avoid duplicates)
                result = self.check_map_pair(
                    centroids[i],
                    centroids[j],
                    f"{cluster_type}_Cluster_{i}",
                    f"{cluster_type}_Cluster_{j}"
                )

                similarity_matrix_original[i, j] = result["ssim_original"]
                similarity_matrix_original[j, i] = result["ssim_original"]  # Symmetric
                similarity_matrix_inverted[i, j] = result["ssim_inverted"]
                similarity_matrix_inverted[j, i] = result["ssim_inverted"]  # Symmetric

                all_pairs.append({
                    "cluster_i": i,
                    "cluster_j": j,
                    **result
                })

                # Check if this pair is redundant (same microstate, opposite polarity)
                if result["relationship"] in ["INVERTED", "SAME"]:
                    redundant_pairs.append({
                        "cluster_i": i,
                        "cluster_j": j,
                        "ssim_original": result["ssim_original"],
                        "ssim_inverted": result["ssim_inverted"],
                        "relationship": result["relationship"]
                    })

        # Fill diagonal with 1.0 (perfect similarity with self)
        np.fill_diagonal(similarity_matrix_original, 1.0)
        np.fill_diagonal(similarity_matrix_inverted, 1.0)

        # Estimate unique clusters (n_clusters minus redundant pairs)
        redundant_cluster_ids = set()
        for pair in redundant_pairs:
            # Mark the higher-indexed cluster as redundant (arbitrary choice)
            redundant_cluster_ids.add(pair["cluster_j"])

        unique_clusters = n_clusters - len(redundant_cluster_ids)

        summary = {
            "cluster_type": cluster_type,
            "n_clusters": n_clusters,
            "similarity_matrix_original": similarity_matrix_original.tolist(),
            "similarity_matrix_inverted": similarity_matrix_inverted.tolist(),
            "redundant_pairs": redundant_pairs,
            "n_redundant_pairs": len(redundant_pairs),
            "redundant_cluster_ids": list(redundant_cluster_ids),
            "unique_clusters_estimate": unique_clusters,
            "all_pair_results": all_pairs,
            "is_polarity_invariant": len(redundant_pairs) == 0,
            "polarity_invariance_score": 1.0 - (len(redundant_pairs) / max(1, n_clusters * (n_clusters - 1) / 2))
        }

        return summary

    def visualize_polarity_comparison(
        self,
        map_a: np.ndarray,
        map_b: np.ndarray,
        result: Dict,
        save_path: Optional[str] = None
    ):
        """Visualize the polarity comparison between two maps."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Ensure 2D
        if map_a.ndim > 2:
            map_a = map_a.squeeze()
        if map_b.ndim > 2:
            map_b = map_b.squeeze()

        vmax = max(np.abs(map_a).max(), np.abs(map_b).max())

        # Map A
        im0 = axes[0].imshow(map_a, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"{result['map_a_name']}")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # Map B
        im1 = axes[1].imshow(map_b, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"{result['map_b_name']}\nSSIM: {result['ssim_original']:.3f}")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Map B Inverted
        im2 = axes[2].imshow(-map_b, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        axes[2].set_title(f"{result['map_b_name']} (Inverted)\nSSIM: {result['ssim_inverted']:.3f}")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        # Summary text
        axes[3].axis('off')
        summary_text = (
            f"Polarity Analysis\n"
            f"{'='*30}\n\n"
            f"SSIM Original: {result['ssim_original']:.4f}\n"
            f"SSIM Inverted: {result['ssim_inverted']:.4f}\n\n"
            f"Relationship: {result['relationship']}\n\n"
            f"Threshold: {result['threshold']}\n"
            f"Is Polarity Inverted: {result['is_polarity_inverted']}"
        )
        axes[3].text(0.1, 0.5, summary_text, transform=axes[3].transAxes,
                     fontsize=12, verticalalignment='center', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle("Polarity Invariance Check", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_results(self, filename: str = "polarity_analysis_results.json"):
        """Save all analysis results to JSON."""
        output_path = self.output_dir / filename

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (bool, int, float, str, type(None))):
                return obj
            else:
                return str(obj)

        serializable_log = convert_to_serializable(self.results_log)

        with open(output_path, 'w') as f:
            json.dump(serializable_log, f, indent=4)

        # Also save human-readable documentation
        doc_path = self.output_dir / "POLARITY_INVERSION_CHECK.txt"
        self._save_polarity_check_documentation(doc_path)

        return output_path

    def _save_polarity_check_documentation(self, filepath: Path):
        """
        Save documentation explaining the polarity inversion check methodology.

        This addresses the question: "Check if maps are inverted -> Two maps
        and then invert 1 of the maps (0,1); SSIM over map A and B see if they are high"
        """
        doc = """
================================================================================
              POLARITY INVERSION CHECK - METHODOLOGY DOCUMENTATION
================================================================================

BACKGROUND: WHY POLARITY INVERSION MATTERS
------------------------------------------
In EEG microstate analysis, brain states are represented by topographic maps
showing voltage distribution across the scalp. A critical property is that
a map and its INVERTED version (multiplied by -1) represent the SAME brain state.

This is because:
1. EEG measures voltage DIFFERENCES, not absolute values
2. The reference electrode choice affects polarity
3. Physiologically, the same neural generators can produce either polarity

Therefore, Map A and -Map A should be treated as IDENTICAL microstates.


METHODOLOGY: HOW WE CHECK FOR INVERSION
---------------------------------------
For each pair of maps (Map_A, Map_B), we compute:

1. ORIGINAL COMPARISON:
   - Normalize both maps to [0,1] range
   - Compute SSIM(Map_A, Map_B)
   - This measures similarity without polarity correction

2. INVERTED COMPARISON:
   - Invert Map_B: Map_B_inverted = -Map_B
   - Normalize both to [0,1] range
   - Compute SSIM(Map_A, Map_B_inverted)
   - This measures similarity with polarity correction

3. DECISION LOGIC:
   - If SSIM_inverted > SSIM_original:
     --> Maps are POLARITY INVERTED versions of each other
   - If SSIM_original > threshold (from config.toml, default 0.95):
     --> Maps are the SAME (no inversion needed)
   - If SSIM_inverted > threshold:
     --> Maps are INVERTED versions of the same state
   - Otherwise:
     --> Maps represent DIFFERENT states


INTERPRETATION GUIDE
--------------------
SSIM Score | Interpretation
-----------|---------------
> 0.95     | Nearly identical (threshold for SAME/INVERTED)
0.85-0.95  | Very similar (high confidence match)
0.70-0.85  | Similar (likely same underlying state)
0.50-0.70  | Moderately similar (possibly related)
< 0.50     | Different maps


EXAMPLE OUTPUT FORMAT
---------------------
{
    "map_a_name": "VAE_Cluster_0",
    "map_b_name": "Baseline_Cluster_2",
    "ssim_original": 0.42,        <-- Low: maps don't match as-is
    "ssim_inverted": 0.96,        <-- High: maps match when inverted!
    "relationship": "INVERTED",   <-- Maps are polarity-inverted
    "is_polarity_inverted": true
}

This tells us: VAE Cluster 0 and Baseline Cluster 2 represent the
SAME brain state, but with opposite polarity.


WHY THIS MATTERS FOR VAE vs BASELINE COMPARISON
-----------------------------------------------
If the VAE learns a microstate with opposite polarity compared to the
baseline, they should still be considered MATCHING clusters. Without
polarity-invariant comparison, we would incorrectly conclude they
found different states.


RESULTS SUMMARY
---------------
"""
        # Add summary of results
        if self.results_log:
            n_inverted = sum(1 for r in self.results_log if r.get("is_polarity_inverted", False))
            n_same = sum(1 for r in self.results_log if r.get("relationship") == "SAME")
            n_different = sum(1 for r in self.results_log if r.get("relationship") == "DIFFERENT")

            doc += f"""
Total comparisons: {len(self.results_log)}
- Maps that are SAME (no inversion): {n_same}
- Maps that are INVERTED (same state, opposite polarity): {n_inverted}
- Maps that are DIFFERENT: {n_different}

Detailed results saved in: polarity_analysis_results.json
Visualizations saved as: match_vae*_baseline*.png
"""
        else:
            doc += "\nNo comparisons performed yet.\n"

        doc += """
================================================================================
"""
        with open(filepath, 'w') as f:
            f.write(doc)

    def visualize_intra_cluster_redundancy(
        self,
        redundancy_result: Dict,
        centroids: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize the intra-cluster redundancy analysis with similarity matrices
        and redundant pair visualizations.

        Args:
            redundancy_result: Output from check_intra_cluster_redundancy()
            centroids: Original centroids array for plotting
            save_path: Path to save the figure
        """
        # Reshape centroids if needed
        if centroids.ndim == 4:
            centroids = centroids.squeeze(1)
        if centroids.ndim == 2 and centroids.shape[1] == 1600:
            centroids = centroids.reshape(-1, 40, 40)

        n_clusters = redundancy_result["n_clusters"]
        cluster_type = redundancy_result["cluster_type"]

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Original SSIM matrix (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        sim_orig = np.array(redundancy_result["similarity_matrix_original"])
        im1 = ax1.imshow(sim_orig, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title(f"{cluster_type} Intra-Cluster SSIM (Original)", fontweight='bold')
        ax1.set_xlabel("Cluster Index")
        ax1.set_ylabel("Cluster Index")
        ax1.set_xticks(range(n_clusters))
        ax1.set_yticks(range(n_clusters))
        plt.colorbar(im1, ax=ax1, label="SSIM")

        # Add text annotations
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    ax1.text(j, i, f"{sim_orig[i, j]:.2f}", ha='center', va='center',
                             fontsize=8, color='white' if sim_orig[i, j] < 0.5 else 'black')

        # 2. Inverted SSIM matrix (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        sim_inv = np.array(redundancy_result["similarity_matrix_inverted"])
        im2 = ax2.imshow(sim_inv, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title(f"{cluster_type} Intra-Cluster SSIM (Inverted)", fontweight='bold')
        ax2.set_xlabel("Cluster Index")
        ax2.set_ylabel("Cluster Index")
        ax2.set_xticks(range(n_clusters))
        ax2.set_yticks(range(n_clusters))
        plt.colorbar(im2, ax=ax2, label="SSIM")

        # Add text annotations and highlight high inverted similarities
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    color = 'white' if sim_inv[i, j] < 0.5 else 'black'
                    weight = 'bold' if sim_inv[i, j] >= self.ssim_threshold else 'normal'
                    ax2.text(j, i, f"{sim_inv[i, j]:.2f}", ha='center', va='center',
                             fontsize=8, color=color, fontweight=weight)

        # 3. Summary statistics (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axis('off')

        summary_text = f"""
INTRA-CLUSTER REDUNDANCY ANALYSIS
==================================
Cluster Type: {cluster_type}
Number of Clusters: {n_clusters}
Total Pairs Compared: {n_clusters * (n_clusters - 1) // 2}

REDUNDANCY DETECTION
--------------------
Redundant Pairs Found: {redundancy_result['n_redundant_pairs']}
Redundant Cluster IDs: {redundancy_result['redundant_cluster_ids']}
Unique Clusters Estimate: {redundancy_result['unique_clusters_estimate']}

POLARITY INVARIANCE
-------------------
Is Polarity Invariant: {'YES' if redundancy_result['is_polarity_invariant'] else 'NO'}
Polarity Invariance Score: {redundancy_result['polarity_invariance_score']:.3f}
(1.0 = fully invariant, 0.0 = all pairs redundant)

INTERPRETATION
--------------
{'✓ All clusters are unique - no redundancy detected!' if redundancy_result['is_polarity_invariant'] else '⚠ Some clusters may be polarity-inverted duplicates!'}
"""
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 4. Redundant pairs visualization (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        redundant_pairs = redundancy_result["redundant_pairs"]

        if redundant_pairs:
            # Show the first few redundant pairs
            n_show = min(3, len(redundant_pairs))
            pair_text = "REDUNDANT PAIRS DETECTED:\n\n"
            for pair in redundant_pairs[:n_show]:
                pair_text += f"Cluster {pair['cluster_i']} ↔ Cluster {pair['cluster_j']}\n"
                pair_text += f"  Original SSIM: {pair['ssim_original']:.3f}\n"
                pair_text += f"  Inverted SSIM: {pair['ssim_inverted']:.3f}\n"
                pair_text += f"  Relationship: {pair['relationship']}\n\n"

            if len(redundant_pairs) > n_show:
                pair_text += f"... and {len(redundant_pairs) - n_show} more pairs"

            ax4.text(0.05, 0.95, pair_text, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "No redundant pairs detected!\n\nAll clusters are unique.",
                     transform=ax4.transAxes, fontsize=14, ha='center', va='center',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax4.axis('off')

        plt.suptitle(f"{cluster_type} Intra-Cluster Redundancy Analysis\n"
                     f"(Detecting polarity-inverted duplicate clusters)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class FairMetricsComputer:
    """
    Compute clustering metrics in a fair and consistent manner for both
    VAE and baseline methods.

    Key principles:
    1. Same feature space for both methods
    2. Same normalization scheme
    3. Full dataset computation (not batch-averaged)
    4. Polarity-invariant cluster assignment option
    """

    def __init__(
        self,
        normalize: bool = True,
        use_polarity_invariance: bool = True,
        feature_space: str = "raw",  # "raw" or "latent"
        logger = None
    ):
        self.normalize = normalize
        self.use_polarity_invariance = use_polarity_invariance
        self.feature_space = feature_space
        self.logger = logger

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """L2 normalize features (row-wise)."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + EPSILON)

    def compute_polarity_invariant_assignment(
        self,
        X: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Assign samples to clusters using polarity-invariant correlation.

        For each sample, computes correlation with each centroid,
        takes absolute value (ignoring polarity), and assigns to
        highest correlation cluster.
        """
        # Normalize for correlation computation
        X_norm = self.normalize_features(X)
        C_norm = self.normalize_features(centroids)

        # Compute correlation (dot product of normalized vectors)
        correlation = np.dot(X_norm, C_norm.T)

        # Ignore polarity (absolute correlation)
        abs_correlation = np.abs(correlation)

        # Assign to highest correlation
        labels = np.argmax(abs_correlation, axis=1)

        return labels

    def compute_gev(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Compute Global Explained Variance (GEV).

        GEV is the standard metric for microstate analysis quality.
        It measures how much of the variance in the data is explained
        by the assigned microstate templates.

        GEV = sum((GFP * correlation)^2) / sum(GFP^2)

        Where GFP (Global Field Power) = std across spatial dimension
        """
        # GFP = spatial standard deviation
        gfp = np.std(X, axis=1)
        gfp_squared_sum = np.sum(gfp ** 2)

        if gfp_squared_sum < EPSILON:
            return 0.0

        # Get assigned centroids for each sample
        assigned_maps = centroids[labels]

        # Center the data for correlation
        X_centered = X - X.mean(axis=1, keepdims=True)
        maps_centered = assigned_maps - assigned_maps.mean(axis=1, keepdims=True)

        # Compute correlation
        numerator = np.sum(X_centered * maps_centered, axis=1)
        denominator = np.sqrt(
            np.sum(X_centered ** 2, axis=1) *
            np.sum(maps_centered ** 2, axis=1)
        )
        correlations = numerator / (denominator + EPSILON)

        # If using polarity invariance, take absolute correlation
        if self.use_polarity_invariance:
            correlations = np.abs(correlations)

        # GEV formula
        gev = np.sum((gfp * correlations) ** 2) / (gfp_squared_sum + EPSILON)

        return float(gev)

    def compute_all_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None,
        method_name: str = "Unknown"
    ) -> Dict:
        """
        Compute all clustering metrics for a given method.

        Args:
            X: Feature matrix (N_samples, N_features)
            labels: Cluster assignments (N_samples,)
            centroids: Cluster centers (N_clusters, N_features), optional
            method_name: Name for logging

        Returns:
            Dict with all metrics
        """
        self._log(f"Computing metrics for {method_name}...")

        # Normalize if requested
        if self.normalize:
            X_metrics = self.normalize_features(X)
        else:
            X_metrics = X

        n_clusters = len(np.unique(labels))

        metrics = {
            "method": method_name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_clusters": n_clusters,
            "normalized": self.normalize,
            "polarity_invariant": self.use_polarity_invariance
        }

        # Clustering metrics (require >= 2 clusters)
        if n_clusters >= 2:
            try:
                # Subsample for silhouette if dataset is large
                if len(X_metrics) > 10000:
                    sample_idx = np.random.choice(
                        len(X_metrics), size=10000, replace=False
                    )
                    metrics["silhouette"] = float(silhouette_score(
                        X_metrics[sample_idx], labels[sample_idx]
                    ))
                else:
                    metrics["silhouette"] = float(silhouette_score(X_metrics, labels))

                metrics["davies_bouldin"] = float(davies_bouldin_score(X_metrics, labels))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_metrics, labels))

            except Exception as e:
                self._log(f"Error computing sklearn metrics: {e}")
                metrics["silhouette"] = -1.0
                metrics["davies_bouldin"] = 10.0
                metrics["calinski_harabasz"] = 0.0
        else:
            self._log(f"Only {n_clusters} cluster(s) found, skipping sklearn metrics")
            metrics["silhouette"] = -1.0
            metrics["davies_bouldin"] = 10.0
            metrics["calinski_harabasz"] = 0.0

        # GEV (requires centroids)
        if centroids is not None:
            # Use original (non-normalized) X for GEV
            metrics["gev"] = self.compute_gev(X, labels, centroids)
        else:
            metrics["gev"] = None

        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        metrics["cluster_distribution"] = {
            int(u): int(c) for u, c in zip(unique, counts)
        }
        metrics["cluster_balance"] = float(counts.min() / counts.max()) if len(counts) > 0 else 0

        self._log(f"  Silhouette: {metrics['silhouette']:.4f}")
        self._log(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        self._log(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.4f}")
        if metrics["gev"] is not None:
            self._log(f"  GEV: {metrics['gev']:.4f}")

        return metrics


def extract_vae_features_and_predictions(
    model,
    data_loader,
    device,
    use_polarity_invariance: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features and predictions from VAE model.

    Returns:
        X_raw: Raw flattened input data (N, 1600)
        Z_latent: Latent representations (N, latent_dim)
        labels_polarity_inv: Polarity-invariant cluster assignments (N,)
        labels_original: Original VAE cluster assignments (N,)
        centroids_decoded: Decoded cluster centroids (K, 1600)
    """
    model.eval()

    X_raw_list = []
    Z_latent_list = []
    labels_polarity_inv_list = []
    labels_original_list = []

    with torch.no_grad():
        # Get decoded centroids
        centroids_decoded = model.decode(model.mu_c).detach().cpu().numpy()
        centroids_decoded = centroids_decoded.reshape(model.nClusters, -1)

        for data, _ in tqdm(data_loader, desc="Extracting VAE features"):
            data = data.to(device)

            # Get latent representation
            mu, _ = model.encode(data)

            # Get ORIGINAL VAE predictions (for latent space metrics)
            original_labels = model.predict(data)
            labels_original_list.append(original_labels)

            # Get polarity-invariant predictions (for fair comparison)
            if use_polarity_invariance:
                X_batch = data.view(data.size(0), -1).cpu().numpy()
                X_norm = X_batch / (np.linalg.norm(X_batch, axis=1, keepdims=True) + EPSILON)
                C_norm = centroids_decoded / (np.linalg.norm(centroids_decoded, axis=1, keepdims=True) + EPSILON)

                correlation = np.dot(X_norm, C_norm.T)
                abs_correlation = np.abs(correlation)
                polarity_inv_labels = np.argmax(abs_correlation, axis=1)
            else:
                polarity_inv_labels = original_labels

            X_raw_list.append(data.view(data.size(0), -1).cpu().numpy())
            Z_latent_list.append(mu.detach().cpu().numpy())
            labels_polarity_inv_list.append(polarity_inv_labels)

    X_raw = np.concatenate(X_raw_list, axis=0)
    Z_latent = np.concatenate(Z_latent_list, axis=0)
    labels_polarity_inv = np.concatenate(labels_polarity_inv_list, axis=0)
    labels_original = np.concatenate(labels_original_list, axis=0)

    return X_raw, Z_latent, labels_polarity_inv, labels_original, centroids_decoded


def compute_fair_comparison(
    vae_model,
    baseline_handler,
    test_loader,
    device,
    output_dir: str,
    logger = None,
    baseline_cluster_metrics: Dict = None
) -> Dict:
    """
    Compute fair comparison metrics between VAE and baseline.

    Both methods are evaluated:
    1. In RAW feature space (1600-dim, normalized)
    2. With polarity-invariant assignments
    3. On the FULL test set
    4. With GEV computation

    Parameters
    ----------
    baseline_cluster_metrics : Dict, optional
        Pycrostates cluster validation metrics (silhouette, calinski-harabasz, dunn, davies-bouldin)
        from baseline_handler.compute_cluster_metrics()
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("=" * 60)
        logger.info("COMPUTING FAIR COMPARISON METRICS")
        logger.info("=" * 60)

    # Initialize metrics computer
    metrics_computer = FairMetricsComputer(
        normalize=True,
        use_polarity_invariance=True,
        logger=logger
    )

    # Extract VAE data with polarity invariance
    if logger:
        logger.info("Extracting VAE features with polarity-invariant assignment...")

    X_raw, Z_latent, vae_labels_polarity_inv, vae_labels_original, vae_centroids = extract_vae_features_and_predictions(
        vae_model, test_loader, device, use_polarity_invariance=True
    )

    # Check if baseline uses standard pycrostates (n-channel) or old (1600-dim) approach
    baseline_n_features = baseline_handler.cluster_centers_.shape[1] if baseline_handler.cluster_centers_ is not None else 0
    use_old_baseline = (baseline_n_features == 1600)

    if not use_old_baseline:
        if logger:
            logger.info(f"Baseline uses {baseline_n_features}-channel electrode space (pycrostates).")
            logger.info("Skipping direct comparison with VAE (different feature spaces).")
            logger.info("Computing VAE metrics only...")

    # Extract baseline data (only if using old 1600-dim approach)
    baseline_labels = None
    baseline_metrics_raw = None

    if use_old_baseline:
        if logger:
            logger.info("Extracting baseline predictions...")

        X_baseline = []
        for data, _ in test_loader:
            X_baseline.append(data.view(data.size(0), -1).numpy())
        X_baseline = np.concatenate(X_baseline, axis=0)

        # Get baseline predictions (already polarity-invariant)
        X_norm = X_baseline / (np.linalg.norm(X_baseline, axis=1, keepdims=True) + EPSILON)
        C_norm = baseline_handler.cluster_centers_ / (
            np.linalg.norm(baseline_handler.cluster_centers_, axis=1, keepdims=True) + EPSILON
        )
        correlation = np.dot(X_norm, C_norm.T)
        baseline_labels = np.argmax(np.abs(correlation), axis=1)

    # Compute metrics in RAW feature space for VAE
    if logger:
        logger.info("\n--- Metrics in RAW Feature Space ---")

    vae_metrics_raw = metrics_computer.compute_all_metrics(
        X_raw, vae_labels_polarity_inv, vae_centroids, "VAE (Polarity-Invariant)"
    )

    # Baseline metrics only if using old 1600-dim approach
    if use_old_baseline:
        baseline_metrics_raw = metrics_computer.compute_all_metrics(
            X_baseline, baseline_labels, baseline_handler.cluster_centers_, "Baseline (ModKMeans)"
        )

    # Also compute VAE metrics in latent space for reference
    # IMPORTANT: Use ORIGINAL VAE labels here, not polarity-invariant ones!
    if logger:
        logger.info("\n--- VAE Metrics in Latent Space (Reference - Original Labels) ---")

    # For latent space, use original VAE predictions (not polarity-invariant)
    vae_metrics_latent = metrics_computer.compute_all_metrics(
        Z_latent, vae_labels_original, vae_model.mu_c.detach().cpu().numpy(), "VAE (Latent Space)"
    )

    # Also compute VAE metrics in RAW space with ORIGINAL labels for comparison
    if logger:
        logger.info("\n--- VAE Metrics in RAW Space (Original Labels - for comparison) ---")

    vae_metrics_raw_original = metrics_computer.compute_all_metrics(
        X_raw, vae_labels_original, vae_centroids, "VAE (Original Labels)"
    )

    # Polarity analysis
    polarity_summary = None
    vae_redundancy = None
    baseline_redundancy = None

    # Always run VAE intra-cluster redundancy check (independent of baseline type)
    if logger:
        logger.info("\n--- VAE Intra-Cluster Redundancy Analysis ---")
        logger.info("Checking if any VAE clusters are polarity-inverted duplicates...")

    polarity_checker = PolarityInvarianceChecker(
        output_dir=str(output_dir / "polarity_analysis")
    )

    # Check VAE clusters for redundancy
    vae_centroids_2d = vae_centroids.reshape(-1, 40, 40)
    vae_redundancy = polarity_checker.check_intra_cluster_redundancy(
        vae_centroids_2d, cluster_type="VAE"
    )

    # Visualize VAE redundancy analysis
    polarity_checker.visualize_intra_cluster_redundancy(
        vae_redundancy, vae_centroids_2d,
        save_path=str(output_dir / "polarity_analysis" / "vae_intra_cluster_redundancy.png")
    )

    if logger:
        logger.info(f"VAE - Redundant pairs: {vae_redundancy['n_redundant_pairs']}, "
                    f"Unique clusters: {vae_redundancy['unique_clusters_estimate']}/{vae_redundancy['n_clusters']}")
        if vae_redundancy['is_polarity_invariant']:
            logger.info("VAE clusters are polarity-invariant (no redundant pairs detected)")
        else:
            logger.info(f"WARNING: VAE has polarity-inverted duplicate clusters: {vae_redundancy['redundant_pairs']}")

    # Additional baseline comparison (only if using old 1600-dim baseline approach)
    if use_old_baseline:
        if logger:
            logger.info("\n--- VAE vs Baseline Polarity Analysis ---")

        # Reshape baseline centroids for comparison
        baseline_centroids_2d = baseline_handler.cluster_centers_.reshape(-1, 40, 40)

        polarity_summary = polarity_checker.check_all_cluster_pairs(
            vae_centroids_2d, baseline_centroids_2d
        )

        # Visualize best matches
        for match in polarity_summary["best_matches"]:
            i, j = match["vae_cluster"], match["baseline_cluster"]
            result = polarity_checker.check_map_pair(
                vae_centroids_2d[i], baseline_centroids_2d[j],
                f"VAE_{i}", f"Baseline_{j}"
            )
            polarity_checker.visualize_polarity_comparison(
                vae_centroids_2d[i], baseline_centroids_2d[j], result,
                save_path=str(output_dir / "polarity_analysis" / f"match_vae{i}_baseline{j}.png")
            )

        # Check Baseline clusters for redundancy
        baseline_redundancy = polarity_checker.check_intra_cluster_redundancy(
            baseline_centroids_2d, cluster_type="Baseline"
        )

        polarity_checker.visualize_intra_cluster_redundancy(
            baseline_redundancy, baseline_centroids_2d,
            save_path=str(output_dir / "polarity_analysis" / "baseline_intra_cluster_redundancy.png")
        )

        if logger:
            logger.info(f"Baseline - Redundant pairs: {baseline_redundancy['n_redundant_pairs']}, "
                        f"Unique clusters: {baseline_redundancy['unique_clusters_estimate']}/{baseline_redundancy['n_clusters']}")

    polarity_checker.save_results()

    # Compile results
    comparison_results = {
        "fair_comparison": {
            "description": "VAE metrics in RAW feature space" if not use_old_baseline else "Both methods evaluated in RAW feature space with polarity invariance",
            "vae": vae_metrics_raw,
            "baseline": baseline_metrics_raw if use_old_baseline else {"note": "Baseline uses N-channel electrode space (pycrostates), not comparable"},
        },
        "vae_original_labels": {
            "description": "VAE metrics in RAW space with ORIGINAL labels (no polarity correction)",
            "metrics": vae_metrics_raw_original
        },
        "vae_latent_reference": {
            "description": "VAE metrics in LATENT space with original labels (VAE representation quality)",
            "metrics": vae_metrics_latent
        },
        "label_comparison": {
            "description": "Shows effect of polarity-invariant vs original label assignment",
            "polarity_invariant_silhouette": vae_metrics_raw.get("silhouette", -1),
            "original_labels_silhouette": vae_metrics_raw_original.get("silhouette", -1),
            "latent_space_silhouette": vae_metrics_latent.get("silhouette", -1),
        },
        "winner_by_metric": {},
        "timestamp": datetime.now().isoformat()
    }

    # Add pycrostates baseline metrics if provided
    if baseline_cluster_metrics is not None:
        comparison_results["baseline_pycrostates"] = {
            "description": "Baseline (ModKMeans) cluster quality metrics from pycrostates",
            "n_clusters": baseline_cluster_metrics.get("n_clusters", baseline_handler.n_clusters),
            "silhouette": baseline_cluster_metrics.get("Silhouette"),
            "calinski_harabasz": baseline_cluster_metrics.get("Calinski-Harabasz"),
            "dunn": baseline_cluster_metrics.get("Dunn"),
            "davies_bouldin": baseline_cluster_metrics.get("Davies-Bouldin"),
            "gev": baseline_handler.modk.GEV_ if hasattr(baseline_handler.modk, 'GEV_') else None
        }
        if logger:
            logger.info("\n--- Baseline (pycrostates) Cluster Metrics ---")
            logger.info(f"  Silhouette:        {baseline_cluster_metrics.get('Silhouette', 'N/A'):.4f}" if baseline_cluster_metrics.get('Silhouette') else "  Silhouette: N/A")
            logger.info(f"  Calinski-Harabasz: {baseline_cluster_metrics.get('Calinski-Harabasz', 'N/A'):.2f}" if baseline_cluster_metrics.get('Calinski-Harabasz') else "  Calinski-Harabasz: N/A")
            logger.info(f"  Dunn Index:        {baseline_cluster_metrics.get('Dunn', 'N/A'):.4f}" if baseline_cluster_metrics.get('Dunn') else "  Dunn Index: N/A")
            logger.info(f"  Davies-Bouldin:    {baseline_cluster_metrics.get('Davies-Bouldin', 'N/A'):.4f}" if baseline_cluster_metrics.get('Davies-Bouldin') else "  Davies-Bouldin: N/A")
            if hasattr(baseline_handler.modk, 'GEV_'):
                logger.info(f"  GEV:               {baseline_handler.modk.GEV_:.4f}")

    # Always add VAE intra-cluster redundancy results
    comparison_results["intra_cluster_redundancy"] = {
        "description": "Detects if clusters within the same method are polarity-inverted duplicates",
        "vae": {
            "n_redundant_pairs": vae_redundancy["n_redundant_pairs"],
            "redundant_cluster_ids": vae_redundancy["redundant_cluster_ids"],
            "unique_clusters_estimate": vae_redundancy["unique_clusters_estimate"],
            "is_polarity_invariant": vae_redundancy["is_polarity_invariant"],
            "polarity_invariance_score": vae_redundancy["polarity_invariance_score"],
            "redundant_pairs": vae_redundancy["redundant_pairs"]
        }
    }

    # Add baseline polarity analysis only if using old baseline
    if use_old_baseline and polarity_summary is not None:
        comparison_results["polarity_analysis"] = {
            "summary": {
                "n_inverted_matches": polarity_summary["n_inverted_matches"],
                "avg_best_ssim": polarity_summary["avg_best_ssim"],
            },
            "best_matches": polarity_summary["best_matches"]
        }
        # Add baseline redundancy to the existing intra_cluster_redundancy
        comparison_results["intra_cluster_redundancy"]["baseline"] = {
            "n_redundant_pairs": baseline_redundancy["n_redundant_pairs"],
            "redundant_cluster_ids": baseline_redundancy["redundant_cluster_ids"],
            "unique_clusters_estimate": baseline_redundancy["unique_clusters_estimate"],
            "is_polarity_invariant": baseline_redundancy["is_polarity_invariant"],
            "polarity_invariance_score": baseline_redundancy["polarity_invariance_score"],
            "redundant_pairs": baseline_redundancy["redundant_pairs"]
        }

    # Determine winners (only if using old baseline)
    if use_old_baseline and baseline_metrics_raw is not None:
        metrics_to_compare = ["silhouette", "davies_bouldin", "calinski_harabasz", "gev"]
        higher_is_better = {"silhouette": True, "davies_bouldin": False, "calinski_harabasz": True, "gev": True}

        for metric in metrics_to_compare:
            vae_val = vae_metrics_raw.get(metric)
            baseline_val = baseline_metrics_raw.get(metric)

            if vae_val is None or baseline_val is None:
                comparison_results["winner_by_metric"][metric] = "N/A"
                continue

            if higher_is_better[metric]:
                winner = "VAE" if vae_val > baseline_val else "Baseline"
            else:
                winner = "VAE" if vae_val < baseline_val else "Baseline"

            comparison_results["winner_by_metric"][metric] = winner
    else:
        comparison_results["winner_by_metric"] = {"note": "Baseline uses different feature space, direct comparison not applicable"}

    # Save results
    results_path = output_dir / "fair_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=4, default=str)

    # Generate comparison report
    report = generate_comparison_report(comparison_results)
    report_path = output_dir / "fair_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    if logger:
        logger.info(f"\nResults saved to {output_dir}")
        logger.info("\n" + report)

    return comparison_results


def generate_comparison_report(results: Dict) -> str:
    """Generate a human-readable comparison report."""
    vae = results["fair_comparison"]["vae"]
    baseline = results["fair_comparison"]["baseline"]
    vae_original = results.get("vae_original_labels", {}).get("metrics", {})
    vae_latent = results.get("vae_latent_reference", {}).get("metrics", {})
    polarity = results.get("polarity_analysis", None)
    label_comparison = results.get("label_comparison", {})
    intra_redundancy = results.get("intra_cluster_redundancy", None)
    baseline_pycrostates = results.get("baseline_pycrostates", None)

    # Check if baseline uses different feature space (pycrostates)
    baseline_incompatible = isinstance(baseline, dict) and "note" in baseline

    report = """
================================================================================
                    FAIR COMPARISON: VAE vs Modified K-Means
================================================================================

METHODOLOGY:
------------
- Both methods evaluated in RAW feature space (1600 dimensions)
- Both use L2 normalization before metric computation
- Both use polarity-invariant cluster assignment
- Metrics computed on FULL test set (not batch-averaged)
- GEV computed for both methods

FAIR COMPARISON (Both Polarity-Invariant, RAW Space):
-----------------------------------------------------
{metric_table}

WINNER BY METRIC:
-----------------
{winner_table}

BASELINE (pycrostates ModKMeans) CLUSTER QUALITY:
-------------------------------------------------
{baseline_pycrostates_table}

EFFECT OF LABEL ASSIGNMENT METHOD:
----------------------------------
{label_effect_table}

This shows how polarity-invariant assignment affects VAE metrics.
"Original Labels" = VAE's native cluster assignment (posterior probability)
"Polarity-Invariant" = Re-assigned using |correlation| with decoded centroids

POLARITY ANALYSIS (VAE vs Baseline):
-------------------------------------
- Maps with inverted polarity detected: {n_inverted}/{n_total}
- Average best SSIM between matched clusters: {avg_ssim:.4f}

CLUSTER MATCHES (VAE -> Baseline):
----------------------------------
{matches}

INTRA-CLUSTER REDUNDANCY ANALYSIS:
----------------------------------
This detects if clusters within the SAME method are polarity-inverted duplicates.

VAE Redundancy:
{vae_redundancy_summary}

Baseline (ModKMeans) Redundancy:
{baseline_redundancy_summary}

INTERPRETATION:
- If unique clusters < requested clusters, some learned patterns are duplicates
{redundancy_comparison}
- Traditional metrics may favor methods with more redundant clusters

NOTES:
------
1. Calinski-Harabasz scales with sample size - higher values on full dataset
   are expected compared to batch-averaged training metrics.
2. The VAE latent space metrics use original labels and measure representation
   quality, not clustering in the comparable feature space.
3. For fair VAE vs Baseline comparison, use the "Fair Comparison" metrics above.

================================================================================
"""

    # Build metric table
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "gev"]
    metric_lines = []
    metric_lines.append(f"{'Metric':<25} | {'VAE':<15} | {'Baseline':<15} | {'Better':<10}")
    metric_lines.append("-" * 70)

    higher_better = {"silhouette": True, "davies_bouldin": False, "calinski_harabasz": True, "gev": True}

    for m in metrics:
        vae_val = vae.get(m) if isinstance(vae, dict) else None
        base_val = baseline.get(m) if isinstance(baseline, dict) and not baseline_incompatible else None

        # Format VAE value
        vae_str = f"{vae_val:.4f}" if vae_val is not None else "N/A"
        # Format baseline value
        base_str = f"{base_val:.4f}" if base_val is not None else "N/A (diff. space)"

        # Determine better
        if vae_val is not None and base_val is not None:
            if higher_better[m]:
                better = "VAE" if vae_val > base_val else "Baseline"
            else:
                better = "VAE" if vae_val < base_val else "Baseline"
        else:
            better = "-"

        metric_lines.append(f"{m:<25} | {vae_str:<15} | {base_str:<15} | {better:<10}")

    metric_table = "\n".join(metric_lines)

    # Build winner table
    winners = results["winner_by_metric"]
    if winners:
        winner_lines = [f"  {m}: {w}" for m, w in winners.items()]
        winner_table = "\n".join(winner_lines)
    else:
        winner_table = "  (Baseline uses different feature space - direct comparison not applicable)"

    # Build label effect table (shows how different label methods affect VAE metrics)
    label_effect_lines = []
    label_effect_lines.append(f"{'Metric':<20} | {'Polarity-Inv':<15} | {'Original':<15} | {'Latent Space':<15}")
    label_effect_lines.append("-" * 70)

    for m in ["silhouette", "davies_bouldin", "calinski_harabasz", "gev"]:
        pol_val = vae.get(m, "N/A")
        orig_val = vae_original.get(m, "N/A")
        lat_val = vae_latent.get(m, "N/A")

        pol_str = f"{pol_val:.4f}" if isinstance(pol_val, (int, float)) else "N/A"
        orig_str = f"{orig_val:.4f}" if isinstance(orig_val, (int, float)) else "N/A"
        lat_str = f"{lat_val:.4f}" if isinstance(lat_val, (int, float)) else "N/A"

        label_effect_lines.append(f"{m:<20} | {pol_str:<15} | {orig_str:<15} | {lat_str:<15}")

    label_effect_table = "\n".join(label_effect_lines)

    # Build matches (only if polarity analysis was performed)
    if polarity is not None:
        match_lines = []
        for match in polarity["best_matches"]:
            inv_str = " (INVERTED)" if match["is_inverted"] else ""
            match_lines.append(
                f"  VAE Cluster {match['vae_cluster']} -> "
                f"Baseline Cluster {match['baseline_cluster']} "
                f"(SSIM: {match['ssim']:.3f}){inv_str}"
            )
        matches = "\n".join(match_lines)
        n_inverted = polarity["summary"]["n_inverted_matches"]
        n_total = len(polarity["best_matches"])
        avg_ssim = polarity["summary"]["avg_best_ssim"]
    else:
        matches = "  N/A - Baseline uses different feature space (pycrostates n-channel)"
        n_inverted = 0
        n_total = 0
        avg_ssim = 0.0

    # Build intra-cluster redundancy summaries
    if intra_redundancy is not None:
        vae_red = intra_redundancy.get("vae", {})
        baseline_red = intra_redundancy.get("baseline", {})
    else:
        vae_red = {}
        baseline_red = {}

    def build_redundancy_summary(red_data: Dict, method_name: str) -> str:
        if not red_data:
            return f"  {method_name}: No data available"

        lines = []
        n_clusters = red_data.get("n_clusters", "N/A")
        unique = red_data.get("unique_clusters_estimate", n_clusters)
        n_redundant = red_data.get("n_redundant_pairs", 0)
        score = red_data.get("polarity_invariance_score", 1.0)
        is_invariant = red_data.get("is_polarity_invariant", True)

        lines.append(f"  - Requested clusters: {n_clusters}")
        lines.append(f"  - Unique clusters found: {unique}")
        lines.append(f"  - Redundant pairs: {n_redundant}")
        lines.append(f"  - Polarity invariance score: {score:.3f}")
        lines.append(f"  - Is fully polarity-invariant: {'Yes' if is_invariant else 'No'}")

        # List redundant pairs if any
        redundant_pairs = red_data.get("redundant_pairs", [])
        if redundant_pairs:
            lines.append(f"  - Redundant pairs detected:")
            for pair in redundant_pairs[:5]:  # Show max 5
                rel = pair.get("relationship", "UNKNOWN")
                lines.append(f"      Cluster {pair['cluster_i']} <-> Cluster {pair['cluster_j']} ({rel})")
            if len(redundant_pairs) > 5:
                lines.append(f"      ... and {len(redundant_pairs) - 5} more")

        return "\n".join(lines)

    vae_redundancy_summary = build_redundancy_summary(vae_red, "VAE")
    baseline_redundancy_summary = build_redundancy_summary(baseline_red, "Baseline")

    # Build dynamic redundancy comparison text
    vae_unique = vae_red.get("unique_clusters_estimate", "N/A")
    baseline_unique = baseline_red.get("unique_clusters_estimate", "N/A")
    if isinstance(vae_unique, (int, float)) and isinstance(baseline_unique, (int, float)):
        if baseline_unique < vae_unique:
            redundancy_comparison = f"- Baseline has MORE redundancy ({baseline_unique} unique) than VAE ({vae_unique} unique)"
        elif vae_unique < baseline_unique:
            redundancy_comparison = f"- VAE has MORE redundancy ({vae_unique} unique) than Baseline ({baseline_unique} unique)"
        else:
            redundancy_comparison = f"- Both methods have same unique clusters ({vae_unique} unique)"
    else:
        redundancy_comparison = f"- VAE unique clusters: {vae_unique}, Baseline unique clusters: {baseline_unique}"

    # Build pycrostates baseline metrics table
    if baseline_pycrostates is not None:
        pycrostates_lines = []
        n_clusters = baseline_pycrostates.get("n_clusters", "N/A")
        silhouette = baseline_pycrostates.get("silhouette")
        ch = baseline_pycrostates.get("calinski_harabasz")
        dunn = baseline_pycrostates.get("dunn")
        db = baseline_pycrostates.get("davies_bouldin")
        gev = baseline_pycrostates.get("gev")

        pycrostates_lines.append(f"  Number of Clusters:  {n_clusters}")
        pycrostates_lines.append(f"  Silhouette Score:    {silhouette:.4f}" if silhouette is not None else "  Silhouette Score:    N/A")
        pycrostates_lines.append(f"  Calinski-Harabasz:   {ch:.2f}" if ch is not None else "  Calinski-Harabasz:   N/A")
        pycrostates_lines.append(f"  Dunn Index:          {dunn:.4f}" if dunn is not None else "  Dunn Index:          N/A")
        pycrostates_lines.append(f"  Davies-Bouldin:      {db:.4f}" if db is not None else "  Davies-Bouldin:      N/A")
        pycrostates_lines.append(f"  GEV:                 {gev:.4f}" if gev is not None else "  GEV:                 N/A")
        pycrostates_lines.append("")
        pycrostates_lines.append("  Note: These metrics are computed in native electrode space")
        pycrostates_lines.append("        (not directly comparable to VAE's 1600-dim topographic space)")
        baseline_pycrostates_table = "\n".join(pycrostates_lines)
    else:
        baseline_pycrostates_table = "  No pycrostates metrics available (call baseline.compute_cluster_metrics() first)"

    return report.format(
        metric_table=metric_table,
        winner_table=winner_table,
        baseline_pycrostates_table=baseline_pycrostates_table,
        label_effect_table=label_effect_table,
        n_inverted=n_inverted,
        n_total=n_total,
        avg_ssim=avg_ssim,
        matches=matches,
        vae_redundancy_summary=vae_redundancy_summary,
        baseline_redundancy_summary=baseline_redundancy_summary,
        redundancy_comparison=redundancy_comparison
    )
