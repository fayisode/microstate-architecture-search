"""
Centroid Redundancy Analysis Module
====================================
Analyzes VAE cluster centroids for redundancy:
1. Computes SSIM between centroids
2. Computes SSIM between centroids and inverted centroids
3. Computes spatial correlation matrices
4. Identifies redundant pairs (similar or polarity-inverted)
5. Merges redundant clusters and recomputes metrics
6. Creates comprehensive visualization

Author: Auto-generated for microstate_eeg project
"""

import numpy as np
import torch
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    plt = None
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import TSNE
from metrics_utils import pycrostates_silhouette_score
from config.config import config as cfg
import json
from typing import Dict, List, Tuple, Optional
import logging


class CentroidAnalyzer:
    """
    Analyzes VAE centroids for redundancy and creates comprehensive visualizations.
    """

    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None, norm_params=None):
        self.output_dir = Path(output_dir) / "centroid_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.norm_params = norm_params

        # Thresholds from config (single source of truth)
        thresholds = cfg.get_merge_thresholds()
        self.ssim_threshold = thresholds["ssim_threshold"]
        self.corr_threshold = thresholds["corr_threshold"]

    def extract_centroids(self, model) -> np.ndarray:
        """
        Extract decoded centroids from VAE model.

        Returns
        -------
        centroids : np.ndarray
            Shape (n_clusters, H, W) - decoded centroid images
        """
        model.eval()
        with torch.no_grad():
            # Get latent centroids
            mu_c = model.mu_c.detach().cpu().numpy()  # (n_clusters, latent_dim)

            # Decode each centroid
            centroids = []
            for k in range(model.nClusters):
                z = torch.tensor(mu_c[k:k+1], dtype=torch.float32).to(next(model.parameters()).device)
                decoded = model.decode(z)
                centroid_img = decoded[0, 0].detach().cpu().numpy()  # (H, W)
                centroids.append(centroid_img)

            centroids = np.array(centroids)

        # Denormalize from z-score back to microvolts if norm_params available
        if self.norm_params is not None:
            z_mean = self.norm_params.get("mean", 0.0)
            z_std = self.norm_params.get("std", 1.0)
            centroids = centroids * z_std + z_mean
            self.logger.info(f"Denormalized centroids to μV (mean={z_mean:.4f}, std={z_std:.4f})")

        self.logger.info(f"Extracted {len(centroids)} centroids with shape {centroids[0].shape}")
        return centroids

    def _normalize_centroids(self, centroids: np.ndarray) -> np.ndarray:
        """Normalize centroids to [0, 1] range for SSIM computation."""
        normalized = np.zeros_like(centroids)
        for i, c in enumerate(centroids):
            c_range = c.max() - c.min()
            if c_range > 1e-8:
                normalized[i] = (c - c.min()) / c_range
        return normalized

    def compute_ssim_matrix(self, centroids: np.ndarray, inverted: bool = False) -> np.ndarray:
        """
        Compute SSIM matrix between centroids.

        Parameters
        ----------
        centroids : np.ndarray
            Shape (n_clusters, H, W)
        inverted : bool
            If True, compare centroids with inverted versions

        Returns
        -------
        ssim_matrix : np.ndarray
            Shape (n_clusters, n_clusters)
        """
        n_clusters = len(centroids)
        centroids_norm = self._normalize_centroids(centroids)
        compare_centroids = 1.0 - centroids_norm if inverted else centroids_norm

        ssim_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                ssim_matrix[i, j] = ssim(centroids_norm[i], compare_centroids[j], data_range=1.0)

        return ssim_matrix

    def compute_correlation_matrix(self, centroids: np.ndarray, inverted: bool = False) -> np.ndarray:
        """
        Compute spatial correlation matrix between centroids.

        Parameters
        ----------
        centroids : np.ndarray
            Shape (n_clusters, H, W)
        inverted : bool
            If True, compare centroids with inverted versions

        Returns
        -------
        corr_matrix : np.ndarray
            Shape (n_clusters, n_clusters)
        """
        n_clusters = len(centroids)
        corr_matrix = np.zeros((n_clusters, n_clusters))

        # Flatten centroids
        flat = centroids.reshape(n_clusters, -1)

        # Invert if requested
        if inverted:
            # Flip around mean
            means = flat.mean(axis=1, keepdims=True)
            compare_flat = 2 * means - flat
        else:
            compare_flat = flat

        # Center the data
        flat_centered = flat - flat.mean(axis=1, keepdims=True)
        compare_centered = compare_flat - compare_flat.mean(axis=1, keepdims=True)

        # Compute norms
        flat_norms = np.linalg.norm(flat_centered, axis=1, keepdims=True)
        compare_norms = np.linalg.norm(compare_centered, axis=1, keepdims=True)

        # Avoid division by zero
        flat_norms[flat_norms == 0] = 1e-8
        compare_norms[compare_norms == 0] = 1e-8

        # Normalize
        flat_normed = flat_centered / flat_norms
        compare_normed = compare_centered / compare_norms

        # Correlation matrix
        corr_matrix = np.dot(flat_normed, compare_normed.T)

        return corr_matrix

    def _is_redundant(self, ssim_val: float, corr_val: float) -> bool:
        """Check if a pair is redundant based on thresholds."""
        return ssim_val >= self.ssim_threshold or abs(corr_val) >= self.corr_threshold

    def detect_redundant_pairs(
        self,
        ssim_matrix: np.ndarray,
        ssim_inverted_matrix: np.ndarray,
        corr_matrix: np.ndarray,
        corr_inverted_matrix: np.ndarray,
    ) -> Dict:
        """
        Detect redundant centroid pairs based on SSIM and correlation.

        Returns
        -------
        redundancy_info : dict
            Contains lists of similar and inverted pairs
        """
        n_clusters = ssim_matrix.shape[0]
        similar_pairs = []
        inverted_pairs = []

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if self._is_redundant(ssim_matrix[i, j], corr_matrix[i, j]):
                    similar_pairs.append({
                        "pair": (i, j),
                        "ssim": float(ssim_matrix[i, j]),
                        "correlation": float(abs(corr_matrix[i, j])),
                        "type": "similar"
                    })

                if self._is_redundant(ssim_inverted_matrix[i, j], corr_inverted_matrix[i, j]):
                    inverted_pairs.append({
                        "pair": (i, j),
                        "ssim_inverted": float(ssim_inverted_matrix[i, j]),
                        "correlation_inverted": float(abs(corr_inverted_matrix[i, j])),
                        "type": "inverted"
                    })

        return {
            "similar_pairs": similar_pairs,
            "inverted_pairs": inverted_pairs,
            "total_redundant": len(similar_pairs) + len(inverted_pairs),
            "thresholds": {"ssim": self.ssim_threshold, "correlation": self.corr_threshold}
        }

    def merge_clusters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        merge_pairs: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Merge redundant clusters and return new labels.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        labels : np.ndarray
            Original cluster labels
        merge_pairs : list of tuples
            Pairs of cluster indices to merge

        Returns
        -------
        new_labels : np.ndarray
            Labels after merging
        merge_info : dict
            Information about the merge
        """
        new_labels = labels.copy()

        # Build merge map
        merge_map = {}
        for i, j in merge_pairs:
            # Always merge higher index into lower
            src, dst = max(i, j), min(i, j)
            merge_map[src] = dst

        # Apply merges
        for src, dst in merge_map.items():
            new_labels[new_labels == src] = dst

        # Remap to contiguous labels
        unique_labels = np.unique(new_labels)
        label_remap = {old: new for new, old in enumerate(unique_labels)}
        new_labels = np.array([label_remap[l] for l in new_labels])

        merge_info = {
            "original_k": len(np.unique(labels)),
            "merged_k": len(np.unique(new_labels)),
            "merge_pairs": merge_pairs,
            "merge_map": {str(k): v for k, v in merge_map.items()},
        }

        return new_labels, merge_info

    def compute_gev(
        self,
        raw_data: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """
        Compute Global Explained Variance (GEV) for given labels and centroids.

        Parameters
        ----------
        raw_data : np.ndarray
            Raw data in electrode space, shape (n_samples, n_channels)
        labels : np.ndarray
            Cluster labels, shape (n_samples,)
        centroids : np.ndarray
            Cluster centroids in electrode space, shape (n_clusters, n_channels)

        Returns
        -------
        gev : float
            Global Explained Variance (0 to 1)
        """
        if raw_data is None or centroids is None:
            return -1.0

        n_samples = len(labels)
        unique_labels = np.unique(labels)

        # Center raw data (zero mean per sample)
        raw_centered = raw_data - raw_data.mean(axis=1, keepdims=True)

        # Compute GFP (Global Field Power) for each sample
        gfp = np.sqrt(np.mean(raw_centered ** 2, axis=1))
        gfp_squared = gfp ** 2
        total_gfp = np.sum(gfp_squared)

        if total_gfp < 1e-10:
            return 0.0

        # Compute explained variance for each sample
        explained_variance = 0.0

        for k in unique_labels:
            mask = labels == k
            if not np.any(mask):
                continue

            # Get centroid for this cluster (remap if needed)
            centroid_idx = k if k < len(centroids) else 0
            centroid = centroids[centroid_idx]
            centroid_centered = centroid - centroid.mean()
            centroid_norm = np.linalg.norm(centroid_centered)

            if centroid_norm < 1e-10:
                continue

            centroid_normalized = centroid_centered / centroid_norm

            # Compute correlation for samples in this cluster
            samples = raw_centered[mask]
            sample_norms = np.linalg.norm(samples, axis=1, keepdims=True)
            sample_norms[sample_norms == 0] = 1e-10
            samples_normalized = samples / sample_norms

            # Correlation (polarity-invariant)
            correlations = np.abs(np.dot(samples_normalized, centroid_normalized))

            # Explained variance = correlation² × GFP²
            explained_variance += np.sum((correlations ** 2) * gfp_squared[mask])

        gev = explained_variance / total_gfp
        return float(np.clip(gev, 0, 1))

    def compute_clustering_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        prefix: str = "",
        raw_data: np.ndarray = None,
        centroids: np.ndarray = None,
    ) -> Dict:
        """
        Compute clustering metrics including GEV and pycrostates silhouette.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix for silhouette computation (latent or decoded)
        labels : np.ndarray
            Cluster labels
        prefix : str
            Prefix for metric keys (e.g., "before_" or "after_")
        raw_data : np.ndarray, optional
            Raw electrode data for GEV computation
        centroids : np.ndarray, optional
            Cluster centroids in electrode space for GEV
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return {
                f"{prefix}n_clusters": n_clusters,
                f"{prefix}silhouette": -1.0,
                f"{prefix}silhouette_correlation": -1.0,
                f"{prefix}davies_bouldin": 999.0,
                f"{prefix}calinski_harabasz": 0.0,
                f"{prefix}gev": -1.0,
                f"{prefix}composite": -1.0,
            }

        # Remap labels to contiguous
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels_remapped = np.array([label_map[l] for l in labels])

        # Standard sklearn silhouette
        sil_sklearn = float(silhouette_score(features, labels_remapped))

        # Correlation-based silhouette (custom implementation, NOT pycrostates library)
        try:
            sil_correlation = float(pycrostates_silhouette_score(features, labels_remapped))
        except Exception:
            sil_correlation = sil_sklearn  # Fallback

        # GEV computation
        gev = -1.0
        if raw_data is not None and centroids is not None:
            # Remap centroids for merged clusters
            remapped_centroids = np.zeros((n_clusters, centroids.shape[1]))
            for old_label, new_label in label_map.items():
                if old_label < len(centroids):
                    remapped_centroids[new_label] = centroids[old_label]
            gev = self.compute_gev(raw_data, labels_remapped, remapped_centroids)

        # Composite score (geometric mean of normalized silhouette and GEV)
        composite = -1.0
        if gev >= 0:
            sil_normalized = (sil_correlation + 1) / 2  # Map [-1, 1] to [0, 1]
            composite = float(np.sqrt(max(0, sil_normalized) * gev))

        return {
            f"{prefix}n_clusters": n_clusters,
            f"{prefix}silhouette": sil_sklearn,
            f"{prefix}silhouette_correlation": sil_correlation,
            f"{prefix}davies_bouldin": float(davies_bouldin_score(features, labels_remapped)),
            f"{prefix}calinski_harabasz": float(calinski_harabasz_score(features, labels_remapped)),
            f"{prefix}gev": gev,
            f"{prefix}composite": composite,
        }

    def create_comprehensive_figure(
        self,
        centroids: np.ndarray,
        ssim_matrix: np.ndarray,
        ssim_inverted_matrix: np.ndarray,
        corr_matrix: np.ndarray,
        corr_inverted_matrix: np.ndarray,
        redundancy_info: Dict,
        metrics_before: Dict,
        metrics_after: Dict,
        cluster_counts: np.ndarray,
    ) -> str:
        """
        Create a single comprehensive figure with all analysis.

        Returns
        -------
        save_path : str
            Path to saved figure
        """
        n_clusters = len(centroids)

        # Create figure with gridspec
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(
            f"Centroid Redundancy Analysis (K={n_clusters})",
            fontsize=16, fontweight='bold', y=0.98
        )

        # === Row 1: Centroid Topomaps ===
        for k in range(min(n_clusters, 4)):
            ax = fig.add_subplot(gs[0, k])
            im = ax.imshow(centroids[k], cmap='RdBu_r', aspect='equal')
            ax.set_title(f"Centroid {k+1}\n(n={cluster_counts[k]})", fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # === Row 2: SSIM Matrices ===
        # SSIM (normal)
        ax_ssim = fig.add_subplot(gs[1, 0:2])
        im = ax_ssim.imshow(ssim_matrix, cmap='viridis', vmin=0, vmax=1)
        ax_ssim.set_title("SSIM Matrix\n(Centroid vs Centroid)", fontsize=11, fontweight='bold')
        ax_ssim.set_xlabel("Centroid")
        ax_ssim.set_ylabel("Centroid")
        ax_ssim.set_xticks(range(n_clusters))
        ax_ssim.set_yticks(range(n_clusters))
        ax_ssim.set_xticklabels([f"C{i+1}" for i in range(n_clusters)])
        ax_ssim.set_yticklabels([f"C{i+1}" for i in range(n_clusters)])
        # Add values
        for i in range(n_clusters):
            for j in range(n_clusters):
                color = 'white' if ssim_matrix[i, j] < 0.5 else 'black'
                ax_ssim.text(j, i, f"{ssim_matrix[i, j]:.2f}", ha='center', va='center',
                           fontsize=9, color=color)
        plt.colorbar(im, ax=ax_ssim, fraction=0.046, pad=0.04)

        # SSIM (inverted)
        ax_ssim_inv = fig.add_subplot(gs[1, 2:4])
        im = ax_ssim_inv.imshow(ssim_inverted_matrix, cmap='viridis', vmin=0, vmax=1)
        ax_ssim_inv.set_title("SSIM Matrix\n(Centroid vs INVERTED Centroid)", fontsize=11, fontweight='bold')
        ax_ssim_inv.set_xlabel("Inverted Centroid")
        ax_ssim_inv.set_ylabel("Centroid")
        ax_ssim_inv.set_xticks(range(n_clusters))
        ax_ssim_inv.set_yticks(range(n_clusters))
        ax_ssim_inv.set_xticklabels([f"-C{i+1}" for i in range(n_clusters)])
        ax_ssim_inv.set_yticklabels([f"C{i+1}" for i in range(n_clusters)])
        for i in range(n_clusters):
            for j in range(n_clusters):
                color = 'white' if ssim_inverted_matrix[i, j] < 0.5 else 'black'
                ax_ssim_inv.text(j, i, f"{ssim_inverted_matrix[i, j]:.2f}", ha='center', va='center',
                               fontsize=9, color=color)
        plt.colorbar(im, ax=ax_ssim_inv, fraction=0.046, pad=0.04)

        # === Row 3: Correlation Matrices ===
        # Correlation (normal)
        ax_corr = fig.add_subplot(gs[2, 0:2])
        im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_corr.set_title("Correlation Matrix\n(Centroid vs Centroid)", fontsize=11, fontweight='bold')
        ax_corr.set_xlabel("Centroid")
        ax_corr.set_ylabel("Centroid")
        ax_corr.set_xticks(range(n_clusters))
        ax_corr.set_yticks(range(n_clusters))
        ax_corr.set_xticklabels([f"C{i+1}" for i in range(n_clusters)])
        ax_corr.set_yticklabels([f"C{i+1}" for i in range(n_clusters)])
        for i in range(n_clusters):
            for j in range(n_clusters):
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                ax_corr.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center',
                           fontsize=9, color=color)
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

        # Correlation (inverted)
        ax_corr_inv = fig.add_subplot(gs[2, 2:4])
        im = ax_corr_inv.imshow(corr_inverted_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_corr_inv.set_title("Correlation Matrix\n(Centroid vs INVERTED Centroid)", fontsize=11, fontweight='bold')
        ax_corr_inv.set_xlabel("Inverted Centroid")
        ax_corr_inv.set_ylabel("Centroid")
        ax_corr_inv.set_xticks(range(n_clusters))
        ax_corr_inv.set_yticks(range(n_clusters))
        ax_corr_inv.set_xticklabels([f"-C{i+1}" for i in range(n_clusters)])
        ax_corr_inv.set_yticklabels([f"C{i+1}" for i in range(n_clusters)])
        for i in range(n_clusters):
            for j in range(n_clusters):
                color = 'white' if abs(corr_inverted_matrix[i, j]) > 0.5 else 'black'
                ax_corr_inv.text(j, i, f"{corr_inverted_matrix[i, j]:.2f}", ha='center', va='center',
                               fontsize=9, color=color)
        plt.colorbar(im, ax=ax_corr_inv, fraction=0.046, pad=0.04)

        # === Row 4: Summary and Metrics ===
        # Redundancy Summary (text)
        ax_summary = fig.add_subplot(gs[3, 0:2])
        ax_summary.axis('off')

        summary_text = "REDUNDANCY ANALYSIS SUMMARY\n"
        summary_text += "=" * 40 + "\n\n"
        summary_text += f"Thresholds: SSIM >= {self.ssim_threshold}, |Corr| >= {self.corr_threshold}\n\n"

        if redundancy_info["similar_pairs"]:
            summary_text += "Similar Pairs (same polarity):\n"
            for p in redundancy_info["similar_pairs"]:
                summary_text += f"  C{p['pair'][0]+1} <-> C{p['pair'][1]+1}: "
                summary_text += f"SSIM={p['ssim']:.2f}, Corr={p['correlation']:.2f}\n"
        else:
            summary_text += "Similar Pairs: None detected\n"

        summary_text += "\n"

        if redundancy_info["inverted_pairs"]:
            summary_text += "Inverted Pairs (opposite polarity):\n"
            for p in redundancy_info["inverted_pairs"]:
                summary_text += f"  C{p['pair'][0]+1} <-> -C{p['pair'][1]+1}: "
                summary_text += f"SSIM={p['ssim_inverted']:.2f}, Corr={p['correlation_inverted']:.2f}\n"
        else:
            summary_text += "Inverted Pairs: None detected\n"

        summary_text += f"\nTotal Redundant Pairs: {redundancy_info['total_redundant']}"

        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Metrics Comparison (table)
        ax_metrics = fig.add_subplot(gs[3, 2:4])
        ax_metrics.axis('off')

        metrics_text = "CLUSTERING METRICS COMPARISON\n"
        metrics_text += "=" * 40 + "\n\n"
        metrics_text += f"{'Metric':<20} {'Before':<12} {'After':<12} {'Change':<10}\n"
        metrics_text += "-" * 54 + "\n"

        for key in ['n_clusters', 'silhouette_correlation', 'davies_bouldin', 'gev', 'composite']:
            before_key = f"before_{key}"
            after_key = f"after_{key}"
            before_val = metrics_before.get(before_key, -1)
            after_val = metrics_after.get(after_key, -1)

            if key == 'n_clusters':
                change = f"{int(after_val - before_val):+d}"
            elif before_val < 0 or after_val < 0:
                change = "N/A"
            else:
                change = f"{after_val - before_val:+.3f}"

            # Format display name
            display_name = key.replace('_', ' ').title()
            if before_val < 0:
                metrics_text += f"{display_name:<20} {'N/A':<12} {after_val:<12.3f} {change:<10}\n"
            elif after_val < 0:
                metrics_text += f"{display_name:<20} {before_val:<12.3f} {'N/A':<12} {change:<10}\n"
            else:
                metrics_text += f"{display_name:<20} {before_val:<12.3f} {after_val:<12.3f} {change:<10}\n"

        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Save figure
        save_path = self.output_dir / "centroid_redundancy_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"Comprehensive figure saved to: {save_path}")
        return str(save_path)

    def create_tsne_visualization(
        self,
        latent_features: np.ndarray,
        labels: np.ndarray,
        latent_centroids: np.ndarray,
        merged_labels: np.ndarray = None,
        merged_centroids: np.ndarray = None,
        modkmeans_labels: np.ndarray = None,
        modkmeans_centroids: np.ndarray = None,
        title_suffix: str = "",
        max_samples: int = 5000,
    ) -> str:
        """
        Create t-SNE visualization of latent space with centroids.

        Parameters
        ----------
        latent_features : np.ndarray
            Latent space features (n_samples, latent_dim)
        labels : np.ndarray
            Original VAE cluster labels
        latent_centroids : np.ndarray
            VAE centroids in latent space (n_clusters, latent_dim)
        merged_labels : np.ndarray, optional
            Labels after polarity merge (for Column 2 visualization)
        merged_centroids : np.ndarray, optional
            Recomputed centroids after merge
        modkmeans_labels : np.ndarray, optional
            ModKMeans cluster labels (if available)
        modkmeans_centroids : np.ndarray, optional
            ModKMeans centroids projected to latent space (if available)
        title_suffix : str
            Additional text for figure title
        max_samples : int
            Maximum samples to use for t-SNE (for speed)

        Returns
        -------
        save_path : str
            Path to saved figure
        """
        self.logger.info("Creating t-SNE visualization...")

        n_samples = len(latent_features)
        n_clusters = len(latent_centroids)

        # Subsample if too many points
        if n_samples > max_samples:
            indices = np.random.choice(n_samples, max_samples, replace=False)
            latent_subset = latent_features[indices]
            labels_subset = labels[indices]
            merged_labels_subset = merged_labels[indices] if merged_labels is not None else None
            modkmeans_labels_subset = modkmeans_labels[indices] if modkmeans_labels is not None else None
        else:
            latent_subset = latent_features
            labels_subset = labels
            merged_labels_subset = merged_labels
            modkmeans_labels_subset = modkmeans_labels

        # Combine data points and centroids for t-SNE
        # This ensures centroids are in the same t-SNE space as data points
        all_points = [latent_subset, latent_centroids]
        centroid_start_idx = len(latent_subset)
        centroid_end_idx = centroid_start_idx + len(latent_centroids)

        if merged_centroids is not None:
            all_points.append(merged_centroids)
            merged_centroid_start = centroid_end_idx
            merged_centroid_end = merged_centroid_start + len(merged_centroids)
        else:
            merged_centroid_start = merged_centroid_end = None

        combined = np.vstack(all_points)

        # Run t-SNE
        self.logger.info(f"Running t-SNE on {len(combined)} points...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(combined) - 1),
                    random_state=42, n_iter=1000)
        embedded = tsne.fit_transform(combined)

        # Split back
        data_embedded = embedded[:centroid_start_idx]
        centroids_embedded = embedded[centroid_start_idx:centroid_end_idx]

        if merged_centroid_start is not None:
            merged_centroids_embedded = embedded[merged_centroid_start:merged_centroid_end]
        else:
            merged_centroids_embedded = None

        # Create figure with subplots
        n_plots = 2 if merged_labels is None else 3
        if modkmeans_labels is not None:
            n_plots += 1

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        # Color palette
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))

        plot_idx = 0

        # --- Plot 1: VAE Polarity-Specific (Original labels) ---
        ax = axes[plot_idx]
        for k in range(n_clusters):
            mask = labels_subset == k
            ax.scatter(data_embedded[mask, 0], data_embedded[mask, 1],
                      c=[colors[k]], alpha=0.3, s=10, label=f'Cluster {k+1}')

        # Plot centroids as large stars
        for k in range(n_clusters):
            ax.scatter(centroids_embedded[k, 0], centroids_embedded[k, 1],
                      c=[colors[k]], marker='*', s=500, edgecolors='black',
                      linewidths=2, zorder=10)

        ax.set_title(f"VAE Polarity-Specific (K={n_clusters})\nLatent Space", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc='upper right', fontsize=8)
        plot_idx += 1

        # --- Plot 2: VAE Polarity-Invariant (After merge) ---
        if merged_labels is not None and merged_labels_subset is not None:
            ax = axes[plot_idx]
            unique_merged = np.unique(merged_labels_subset)
            k_merged = len(unique_merged)

            for i, k in enumerate(unique_merged):
                mask = merged_labels_subset == k
                ax.scatter(data_embedded[mask, 0], data_embedded[mask, 1],
                          c=[colors[i]], alpha=0.3, s=10, label=f'Cluster {k+1}')

            # Plot merged centroids
            if merged_centroids_embedded is not None:
                for i in range(len(merged_centroids_embedded)):
                    ax.scatter(merged_centroids_embedded[i, 0], merged_centroids_embedded[i, 1],
                              c=[colors[i]], marker='*', s=500, edgecolors='black',
                              linewidths=2, zorder=10)

            ax.set_title(f"VAE Polarity-Invariant (K={k_merged})\nAfter Merge", fontsize=12, fontweight='bold')
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(loc='upper right', fontsize=8)
            plot_idx += 1

        # --- Plot 3: ModKMeans labels in VAE latent space ---
        if modkmeans_labels is not None and modkmeans_labels_subset is not None:
            ax = axes[plot_idx]
            unique_modk = np.unique(modkmeans_labels_subset)
            k_modk = len(unique_modk)

            for i, k in enumerate(unique_modk):
                mask = modkmeans_labels_subset == k
                ax.scatter(data_embedded[mask, 0], data_embedded[mask, 1],
                          c=[colors[i]], alpha=0.3, s=10, label=f'MS {k+1}')

            ax.set_title(f"ModKMeans Labels (K={k_modk})\nin VAE Latent Space", fontsize=12, fontweight='bold')
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(loc='upper right', fontsize=8)
            plot_idx += 1

        # --- Final Plot: Comparison overlay ---
        ax = axes[plot_idx]

        # Plot all data points in gray
        ax.scatter(data_embedded[:, 0], data_embedded[:, 1],
                  c='lightgray', alpha=0.2, s=5, label='Data points')

        # Plot VAE centroids as stars
        for k in range(n_clusters):
            ax.scatter(centroids_embedded[k, 0], centroids_embedded[k, 1],
                      c='blue', marker='*', s=400, edgecolors='black',
                      linewidths=2, zorder=10,
                      label='VAE Centroid' if k == 0 else None)

        # Plot merged centroids as diamonds
        if merged_centroids_embedded is not None:
            for i in range(len(merged_centroids_embedded)):
                ax.scatter(merged_centroids_embedded[i, 0], merged_centroids_embedded[i, 1],
                          c='green', marker='D', s=300, edgecolors='black',
                          linewidths=2, zorder=10,
                          label='Merged Centroid' if i == 0 else None)

        ax.set_title("Centroid Comparison\n(★=Original, ◆=Merged)", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc='upper right', fontsize=8)

        plt.suptitle(f"t-SNE Latent Space Visualization {title_suffix}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save figure
        save_path = self.output_dir / "tsne_latent_space_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"t-SNE visualization saved to: {save_path}")
        return str(save_path)

    def run_full_analysis(
        self,
        model,
        features: np.ndarray,
        labels: np.ndarray,
        auto_merge: bool = True,
        raw_data: np.ndarray = None,
        electrode_centroids: np.ndarray = None,
        latent_features: np.ndarray = None,
        latent_centroids: np.ndarray = None,
        modkmeans_labels: np.ndarray = None,
    ) -> Dict:
        """
        Run complete centroid redundancy analysis.

        Parameters
        ----------
        model : MyModel
            Trained VAE model
        features : np.ndarray
            Feature matrix (n_samples, n_features) - decoded topomaps or latent
        labels : np.ndarray
            Cluster labels from VAE
        auto_merge : bool
            If True, automatically merge redundant clusters
        raw_data : np.ndarray, optional
            Raw electrode data (n_samples, n_channels) for GEV computation
        electrode_centroids : np.ndarray, optional
            Cluster centroids in electrode space (n_clusters, n_channels)
        latent_features : np.ndarray, optional
            Latent space features (n_samples, latent_dim) for pycrostates silhouette
        latent_centroids : np.ndarray, optional
            Latent space centroids (n_clusters, latent_dim)
        modkmeans_labels : np.ndarray, optional
            ModKMeans cluster labels for same data (for t-SNE comparison)

        Returns
        -------
        results : dict
            Complete analysis results
        """
        self.logger.info("=" * 60)
        self.logger.info("CENTROID REDUNDANCY ANALYSIS")
        self.logger.info("=" * 60)

        # 1. Extract centroids
        centroids = self.extract_centroids(model)
        n_clusters = len(centroids)

        # 2. Compute cluster counts
        cluster_counts = np.array([np.sum(labels == k) for k in range(n_clusters)])
        self.logger.info(f"Cluster distribution: {cluster_counts}")

        # 3. Compute SSIM matrices
        self.logger.info("Computing SSIM matrices...")
        ssim_matrix = self.compute_ssim_matrix(centroids, inverted=False)
        ssim_inverted_matrix = self.compute_ssim_matrix(centroids, inverted=True)

        # 4. Compute correlation matrices
        self.logger.info("Computing correlation matrices...")
        corr_matrix = self.compute_correlation_matrix(centroids, inverted=False)
        corr_inverted_matrix = self.compute_correlation_matrix(centroids, inverted=True)

        # 5. Detect redundant pairs
        self.logger.info("Detecting redundant pairs...")
        redundancy_info = self.detect_redundant_pairs(
            ssim_matrix, ssim_inverted_matrix,
            corr_matrix, corr_inverted_matrix
        )

        self.logger.info(f"Found {len(redundancy_info['similar_pairs'])} similar pairs")
        self.logger.info(f"Found {len(redundancy_info['inverted_pairs'])} inverted pairs")

        # 6. Compute metrics before merge
        # Use latent features if available (lecturer's recommendation), otherwise use decoded features
        metric_features = latent_features if latent_features is not None else features
        metric_centroids = latent_centroids if latent_centroids is not None else None

        self.logger.info("Computing metrics before merge...")
        metrics_before = self.compute_clustering_metrics(
            metric_features, labels, prefix="before_",
            raw_data=raw_data, centroids=electrode_centroids
        )

        # 7. Merge if redundant pairs found and auto_merge enabled
        merge_info = {"merged": False}
        merge_pairs = [p['pair'] for p in redundancy_info['similar_pairs']] + \
                      [p['pair'] for p in redundancy_info['inverted_pairs']]

        merged_labels = None
        merged_centroids = None

        if auto_merge and merge_pairs:
            self.logger.info(f"Merging {len(merge_pairs)} redundant pairs...")
            new_labels, merge_info = self.merge_clusters(metric_features, labels, merge_pairs)
            merge_info["merged"] = True
            merged_labels = new_labels

            # Recompute centroids for merged clusters (mean of points in each merged cluster)
            if latent_features is not None:
                unique_merged = np.unique(new_labels)
                merged_centroids = np.zeros((len(unique_merged), latent_features.shape[1]))
                for i, k in enumerate(unique_merged):
                    mask = new_labels == k
                    merged_centroids[i] = latent_features[mask].mean(axis=0)
                self.logger.info(f"Recomputed merged centroids: {merged_centroids.shape}")

            metrics_after = self.compute_clustering_metrics(
                metric_features, new_labels, prefix="after_",
                raw_data=raw_data, centroids=electrode_centroids
            )
            self.logger.info(f"Clusters reduced: {merge_info['original_k']} -> {merge_info['merged_k']}")
        else:
            metrics_after = {k.replace('before_', 'after_'): v for k, v in metrics_before.items()}

        # 8. Create comprehensive figure
        self.logger.info("Creating comprehensive visualization...")
        figure_path = self.create_comprehensive_figure(
            centroids=centroids,
            ssim_matrix=ssim_matrix,
            ssim_inverted_matrix=ssim_inverted_matrix,
            corr_matrix=corr_matrix,
            corr_inverted_matrix=corr_inverted_matrix,
            redundancy_info=redundancy_info,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            cluster_counts=cluster_counts,
        )

        # 9. Save individual centroid images
        self.logger.info("Saving individual centroid images...")
        for k in range(n_clusters):
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(centroids[k], cmap='RdBu_r')
            ax.set_title(f"Centroid {k+1} (n={cluster_counts[k]})", fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.savefig(self.output_dir / f"centroid_{k+1}.png", dpi=200, bbox_inches='tight')
            plt.close()

        # 10. Create t-SNE visualization of latent space
        tsne_path = None
        if latent_features is not None and latent_centroids is not None:
            try:
                tsne_path = self.create_tsne_visualization(
                    latent_features=latent_features,
                    labels=labels,
                    latent_centroids=latent_centroids,
                    merged_labels=merged_labels,
                    merged_centroids=merged_centroids,
                    modkmeans_labels=modkmeans_labels,
                )
            except Exception as e:
                self.logger.warning(f"t-SNE visualization failed: {e}")

        # 11. Compile results
        results = {
            "n_clusters": n_clusters,
            "cluster_counts": cluster_counts.tolist(),
            "ssim_matrix": ssim_matrix.tolist(),
            "ssim_inverted_matrix": ssim_inverted_matrix.tolist(),
            "correlation_matrix": corr_matrix.tolist(),
            "correlation_inverted_matrix": corr_inverted_matrix.tolist(),
            "redundancy_info": redundancy_info,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "merge_info": merge_info,
            "figure_path": figure_path,
            "tsne_path": tsne_path,
            "output_dir": str(self.output_dir),
        }

        # 11. Save JSON results
        with open(self.output_dir / "centroid_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)

        return results


def run_centroid_analysis(
    model,
    data_loader,
    output_dir: str,
    logger: Optional[logging.Logger] = None,
    auto_merge: bool = True,
    raw_mne=None,
    modkmeans_labels: np.ndarray = None,
    norm_params=None,
) -> Dict:
    """
    Convenience function to run centroid analysis with GEV computation.

    Parameters
    ----------
    model : MyModel
        Trained VAE model
    data_loader : DataLoader
        DataLoader with test/validation data
    output_dir : str
        Output directory for results
    logger : Logger, optional
        Logger instance
    auto_merge : bool
        If True, automatically merge redundant clusters
    raw_mne : mne.io.Raw, optional
        Raw MNE object for electrode positions (enables GEV computation)
    modkmeans_labels : np.ndarray, optional
        ModKMeans cluster labels for same test data (for t-SNE comparison)

    Returns
    -------
    results : dict
        Analysis results including GEV, composite scores, and t-SNE visualization
    """
    import torch

    logger = logger or logging.getLogger(__name__)
    device = next(model.parameters()).device

    model.set_eval_mode() if hasattr(model, 'set_eval_mode') else model.eval()
    all_features = []
    all_labels = []
    all_latent = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon, mu, _ = model(data)
            all_features.append(recon.view(recon.size(0), -1).detach().cpu().numpy())
            all_labels.append(model.predict(data))
            all_latent.append(mu.detach().cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    latent_features = np.concatenate(all_latent, axis=0)

    # Get latent centroids from model
    latent_centroids = model.mu_c.detach().cpu().numpy()

    # Try to extract electrode centroids for GEV computation
    electrode_centroids = None
    raw_data = None

    if raw_mne is not None:
        try:
            # Decode centroids to topomaps
            with torch.no_grad():
                decoded = model.decode(model.mu_c)
                decoded_centroids = decoded.squeeze(1).detach().cpu().numpy()

            # Get electrode positions from MNE
            info = raw_mne.info
            pos = np.array([info['chs'][i]['loc'][:2] for i in range(len(info.ch_names))])
            pos_min = pos.min(axis=0)
            pos_max = pos.max(axis=0)
            pos_range = pos_max - pos_min
            pos_range[pos_range < 1e-10] = 1e-10
            pos_normalized = (pos - pos_min) / pos_range

            img_size = decoded_centroids.shape[1]
            margin = 0.1
            pos_img = pos_normalized * (1 - 2 * margin) + margin
            pos_img = (pos_img * (img_size - 1)).astype(int)
            pos_img = np.clip(pos_img, 0, img_size - 1)

            # Extract electrode values from each decoded centroid
            n_clusters = model.nClusters
            n_channels = len(info.ch_names)
            electrode_centroids = np.zeros((n_clusters, n_channels))

            for k in range(n_clusters):
                centroid_img = decoded_centroids[k]
                for ch_idx in range(n_channels):
                    x, y = pos_img[ch_idx]
                    electrode_centroids[k, ch_idx] = centroid_img[y, x]
                # Center the centroid
                electrode_centroids[k] = electrode_centroids[k] - electrode_centroids[k].mean()

            logger.info(f"Extracted electrode centroids: {electrode_centroids.shape}")

            # Get raw data if available in dataset
            if hasattr(data_loader.dataset, 'raw_data'):
                raw_data = data_loader.dataset.raw_data
                logger.info(f"Raw data shape for GEV: {raw_data.shape}")

        except Exception as e:
            logger.warning(f"Could not extract electrode data for GEV: {e}")

    analyzer = CentroidAnalyzer(output_dir, logger, norm_params=norm_params)
    return analyzer.run_full_analysis(
        model, features, labels,
        auto_merge=auto_merge,
        raw_data=raw_data,
        electrode_centroids=electrode_centroids,
        latent_features=latent_features,
        latent_centroids=latent_centroids,
        modkmeans_labels=modkmeans_labels,
    )
