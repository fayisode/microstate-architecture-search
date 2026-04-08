"""
Centroid-Based Clustering Metrics (for VAE)
===========================================

WARNING: The "pycrostates" methods in this module are CUSTOM implementations
using correlation-based distance. They are NOT the actual pycrostates library
functions. For true pycrostates library metrics, use:

    from pycrostates.metrics import silhouette_score, dunn_score, etc.
    silhouette_score(fitted_ModKMeans)  # Takes fitted ModKMeans object

This module computes clustering quality metrics using centroid distances instead
of pairwise distances. This is:
1. Faster: O(N×K) instead of O(N²)
2. More aligned with VaDE model which explicitly learns centroids
3. Uses correlation-based distance similar to pycrostates' approach

Provides both sklearn-style (Euclidean) and correlation-based metrics.

Author: Microstate EEG Project
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

EPSILON = 1e-8


class CentroidMetrics:
    """
    Compute clustering metrics using centroid-based distances.

    NOTE: The "pycrostates" methods in this class are CUSTOM correlation-based
    implementations, NOT the actual pycrostates library functions. They use
    similar correlation-based distance concepts but are computed differently.

    Supports two distance types:
    - Euclidean (sklearn-style): d(x, c) = ||x - c||
    - Correlation (custom): d(x, c) = 1 - |corr(x, c)|

    Usage:
    ------
    metrics = CentroidMetrics()
    results = metrics.compute_all(
        features=latent_features,  # (N, D)
        labels=cluster_labels,     # (N,)
        centroids=mu_c             # (K, D)
    )
    metrics.save_results(results, "test_metrics.json")
    """

    def __init__(self, use_polarity_invariance: bool = True):
        """
        Args:
            use_polarity_invariance: If True, correlation uses |corr| (for EEG)
        """
        self.use_polarity_invariance = use_polarity_invariance

    # =========================================================================
    # Distance Computation
    # =========================================================================

    def euclidean_distance_to_centroids(
        self,
        features: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Compute Euclidean distance from each sample to each centroid.

        Args:
            features: (N, D) feature matrix
            centroids: (K, D) centroid matrix

        Returns:
            distances: (N, K) distance matrix
        """
        # Using broadcasting: (N, 1, D) - (1, K, D) -> (N, K, D) -> (N, K)
        diff = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances

    def correlation_distance_to_centroids(
        self,
        features: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlation-based distance from each sample to each centroid.

        Distance = 1 - |correlation| (polarity-invariant)

        Args:
            features: (N, D) feature matrix
            centroids: (K, D) centroid matrix

        Returns:
            distances: (N, K) distance matrix (0 = identical, 1 = orthogonal)
        """
        # Normalize features (zero mean, unit norm)
        features_centered = features - features.mean(axis=1, keepdims=True)
        features_norm = np.linalg.norm(features_centered, axis=1, keepdims=True)
        features_norm[features_norm == 0] = EPSILON
        features_normalized = features_centered / features_norm

        # Normalize centroids
        centroids_centered = centroids - centroids.mean(axis=1, keepdims=True)
        centroids_norm = np.linalg.norm(centroids_centered, axis=1, keepdims=True)
        centroids_norm[centroids_norm == 0] = EPSILON
        centroids_normalized = centroids_centered / centroids_norm

        # Correlation = dot product of normalized vectors
        correlation = np.dot(features_normalized, centroids_normalized.T)

        # TRUE pycrostates distance: |1/correlation| - 1
        # corr=1.0 -> dist=0, corr=0.5 -> dist=1, corr=0.2 -> dist=4
        MAX_DIST = 100.0
        if self.use_polarity_invariance:
            abs_corr = np.abs(correlation)
        else:
            abs_corr = correlation
        abs_corr = np.clip(abs_corr, EPSILON, None)
        distances = np.abs(1.0 / abs_corr) - 1.0
        distances = np.clip(distances, 0.0, MAX_DIST)

        return distances

    # =========================================================================
    # Centroid-Based Silhouette
    # =========================================================================

    def silhouette_sklearn(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Centroid-based silhouette using Euclidean distance.

        For each sample:
            a(i) = distance to its own centroid
            b(i) = distance to nearest other centroid
            s(i) = (b(i) - a(i)) / max(a(i), b(i))

        Returns:
            mean_silhouette: float
            per_sample_silhouette: (N,) array
        """
        n_samples = len(features)
        n_clusters = len(centroids)

        if n_clusters < 2:
            return -1.0, np.full(n_samples, -1.0)

        # Compute distances to all centroids
        distances = self.euclidean_distance_to_centroids(features, centroids)

        silhouette_values = np.zeros(n_samples)

        for i in range(n_samples):
            cluster_i = labels[i]

            # a(i) = distance to own centroid
            a_i = distances[i, cluster_i]

            # b(i) = distance to nearest OTHER centroid
            other_distances = np.delete(distances[i], cluster_i)
            b_i = np.min(other_distances)

            # Silhouette
            max_ab = max(a_i, b_i)
            if max_ab > 0:
                silhouette_values[i] = (b_i - a_i) / max_ab
            else:
                silhouette_values[i] = 0.0

        return float(np.mean(silhouette_values)), silhouette_values

    def silhouette_pycrostates(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Centroid-based silhouette using correlation distance.

        NOTE: This is a CUSTOM implementation, NOT the pycrostates library function.
        Uses polarity-invariant correlation: d = 1 - |corr(x, c)|

        Returns:
            mean_silhouette: float
            per_sample_silhouette: (N,) array
        """
        n_samples = len(features)
        n_clusters = len(centroids)

        if n_clusters < 2:
            return -1.0, np.full(n_samples, -1.0)

        # Compute correlation distances to all centroids
        distances = self.correlation_distance_to_centroids(features, centroids)

        silhouette_values = np.zeros(n_samples)

        for i in range(n_samples):
            cluster_i = labels[i]

            # a(i) = correlation distance to own centroid
            a_i = distances[i, cluster_i]

            # b(i) = correlation distance to nearest OTHER centroid
            other_distances = np.delete(distances[i], cluster_i)
            b_i = np.min(other_distances)

            # Silhouette
            max_ab = max(a_i, b_i)
            if max_ab > 0:
                silhouette_values[i] = (b_i - a_i) / max_ab
            else:
                silhouette_values[i] = 0.0

        return float(np.mean(silhouette_values)), silhouette_values

    # =========================================================================
    # Centroid-Based Davies-Bouldin
    # =========================================================================

    def davies_bouldin_sklearn(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Centroid-based Davies-Bouldin index using Euclidean distance.

        DB = (1/K) * sum_i(max_j≠i((S_i + S_j) / d(c_i, c_j)))

        Where:
            S_i = mean distance of cluster i samples to centroid i
            d(c_i, c_j) = distance between centroids

        Lower is better.
        """
        n_clusters = len(centroids)

        if n_clusters < 2:
            return 0.0

        # Compute intra-cluster scatter (mean dist to centroid)
        scatter = np.zeros(n_clusters)
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_features = features[mask]
                distances = np.linalg.norm(cluster_features - centroids[k], axis=1)
                scatter[k] = np.mean(distances)

        # Compute inter-centroid distances
        centroid_distances = self.euclidean_distance_to_centroids(centroids, centroids)

        # Compute DB index
        db_values = np.zeros(n_clusters)
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j and centroid_distances[i, j] > 0:
                    ratio = (scatter[i] + scatter[j]) / centroid_distances[i, j]
                    max_ratio = max(max_ratio, ratio)
            db_values[i] = max_ratio

        return float(np.mean(db_values))

    def davies_bouldin_pycrostates(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Centroid-based Davies-Bouldin using correlation distance.

        NOTE: This is a CUSTOM implementation, NOT the pycrostates library function.
        Lower is better.
        """
        n_clusters = len(centroids)

        if n_clusters < 2:
            return 0.0

        # Compute intra-cluster scatter using correlation distance
        scatter = np.zeros(n_clusters)
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_features = features[mask]
                distances = self.correlation_distance_to_centroids(
                    cluster_features, centroids[k:k+1]
                ).flatten()
                scatter[k] = np.mean(distances)

        # Compute inter-centroid correlation distances
        centroid_distances = self.correlation_distance_to_centroids(centroids, centroids)

        # Compute DB index
        db_values = np.zeros(n_clusters)
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j and centroid_distances[i, j] > 0:
                    ratio = (scatter[i] + scatter[j]) / centroid_distances[i, j]
                    max_ratio = max(max_ratio, ratio)
            db_values[i] = max_ratio

        return float(np.mean(db_values))

    # =========================================================================
    # Centroid-Based Calinski-Harabasz
    # =========================================================================

    def calinski_harabasz_sklearn(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Centroid-based Calinski-Harabasz index using Euclidean distance.

        CH = [B / (K-1)] / [W / (N-K)]

        Where:
            B = between-cluster dispersion (weighted centroid distances to global mean)
            W = within-cluster dispersion (sum of distances to centroids)

        Higher is better.
        """
        n_samples = len(features)
        n_clusters = len(centroids)

        if n_clusters < 2:
            return 0.0

        # Global mean
        global_mean = np.mean(features, axis=0)

        # Between-cluster dispersion
        B = 0.0
        for k in range(n_clusters):
            n_k = np.sum(labels == k)
            if n_k > 0:
                B += n_k * np.sum((centroids[k] - global_mean) ** 2)

        # Within-cluster dispersion
        W = 0.0
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_features = features[mask]
                W += np.sum((cluster_features - centroids[k]) ** 2)

        # CH index
        if W == 0 or (n_samples - n_clusters) == 0:
            return 0.0

        ch = (B / (n_clusters - 1)) / (W / (n_samples - n_clusters))
        return float(ch)

    def calinski_harabasz_pycrostates(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Centroid-based Calinski-Harabasz using correlation distance.

        NOTE: This is a CUSTOM implementation, NOT the pycrostates library function.
        Higher is better.
        """
        n_samples = len(features)
        n_clusters = len(centroids)

        if n_clusters < 2:
            return 0.0

        # Global mean centroid
        global_mean = np.mean(features, axis=0, keepdims=True)

        # Between-cluster dispersion (correlation distance from centroids to global mean)
        B = 0.0
        centroid_to_mean_dist = self.correlation_distance_to_centroids(centroids, global_mean)
        for k in range(n_clusters):
            n_k = np.sum(labels == k)
            if n_k > 0:
                # Use squared distance for dispersion
                B += n_k * (centroid_to_mean_dist[k, 0] ** 2)

        # Within-cluster dispersion
        W = 0.0
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_features = features[mask]
                distances = self.correlation_distance_to_centroids(
                    cluster_features, centroids[k:k+1]
                ).flatten()
                W += np.sum(distances ** 2)

        # CH index
        if W == 0 or (n_samples - n_clusters) == 0:
            return 0.0

        ch = (B / (n_clusters - 1)) / (W / (n_samples - n_clusters))
        return float(ch)

    # =========================================================================
    # Centroid-Based Dunn Index
    # =========================================================================

    def dunn_index_sklearn(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Centroid-based Dunn index using Euclidean distance.

        Dunn = min(inter-centroid distance) / max(intra-cluster diameter)

        Higher is better.
        """
        n_clusters = len(centroids)

        if n_clusters < 2:
            return 0.0

        # Min inter-centroid distance
        centroid_distances = self.euclidean_distance_to_centroids(centroids, centroids)
        np.fill_diagonal(centroid_distances, np.inf)
        min_inter = np.min(centroid_distances)

        # Max intra-cluster diameter (max distance from any point to its centroid)
        max_intra = 0.0
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 1:
                cluster_features = features[mask]
                distances = np.linalg.norm(cluster_features - centroids[k], axis=1)
                max_intra = max(max_intra, np.max(distances) * 2)  # diameter = 2 * radius

        if max_intra == 0:
            return 0.0

        return float(min_inter / max_intra)

    def dunn_index_pycrostates(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """
        Centroid-based Dunn index using correlation distance.

        NOTE: This is a CUSTOM implementation, NOT the pycrostates library function.
        Higher is better.
        """
        n_clusters = len(centroids)

        if n_clusters < 2:
            return 0.0

        # Min inter-centroid correlation distance
        centroid_distances = self.correlation_distance_to_centroids(centroids, centroids)
        np.fill_diagonal(centroid_distances, np.inf)
        min_inter = np.min(centroid_distances)

        # Max intra-cluster diameter
        max_intra = 0.0
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 1:
                cluster_features = features[mask]
                distances = self.correlation_distance_to_centroids(
                    cluster_features, centroids[k:k+1]
                ).flatten()
                max_intra = max(max_intra, np.max(distances) * 2)

        if max_intra == 0:
            return 0.0

        return float(min_inter / max_intra)

    # =========================================================================
    # Composite Scores (Silhouette + GEV)
    # =========================================================================

    @staticmethod
    def normalize_silhouette(silhouette: float) -> float:
        """
        Normalize silhouette from [-1, 1] to [0, 1].

        Args:
            silhouette: Raw silhouette score in [-1, 1]

        Returns:
            Normalized score in [0, 1]
        """
        return (silhouette + 1) / 2

    @staticmethod
    def composite_weighted_average(
        silhouette: float,
        gev: float,
        weight_silhouette: float = 0.5,
        weight_gev: float = 0.5
    ) -> float:
        """
        Compute composite score using weighted average (Option A).

        Formula: composite = w1 * sil_norm + w2 * GEV

        Args:
            silhouette: Silhouette score in [-1, 1]
            gev: GEV score in [0, 1]
            weight_silhouette: Weight for silhouette (default 0.5)
            weight_gev: Weight for GEV (default 0.5)

        Returns:
            Composite score in [0, 1]
        """
        sil_norm = (silhouette + 1) / 2  # Map [-1,1] → [0,1]

        # Handle edge cases
        if gev < 0:
            gev = 0.0

        composite = weight_silhouette * sil_norm + weight_gev * gev
        return float(composite)

    @staticmethod
    def composite_geometric_mean(silhouette: float, gev: float) -> float:
        """
        Compute composite score using geometric mean (Option B - Recommended).

        Formula: composite = sqrt(sil_norm * GEV)

        This ensures BOTH metrics must be good for a high composite score.
        A model with great silhouette but poor GEV (or vice versa) will score lower.

        Args:
            silhouette: Silhouette score in [-1, 1]
            gev: GEV score in [0, 1]

        Returns:
            Composite score in [0, 1]
        """
        sil_norm = (silhouette + 1) / 2  # Map [-1,1] → [0,1]

        # Handle edge cases
        if gev < 0:
            gev = 0.0
        if sil_norm < 0:
            sil_norm = 0.0

        composite = np.sqrt(sil_norm * gev)
        return float(composite)

    def compute_composite_scores(
        self,
        silhouette_sklearn: float,
        silhouette_correlation: float,
        gev: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all composite scores (dual silhouette x dual formula).

        Returns 4 composite scores:
        - sklearn_weighted: Weighted average with sklearn silhouette
        - sklearn_geometric: Geometric mean with sklearn silhouette
        - correlation_weighted: Weighted average with correlation-based silhouette
        - correlation_geometric: Geometric mean with correlation-based silhouette

        Args:
            silhouette_sklearn: Sklearn silhouette (Euclidean-based)
            silhouette_correlation: Correlation-based silhouette (custom implementation)
            gev: Global Explained Variance

        Returns:
            Dictionary with all composite scores
        """
        return {
            "sklearn": {
                "description": "Composite using sklearn silhouette (Euclidean)",
                "silhouette_normalized": self.normalize_silhouette(silhouette_sklearn),
                "weighted_average": self.composite_weighted_average(silhouette_sklearn, gev),
                "geometric_mean": self.composite_geometric_mean(silhouette_sklearn, gev),
            },
            "correlation_based": {
                "description": "Composite using correlation-based silhouette (custom, polarity-invariant)",
                "silhouette_normalized": self.normalize_silhouette(silhouette_correlation),
                "weighted_average": self.composite_weighted_average(silhouette_correlation, gev),
                "geometric_mean": self.composite_geometric_mean(silhouette_correlation, gev),
            },
            "gev": gev,
            "recommended": {
                "description": "Recommended composite: correlation-based geometric mean (polarity-invariant, balanced)",
                "score": self.composite_geometric_mean(silhouette_correlation, gev),
            }
        }

    # =========================================================================
    # Compute All Metrics
    # =========================================================================

    def compute_all(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        feature_space: str = "latent",
        split: str = "test",
        gev: Optional[float] = None
    ) -> Dict:
        """
        Compute all centroid-based metrics including composite scores.

        Args:
            features: (N, D) feature matrix
            labels: (N,) cluster assignments
            centroids: (K, D) cluster centroids
            feature_space: "latent", "raw", or "decoded" (for metadata)
            split: "train", "val", or "test" (for metadata)
            gev: Optional GEV score for composite score computation

        Returns:
            Dictionary with all metrics including composite scores
        """
        n_samples = len(features)
        n_clusters = len(centroids)
        unique_labels = np.unique(labels)

        # Ensure labels are contiguous 0 to K-1
        if len(unique_labels) != n_clusters or np.max(labels) >= n_clusters:
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])

        # Compute sklearn-style metrics (Euclidean)
        sil_sklearn, sil_sklearn_per_sample = self.silhouette_sklearn(features, labels, centroids)
        db_sklearn = self.davies_bouldin_sklearn(features, labels, centroids)
        ch_sklearn = self.calinski_harabasz_sklearn(features, labels, centroids)
        dunn_sklearn = self.dunn_index_sklearn(features, labels, centroids)

        # Compute pycrostates-style metrics (Correlation)
        sil_pycrostates, sil_pycrostates_per_sample = self.silhouette_pycrostates(features, labels, centroids)
        db_pycrostates = self.davies_bouldin_pycrostates(features, labels, centroids)
        ch_pycrostates = self.calinski_harabasz_pycrostates(features, labels, centroids)
        dunn_pycrostates = self.dunn_index_pycrostates(features, labels, centroids)

        # Per-cluster statistics
        per_cluster_stats = {}
        for k in range(n_clusters):
            mask = labels == k
            n_k = np.sum(mask)
            if n_k > 0:
                cluster_features = features[mask]

                # Euclidean stats
                euc_distances = np.linalg.norm(cluster_features - centroids[k], axis=1)

                # Correlation stats
                corr_distances = self.correlation_distance_to_centroids(
                    cluster_features, centroids[k:k+1]
                ).flatten()

                per_cluster_stats[f"cluster_{k}"] = {
                    "n_samples": int(n_k),
                    "proportion": float(n_k / n_samples),
                    "euclidean": {
                        "mean_dist_to_centroid": float(np.mean(euc_distances)),
                        "std_dist_to_centroid": float(np.std(euc_distances)),
                        "max_dist_to_centroid": float(np.max(euc_distances)),
                    },
                    "correlation": {
                        "mean_dist_to_centroid": float(np.mean(corr_distances)),
                        "std_dist_to_centroid": float(np.std(corr_distances)),
                        "max_dist_to_centroid": float(np.max(corr_distances)),
                    },
                    "silhouette_sklearn_mean": float(np.mean(sil_sklearn_per_sample[mask])),
                    "silhouette_pycrostates_mean": float(np.mean(sil_pycrostates_per_sample[mask])),
                }

        results = {
            "metadata": {
                "n_clusters": n_clusters,
                "n_samples": n_samples,
                "feature_dim": features.shape[1],
                "feature_space": feature_space,
                "split": split,
                "polarity_invariant": self.use_polarity_invariance,
                "timestamp": datetime.now().isoformat(),
            },

            "centroid_based_metrics": {
                "sklearn": {
                    "description": "Euclidean distance to centroids",
                    "silhouette": sil_sklearn,
                    "davies_bouldin": db_sklearn,
                    "calinski_harabasz": ch_sklearn,
                    "dunn": dunn_sklearn,
                },
                "correlation_based": {
                    "description": "Custom correlation-based metrics (NOT pycrostates library)",
                    "silhouette": sil_pycrostates,
                    "davies_bouldin": db_pycrostates,
                    "calinski_harabasz": ch_pycrostates,
                    "dunn": dunn_pycrostates,
                },
            },

            "per_cluster_stats": per_cluster_stats,

            "label_distribution": {
                f"cluster_{k}": int(np.sum(labels == k)) for k in range(n_clusters)
            },
        }

        # Add composite scores if GEV is provided
        if gev is not None:
            results["composite_scores"] = self.compute_composite_scores(
                silhouette_sklearn=sil_sklearn,
                silhouette_correlation=sil_pycrostates,
                gev=gev
            )

        return results

    # =========================================================================
    # Save Results
    # =========================================================================

    def save_results(
        self,
        results: Dict,
        filepath: Union[str, Path],
        append_if_exists: bool = False
    ) -> Path:
        """
        Save results to JSON file.

        Args:
            results: Dictionary of metrics
            filepath: Output path
            append_if_exists: If True, append to existing file (for epoch tracking)

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if append_if_exists and filepath.exists():
            with open(filepath, 'r') as f:
                existing = json.load(f)

            # Append as new entry with timestamp key
            if isinstance(existing, list):
                existing.append(results)
            else:
                existing = [existing, results]

            with open(filepath, 'w') as f:
                json.dump(existing, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)

        logger.info(f"Metrics saved to: {filepath}")
        return filepath

    def print_summary(self, results: Dict):
        """Print a formatted summary of the metrics including composite scores."""
        meta = results["metadata"]
        sklearn_m = results["centroid_based_metrics"]["sklearn"]
        corr_m = results["centroid_based_metrics"]["correlation_based"]

        print("\n" + "="*60)
        print(f"CENTROID-BASED METRICS SUMMARY ({meta['split'].upper()})")
        print("="*60)
        print(f"Feature space: {meta['feature_space']} ({meta['feature_dim']}D)")
        print(f"Samples: {meta['n_samples']}, Clusters: {meta['n_clusters']}")
        print("-"*60)
        print(f"{'Metric':<25} {'sklearn':<15} {'correlation':<15}")
        print("-"*60)
        print(f"{'Silhouette':<25} {sklearn_m['silhouette']:<15.4f} {corr_m['silhouette']:<15.4f}")
        print(f"{'Davies-Bouldin':<25} {sklearn_m['davies_bouldin']:<15.4f} {corr_m['davies_bouldin']:<15.4f}")
        print(f"{'Calinski-Harabasz':<25} {sklearn_m['calinski_harabasz']:<15.2f} {corr_m['calinski_harabasz']:<15.2f}")
        print(f"{'Dunn':<25} {sklearn_m['dunn']:<15.4f} {corr_m['dunn']:<15.4f}")

        # Print composite scores if available
        if "composite_scores" in results:
            comp = results["composite_scores"]
            print("-"*60)
            print("COMPOSITE SCORES (Silhouette + GEV)")
            print("-"*60)
            print(f"{'GEV':<25} {comp['gev']:.4f}")
            print(f"{'Metric':<25} {'sklearn':<15} {'correlation':<15}")
            print(f"{'Sil (normalized)':<25} {comp['sklearn']['silhouette_normalized']:<15.4f} {comp['correlation_based']['silhouette_normalized']:<15.4f}")
            print(f"{'Weighted Average':<25} {comp['sklearn']['weighted_average']:<15.4f} {comp['correlation_based']['weighted_average']:<15.4f}")
            print(f"{'Geometric Mean':<25} {comp['sklearn']['geometric_mean']:<15.4f} {comp['correlation_based']['geometric_mean']:<15.4f}")
            print("-"*60)
            print(f"RECOMMENDED (correlation geometric): {comp['recommended']['score']:.4f}")

        print("="*60)


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_centroid_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    feature_space: str = "latent",
    split: str = "test",
    gev: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    print_summary: bool = True
) -> Dict:
    """
    Convenience function to compute and optionally save centroid-based metrics.

    Args:
        features: (N, D) feature matrix
        labels: (N,) cluster assignments
        centroids: (K, D) cluster centroids
        feature_space: "latent", "raw", or "decoded"
        split: "train", "val", or "test"
        gev: Optional GEV score for composite score computation
        save_path: Optional path to save results
        print_summary: Whether to print summary

    Returns:
        Dictionary with all metrics including composite scores (if GEV provided)
    """
    metrics = CentroidMetrics()
    results = metrics.compute_all(
        features=features,
        labels=labels,
        centroids=centroids,
        feature_space=feature_space,
        split=split,
        gev=gev
    )

    if print_summary:
        metrics.print_summary(results)

    if save_path:
        metrics.save_results(results, save_path)

    return results


class TrainingMetricsTracker:
    """
    Track metrics during training across epochs.

    Usage:
    ------
    tracker = TrainingMetricsTracker(output_dir="outputs/010004/run_xxx/cluster_4")

    for epoch in range(n_epochs):
        # ... training ...

        tracker.log_epoch(
            epoch=epoch,
            train_features=train_z,
            train_labels=train_labels,
            val_features=val_z,
            val_labels=val_labels,
            centroids=model.mu_c.detach().cpu().numpy()
        )

    tracker.save_final_report()
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        feature_space: str = "latent"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_space = feature_space
        self.metrics_computer = CentroidMetrics()

        self.train_history = []
        self.val_history = []

    def log_epoch(
        self,
        epoch: int,
        centroids: np.ndarray,
        train_features: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        additional_metrics: Optional[Dict] = None
    ):
        """Log metrics for a single epoch."""

        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_metrics:
            epoch_data["additional"] = additional_metrics

        # Compute train metrics
        if train_features is not None and train_labels is not None:
            train_results = self.metrics_computer.compute_all(
                features=train_features,
                labels=train_labels,
                centroids=centroids,
                feature_space=self.feature_space,
                split="train"
            )
            epoch_data["train"] = train_results["centroid_based_metrics"]
            self.train_history.append({
                "epoch": epoch,
                **train_results["centroid_based_metrics"]
            })

        # Compute validation metrics
        if val_features is not None and val_labels is not None:
            val_results = self.metrics_computer.compute_all(
                features=val_features,
                labels=val_labels,
                centroids=centroids,
                feature_space=self.feature_space,
                split="val"
            )
            epoch_data["val"] = val_results["centroid_based_metrics"]
            self.val_history.append({
                "epoch": epoch,
                **val_results["centroid_based_metrics"]
            })

        # Save epoch data
        epoch_path = self.output_dir / f"metrics_epoch_{epoch:04d}.json"
        with open(epoch_path, 'w') as f:
            json.dump(epoch_data, f, indent=2)

        logger.info(f"Epoch {epoch} metrics saved")

        return epoch_data

    def save_final_report(
        self,
        test_features: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        centroids: Optional[np.ndarray] = None
    ) -> Path:
        """Save final comprehensive report with all epochs and test metrics."""

        report = {
            "training_history": {
                "train": self.train_history,
                "val": self.val_history,
            },
            "summary": {
                "n_epochs": len(self.train_history),
                "feature_space": self.feature_space,
            }
        }

        # Add test metrics if provided
        if test_features is not None and test_labels is not None and centroids is not None:
            test_results = self.metrics_computer.compute_all(
                features=test_features,
                labels=test_labels,
                centroids=centroids,
                feature_space=self.feature_space,
                split="test"
            )
            report["test_metrics"] = test_results
            self.metrics_computer.print_summary(test_results)

        # Find best epochs
        if self.val_history:
            silhouettes = [h["correlation_based"]["silhouette"] for h in self.val_history]
            best_epoch = self.val_history[np.argmax(silhouettes)]["epoch"]
            report["summary"]["best_epoch_by_silhouette"] = best_epoch
            report["summary"]["best_val_silhouette"] = max(silhouettes)

        report_path = self.output_dir / "training_metrics_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Final report saved to: {report_path}")
        return report_path
