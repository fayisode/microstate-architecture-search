import numpy as np
from pathlib import Path
import json

# EEG/Visualization imports — lazy for CPU-only environments
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import mne
    from pycrostates.cluster import ModKMeans
    from pycrostates.io import ChData
    from pycrostates.viz import (
        plot_raw_segmentation,
    )
    from pycrostates.metrics import (
        silhouette_score as pycrostates_silhouette,
        calinski_harabasz_score as pycrostates_ch,
        dunn_score as pycrostates_dunn,
        davies_bouldin_score as pycrostates_db,
    )
except ImportError:
    mne = None
from sklearn.metrics import (
    silhouette_score as sklearn_silhouette,
    calinski_harabasz_score as sklearn_ch,
    davies_bouldin_score as sklearn_db,
)
from metrics_utils import (
    align_polarities,
    pycrostates_silhouette_score as custom_pycrostates_silhouette,
    pycrostates_calinski_harabasz_score as custom_pycrostates_ch,
    pycrostates_davies_bouldin_score as custom_pycrostates_db,
    pycrostates_dunn_score as custom_pycrostates_dunn,
)


class BaselineHandler:
    def __init__(self, n_clusters, device, logger, output_dir):
        self.n_clusters = n_clusters
        self.device = device
        self.logger = logger
        self.output_dir = Path(output_dir) / "baseline_modkmeans"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ModKMeans
        # n_init=10 restarts clustering 10 times to find the global optimum
        self.modk = ModKMeans(
            n_clusters=n_clusters, n_init=10, max_iter=100, random_state=42
        )
        self.fitted = False
        self.cluster_centers_ = None
        self.gfp_peaks = None  # Store for metrics computation

    def _assign_labels_gfp_peaks(self, peak_data):
        """
        Assign cluster labels to GFP peaks using spatial correlation with centroids.

        pycrostates predict() requires Raw/Epochs for temporal segmentation.
        For GFP peaks (ChData), we manually compute labels using absolute spatial
        correlation (polarity-invariant) with the fitted centroids.

        Parameters
        ----------
        peak_data : ndarray
            Shape (n_channels, n_peaks) - the GFP peak topographies.

        Returns
        -------
        labels : ndarray
            Cluster labels for each peak (shape: n_peaks).
        gev : float
            Global Explained Variance for this assignment.
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)
        n_channels, n_peaks = peak_data.shape

        # Compute spatial correlation between each peak and each centroid
        # Using absolute correlation for polarity invariance
        labels = np.zeros(n_peaks, dtype=int)
        correlations = np.zeros((n_peaks, self.n_clusters))

        for i in range(n_peaks):
            peak = peak_data[:, i]  # (n_channels,)
            peak_norm = peak - peak.mean()
            peak_std = peak_norm.std()
            if peak_std < 1e-10:
                peak_std = 1e-10

            for k in range(self.n_clusters):
                centroid = centroids[k]  # (n_channels,)
                centroid_norm = centroid - centroid.mean()
                centroid_std = centroid_norm.std()
                if centroid_std < 1e-10:
                    centroid_std = 1e-10

                # Pearson correlation
                corr = np.mean(peak_norm * centroid_norm) / (peak_std * centroid_std)
                correlations[i, k] = np.abs(corr)  # Polarity-invariant

            labels[i] = np.argmax(correlations[i])

        # Compute GEV: sum of (GFP^2 * correlation^2) / sum(GFP^2)
        # GFP at each peak
        gfp = np.std(peak_data, axis=0)  # (n_peaks,)
        gfp_squared = gfp ** 2

        # Best correlation for each peak
        best_corr = correlations[np.arange(n_peaks), labels]

        # GEV = sum(GFP^2 * r^2) / sum(GFP^2)
        gev = np.sum(gfp_squared * (best_corr ** 2)) / np.sum(gfp_squared)

        return labels, float(gev)

    def get_labels_for_data(self, peak_data: np.ndarray) -> np.ndarray:
        """
        Get ModKMeans cluster labels for given electrode data.

        Parameters
        ----------
        peak_data : np.ndarray
            Shape (n_channels, n_samples) or (n_samples, n_channels)
            If shape is (n_samples, n_channels), will be transposed.

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each sample
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Ensure shape is (n_channels, n_samples)
        if peak_data.shape[0] != self.cluster_centers_.shape[1]:
            peak_data = peak_data.T

        labels, _ = self._assign_labels_gfp_peaks(peak_data)
        return labels

    def get_test_labels(self) -> np.ndarray:
        """
        Get stored test labels from the last evaluate() call.

        Returns
        -------
        labels : np.ndarray or None
            Test labels if available, None otherwise
        """
        return getattr(self, 'test_labels_', None)

    def _create_chdata_object(self, peak_data, gfp_peaks_info):
        """Create pycrostates ChData from peak data for ModKMeans compatibility."""
        # Handle both dict-like access and attribute access for MNE Info
        if hasattr(gfp_peaks_info, 'get'):
            ch_names = gfp_peaks_info.get('ch_names', gfp_peaks_info.ch_names if hasattr(gfp_peaks_info, 'ch_names') else None)
            sfreq = gfp_peaks_info.get('sfreq', getattr(gfp_peaks_info, 'sfreq', 250.0))
        else:
            ch_names = gfp_peaks_info.ch_names if hasattr(gfp_peaks_info, 'ch_names') else None
            sfreq = getattr(gfp_peaks_info, 'sfreq', 250.0)

        # Fallback if ch_names still None
        if ch_names is None:
            ch_names = [f'EEG{i:03d}' for i in range(peak_data.shape[0])]

        info = mne.create_info(
            ch_names=list(ch_names),
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Set montage if available
        try:
            dig = gfp_peaks_info.get('dig') if hasattr(gfp_peaks_info, 'get') else getattr(gfp_peaks_info, 'dig', None)
            chs = gfp_peaks_info.get('chs') if hasattr(gfp_peaks_info, 'get') else getattr(gfp_peaks_info, 'chs', None)
            if dig is not None and chs is not None:
                montage = mne.channels.make_dig_montage(
                    ch_pos={ch: chs[i]['loc'][:3] for i, ch in enumerate(ch_names)},
                    coord_frame='head'
                )
                info.set_montage(montage)
        except Exception:
            pass  # Skip montage if it fails

        # Create ChData object (required by pycrostates ModKMeans)
        return ChData(peak_data, info)

    def fit(self, gfp_peaks):
        """
        Fit ModKMeans on ALL GFP peaks (100% data mode).

        Parameters
        ----------
        gfp_peaks : pycrostates ChData
            GFP peaks extracted from process_eeg_signals.gfp_peaks
            Shape: (n_channels, N peaks)

        Note: 100% data mode - ModKMeans fits on ALL data, same as VAE training.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"BASELINE: ModKMeans (K={self.n_clusters}) - 100% Data Mode")
        self.logger.info("=" * 60)

        peak_data = gfp_peaks.get_data()
        n_channels, n_peaks = peak_data.shape

        # Store full GFP peaks timestamps
        self.all_gfp_peak_times = gfp_peaks.times if hasattr(gfp_peaks, 'times') else None
        if self.all_gfp_peak_times is not None:
            self.logger.info(f"Stored GFP peak timestamps: {len(self.all_gfp_peak_times)} peaks, "
                           f"time range [{self.all_gfp_peak_times[0]:.2f}s - {self.all_gfp_peak_times[-1]:.2f}s]")

        # 100% data mode: fit on ALL GFP peaks
        self.logger.info(f"FITTING on ALL {n_peaks} GFP peaks (100% data mode)")

        # Store data for metrics computation
        self.train_gfp_peaks = gfp_peaks
        self.n_train_peaks = n_peaks
        self.gfp_peaks = gfp_peaks

        try:
            self.modk.fit(self.gfp_peaks, n_jobs=-1)

            self.cluster_centers_ = self.modk.cluster_centers_
            self.fitted = True

            self.logger.info(f"GEV: {self.modk.GEV_:.4f}")
            self.logger.info(f"Centroid shape: {self.cluster_centers_.shape}")
            self.logger.info(f"Fitting complete on 100% of data ({n_peaks} peaks).")

            # Save cluster centers for later comparison with VAE
            centers_path = self.output_dir / "cluster_centers.npy"
            np.save(centers_path, self.cluster_centers_)
            self.logger.info(f"Saved cluster centers to: {centers_path}")

        except Exception as e:
            self.logger.error(f"Fitting failed: {e}")
            import traceback
            traceback.print_exc()

    def plot(self):
        """
        Plot microstate topomaps using pycrostates' built-in .plot() method.
        """
        if not self.fitted:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        self.logger.info("Plotting cluster centers...")

        try:
            fig = self.modk.plot(block=False)

            if fig is not None:
                fig.suptitle(
                    f"Microstate Topomaps (K={self.n_clusters})\nGEV = {self.modk.GEV_:.2%}",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.tight_layout()

                save_path = self.output_dir / "microstate_topomaps.png"
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Saved to: {save_path}")
                plt.close(fig)

            return fig

        except Exception as e:
            self.logger.error(f"Plotting failed: {e}")
            return None

    def plot_merged_centroids(self):
        """
        Plot ModKMeans cluster centroids as publication-ready microstate maps.

        Creates a visualization matching the VAE's merged_centroids.png style:
        - Circular head mask for each microstate
        - Nose and ear indicators
        - Standard microstate labels (A, B, C, D...)
        - Consistent color scale across all maps
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        self.logger.info(f"Creating merged_centroids visualization for {self.n_clusters} clusters...")

        try:
            # Get cluster centers (n_clusters, n_channels)
            centroids = self.modk.cluster_centers_.copy()

            # Center each centroid (zero mean) for proper polarity visualization
            for i in range(self.n_clusters):
                centroids[i] = centroids[i] - centroids[i].mean()

            # Get electrode positions from the stored info
            if not hasattr(self, 'train_gfp_peaks') or self.train_gfp_peaks is None:
                self.logger.warning("No GFP peaks info available for electrode positions.")
                return None

            info = self.train_gfp_peaks.info
            montage = info.get_montage()
            if montage is None:
                self.logger.warning("No montage found. Cannot create merged centroids plot.")
                return None

            # Get 2D positions for electrodes
            pos_dict = montage.get_positions()
            ch_pos = pos_dict["ch_pos"]
            ch_names = info.ch_names

            positions_3d = np.array([ch_pos[ch] for ch in ch_names])
            positions_2d = positions_3d[:, :2]  # x, y only

            # Normalize to [0, 1]
            pos_min = positions_2d.min(axis=0)
            pos_max = positions_2d.max(axis=0)
            pos_norm = (positions_2d - pos_min) / (pos_max - pos_min + 1e-8)

            # Create interpolated 2D images (40x40) for each centroid
            from scipy.interpolate import griddata

            img_size = 40
            margin = 0.1
            grid_x, grid_y = np.mgrid[0:1:complex(img_size), 0:1:complex(img_size)]

            centroid_images = np.zeros((self.n_clusters, img_size, img_size))
            for k in range(self.n_clusters):
                # Interpolate electrode values to grid
                centroid_images[k] = griddata(
                    pos_norm, centroids[k],
                    (grid_x, grid_y),
                    method='cubic',
                    fill_value=0
                )

            # Apply circular mask
            h, w = img_size, img_size
            y, x = np.ogrid[:h, :w]
            center = (h / 2, w / 2)
            radius = min(h, w) / 2 * 0.95
            mask = ((x - center[1])**2 + (y - center[0])**2) > radius**2

            centroid_images_masked = np.ma.array(centroid_images, mask=np.broadcast_to(mask, centroid_images.shape))

            # Layout
            n_cols = min(4, self.n_clusters)
            n_rows = int(np.ceil(self.n_clusters / n_cols))

            # Consistent color scale
            valid_data = centroid_images[~np.broadcast_to(mask, centroid_images.shape)]
            abs_max = np.percentile(np.abs(valid_data), 99)

            # Create figure
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), facecolor='white')
            if self.n_clusters == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # Microstate labels (A, B, C, D...)
            labels = [chr(65 + i) for i in range(self.n_clusters)]

            for i in range(self.n_clusters):
                ax = axes[i]
                ax.set_facecolor('white')

                # Plot the masked centroid
                im = ax.imshow(
                    centroid_images_masked[i],
                    cmap="RdBu_r",
                    origin="lower",
                    vmin=-abs_max,
                    vmax=abs_max,
                    interpolation="bilinear",
                )

                # Add head outline circle
                from matplotlib.patches import Circle
                circle = Circle(
                    (w / 2, h / 2),
                    radius,
                    fill=False,
                    color="black",
                    linewidth=2.5,
                )
                ax.add_patch(circle)

                # Add nose indicator (top)
                nose_x = w / 2
                nose_y = h / 2 + radius * 0.95
                ax.plot([nose_x - 3, nose_x, nose_x + 3],
                       [nose_y - 5, nose_y + 3, nose_y - 5],
                       'k-', linewidth=2)

                # Add ear indicators
                ear_y = h / 2
                ax.plot([w/2 - radius - 2, w/2 - radius - 5, w/2 - radius - 2],
                       [ear_y - 4, ear_y, ear_y + 4], 'k-', linewidth=1.5)
                ax.plot([w/2 + radius + 2, w/2 + radius + 5, w/2 + radius + 2],
                       [ear_y - 4, ear_y, ear_y + 4], 'k-', linewidth=1.5)

                ax.set_title(f"Microstate {labels[i]}", fontsize=14, fontweight="bold", pad=10)
                ax.axis("off")
                ax.set_xlim(-5, w + 5)
                ax.set_ylim(-5, h + 8)

            # Hide empty subplots
            for i in range(self.n_clusters, len(axes)):
                axes[i].axis("off")

            # Add shared colorbar
            cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Amplitude (μV, centered)", fontsize=11)
            cbar.ax.tick_params(labelsize=10)

            # Add GEV to title
            gev = self.modk.GEV_ if hasattr(self.modk, 'GEV_') else 0
            plt.suptitle(
                f"ModKMeans Microstate Templates (K={self.n_clusters})\nGEV = {gev:.2%}",
                fontsize=16,
                fontweight="bold",
                y=0.98
            )
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])

            save_path = self.output_dir / "baseline_merged_centroids.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()

            self.logger.info(f"Saved ModKMeans centroid visualization: {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Merged centroids plotting failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def reorder_and_rename_clusters(self, order: list = None, names: list = None):
        """
        Reorder and rename clusters using pycrostates conventions.

        If no order provided, clusters are ordered by GEV contribution (highest first).
        If no names provided, uses standard microstate naming: A, B, C, D, E, F, G...

        Parameters
        ----------
        order : list, optional
            Custom order of cluster indices. If None, orders by GEV contribution.
        names : list, optional
            Custom names for clusters. If None, uses A, B, C, D...

        Returns
        -------
        dict
            Information about the reordering
        """
        if not self.fitted:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        n_clusters = self.n_clusters

        # Determine order (by GEV contribution if not specified)
        if order is None:
            # Get cluster centers and compute their GEV contribution
            # Order by variance explained (approximate using centroid norms)
            centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)
            centroid_norms = np.linalg.norm(centroids, axis=1)
            order = np.argsort(centroid_norms)[::-1].tolist()  # Highest norm first
            self.logger.info(f"Auto-ordering by centroid magnitude: {order}")

        # Determine names
        if names is None:
            # Standard microstate naming: A, B, C, D, E, F, G...
            names = [chr(65 + i) for i in range(n_clusters)]  # A=65 in ASCII

        try:
            # Reorder clusters
            self.modk.reorder_clusters(order=order)
            self.logger.info(f"Reordered clusters: {order}")

            # Rename clusters
            self.modk.rename_clusters(new_names=names)
            self.logger.info(f"Renamed clusters: {names}")

            return {
                "order": order,
                "names": names,
                "n_clusters": n_clusters
            }

        except Exception as e:
            self.logger.error(f"Reorder/rename failed: {e}")
            return None

    def plot_microstate_topomaps_named(self, show_gev: bool = True):
        """
        Plot microstate topomaps with standard naming (A, B, C, D...).

        This creates publication-ready microstate visualizations following
        the standard conventions in EEG microstate literature.

        Parameters
        ----------
        show_gev : bool
            Whether to show GEV in the title

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        if not self.fitted:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        try:
            # Plot using pycrostates
            fig = self.modk.plot(block=False)

            if fig is not None:
                title = f"Microstate Topomaps (K={self.n_clusters})"
                if show_gev and hasattr(self.modk, 'GEV_'):
                    title += f"\nGlobal Explained Variance = {self.modk.GEV_:.2%}"

                fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
                plt.tight_layout()

                save_path = self.output_dir / "microstate_topomaps_named.png"
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Saved named topomaps to: {save_path}")
                plt.close(fig)

            return fig

        except Exception as e:
            self.logger.error(f"Named topomap plotting failed: {e}")
            return None

    def get_cluster_centers_for_comparison(self) -> dict:
        """
        Get cluster centers in a format suitable for comparison with VAE.

        Returns the electrode values for each microstate,
        which can be directly compared with VAE extracted electrode values.

        Returns
        -------
        dict
            Contains centroids array and metadata
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)

        # Get cluster names if available
        try:
            names = list(self.modk.cluster_names)
        except Exception:
            names = [f"MS{i+1}" for i in range(self.n_clusters)]

        return {
            "centroids": centroids,  # (K, 32) array
            "n_clusters": self.n_clusters,
            "n_channels": centroids.shape[1],
            "names": names,
            "gev": float(self.modk.GEV_) if hasattr(self.modk, 'GEV_') else None,
            "method": "ModKMeans (pycrostates)"
        }

    def plot_centroid_spatial_correlations(self):
        """
        Compute and visualize spatial correlation matrix between ModKMeans centroids.

        Creates two visualizations:
        1. Full correlation matrix (showing polarity)
        2. Absolute correlation matrix (polarity-invariant)

        Returns
        -------
        dict
            Correlation analysis results including matrix and identified pairs
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        from scipy.stats import pearsonr

        self.logger.info("Computing spatial correlation matrix between ModKMeans centroids...")

        centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)
        n_clusters = centroids.shape[0]

        # Compute full correlation matrix
        corr_matrix = np.zeros((n_clusters, n_clusters))
        pval_matrix = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                r, p = pearsonr(centroids[i], centroids[j])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p

        # Create labels for centroids (matching pycrostates convention)
        labels = [f"MS{i+1}" for i in range(n_clusters)]

        # ===== VISUALIZATION 1: Full Correlation Matrix =====
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spatial Correlation (Pearson r)', fontsize=12)

        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        # Add correlation values as text
        for i in range(n_clusters):
            for j in range(n_clusters):
                r = corr_matrix[i, j]
                text_color = 'white' if abs(r) > 0.7 else 'black'
                ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                       fontsize=10, color=text_color, fontweight='bold')

        ax.set_xlabel('Microstate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Microstate', fontsize=12, fontweight='bold')
        ax.set_title(f'ModKMeans Centroid Spatial Correlation Matrix (K={n_clusters})\n'
                    f'GEV = {self.modk.GEV_:.2%}',
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / "centroid_correlation_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved correlation matrix to: {save_path}")

        # ===== VISUALIZATION 2: Absolute Correlation (Polarity-Invariant) =====
        abs_corr_matrix = np.abs(corr_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(abs_corr_matrix, cmap='Reds', vmin=0, vmax=1, aspect='equal')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('|Spatial Correlation|', fontsize=12)

        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        for i in range(n_clusters):
            for j in range(n_clusters):
                r = abs_corr_matrix[i, j]
                text_color = 'white' if r > 0.7 else 'black'
                ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                       fontsize=10, color=text_color, fontweight='bold')

        ax.set_xlabel('Microstate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Microstate', fontsize=12, fontweight='bold')
        ax.set_title(f'ModKMeans Absolute Correlation (Polarity-Invariant)\n'
                    'High values indicate similar/inverted topographies',
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path_abs = self.output_dir / "centroid_correlation_absolute.png"
        plt.savefig(save_path_abs, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved absolute correlation matrix to: {save_path_abs}")

        # Identify highly correlated/anti-correlated pairs (excluding diagonal)
        redundant_pairs = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if abs(corr_matrix[i, j]) > 0.8:
                    redundant_pairs.append({
                        "pair": (labels[i], labels[j]),
                        "correlation": float(corr_matrix[i, j]),
                        "type": "similar" if corr_matrix[i, j] > 0 else "polarity-inverted"
                    })

        if redundant_pairs:
            self.logger.warning(f"Found {len(redundant_pairs)} highly correlated pairs:")
            for pair in redundant_pairs:
                self.logger.warning(f"  {pair['pair']}: r={pair['correlation']:.3f} ({pair['type']})")

        return {
            "correlation_matrix": corr_matrix.tolist(),
            "absolute_correlation_matrix": abs_corr_matrix.tolist(),
            "redundant_pairs": redundant_pairs,
            "labels": labels,
        }

    def plot_centroid_summary(self, info=None):
        """
        Create comprehensive centroid visualization combining topomaps and correlation.

        Parameters
        ----------
        info : mne.Info, optional
            MNE Info object with channel positions. If None, uses stored info.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        from scipy.stats import pearsonr

        centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)
        n_clusters = centroids.shape[0]
        n_channels = centroids.shape[1]

        self.logger.info(f"Creating centroid summary visualization (K={n_clusters}, {n_channels} channels)")

        # Compute correlation matrix
        corr_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                r, _ = pearsonr(centroids[i], centroids[j])
                corr_matrix[i, j] = r

        # Create figure with topomaps on top row and correlation matrix below
        fig = plt.figure(figsize=(4 * n_clusters, 10))

        # Top row: Topomaps (use pycrostates internal plotting or bar plots)
        # Since we might not have MNE info, we'll create a simpler visualization
        gs = fig.add_gridspec(2, n_clusters, height_ratios=[1, 1.2], hspace=0.3)

        # Row 1: Channel activity bar plots for each centroid
        for k in range(n_clusters):
            ax = fig.add_subplot(gs[0, k])
            centroid_vals = centroids[k]

            # Sort channels by value for better visualization
            sorted_idx = np.argsort(centroid_vals)
            colors = ['blue' if v < 0 else 'red' for v in centroid_vals[sorted_idx]]

            ax.barh(range(n_channels), centroid_vals[sorted_idx], color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_title(f'MS{k+1}', fontsize=12, fontweight='bold')

            if k == 0:
                ax.set_ylabel('Channels (sorted)', fontsize=10)
            else:
                ax.set_yticks([])

            ax.set_xlabel('Amplitude', fontsize=9)

            # Add stats annotation
            ax.text(0.95, 0.95, f'max: {centroid_vals.max():.2f}\nmin: {centroid_vals.min():.2f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Row 2: Correlation matrix (spanning all columns)
        ax_corr = fig.add_subplot(gs[1, :])
        im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

        cbar = plt.colorbar(im, ax=ax_corr, shrink=0.6, orientation='horizontal', pad=0.15)
        cbar.set_label('Spatial Correlation', fontsize=11)

        labels = [f'MS{i+1}' for i in range(n_clusters)]
        ax_corr.set_xticks(range(n_clusters))
        ax_corr.set_yticks(range(n_clusters))
        ax_corr.set_xticklabels(labels, fontsize=11)
        ax_corr.set_yticklabels(labels, fontsize=11)

        # Add values
        for i in range(n_clusters):
            for j in range(n_clusters):
                r = corr_matrix[i, j]
                text_color = 'white' if abs(r) > 0.7 else 'black'
                ax_corr.text(j, i, f'{r:.2f}', ha='center', va='center',
                           fontsize=10, color=text_color, fontweight='bold')

        ax_corr.set_title('Centroid Spatial Correlation Matrix', fontsize=12, fontweight='bold')

        # Overall title
        fig.suptitle(f'ModKMeans Microstate Centroids (K={n_clusters})\n'
                    f'GEV = {self.modk.GEV_:.2%}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        save_path = self.output_dir / "centroid_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved centroid summary to: {save_path}")

        return fig

    def plot_centroid_topomaps_mne(self, info):
        """
        Plot centroids as MNE topomaps (requires valid MNE Info with channel positions).

        Parameters
        ----------
        info : mne.Info
            MNE Info object with electrode positions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)
        n_clusters = centroids.shape[0]

        self.logger.info(f"Creating MNE topomaps for {n_clusters} centroids...")

        try:
            # Create figure
            fig, axes = plt.subplots(1, n_clusters, figsize=(3 * n_clusters, 4))
            if n_clusters == 1:
                axes = [axes]

            # Determine global vmin/vmax for consistent colormap
            vmax = np.abs(centroids).max()
            vmin = -vmax

            for k, ax in enumerate(axes):
                # Plot topomap
                mne.viz.plot_topomap(
                    centroids[k],
                    info,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(vmin, vmax),
                    contours=6,
                )
                ax.set_title(f'MS{k+1}', fontsize=12, fontweight='bold')

            # Clean title without colorbar (amplitude range in subtitle)
            fig.suptitle(f'ModKMeans Microstate Topomaps (K={n_clusters})\n'
                        f'GEV = {self.modk.GEV_:.2%} | Amplitude: ±{vmax:.2f} a.u.',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()
            save_path = self.output_dir / "centroid_topomaps_mne.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved MNE topomaps to: {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"MNE topomap plotting failed: {e}")
            self.logger.info("Falling back to bar plot visualization")
            return self.plot_centroid_summary()

    def plot_vae_comparison(self, vae_centroids_electrode: np.ndarray, vae_gev: float = None):
        """
        Create comparison visualization between VAE and ModKMeans centroids.

        Computes cross-correlation to find best matching pairs (accounting for polarity).

        Parameters
        ----------
        vae_centroids_electrode : np.ndarray
            VAE centroids in electrode space, shape (n_vae_clusters, n_channels).
            These should be extracted from the VAE's decoded topomaps.
        vae_gev : float, optional
            VAE's GEV for display.

        Returns
        -------
        dict
            Comparison results including cross-correlation and best matches
        """
        if not self.fitted or self.modk.cluster_centers_ is None:
            self.logger.warning("Model not fitted. Call fit() first.")
            return None

        from scipy.stats import pearsonr
        from scipy.optimize import linear_sum_assignment

        modk_centroids = self.modk.cluster_centers_  # (n_modk, n_channels)
        n_modk = modk_centroids.shape[0]
        n_vae = vae_centroids_electrode.shape[0]

        self.logger.info(f"Comparing VAE ({n_vae} clusters) vs ModKMeans ({n_modk} clusters) centroids...")

        # Compute cross-correlation matrix (absolute for polarity invariance)
        cross_corr = np.zeros((n_modk, n_vae))
        cross_corr_signed = np.zeros((n_modk, n_vae))

        for i in range(n_modk):
            for j in range(n_vae):
                r, _ = pearsonr(modk_centroids[i], vae_centroids_electrode[j])
                cross_corr_signed[i, j] = r
                cross_corr[i, j] = abs(r)

        # Find optimal matching using Hungarian algorithm (maximize correlation)
        min_k = min(n_modk, n_vae)
        cost_matrix = 1 - cross_corr[:min_k, :min_k]  # Convert to minimization
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind):
            matches.append({
                "modk": f"MS{i+1}",
                "vae": f"C{j+1}",
                "correlation": float(cross_corr_signed[i, j]),
                "abs_correlation": float(cross_corr[i, j]),
                "polarity": "same" if cross_corr_signed[i, j] > 0 else "inverted"
            })

        # Log matching results
        self.logger.info("Best centroid matches (ModKMeans -> VAE):")
        for m in matches:
            self.logger.info(f"  {m['modk']} -> {m['vae']}: r={m['correlation']:.3f} ({m['polarity']})")

        avg_match_corr = np.mean([m['abs_correlation'] for m in matches])
        self.logger.info(f"Average match correlation: {avg_match_corr:.3f}")

        # ===== VISUALIZATION 1: Cross-Correlation Matrix =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Signed correlation
        ax1 = axes[0]
        im1 = ax1.imshow(cross_corr_signed, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        ax1.set_xticks(range(n_vae))
        ax1.set_yticks(range(n_modk))
        ax1.set_xticklabels([f'VAE C{i+1}' for i in range(n_vae)], fontsize=10)
        ax1.set_yticklabels([f'MS{i+1}' for i in range(n_modk)], fontsize=10)
        ax1.set_xlabel('VAE Centroids', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ModKMeans Centroids', fontsize=12, fontweight='bold')
        ax1.set_title('Signed Cross-Correlation', fontsize=12, fontweight='bold')

        # Add values
        for i in range(n_modk):
            for j in range(n_vae):
                r = cross_corr_signed[i, j]
                text_color = 'white' if abs(r) > 0.6 else 'black'
                ax1.text(j, i, f'{r:.2f}', ha='center', va='center',
                        fontsize=9, color=text_color)

        # Highlight best matches
        for m in matches:
            i = int(m['modk'][2:]) - 1
            j = int(m['vae'][1:]) - 1
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                 edgecolor='lime', linewidth=3)
            ax1.add_patch(rect)

        # Absolute correlation (polarity-invariant)
        ax2 = axes[1]
        im2 = ax2.imshow(cross_corr, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        ax2.set_xticks(range(n_vae))
        ax2.set_yticks(range(n_modk))
        ax2.set_xticklabels([f'VAE C{i+1}' for i in range(n_vae)], fontsize=10)
        ax2.set_yticklabels([f'MS{i+1}' for i in range(n_modk)], fontsize=10)
        ax2.set_xlabel('VAE Centroids', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ModKMeans Centroids', fontsize=12, fontweight='bold')
        ax2.set_title('Absolute Cross-Correlation (Polarity-Invariant)', fontsize=12, fontweight='bold')

        for i in range(n_modk):
            for j in range(n_vae):
                r = cross_corr[i, j]
                text_color = 'white' if r > 0.6 else 'black'
                ax2.text(j, i, f'{r:.2f}', ha='center', va='center',
                        fontsize=9, color=text_color)

        # Highlight best matches
        for m in matches:
            i = int(m['modk'][2:]) - 1
            j = int(m['vae'][1:]) - 1
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                 edgecolor='lime', linewidth=3)
            ax2.add_patch(rect)

        # Title with GEV comparison
        gev_str = f"VAE GEV: {vae_gev:.2%}" if vae_gev else "VAE GEV: N/A"
        fig.suptitle(f'VAE vs ModKMeans Centroid Comparison\n'
                    f'ModKMeans GEV: {self.modk.GEV_:.2%} | {gev_str}\n'
                    f'Avg Match Correlation: {avg_match_corr:.3f}',
                    fontsize=13, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / "vae_vs_modkmeans_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved comparison to: {save_path}")

        # ===== VISUALIZATION 2: Matched Centroid Profiles =====
        fig2, axes2 = plt.subplots(min_k, 2, figsize=(12, 3 * min_k))
        if min_k == 1:
            axes2 = axes2.reshape(1, -1)

        for idx, m in enumerate(matches):
            i = int(m['modk'][2:]) - 1
            j = int(m['vae'][1:]) - 1

            ax_modk = axes2[idx, 0]
            ax_vae = axes2[idx, 1]

            n_channels = modk_centroids.shape[1]

            # ModKMeans centroid
            ax_modk.bar(range(n_channels), modk_centroids[i],
                       color='steelblue', alpha=0.7)
            ax_modk.axhline(y=0, color='black', linewidth=0.5)
            ax_modk.set_title(f'{m["modk"]} (ModKMeans)', fontsize=11, fontweight='bold')
            ax_modk.set_ylabel('Amplitude', fontsize=10)
            if idx == min_k - 1:
                ax_modk.set_xlabel('Channel', fontsize=10)

            # VAE centroid (flip if polarity inverted for visual comparison)
            vae_vals = vae_centroids_electrode[j]
            if m['polarity'] == 'inverted':
                vae_vals = -vae_vals
                label = f'{m["vae"]} (VAE, flipped)'
            else:
                label = f'{m["vae"]} (VAE)'

            ax_vae.bar(range(n_channels), vae_vals,
                      color='darkorange', alpha=0.7)
            ax_vae.axhline(y=0, color='black', linewidth=0.5)
            ax_vae.set_title(f'{label} | r={m["correlation"]:.3f}', fontsize=11, fontweight='bold')
            if idx == min_k - 1:
                ax_vae.set_xlabel('Channel', fontsize=10)

        fig2.suptitle('Matched Centroid Profiles (ModKMeans vs VAE)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path2 = self.output_dir / "matched_centroid_profiles.png"
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved matched profiles to: {save_path2}")

        # Save results to JSON
        results = {
            "n_modk_clusters": n_modk,
            "n_vae_clusters": n_vae,
            "modk_gev": float(self.modk.GEV_),
            "vae_gev": float(vae_gev) if vae_gev else None,
            "cross_correlation_matrix": cross_corr_signed.tolist(),
            "abs_cross_correlation_matrix": cross_corr.tolist(),
            "best_matches": matches,
            "average_match_correlation": float(avg_match_corr),
        }
        with open(self.output_dir / "vae_comparison_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def evaluate(self):
        """
        Evaluates the baseline on ALL data using pycrostates metrics (100% data mode).

        Computes:
        - GEV on full data (using fitted centroids)
        - Clustering metrics (silhouette, CH, DB, Dunn) on full data
        """
        if not self.fitted:
            self.logger.warning("Baseline not fitted. Skipping evaluation.")
            return {}

        self.logger.info("Evaluating Baseline Performance on ALL data (100% data mode)...")

        # 100% data mode: evaluate on all data
        eval_chdata = self.gfp_peaks
        n_peaks = eval_chdata.get_data().shape[1] if eval_chdata is not None else 0
        self.logger.info(f"Evaluating on ALL data: {n_peaks} peaks")

        # Get numpy array from ChData for manual label assignment
        eval_data = eval_chdata.get_data()  # (n_channels, n_peaks)

        # Compute labels and GEV using spatial correlation with fitted centroids
        # pycrostates predict() requires Raw/Epochs, not ChData, so we use manual assignment
        try:
            all_labels, all_gev = self._assign_labels_gfp_peaks(eval_data)
            self.all_labels_ = all_labels  # Store for later use (e.g., t-SNE visualization)
            self.logger.info(f"GEV: {all_gev:.4f}")
            self.logger.info(f"Label distribution: {np.bincount(all_labels, minlength=self.n_clusters)}")
        except Exception as e:
            self.logger.warning(f"Could not compute GEV: {e}")
            all_gev = self.modk.GEV_  # Fall back to fitted GEV
            all_labels = None
            self.all_labels_ = None

        # Subsample if dataset is too large for O(N^2) metric computations
        MAX_METRIC_SAMPLES = 50000
        if all_labels is not None and n_peaks > MAX_METRIC_SAMPLES:
            rng = np.random.default_rng(42)
            subsample_idx = rng.choice(n_peaks, size=MAX_METRIC_SAMPLES, replace=False)
            eval_data = eval_data[:, subsample_idx]
            all_labels = all_labels[subsample_idx]
            n_peaks = MAX_METRIC_SAMPLES
            self.logger.info(f"Subsampled to {MAX_METRIC_SAMPLES} peaks for metric computation")

        # Compute clustering metrics on full data using sklearn
        # Note: sklearn metrics use Euclidean distance, pycrostates uses spatial correlation
        if all_labels is None:
            self.logger.warning("No test labels available - skipping metrics")
            sil_score = -999.0
            ch_score = -999.0
            db_score = -999.0
            dunn = -999.0
        else:
            try:
                # sklearn metrics expect (n_samples, n_features) shape
                X_all = eval_data.T  # (n_peaks, n_channels)

                # ============================================================
                # METRICS WITHOUT POLARITY ALIGNMENT (raw sklearn)
                # ============================================================
                sil_score_raw = sklearn_silhouette(X_all, all_labels)
                ch_score_raw = sklearn_ch(X_all, all_labels)
                db_score_raw = sklearn_db(X_all, all_labels)
                # Use correlation-based Dunn (Euclidean-based _compute_dunn_index gives extreme values)
                dunn_raw = custom_pycrostates_dunn(X_all, all_labels)

                self.logger.info("Computed metrics WITHOUT polarity alignment (raw)")

                # ============================================================
                # METRICS WITH POLARITY ALIGNMENT (pycrostates-style)
                # ============================================================
                # Microstates are polarity-invariant, but sklearn uses Euclidean distance
                # Without alignment, opposite-polarity samples appear far apart
                # This follows pycrostates' implementation in their metrics module
                centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)
                centroid_for_each_sample = centroids[all_labels]  # (n_peaks, n_channels)

                # Compute sign of dot product between each sample and its centroid
                # Positive = same polarity, Negative = opposite polarity
                dot_products = np.sum(X_all * centroid_for_each_sample, axis=1)
                signs = np.sign(dot_products)
                signs[signs == 0] = 1  # Handle edge case of zero dot product

                # Flip samples to match centroid polarity
                X_all_aligned = X_all * signs[:, np.newaxis]

                sil_score = sklearn_silhouette(X_all_aligned, all_labels)
                ch_score = sklearn_ch(X_all_aligned, all_labels)
                db_score = sklearn_db(X_all_aligned, all_labels)
                # Use correlation-based Dunn (Euclidean-based _compute_dunn_index gives extreme values)
                dunn = custom_pycrostates_dunn(X_all_aligned, all_labels)

                self.logger.info("Computed metrics WITH polarity alignment (sklearn + correlation-based Dunn)")

                # ============================================================
                # PYCROSTATES-STYLE METRICS (correlation-based distance)
                # ============================================================
                # These use spatial correlation instead of Euclidean distance,
                # which is more appropriate for EEG topographies
                sil_score_corr = custom_pycrostates_silhouette(X_all, all_labels)
                ch_score_corr = custom_pycrostates_ch(X_all, all_labels, centroids)
                db_score_corr = custom_pycrostates_db(X_all, all_labels, centroids)
                dunn_corr = custom_pycrostates_dunn(X_all, all_labels)

                self.logger.info("Computed metrics WITH correlation-based distance (pycrostates-style)")

            except Exception as e:
                self.logger.warning(f"Could not compute some metrics: {e}")
                import traceback
                traceback.print_exc()
                sil_score = 0.0
                ch_score = 0.0
                db_score = 0.0
                dunn = 0.0
                sil_score_raw = 0.0
                ch_score_raw = 0.0
                db_score_raw = 0.0
                dunn_raw = 0.0
                sil_score_corr = 0.0
                ch_score_corr = 0.0
                db_score_corr = 0.0
                dunn_corr = 0.0

        # Initialize scores if they weren't set (e.g., skipped due to large dataset)
        if 'sil_score_raw' not in dir():
            sil_score_raw = -999.0
            ch_score_raw = -999.0
            db_score_raw = -999.0
            dunn_raw = -999.0
        if 'sil_score_corr' not in dir():
            sil_score_corr = -999.0
            ch_score_corr = -999.0
            db_score_corr = -999.0
            dunn_corr = -999.0

        metrics = {
            "strategy_name": "Baseline (ModKMeans)",
            "gev": float(all_gev),
            # Sklearn with polarity alignment (Euclidean distance)
            "silhouette_scores": float(sil_score),
            "ch_scores": float(ch_score),
            "db_scores": float(db_score),
            "dunn_score": float(dunn),
            # Sklearn without polarity alignment (raw)
            "silhouette_scores_raw": float(sil_score_raw),
            "ch_scores_raw": float(ch_score_raw),
            "db_scores_raw": float(db_score_raw),
            "dunn_score_raw": float(dunn_raw),
            # Pycrostates-style (correlation-based distance)
            "silhouette_corr": float(sil_score_corr),
            "ch_corr": float(ch_score_corr),
            "db_corr": float(db_score_corr),
            "dunn_corr": float(dunn_corr),
            "n_clusters": self.n_clusters,
            "n_peaks": n_peaks,
        }

        self.logger.info(f"Baseline Results (100% Data Mode):")
        self.logger.info(f"   GEV: {all_gev:.4f}")
        if sil_score == -999.0:
            self.logger.info(f"   Metrics: SKIPPED (dataset too large)")
        else:
            self.logger.info(f"   --- 1. Sklearn RAW (no alignment, Euclidean) ---")
            self.logger.info(f"   Silhouette:         {sil_score_raw:.4f}")
            self.logger.info(f"   Calinski-Harabasz:  {ch_score_raw:.4f}")
            self.logger.info(f"   Davies-Bouldin:     {db_score_raw:.4f}")
            self.logger.info(f"   Dunn:               {dunn_raw:.4f}")
            self.logger.info(f"   --- 2. Sklearn ALIGNED (polarity aligned, Euclidean) ---")
            self.logger.info(f"   Silhouette:         {sil_score:.4f}")
            self.logger.info(f"   Calinski-Harabasz:  {ch_score:.4f}")
            self.logger.info(f"   Davies-Bouldin:     {db_score:.4f}")
            self.logger.info(f"   Dunn:               {dunn:.4f}")
            self.logger.info(f"   --- 3. PYCROSTATES-STYLE (correlation-based distance) ---")
            self.logger.info(f"   Silhouette:         {sil_score_corr:.4f}")
            self.logger.info(f"   Calinski-Harabasz:  {ch_score_corr:.4f}")
            self.logger.info(f"   Davies-Bouldin:     {db_score_corr:.4f}")
            self.logger.info(f"   Dunn:               {dunn_corr:.4f}")

        with open(self.output_dir / "baseline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def evaluate_on_raw(self, raw):
        """
        Evaluate ModKMeans using pycrostates predict() on continuous Raw data (100% data mode).

        This uses the actual pycrostates temporal segmentation method (predict),
        which assigns labels to ALL timepoints using spatial correlation with
        fitted centroids, optionally with temporal smoothing.

        Parameters
        ----------
        raw : mne.io.Raw
            MNE Raw object with EEG data.

        Returns
        -------
        dict
            Dictionary containing metrics from predict-based evaluation on full data.
        """
        if not self.fitted:
            self.logger.warning("Model not fitted. Call fit() first.")
            return {}

        self.logger.info("=" * 60)
        self.logger.info("EVALUATING ModKMeans using pycrostates predict() on Raw (100% Data Mode)")
        self.logger.info("=" * 60)

        try:
            # Use pycrostates predict() for temporal segmentation
            # factor=0 means no temporal smoothing (pure spatial correlation)
            # reject_edges=False to keep all labels
            segmentation = self.modk.predict(
                raw, factor=0, reject_edges=False, reject_by_annotation=True
            )
            all_labels = segmentation._labels
            n_total = len(all_labels)

            self.logger.info(f"Total timepoints segmented: {n_total}")
            self.logger.info(f"Label distribution: {np.bincount(all_labels[all_labels >= 0], minlength=self.n_clusters)}")

            # Get raw data for metrics computation
            raw_data = raw.get_data()  # (n_channels, n_timepoints)
            sfreq = raw.info['sfreq']

            # 100% data mode: use ALL data
            eval_labels = all_labels
            eval_data = raw_data  # (n_channels, n_samples)

            n_eval = len(eval_labels)
            self.logger.info(f"Evaluating on ALL data: {n_eval} samples ({n_eval/sfreq:.1f} seconds)")

            # Filter out unlabeled samples (-1) for metrics
            valid_mask = eval_labels >= 0
            eval_labels_valid = eval_labels[valid_mask]
            eval_data_valid = eval_data[:, valid_mask].T  # (n_valid_samples, n_channels)

            n_valid = len(eval_labels_valid)
            self.logger.info(f"Valid samples (non-rejected): {n_valid}")
            self.logger.info(f"Label distribution: {np.bincount(eval_labels_valid, minlength=self.n_clusters)}")

            # Compute clustering metrics on full data
            if n_valid < 50:
                self.logger.warning(f"Too few valid samples ({n_valid}) for reliable metrics")
                sil_score = -999.0
                ch_score = -999.0
                db_score = -999.0
                sil_score_raw = -999.0
                ch_score_raw = -999.0
                db_score_raw = -999.0
            else:
                # WITHOUT polarity alignment (raw)
                sil_score_raw = sklearn_silhouette(eval_data_valid, eval_labels_valid)
                ch_score_raw = sklearn_ch(eval_data_valid, eval_labels_valid)
                db_score_raw = sklearn_db(eval_data_valid, eval_labels_valid)

                # WITH polarity alignment (pycrostates-style)
                centroids = self.modk.cluster_centers_
                centroid_for_each_sample = centroids[eval_labels_valid]
                dot_products = np.sum(eval_data_valid * centroid_for_each_sample, axis=1)
                signs = np.sign(dot_products)
                signs[signs == 0] = 1
                eval_data_aligned = eval_data_valid * signs[:, np.newaxis]

                sil_score = sklearn_silhouette(eval_data_aligned, eval_labels_valid)
                ch_score = sklearn_ch(eval_data_aligned, eval_labels_valid)
                db_score = sklearn_db(eval_data_aligned, eval_labels_valid)

            # Compute GEV on full data
            # GEV measures how well centroids explain variance at each timepoint
            eval_gfp = np.std(eval_data, axis=0)
            eval_gfp_squared = eval_gfp ** 2

            # Get correlations for GEV calculation
            centroids = self.modk.cluster_centers_
            eval_correlations = np.zeros(n_eval)
            for i in range(n_eval):
                if eval_labels[i] >= 0:
                    sample = eval_data[:, i]
                    centroid = centroids[eval_labels[i]]
                    # Pearson correlation
                    sample_norm = sample - sample.mean()
                    centroid_norm = centroid - centroid.mean()
                    corr = np.abs(np.sum(sample_norm * centroid_norm) /
                                 (np.std(sample) * np.std(centroid) * len(sample)))
                    eval_correlations[i] = corr

            # GEV = sum(GFP^2 * r^2) / sum(GFP^2)
            eval_gev = np.sum(eval_gfp_squared * eval_correlations**2) / np.sum(eval_gfp_squared)

            metrics = {
                "strategy_name": "Baseline predict() on Raw (100% Data Mode)",
                "gev": float(eval_gev),
                # With polarity alignment (pycrostates-style)
                "silhouette_scores": float(sil_score),
                "ch_scores": float(ch_score),
                "db_scores": float(db_score),
                # Without polarity alignment (raw sklearn)
                "silhouette_scores_raw": float(sil_score_raw),
                "ch_scores_raw": float(ch_score_raw),
                "db_scores_raw": float(db_score_raw),
                "n_clusters": self.n_clusters,
                "n_samples": n_eval,
                "n_valid_samples": n_valid,
            }

            self.logger.info(f"Results (predict on Raw, 100% data mode):")
            self.logger.info(f"   GEV: {eval_gev:.4f}")
            if sil_score != -999.0:
                self.logger.info(f"   --- WITHOUT Polarity Alignment (raw) ---")
                self.logger.info(f"   Silhouette (raw):         {sil_score_raw:.4f}")
                self.logger.info(f"   Calinski-Harabasz (raw):  {ch_score_raw:.4f}")
                self.logger.info(f"   Davies-Bouldin (raw):     {db_score_raw:.4f}")
                self.logger.info(f"   --- WITH Polarity Alignment (pycrostates-style) ---")
                self.logger.info(f"   Silhouette (aligned):     {sil_score:.4f}")
                self.logger.info(f"   Calinski-Harabasz (aligned): {ch_score:.4f}")
                self.logger.info(f"   Davies-Bouldin (aligned): {db_score:.4f}")
            else:
                self.logger.info("   Clustering metrics: SKIPPED (insufficient samples)")

            # Save metrics
            with open(self.output_dir / "baseline_predict_raw_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            return metrics

        except Exception as e:
            self.logger.error(f"evaluate_on_raw() failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def plot_segmentation(self, raw, n_samples=500):
        """
        Plot temporal segmentation of microstates using pycrostates predict.

        Shows how microstate labels change over time.

        Parameters
        ----------
        raw : mne.io.Raw
            MNE Raw object with EEG data.
        n_samples : int
            Number of samples to visualize (for readability).
        """
        if not self.fitted:
            self.logger.warning("Baseline not fitted. Cannot plot segmentation.")
            return

        self.logger.info("Generating microstate segmentation plot...")

        try:
            # Predict labels using pycrostates
            segmentation = self.modk.predict(raw)
            labels = segmentation._labels

            # Get sampling frequency from raw
            sfreq = raw.info['sfreq']

            # Limit samples for visualization
            n_samples = min(n_samples, len(labels))
            labels_plot = labels[:n_samples]
            time = np.arange(n_samples) / sfreq

            # Calculate GFP from raw data
            data = raw.get_data()[:, :n_samples]
            gfp = np.std(data, axis=0)

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

            # Top: GFP with colored background by microstate
            ax1 = axes[0]
            ax1.plot(time, gfp, color="black", linewidth=0.8, label="GFP")

            # Color background by microstate
            colors = plt.cm.Set1(np.linspace(0, 1, self.n_clusters))
            for i in range(len(labels_plot) - 1):
                if labels_plot[i] >= 0:  # Skip unlabeled segments (-1)
                    ax1.axvspan(
                        time[i], time[i + 1],
                        alpha=0.3,
                        color=colors[labels_plot[i]],
                        linewidth=0,
                    )

            ax1.set_ylabel("GFP (a.u.)", fontweight="bold")
            ax1.set_title(
                f"Microstate Segmentation (K={self.n_clusters})",
                fontweight="bold",
                fontsize=14,
            )
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            # Bottom: Microstate labels as step plot
            ax2 = axes[1]
            ax2.step(time, labels_plot, where="mid", color="black", linewidth=1.5)
            ax2.set_ylim(-0.5, self.n_clusters - 0.5)
            ax2.set_yticks(range(self.n_clusters))
            ax2.set_yticklabels([f"MS {i+1}" for i in range(self.n_clusters)])
            ax2.set_xlabel("Time (s)", fontweight="bold")
            ax2.set_ylabel("Microstate", fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="x")

            # Add legend for microstate colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[i], alpha=0.5, label=f"MS {i+1}")
                for i in range(self.n_clusters)
            ]
            ax2.legend(
                handles=legend_elements,
                loc="upper right",
                ncol=min(self.n_clusters, 4),
            )

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "baseline_segmentation.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            self.logger.info(f"Saved segmentation plot to: {self.output_dir / 'baseline_segmentation.png'}")

            return labels

        except Exception as e:
            self.logger.error(f"Segmentation plot failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_microstate_statistics(self, raw):
        """
        Plot microstate temporal statistics: duration, occurrence, coverage.

        Parameters
        ----------
        raw : mne.io.Raw
            MNE Raw object with EEG data.
        """
        if not self.fitted:
            return

        self.logger.info("Computing microstate statistics...")

        try:
            # Predict labels using pycrostates
            segmentation = self.modk.predict(raw)
            labels = segmentation._labels
            sfreq = raw.info['sfreq']

            # Filter out unlabeled segments (-1)
            valid_mask = labels >= 0
            labels = labels[valid_mask]

            # Calculate statistics
            n_samples = len(labels)
            total_time = n_samples / sfreq

            # Duration: length of continuous segments
            durations = {i: [] for i in range(self.n_clusters)}
            current_label = labels[0]
            current_duration = 1

            for i in range(1, len(labels)):
                if labels[i] == current_label:
                    current_duration += 1
                else:
                    durations[current_label].append(current_duration / sfreq * 1000)  # ms
                    current_label = labels[i]
                    current_duration = 1
            durations[current_label].append(current_duration / sfreq * 1000)

            # Coverage: percentage of time in each state
            unique, counts = np.unique(labels, return_counts=True)
            coverage = np.zeros(self.n_clusters)
            for u, c in zip(unique, counts):
                coverage[u] = c / n_samples * 100

            # Occurrence: number of times each state appears per second
            occurrence = np.zeros(self.n_clusters)
            for i in range(self.n_clusters):
                occurrence[i] = len(durations[i]) / total_time

            # Mean duration
            mean_duration = [
                np.mean(durations[i]) if durations[i] else 0
                for i in range(self.n_clusters)
            ]
            std_duration = [
                np.std(durations[i]) if len(durations[i]) > 1 else 0
                for i in range(self.n_clusters)
            ]

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            x = np.arange(self.n_clusters)
            colors = plt.cm.Set1(np.linspace(0, 1, self.n_clusters))

            # Duration
            axes[0].bar(x, mean_duration, yerr=std_duration, color=colors, alpha=0.8, capsize=5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f"MS {i+1}" for i in range(self.n_clusters)])
            axes[0].set_ylabel("Duration (ms)", fontweight="bold")
            axes[0].set_title("Mean Duration", fontweight="bold")
            axes[0].grid(True, alpha=0.3, axis="y")

            # Coverage
            axes[1].bar(x, coverage, color=colors, alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"MS {i+1}" for i in range(self.n_clusters)])
            axes[1].set_ylabel("Coverage (%)", fontweight="bold")
            axes[1].set_title("Time Coverage", fontweight="bold")
            axes[1].grid(True, alpha=0.3, axis="y")

            # Occurrence
            axes[2].bar(x, occurrence, color=colors, alpha=0.8)
            axes[2].set_xticks(x)
            axes[2].set_xticklabels([f"MS {i+1}" for i in range(self.n_clusters)])
            axes[2].set_ylabel("Occurrence (per second)", fontweight="bold")
            axes[2].set_title("Occurrence Rate", fontweight="bold")
            axes[2].grid(True, alpha=0.3, axis="y")

            plt.suptitle(
                f"Microstate Statistics (K={self.n_clusters})",
                fontweight="bold",
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "baseline_microstate_statistics.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Save statistics to JSON
            stats = {
                "mean_duration_ms": mean_duration,
                "std_duration_ms": std_duration,
                "coverage_percent": coverage.tolist(),
                "occurrence_per_sec": occurrence.tolist(),
                "n_clusters": self.n_clusters,
                "total_samples": n_samples,
                "sfreq": sfreq,
            }
            with open(self.output_dir / "baseline_microstate_stats.json", "w") as f:
                json.dump(stats, f, indent=4)

            self.logger.info(f"Saved microstate statistics to: {self.output_dir / 'baseline_microstate_statistics.png'}")

        except Exception as e:
            self.logger.error(f"Statistics computation failed: {e}")
            import traceback
            traceback.print_exc()

    def _compute_baseline_composite_scores(self, silhouette: float, gev: float) -> dict:
        """
        Compute composite scores combining silhouette and GEV for baseline.

        Uses the same formulas as VAE evaluation for fair comparison:
        - Weighted Average: 0.5 * sil_norm + 0.5 * GEV
        - Geometric Mean: sqrt(sil_norm * GEV)

        Parameters
        ----------
        silhouette : float
            Silhouette score in [-1, 1]
        gev : float
            Global Explained Variance in [0, 1]

        Returns
        -------
        dict
            Dictionary with composite scores
        """
        # Normalize silhouette from [-1, 1] to [0, 1]
        sil_norm = (silhouette + 1) / 2

        # Handle edge cases
        if gev < 0:
            gev = 0.0
        if sil_norm < 0:
            sil_norm = 0.0

        # Weighted average (Option A)
        weighted_avg = 0.5 * sil_norm + 0.5 * gev

        # Geometric mean (Option B - Recommended)
        geometric = np.sqrt(sil_norm * gev)

        return {
            "silhouette_normalized": float(sil_norm),
            "gev": float(gev),
            "weighted_average": float(weighted_avg),
            "geometric_mean": float(geometric),
            "recommended": float(geometric),  # Same as geometric mean
        }

    def compute_cluster_metrics(self):
        """
        Compute cluster validation metrics using native pycrostates library.

        Uses ONLY the pycrostates library functions directly on the fitted ModKMeans:
            silhouette_score(ModK)
            calinski_harabasz_score(ModK)
            dunn_score(ModK)
            davies_bouldin_score(ModK)

        Returns
        -------
        dict
            Dictionary containing:
            - silhouette, calinski_harabasz, dunn, davies_bouldin
            - gev, composite_scores
        """
        if not self.fitted:
            self.logger.warning("Model not fitted. Call fit() first.")
            return {}

        self.logger.info("=" * 60)
        self.logger.info(f"COMPUTING CLUSTER VALIDATION METRICS (K={self.n_clusters})")
        self.logger.info("Using native pycrostates library functions")
        self.logger.info("=" * 60)

        scores = {}

        # Native pycrostates metrics - uses fitted ModKMeans directly
        try:
            scores["silhouette"] = float(pycrostates_silhouette(self.modk))
            scores["calinski_harabasz"] = float(pycrostates_ch(self.modk))
            scores["dunn"] = float(pycrostates_dunn(self.modk))
            scores["davies_bouldin"] = float(pycrostates_db(self.modk))
            self.logger.info(f"  Silhouette:        {scores['silhouette']:.4f}")
            self.logger.info(f"  Calinski-Harabasz: {scores['calinski_harabasz']:.2f}")
            self.logger.info(f"  Dunn:              {scores['dunn']:.6f}")
            self.logger.info(f"  Davies-Bouldin:    {scores['davies_bouldin']:.6f}")
        except Exception as e:
            self.logger.error(f"Pycrostates metrics failed: {e}")
            import traceback
            traceback.print_exc()
            scores["silhouette"] = None
            scores["calinski_harabasz"] = None
            scores["dunn"] = None
            scores["davies_bouldin"] = None

        # Get GEV from fitted model
        gev = float(self.modk.GEV_) if hasattr(self.modk, 'GEV_') else 0.0
        scores["gev"] = gev
        self.logger.info(f"  GEV:               {gev:.4f}")

        # Compute composite scores (silhouette + GEV)
        sil_for_composite = scores["silhouette"] if scores["silhouette"] is not None else 0.0
        composite_scores = self._compute_baseline_composite_scores(sil_for_composite, gev)
        scores["composite_scores"] = composite_scores

        self.logger.info(f"\n  COMPOSITE SCORES:")
        self.logger.info(f"    Silhouette (normalized): {composite_scores['silhouette_normalized']:.4f}")
        self.logger.info(f"    Weighted Average:        {composite_scores['weighted_average']:.4f}")
        self.logger.info(f"    Geometric Mean:          {composite_scores['geometric_mean']:.4f}")

        # Save scores to JSON - flat structure with pycrostates library metrics
        scores_json = {
            "n_clusters": self.n_clusters,
            "silhouette": scores["silhouette"],
            "calinski_harabasz": scores["calinski_harabasz"],
            "dunn": scores["dunn"],
            "davies_bouldin": scores["davies_bouldin"],
            "gev": gev,
            "composite_scores": composite_scores,
            "metric_source": "pycrostates library native functions",
        }
        with open(self.output_dir / "cluster_validation_metrics.json", "w") as f:
            json.dump(scores_json, f, indent=4)

        self.logger.info(f"Saved cluster validation metrics to: {self.output_dir / 'cluster_validation_metrics.json'}")

        return scores

    def compute_all_split_metrics(self) -> dict:
        """
        Compute metrics on ALL data splits (train, val, test) for comprehensive evaluation.

        This provides:
        - training_metrics: Metrics on train data (primary per lecturer requirement)
        - val_metrics: Metrics on validation data
        - test_metrics: Metrics on test data (unbiased generalization)

        Returns
        -------
        dict
            Dictionary with metrics for each split
        """
        if not self.fitted:
            self.logger.warning("Model not fitted. Call fit() first.")
            return {}

        self.logger.info("=" * 60)
        self.logger.info("COMPUTING METRICS ON ALL DATA SPLITS")
        self.logger.info("=" * 60)

        all_metrics = {
            "n_clusters": self.n_clusters,
            "training_metrics": {},
            "val_metrics": {},
            "test_metrics": {},
        }

        # --- Training Metrics (using pycrostates native functions on fitted model) ---
        self.logger.info("\n--- TRAINING METRICS (fitted data) ---")
        try:
            train_metrics = {
                "silhouette": float(pycrostates_silhouette(self.modk)),
                "calinski_harabasz": float(pycrostates_ch(self.modk)),
                "dunn": float(pycrostates_dunn(self.modk)),
                "davies_bouldin": float(pycrostates_db(self.modk)),
                "gev": float(self.modk.GEV_) if hasattr(self.modk, 'GEV_') else 0.0,
            }
            # Composite scores
            composite = self._compute_baseline_composite_scores(
                train_metrics["silhouette"], train_metrics["gev"]
            )
            train_metrics["composite_scores"] = composite

            all_metrics["training_metrics"] = train_metrics

            self.logger.info(f"  Silhouette:     {train_metrics['silhouette']:.4f}")
            self.logger.info(f"  GEV:            {train_metrics['gev']:.4f}")
            self.logger.info(f"  Composite (GM): {composite['geometric_mean']:.4f}")
        except Exception as e:
            self.logger.error(f"  Training metrics failed: {e}")
            all_metrics["training_metrics"] = {"error": str(e)}

        # --- Validation Metrics (if val data available) ---
        if hasattr(self, 'val_gfp_peaks') and self.val_gfp_peaks is not None:
            self.logger.info("\n--- VALIDATION METRICS ---")
            try:
                val_metrics = self._compute_metrics_on_data(self.val_gfp_peaks, "val")
                all_metrics["val_metrics"] = val_metrics
                val_sil = val_metrics.get('silhouette')
                val_gev = val_metrics.get('gev')
                self.logger.info(f"  Silhouette:     {val_sil:.4f}" if val_sil is not None else "  Silhouette:     N/A")
                self.logger.info(f"  GEV:            {val_gev:.4f}" if val_gev is not None else "  GEV:            N/A")
            except Exception as e:
                self.logger.error(f"  Validation metrics failed: {e}")
                all_metrics["val_metrics"] = {"error": str(e)}
        else:
            self.logger.info("\n--- VALIDATION METRICS: Not available (no val data) ---")

        # --- Test Metrics (if test data available) ---
        if hasattr(self, 'test_gfp_peaks') and self.test_gfp_peaks is not None:
            self.logger.info("\n--- TEST METRICS ---")
            try:
                test_metrics = self._compute_metrics_on_data(self.test_gfp_peaks, "test")
                all_metrics["test_metrics"] = test_metrics
                test_sil = test_metrics.get('silhouette')
                test_gev = test_metrics.get('gev')
                self.logger.info(f"  Silhouette:     {test_sil:.4f}" if test_sil is not None else "  Silhouette:     N/A")
                self.logger.info(f"  GEV:            {test_gev:.4f}" if test_gev is not None else "  GEV:            N/A")
            except Exception as e:
                self.logger.error(f"  Test metrics failed: {e}")
                all_metrics["test_metrics"] = {"error": str(e)}
        else:
            self.logger.info("\n--- TEST METRICS: Not available (no test data) ---")

        # Save comprehensive metrics to JSON
        metrics_file = self.output_dir / "all_split_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=4)
        self.logger.info(f"\nSaved all split metrics to: {metrics_file}")

        # Also update the main cluster_validation_metrics.json with training_metrics as primary
        main_metrics = {
            "n_clusters": self.n_clusters,
            **all_metrics["training_metrics"],
            "data_split": "training",
            "val_metrics": all_metrics.get("val_metrics", {}),
            "test_metrics": all_metrics.get("test_metrics", {}),
        }
        main_file = self.output_dir / "cluster_validation_metrics.json"
        with open(main_file, "w") as f:
            json.dump(main_metrics, f, indent=4)
        self.logger.info(f"Updated main metrics file: {main_file}")

        return all_metrics

    def _compute_metrics_on_data(self, gfp_peaks_data, split_name: str) -> dict:
        """
        Compute metrics on arbitrary GFP peaks data using fitted centroids.

        Parameters
        ----------
        gfp_peaks_data : ChData
            GFP peaks data to evaluate
        split_name : str
            Name of the split (for logging)

        Returns
        -------
        dict
            Dictionary with computed metrics
        """
        data = gfp_peaks_data.get_data()  # (n_channels, n_peaks)
        n_channels, n_peaks = data.shape
        centroids = self.modk.cluster_centers_  # (n_clusters, n_channels)

        # Assign labels using spatial correlation (polarity-invariant)
        labels, gev = self._assign_labels_gfp_peaks(data)

        # CRITICAL: Align polarities before computing metrics (matches pycrostates)
        # This prevents extreme DB/Dunn values from polarity-inverted samples
        centroid_for_each_sample = centroids[labels]  # (n_peaks, n_channels)
        dot_products = np.sum(data.T * centroid_for_each_sample, axis=1)
        signs = np.sign(dot_products)
        signs[signs == 0] = 1  # Handle edge case
        data_aligned = data * signs  # Flip data to match centroid polarity

        # Now use aligned data for metrics
        X = data_aligned.T  # (n_peaks, n_channels)

        # Check we have enough samples per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            return {
                "silhouette": -1.0,
                "gev": gev,
                "n_peaks": n_peaks,
                "n_clusters_used": len(unique_labels),
                "note": "Not enough clusters for silhouette computation"
            }

        # Compute using custom pycrostates-style metrics
        try:
            sil = custom_pycrostates_silhouette(X, labels)
            ch = custom_pycrostates_ch(X, labels, centroids)
            db = custom_pycrostates_db(X, labels, centroids)
            dunn = custom_pycrostates_dunn(X, labels)
        except Exception as e:
            self.logger.warning(f"Custom metrics failed for {split_name}: {e}")
            sil, ch, db, dunn = -1.0, 0.0, 999.0, 0.0

        metrics = {
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(db),
            "dunn": float(dunn),
            "gev": float(gev),
            "n_peaks": n_peaks,
        }

        # Composite scores
        composite = self._compute_baseline_composite_scores(sil, gev)
        metrics["composite_scores"] = composite

        return metrics

    def generate_electrode_space_visualization(self, save_prefix="all"):
        """
        Generate comprehensive t-SNE visualizations for electrode space analysis.

        This is the ModKMeans equivalent of VAE's latent space analysis.
        Since ModKMeans works directly in electrode space (61 channels for LEMON),
        we apply t-SNE to reduce dimensionality for visualization.

        Creates multiple visualizations:
        1. t-SNE colored by cluster assignments with density contours
        2. Inter-cluster distance heatmap
        3. Cluster separation metrics visualization
        4. Cluster center projections onto t-SNE space

        Parameters
        ----------
        save_prefix : str
            Prefix for saved files ("train", "val", "test", or "all")
        """
        from sklearn.manifold import TSNE
        from sklearn.impute import SimpleImputer
        from scipy.stats import gaussian_kde
        from scipy.spatial.distance import pdist, squareform
        import matplotlib.gridspec as gridspec

        self.logger.info("=" * 60)
        self.logger.info(f"GENERATING ELECTRODE SPACE VISUALIZATIONS [{save_prefix}]")
        self.logger.info("=" * 60)

        if not self.fitted:
            self.logger.error("Model not fitted. Call fit() first.")
            return

        # Get data based on prefix
        if save_prefix == "train" and hasattr(self, 'train_gfp_peaks') and self.train_gfp_peaks is not None:
            data_source = self.train_gfp_peaks
            self.logger.info(f"Using TRAIN data: {self.n_train_peaks} peaks")
        elif save_prefix == "val" and hasattr(self, 'val_gfp_peaks') and self.val_gfp_peaks is not None:
            data_source = self.val_gfp_peaks
            self.logger.info(f"Using VAL data: {self.n_val_peaks} peaks")
        elif save_prefix == "test" and hasattr(self, 'test_gfp_peaks') and self.test_gfp_peaks is not None:
            data_source = self.test_gfp_peaks
            self.logger.info(f"Using TEST data: {self.n_test_peaks} peaks")
        else:
            # Default to train data
            data_source = self.train_gfp_peaks if hasattr(self, 'train_gfp_peaks') else self.gfp_peaks
            save_prefix = "train"
            self.logger.info(f"Defaulting to TRAIN data")

        if data_source is None:
            self.logger.error("No data available for visualization")
            return

        # Get data and compute labels
        peak_data = data_source.get_data()  # (n_channels, n_peaks)
        X = peak_data.T  # (n_peaks, n_channels) for sklearn

        # Assign cluster labels
        labels, gev = self._assign_labels_gfp_peaks(peak_data)

        n_samples, n_channels = X.shape
        self.logger.info(f"Data shape: {n_samples} samples x {n_channels} channels")
        self.logger.info(f"GEV on this split: {gev:.4f}")

        # Handle NaN values
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)
            self.logger.info("Imputed NaN values in data")

        # Create output directory
        output_dir = self.output_dir / "electrode_space_analysis"
        output_dir.mkdir(exist_ok=True)

        # Subsample for t-SNE if too large (t-SNE is O(n^2))
        max_tsne_samples = 10000
        if n_samples > max_tsne_samples:
            self.logger.info(f"Subsampling {max_tsne_samples} of {n_samples} for t-SNE")
            # Stratified sampling
            indices = []
            for label in np.unique(labels):
                label_indices = np.where(labels == label)[0]
                n_label = len(label_indices)
                n_sample = int(max_tsne_samples * n_label / n_samples)
                n_sample = max(10, n_sample)  # At least 10 per cluster
                sampled = np.random.choice(label_indices, size=min(n_sample, n_label), replace=False)
                indices.extend(sampled)
            indices = np.array(indices)
            X_tsne = X[indices]
            labels_tsne = labels[indices]
        else:
            X_tsne = X
            labels_tsne = labels
            indices = np.arange(n_samples)

        # Apply t-SNE
        self.logger.info("Applying t-SNE dimensionality reduction...")
        perplexity = min(30, len(X_tsne) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        X_2d = tsne.fit_transform(X_tsne)

        # Get cluster centers
        cluster_centers = self.modk.cluster_centers_  # (n_clusters, n_channels)
        n_clusters = cluster_centers.shape[0]

        # =====================================================================
        # VISUALIZATION 1: t-SNE with Cluster Assignments and Density Contours
        # =====================================================================
        self.logger.info("Creating cluster assignment visualization with density...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        unique_clusters = np.unique(labels_tsne)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        # Left: Scatter plot with cluster colors
        for idx, cluster_id in enumerate(unique_clusters):
            mask = labels_tsne == cluster_id
            axes[0].scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[colors[idx]],
                label=f"Cluster {cluster_id} (n={np.sum(mask)})",
                alpha=0.6,
                s=20,
            )

        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title(f"Electrode Space: Cluster Assignments [{save_prefix}]", fontsize=14)
        axes[0].legend(loc="best", fontsize=9)

        # Right: Density contours per cluster
        for idx, cluster_id in enumerate(unique_clusters):
            mask = labels_tsne == cluster_id
            if np.sum(mask) > 10:
                try:
                    xy = X_2d[mask].T
                    kde = gaussian_kde(xy)

                    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)

                    axes[1].contour(xx, yy, density, levels=5, colors=[colors[idx]], alpha=0.7)
                    axes[1].contourf(xx, yy, density, levels=5, colors=[colors[idx]], alpha=0.2)
                except Exception:
                    pass

        # Add cluster centroids (mean of t-SNE coordinates per cluster)
        for idx, cluster_id in enumerate(unique_clusters):
            mask = labels_tsne == cluster_id
            centroid = X_2d[mask].mean(axis=0)
            axes[1].scatter(
                centroid[0], centroid[1], c=[colors[idx]], s=200,
                marker="*", edgecolors="black", linewidths=1.5,
                label=f"Center {cluster_id}"
            )

        axes[1].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[1].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[1].set_title(f"Electrode Space: Density Contours [{save_prefix}]", fontsize=14)
        axes[1].legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f"tsne_clusters_{save_prefix}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 2: Inter-Cluster Distance Heatmap
        # =====================================================================
        self.logger.info("Creating inter-cluster distance heatmap...")

        # Compute correlation-based distances between cluster centers
        from scipy.spatial.distance import cdist

        # Use correlation distance (1 - |correlation|) for polarity invariance
        corr_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                corr = np.corrcoef(cluster_centers[i], cluster_centers[j])[0, 1]
                corr_distances[i, j] = 1 - np.abs(corr)  # Polarity-invariant

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_distances, cmap='viridis', aspect='equal')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Distance (1 - |r|)', fontsize=12)

        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels([f"MS{i+1}" for i in range(n_clusters)])
        ax.set_yticklabels([f"MS{i+1}" for i in range(n_clusters)])

        # Add text annotations
        for i in range(n_clusters):
            for j in range(n_clusters):
                ax.text(j, i, f"{corr_distances[i, j]:.2f}",
                       ha="center", va="center", color="white" if corr_distances[i, j] > 0.5 else "black")

        ax.set_title(f"Inter-Cluster Correlation Distance [{save_prefix}]", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"cluster_distance_heatmap_{save_prefix}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 3: Cluster Size Distribution
        # =====================================================================
        self.logger.info("Creating cluster size distribution...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Bar plot of cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        colors_bar = plt.cm.tab10(np.linspace(0, 1, len(unique)))

        bars = axes[0].bar(unique, counts, color=colors_bar, alpha=0.8, edgecolor='black')
        axes[0].set_xlabel("Cluster", fontsize=12)
        axes[0].set_ylabel("Number of Samples", fontsize=12)
        axes[0].set_title(f"Cluster Size Distribution [{save_prefix}]", fontsize=14)
        axes[0].set_xticks(unique)
        axes[0].set_xticklabels([f"MS{i+1}" for i in unique])

        # Add percentage labels
        total = len(labels)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].annotate(f'{count/total*100:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

        # Right: Pie chart of cluster proportions
        axes[1].pie(counts, labels=[f"MS{i+1}" for i in unique],
                   colors=colors_bar, autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f"Cluster Proportions [{save_prefix}]", fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / f"cluster_sizes_{save_prefix}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 4: Comprehensive Summary Figure
        # =====================================================================
        self.logger.info("Creating comprehensive summary figure...")

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. t-SNE scatter
        ax1 = fig.add_subplot(gs[0, 0])
        for idx, cluster_id in enumerate(unique_clusters):
            mask = labels_tsne == cluster_id
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[idx]],
                       label=f"MS{cluster_id+1}", alpha=0.6, s=15)
        ax1.set_title("t-SNE: Cluster Assignments", fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

        # 2. Density contours
        ax2 = fig.add_subplot(gs[0, 1])
        for idx, cluster_id in enumerate(unique_clusters):
            mask = labels_tsne == cluster_id
            if np.sum(mask) > 10:
                try:
                    xy = X_2d[mask].T
                    kde = gaussian_kde(xy)
                    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                    xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)
                    ax2.contourf(xx, yy, density, levels=5, colors=[colors[idx]], alpha=0.3)
                except Exception:
                    pass
        ax2.set_title("t-SNE: Density Contours", fontsize=12, fontweight='bold')
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")

        # 3. Cluster distance heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(corr_distances, cmap='viridis', aspect='equal')
        ax3.set_xticks(range(n_clusters))
        ax3.set_yticks(range(n_clusters))
        ax3.set_xticklabels([f"MS{i+1}" for i in range(n_clusters)])
        ax3.set_yticklabels([f"MS{i+1}" for i in range(n_clusters)])
        ax3.set_title("Inter-Cluster Distance", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax3, shrink=0.8)

        # 4. Cluster sizes bar
        ax4 = fig.add_subplot(gs[1, 0])
        bars = ax4.bar(unique, counts, color=colors_bar, alpha=0.8)
        ax4.set_xlabel("Cluster")
        ax4.set_ylabel("Samples")
        ax4.set_title("Cluster Sizes", fontsize=12, fontweight='bold')
        ax4.set_xticks(unique)
        ax4.set_xticklabels([f"MS{i+1}" for i in unique])

        # 5. Metrics summary text
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')

        # Compute metrics for this split
        try:
            sil = custom_pycrostates_silhouette(X, labels)
            ch = custom_pycrostates_ch(X, labels, cluster_centers)
            db = custom_pycrostates_db(X, labels, cluster_centers)
        except Exception:
            sil, ch, db = -1, 0, 999

        metrics_text = f"""
        ELECTRODE SPACE ANALYSIS SUMMARY
        ================================
        Split: {save_prefix.upper()}
        Samples: {n_samples:,}
        Channels: {n_channels}
        Clusters: {n_clusters}

        METRICS:
        --------
        GEV: {gev:.4f}
        Silhouette: {sil:.4f}
        Calinski-Harabasz: {ch:.2f}
        Davies-Bouldin: {db:.4f}

        CLUSTER SIZES:
        --------------
        """ + "\n        ".join([f"MS{i+1}: {c:,} ({c/total*100:.1f}%)" for i, c in zip(unique, counts)])

        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        # 6. Pie chart
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.pie(counts, labels=[f"MS{i+1}" for i in unique],
               colors=colors_bar, autopct='%1.1f%%', startangle=90)
        ax6.set_title("Cluster Proportions", fontsize=12, fontweight='bold')

        fig.suptitle(f"ModKMeans Electrode Space Analysis [{save_prefix.upper()}]",
                    fontsize=16, fontweight='bold', y=1.02)

        plt.savefig(output_dir / f"comprehensive_summary_{save_prefix}.png",
                   dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved electrode space visualizations to: {output_dir}")
        self.logger.info(f"  - tsne_clusters_{save_prefix}.png")
        self.logger.info(f"  - cluster_distance_heatmap_{save_prefix}.png")
        self.logger.info(f"  - cluster_sizes_{save_prefix}.png")
        self.logger.info(f"  - comprehensive_summary_{save_prefix}.png")

        return {
            "output_dir": str(output_dir),
            "n_samples": n_samples,
            "n_channels": n_channels,
            "gev": float(gev),
            "split": save_prefix,
        }
