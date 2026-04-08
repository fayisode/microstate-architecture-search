"""Trainer Metrics Mixin — metric computation methods for VAEClusteringTrainer."""
import gc
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
from tqdm import tqdm
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import metrics_utils as _mu
import centroid_metrics as _cm


def clear_gpu(logger=None):
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        if logger:
            logger.warning(f"Error clearing GPU memory: {e}")


class TrainerMetricsMixin:
    """Mixin providing metric computation methods for VAEClusteringTrainer."""

    def _compute_metrics_and_losses(
        self,
        loader: DataLoader,
        epoch: int,
        is_test: bool = False,
        strategy_name: str = "",
        retry: int = 0,
    ) -> Tuple[float, Dict]:
        self.model.set_eval_mode()
        total_loss, recon_loss_sum, kld_loss_sum, valid_samples = 0.0, 0.0, 0.0, 0

        # Accumulators for full-dataset metric computation (accumulate-then-compute approach)
        all_decoded = []    # Flattened decoded topomaps (N x 1600)
        all_input = []      # Flattened input topomaps (N x 1600) for spatial_corr
        all_latents = []    # Latent representations (N x latent_dim)
        all_clusters = []   # Cluster assignments (N,)
        all_ssim = []

        # Default 4-quadrant metrics
        default_q1 = {"q1_latent_eucl_silhouette": -1, "q1_latent_eucl_db": 10,
                       "q1_latent_eucl_ch": 0, "q1_latent_eucl_dunn": 0}
        default_q2 = {"q2_latent_corr_silhouette": -1, "q2_latent_corr_db": 10,
                       "q2_latent_corr_ch": 0, "q2_latent_corr_dunn": 0}
        default_q3 = {"q3_topo_eucl_silhouette": -1, "q3_topo_eucl_db": 10,
                       "q3_topo_eucl_ch": 0, "q3_topo_eucl_dunn": 0}
        default_q4 = {"q4_topo_corr_silhouette": -1, "q4_topo_corr_db": 10,
                       "q4_topo_corr_ch": 0, "q4_topo_corr_dunn": 0}

        desc = "Test Set" if is_test else "Metrics"
        progress_bar = tqdm(loader, desc=desc, leave=False)

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(progress_bar):
                data = data.to(self.device)
                try:
                    recon, mu, logvar = self.model(data)
                    re_loss, kl_loss, loss = self.model.loss_function(
                        recon, data, mu, logvar, epoch, self.epochs,
                        is_pretraining=True,  # skip clustering losses — only need recon/KLD
                    )

                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        self.logger.warning(f"Batch {batch_idx}: NaN/Inf loss, skipping")
                        continue

                    latents = mu.detach().cpu().numpy()

                    try:
                        clusters = self.model.predict(data)
                        if clusters is None or len(clusters) != data.size(0):
                            continue
                    except Exception:
                        continue

                    unique_clusters = len(np.unique(clusters))
                    if unique_clusters >= 2:
                        decoded_flat = recon.view(recon.size(0), -1).detach().cpu().numpy()
                        input_flat = data.view(data.size(0), -1).cpu().numpy()
                        all_decoded.append(decoded_flat)
                        all_input.append(input_flat)
                        all_latents.append(latents)
                        all_clusters.append(clusters)
                    else:
                        continue

                    total_loss += loss.item() * data.size(0)
                    recon_loss_sum += re_loss.item() * data.size(0)
                    kld_loss_sum += kl_loss.item() * data.size(0)
                    valid_samples += data.size(0)

                    recon_np, data_np = recon.detach().cpu().numpy(), data.cpu().numpy()
                    for j in range(data.size(0)):
                        all_ssim.append(
                            ssim(data_np[j, 0], recon_np[j, 0], data_range=self.ssim_data_range)
                        )

                except Exception as e:
                    self.logger.warning(f"Batch {batch_idx} failed: {e}")
                    continue

        if valid_samples == 0:
            self.logger.error("No valid samples processed!")
            if retry < 2:
                clear_gpu(self.logger)
                return self._compute_metrics_and_losses(
                    loader, epoch, is_test, strategy_name, retry + 1
                )
            return 50.0, {
                "recon_loss": 50, "kld_loss": 0, "beta": 1.0,
                "ssim_score": 0, "spatial_corr": 0,
                "clustering_failed": True,
                **default_q1, **default_q2, **default_q3, **default_q4,
                "silhouette_scores": -1, "db_scores": 10, "ch_scores": 0,
            }

        # === Compute 4-quadrant clustering metrics on full accumulated data ===
        try:
            if len(all_decoded) > 0:
                full_decoded = np.concatenate(all_decoded, axis=0)
                full_input = np.concatenate(all_input, axis=0)
                full_latents = np.concatenate(all_latents, axis=0)
                full_clusters = np.concatenate(all_clusters, axis=0)
                n_samples = len(full_clusters)

                # Cache forward pass for reuse by viz and centroid metrics
                if not hasattr(self, '_cached_forward_passes'):
                    self._cached_forward_passes = {}
                self._cached_forward_passes[id(loader)] = {
                    "latents": full_latents,
                    "decoded": full_decoded,
                    "input": full_input,
                    "clusters": full_clusters,
                }

                unique_clusters = len(np.unique(full_clusters))
                if unique_clusters < 2:
                    raise ValueError("Need at least 2 clusters for metrics")

                # Q1: Latent + Euclidean
                q1 = {
                    "q1_latent_eucl_silhouette": silhouette_score(full_latents, full_clusters),
                    "q1_latent_eucl_db": davies_bouldin_score(full_latents, full_clusters),
                    "q1_latent_eucl_ch": calinski_harabasz_score(full_latents, full_clusters),
                    "q1_latent_eucl_dunn": _mu.dunn_score_euclidean(full_latents, full_clusters),
                }

                # Q2: Latent + Correlation (|1/corr|-1) — single corrcoef for sil+dunn
                try:
                    q2_distances, q2_labels = _mu.correlation_distance_matrix_with_mask(full_latents, full_clusters)
                    q2 = {
                        "q2_latent_corr_silhouette": _mu.pycrostates_silhouette_score_precomputed(q2_distances, q2_labels),
                        "q2_latent_corr_db": _mu.pycrostates_davies_bouldin_score(full_latents, full_clusters),
                        "q2_latent_corr_ch": _mu.pycrostates_calinski_harabasz_score(full_latents, full_clusters),
                        "q2_latent_corr_dunn": _mu.pycrostates_dunn_score_precomputed(q2_distances, q2_labels),
                    }
                    del q2_distances
                except Exception as e:
                    self.logger.debug(f"Q2 latent correlation metrics failed: {e}")
                    q2 = default_q2.copy()

                # Q3: Topomap + Euclidean
                q3 = {
                    "q3_topo_eucl_silhouette": silhouette_score(full_decoded, full_clusters),
                    "q3_topo_eucl_db": davies_bouldin_score(full_decoded, full_clusters),
                    "q3_topo_eucl_ch": calinski_harabasz_score(full_decoded, full_clusters),
                    "q3_topo_eucl_dunn": _mu.dunn_score_euclidean(full_decoded, full_clusters),
                }

                # Q4: Topomap + Correlation (|1/corr|-1) — single corrcoef for sil+dunn
                try:
                    q4_distances, q4_labels = _mu.correlation_distance_matrix_with_mask(full_decoded, full_clusters)
                    q4 = {
                        "q4_topo_corr_silhouette": _mu.pycrostates_silhouette_score_precomputed(q4_distances, q4_labels),
                        "q4_topo_corr_db": _mu.pycrostates_davies_bouldin_score(full_decoded, full_clusters),
                        "q4_topo_corr_ch": _mu.pycrostates_calinski_harabasz_score(full_decoded, full_clusters),
                        "q4_topo_corr_dunn": _mu.pycrostates_dunn_score_precomputed(q4_distances, q4_labels),
                    }
                    del q4_distances
                except Exception as e:
                    self.logger.debug(f"Q4 topo correlation metrics failed: {e}")
                    q4 = default_q4.copy()

                # Spatial correlation: vectorized mean |Pearson r| between input and reconstruction
                inp = full_input - full_input.mean(axis=1, keepdims=True)
                dec = full_decoded - full_decoded.mean(axis=1, keepdims=True)
                num = np.sum(inp * dec, axis=1)
                den = np.sqrt(np.sum(inp ** 2, axis=1) * np.sum(dec ** 2, axis=1))
                corrs = np.where(den > 1e-10, np.abs(num / den), 0.0)
                spatial_corr = float(np.mean(corrs))

                clustering_metrics = {
                    **q1, **q2, **q3, **q4,
                    "spatial_corr": spatial_corr,
                    # Legacy aliases for backward compat (Q3 = topomap euclidean)
                    "silhouette_scores": q3["q3_topo_eucl_silhouette"],
                    "db_scores": q3["q3_topo_eucl_db"],
                    "ch_scores": q3["q3_topo_eucl_ch"],
                    "clustering_failed": False,
                }

                # Log 4-quadrant table
                self.logger.info(f"Clustering metrics ({n_samples} samples):")
                self.logger.info("                  | Sklearn (Euclidean)       | Pycrostates (|1/r|-1)")
                self.logger.info(f"  Latent  {full_latents.shape[1]:>3}D    "
                                 f"| Sil={q1['q1_latent_eucl_silhouette']:+.3f} Dunn={q1['q1_latent_eucl_dunn']:.3f} "
                                 f"| Sil={q2['q2_latent_corr_silhouette']:+.3f} Dunn={q2['q2_latent_corr_dunn']:.3f}")
                self.logger.info(f"  Topo  {full_decoded.shape[1]:>5}D    "
                                 f"| Sil={q3['q3_topo_eucl_silhouette']:+.3f} Dunn={q3['q3_topo_eucl_dunn']:.3f} "
                                 f"| Sil={q4['q4_topo_corr_silhouette']:+.3f} Dunn={q4['q4_topo_corr_dunn']:.3f}")
                self.logger.info(f"  Spatial Corr: {spatial_corr:.4f}")

            else:
                self.logger.warning("No valid clustering data accumulated!")
                clustering_metrics = {
                    **default_q1, **default_q2, **default_q3, **default_q4,
                    "spatial_corr": 0,
                    "silhouette_scores": -1, "db_scores": 10, "ch_scores": 0,
                    "clustering_failed": True,
                }

        except Exception as aggregation_error:
            self.logger.error(f"Metric computation failed: {aggregation_error}")
            clustering_metrics = {
                **default_q1, **default_q2, **default_q3, **default_q4,
                "spatial_corr": 0,
                "silhouette_scores": -1, "db_scores": 10, "ch_scores": 0,
                "clustering_failed": True,
            }

        avg_loss = total_loss / valid_samples
        avg_recon = recon_loss_sum / valid_samples
        avg_kld = kld_loss_sum / valid_samples
        beta = self._safe_beta_calculation(epoch, self.epochs, avg_recon, avg_kld)

        base_metrics = {
            "total_loss": avg_loss,
            "recon_loss": avg_recon,
            "kld_loss": avg_kld,
            "beta": beta,
            "ssim_score": np.mean(all_ssim) if all_ssim else 0,
        }

        return avg_loss, {**base_metrics, **clustering_metrics}

    def _compute_training_metrics(self, epoch, retry=0):
        self.model.set_eval_mode()

        # Use a subset of training data to avoid memory issues
        max_batches = min(10, len(self.train_loader))
        total_loss, recon_loss_sum, kld_loss_sum, valid_samples = 0.0, 0.0, 0.0, 0

        all_decoded = []
        all_latents = []
        all_clusters = []
        all_ssim = []
        all_per_dim_kl = []

        default_q1 = {"q1_latent_eucl_silhouette": -1, "q1_latent_eucl_db": 10,
                       "q1_latent_eucl_ch": 0, "q1_latent_eucl_dunn": 0}
        default_q3 = {"q3_topo_eucl_silhouette": -1, "q3_topo_eucl_db": 10,
                       "q3_topo_eucl_ch": 0, "q3_topo_eucl_dunn": 0}

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                if batch_idx >= max_batches:
                    break
                data = data.to(self.device)
                try:
                    recon, mu, logvar = self.model(data)
                    re_loss, kl_loss, loss = self.model.loss_function(
                        recon, data, mu, logvar, epoch, self.epochs,
                        is_pretraining=True,  # skip clustering losses — only need recon/KLD
                    )
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        continue

                    # Per-dim KL: standard diagonal-Gaussian KL for collapse detection
                    per_dim_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                    all_per_dim_kl.append(per_dim_kl.detach().cpu().numpy())

                    latents = mu.detach().cpu().numpy()
                    try:
                        clusters = self.model.predict(data)
                        if clusters is None or len(clusters) != data.size(0):
                            continue
                    except Exception:
                        continue

                    unique_clusters = len(np.unique(clusters))
                    if unique_clusters >= 2:
                        decoded_flat = recon.view(recon.size(0), -1).detach().cpu().numpy()
                        all_decoded.append(decoded_flat)
                        all_latents.append(latents)
                        all_clusters.append(clusters)
                    else:
                        continue

                    total_loss += loss.item() * data.size(0)
                    recon_loss_sum += re_loss.item() * data.size(0)
                    kld_loss_sum += kl_loss.item() * data.size(0)
                    valid_samples += data.size(0)

                    recon_np, data_np = recon.detach().cpu().numpy(), data.cpu().numpy()
                    for j in range(data.size(0)):
                        all_ssim.append(
                            ssim(data_np[j, 0], recon_np[j, 0], data_range=self.ssim_data_range)
                        )
                except Exception:
                    continue

        self.model.train()

        if valid_samples == 0:
            self.logger.warning("No valid samples for training metrics!")
            if retry < 2:
                clear_gpu(self.logger)
                return self._compute_training_metrics(epoch, retry + 1)
            return {
                "total_loss": 0, "recon_loss": 50, "kld_loss": 0,
                "beta": 1.0, "ssim_score": 0,
                "clustering_failed": True,
                **default_q1, **default_q3,
                "silhouette_scores": -1, "db_scores": 10, "ch_scores": 0,
            }

        try:
            if len(all_decoded) > 0:
                full_decoded = np.concatenate(all_decoded, axis=0)
                full_latents = np.concatenate(all_latents, axis=0)
                full_clusters = np.concatenate(all_clusters, axis=0)
                n_samples = len(full_clusters)

                unique_clusters = len(np.unique(full_clusters))
                if unique_clusters < 2:
                    raise ValueError("Need at least 2 clusters")

                # Q1: Latent + Euclidean (fast, for epoch summary)
                q1 = {
                    "q1_latent_eucl_silhouette": silhouette_score(full_latents, full_clusters),
                    "q1_latent_eucl_db": davies_bouldin_score(full_latents, full_clusters),
                    "q1_latent_eucl_ch": calinski_harabasz_score(full_latents, full_clusters),
                    "q1_latent_eucl_dunn": _mu.dunn_score_euclidean(full_latents, full_clusters),
                }

                # Q3: Topomap + Euclidean (fast, for epoch summary)
                q3 = {
                    "q3_topo_eucl_silhouette": silhouette_score(full_decoded, full_clusters),
                    "q3_topo_eucl_db": davies_bouldin_score(full_decoded, full_clusters),
                    "q3_topo_eucl_ch": calinski_harabasz_score(full_decoded, full_clusters),
                    "q3_topo_eucl_dunn": _mu.dunn_score_euclidean(full_decoded, full_clusters),
                }

                clustering_metrics = {
                    **q1, **q3,
                    "silhouette_scores": q3["q3_topo_eucl_silhouette"],
                    "db_scores": q3["q3_topo_eucl_db"],
                    "ch_scores": q3["q3_topo_eucl_ch"],
                    "clustering_failed": False,
                }
                self.logger.debug(f"Training metrics ({n_samples} samples)")
            else:
                clustering_metrics = {
                    **default_q1, **default_q3,
                    "silhouette_scores": -1, "db_scores": 10, "ch_scores": 0,
                    "clustering_failed": True,
                }
        except Exception as e:
            self.logger.error(f"Training metric computation failed: {e}")
            clustering_metrics = {
                **default_q1, **default_q3,
                "silhouette_scores": -1, "db_scores": 10, "ch_scores": 0,
                "clustering_failed": True,
            }

        avg_loss = total_loss / valid_samples
        avg_recon = recon_loss_sum / valid_samples
        avg_kld = kld_loss_sum / valid_samples
        beta = self._safe_beta_calculation(epoch, self.epochs, avg_recon, avg_kld)

        # Per-dimension KL collapse analysis
        kl_collapse_metrics = {}
        if all_per_dim_kl:
            full_per_dim_kl = np.concatenate(all_per_dim_kl, axis=0)  # (N, D)
            mean_per_dim = full_per_dim_kl.mean(axis=0)  # (D,)
            latent_dim = mean_per_dim.shape[0]
            n_collapsed = int(np.sum(mean_per_dim < 0.05))
            n_active = latent_dim - n_collapsed
            kl_collapse_metrics = {
                "kl_per_dim_mean": float(mean_per_dim.mean()),
                "kl_active_dims": n_active,
                "kl_collapsed_dims": n_collapsed,
                "kl_collapse_ratio": float(n_collapsed / latent_dim),
                "kl_per_dim_values": mean_per_dim.tolist(),
            }

        return {
            "total_loss": avg_loss, "recon_loss": avg_recon,
            "kld_loss": avg_kld, "beta": beta,
            "ssim_score": np.mean(all_ssim) if all_ssim else 0,
            **clustering_metrics,
            **kl_collapse_metrics,
        }

    def _get_decoded_centroids_flat(self):
        """Return decoded centroids, cached per model state (invalidated on model update)."""
        mu_c_hash = self.model.mu_c.data_ptr()
        cached = getattr(self, '_decoded_centroids_cache', None)
        if cached is not None and cached[0] == mu_c_hash:
            return cached[1]
        with torch.no_grad():
            decoded = self.model.decode(self.model.mu_c).detach().cpu().numpy()
            decoded = decoded.squeeze(1)
            flat = decoded.reshape(decoded.shape[0], -1)
        self._decoded_centroids_cache = (mu_c_hash, flat)
        return flat

    def _compute_gev(self, loader: DataLoader) -> float:
        """
        Compute Global Explained Variance (GEV) on any data loader.
        GEV is polarity-invariant and measures how well cluster centroids explain data variance.
        Higher is better (range 0-1).

        GEV is computed on INPUT data (not reconstructions) to ensure stable metrics
        from epoch 1. This measures how well the learned centroids explain the actual data.

        Parameters
        ----------
        loader : DataLoader
            Any data loader (train or evaluation).
        """
        self.model.set_eval_mode()
        EPS = 1e-10

        all_inputs = []
        all_clusters = []

        with torch.no_grad():
            decoded_centroids_flat = self._get_decoded_centroids_flat()

            for data, _ in loader:
                data = data.to(self.device)
                clusters = self.model.predict(data)
                input_flat = data.view(data.size(0), -1).cpu().numpy()
                all_inputs.append(input_flat)
                all_clusters.append(clusters)

        X = np.concatenate(all_inputs, axis=0)
        labels = np.concatenate(all_clusters, axis=0)

        gfp = np.std(X, axis=1)
        gfp_squared_sum = np.sum(gfp ** 2)

        if gfp_squared_sum < EPS:
            self.model.set_train_mode()
            return 0.0

        assigned_maps = decoded_centroids_flat[labels]

        X_centered = X - X.mean(axis=1, keepdims=True)
        maps_centered = assigned_maps - assigned_maps.mean(axis=1, keepdims=True)

        numerator = np.sum(X_centered * maps_centered, axis=1)
        denominator = np.sqrt(
            np.sum(X_centered ** 2, axis=1) *
            np.sum(maps_centered ** 2, axis=1)
        )
        correlations = np.abs(numerator / (denominator + EPS))

        gev = np.sum((gfp * correlations) ** 2) / (gfp_squared_sum + EPS)

        self.model.set_train_mode()
        return float(gev)

    def _compute_training_gev(self):
        """Compute GEV on training set (backward compat wrapper)."""
        return self._compute_gev(self.train_loader)

    def _compute_eval_recon_loss(self):
        """Compute reconstruction loss on eval set (lightweight, no clustering metrics)."""
        if self.eval_loader is None:
            return None
        self.model.set_eval_mode()
        total_loss, total_samples = 0.0, 0
        with torch.no_grad():
            for data, _ in self.eval_loader:
                data = data.to(self.device)
                recon, mu, logvar = self.model(data)
                re_loss = self.model.RE(recon, data, normalize=False)
                total_loss += re_loss.item() * data.size(0)
                total_samples += data.size(0)
        self.model.set_train_mode()
        return total_loss / total_samples if total_samples > 0 else None

    def _compute_electrode_gev(self) -> Optional[float]:
        """
        Compute GEV in electrode space for final comparison with baseline ModKMeans.

        Maps 40x40 decoded centroids back to 61 electrode channels, then computes
        GEV against ALL GFP peaks. Too expensive for per-epoch; used only at end.
        """
        electrode_values = self._extract_vae_electrode_values()
        if electrode_values is None or self.gfp_peaks is None:
            return None

        try:
            # Get GFP peak data in electrode space
            peak_data = self.gfp_peaks.get_data().T  # (n_peaks, n_channels)
            n_peaks = peak_data.shape[0]

            # Assign each peak to nearest centroid (polarity-invariant)
            EPS = 1e-10
            peak_centered = peak_data - peak_data.mean(axis=1, keepdims=True)
            centroid_centered = electrode_values - electrode_values.mean(axis=1, keepdims=True)

            # Correlation of each peak with each centroid
            labels = np.zeros(n_peaks, dtype=int)
            for i in range(n_peaks):
                best_corr = -1
                for k in range(electrode_values.shape[0]):
                    num = np.sum(peak_centered[i] * centroid_centered[k])
                    den = np.sqrt(np.sum(peak_centered[i]**2) * np.sum(centroid_centered[k]**2)) + EPS
                    corr = abs(num / den)
                    if corr > best_corr:
                        best_corr = corr
                        labels[i] = k

            # Compute GEV
            gfp = np.std(peak_data, axis=1)
            gfp_sq_sum = np.sum(gfp**2)
            if gfp_sq_sum < EPS:
                return 0.0

            assigned = centroid_centered[labels]
            num = np.sum(peak_centered * assigned, axis=1)
            den = np.sqrt(np.sum(peak_centered**2, axis=1) * np.sum(assigned**2, axis=1)) + EPS
            correlations = np.abs(num / den)

            gev = np.sum((gfp * correlations)**2) / (gfp_sq_sum + EPS)
            self.logger.info(f"Electrode-space GEV: {gev:.4f}")
            return float(gev)

        except Exception as e:
            self.logger.warning(f"Electrode-space GEV failed: {e}")
            return None

    def _extract_vae_electrode_values(self) -> Optional[np.ndarray]:
        """
        Extract electrode values from VAE decoded centroids for comparison with ModKMeans.

        Maps the 40x40 topomap centroids to electrode positions to get
        (n_clusters, n_channels) array comparable to ModKMeans centroids.

        Returns
        -------
        electrode_values : np.ndarray or None
            Shape (n_clusters, n_channels) with electrode values for each centroid.
            Returns None if extraction fails.
        """
        if self.raw_mne is None or not hasattr(self.raw_mne, 'info'):
            self.logger.warning("Cannot extract electrode values: raw_mne not available")
            return None

        try:
            import mne

            self.model.set_eval_mode()

            # Decode centroids to 40x40 topomaps
            with torch.no_grad():
                decoded = self.model.decode(self.model.mu_c)  # (n_clusters, 1, H, W)
                decoded_centroids = decoded.squeeze(1).detach().cpu().numpy()  # (n_clusters, H, W)

            # Get electrode positions from MNE info
            info = self.raw_mne.info
            pos = np.array([info['chs'][i]['loc'][:2] for i in range(len(info.ch_names))])

            # Normalize positions to [0, 1]
            pos_min = pos.min(axis=0)
            pos_max = pos.max(axis=0)
            pos_range = pos_max - pos_min
            pos_range[pos_range < 1e-10] = 1e-10
            pos_normalized = (pos - pos_min) / pos_range

            # Map to image coordinates (40x40)
            img_size = decoded_centroids.shape[1]
            margin = 0.1
            pos_img = pos_normalized * (1 - 2 * margin) + margin
            pos_img = (pos_img * (img_size - 1)).astype(int)
            pos_img = np.clip(pos_img, 0, img_size - 1)

            # Extract electrode values from each centroid
            n_clusters = self.model.nClusters
            n_channels = len(info.ch_names)
            electrode_values = np.zeros((n_clusters, n_channels))

            for k in range(n_clusters):
                centroid_img = decoded_centroids[k]
                for ch_idx in range(n_channels):
                    x, y = pos_img[ch_idx]
                    electrode_values[k, ch_idx] = centroid_img[y, x]

            # Denormalize from z-score space to original scale (Volts)
            if hasattr(self, 'norm_params') and self.norm_params is not None:
                z_mean = self.norm_params.get("mean", 0.0)
                z_std = self.norm_params.get("std", 1.0)
                electrode_values = electrode_values * z_std + z_mean

            # Center each centroid (zero mean)
            for k in range(n_clusters):
                electrode_values[k] = electrode_values[k] - electrode_values[k].mean()

            self.logger.info(f"Extracted VAE electrode values: {electrode_values.shape}")
            self.model.set_train_mode()
            return electrode_values

        except Exception as e:
            self.logger.error(f"Failed to extract VAE electrode values: {e}")
            return None

    def _compute_centroid_based_metrics(
        self,
        data_loader: DataLoader,
        split: str = "val",
        save_to_file: bool = False
    ) -> Dict:
        """
        Compute centroid-based clustering metrics on the full dataset.

        Uses centroids as reference points instead of pairwise distances.
        Computes both sklearn (Euclidean) and pycrostates (correlation) metrics.

        Args:
            data_loader: DataLoader to evaluate
            split: "train", "val", or "test"
            save_to_file: Whether to save results to JSON

        Returns:
            Dictionary with centroid-based metrics
        """
        self.model.set_eval_mode()

        # Reuse cached forward pass if available (from _compute_metrics_and_losses)
        cached = getattr(self, '_cached_forward_passes', {}).get(id(data_loader))
        if cached is not None:
            self.logger.info(f"Reusing cached forward pass ({len(cached['latents'])} samples)")
            latent_features = cached["latents"]
            labels = cached["clusters"]
            raw_features = cached["input"]
        else:
            all_latent = []
            all_labels = []
            all_raw = []

            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    mu, _ = self.model.encode(data)
                    all_latent.append(mu.detach().cpu().numpy())
                    predictions = self.model.predict(data)
                    all_labels.append(predictions)
                    all_raw.append(data.view(data.size(0), -1).cpu().numpy())

            latent_features = np.concatenate(all_latent, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            raw_features = np.concatenate(all_raw, axis=0)

        # Get centroids
        latent_centroids = self.model.mu_c.detach().cpu().numpy()

        # Decode centroids to get raw space centroids
        with torch.no_grad():
            decoded_centroids = self.model.decode(self.model.mu_c)
            raw_centroids = decoded_centroids.view(decoded_centroids.size(0), -1).detach().cpu().numpy()

        # Compute GEV for composite score (for all data)
        gev = self._compute_training_gev()

        # Compute metrics in LATENT space (primary - what the lecturer suggested)
        latent_metrics = self.centroid_metrics.compute_all(
            features=latent_features,
            labels=labels,
            centroids=latent_centroids,
            feature_space="latent",
            split=split,
            gev=gev  # Include GEV for composite scores
        )

        # Compute metrics in RAW space (secondary - for comparison)
        raw_metrics = self.centroid_metrics.compute_all(
            features=raw_features,
            labels=labels,
            centroids=raw_centroids,
            feature_space="raw",
            split=split,
            gev=gev  # Include GEV for composite scores
        )

        results = {
            "latent_space": latent_metrics,
            "raw_space": raw_metrics,
            "epoch": self.current_epoch,
            "split": split,
            "gev": gev,
        }

        # Save to file if requested
        if save_to_file:
            save_path = self.output_dir / f"centroid_metrics_{split}.json"
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Centroid metrics saved to: {save_path}")

        self.model.set_train_mode()
        return results

    def _compute_composite_score(self, silhouette, gev):
        """Compute geometric mean composite score for early stopping.

        Uses geometric mean of normalized silhouette and GEV, ensuring both
        metrics must be good for a high score. This is independent of beta
        fluctuations from cyclical annealing.

        Args:
            silhouette: Silhouette score in range [-1, 1]
            gev: Global explained variance in range [0, 1]

        Returns:
            Composite score in range [0, 1], higher is better
        """
        import numpy as np
        # Normalize silhouette from [-1, 1] to [0, 1]
        sil_norm = (silhouette + 1) / 2
        # Clamp to valid range
        sil_norm = max(0.0, min(1.0, sil_norm))
        gev_clamped = max(0.0, min(1.0, gev if gev is not None else 0.0))
        # Geometric mean ensures both metrics must be good
        return float(np.sqrt(sil_norm * gev_clamped))

    def _compute_pairwise_latent_metrics(
        self,
        latent_features: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute pycrostates-style pairwise metrics on latent space.

        These metrics use the true pycrostates distance formula (|1/corr| - 1)
        and compute pairwise distances between all samples, providing a
        complementary view to centroid-based metrics.

        Parameters
        ----------
        latent_features : np.ndarray
            Latent representations of shape (n_samples, latent_dim)
        labels : np.ndarray
            Cluster labels of shape (n_samples,)
        centroids : np.ndarray, optional
            Cluster centers for polarity alignment (if needed)

        Returns
        -------
        Dict with silhouette, dunn, davies_bouldin, calinski_harabasz scores
        """
        metrics = {}

        # Silhouette using pycrostates correlation distance
        try:
            metrics["silhouette"] = float(
                _mu.pycrostates_silhouette_score(latent_features, labels)
            )
        except Exception as e:
            self.logger.warning(f"Pairwise silhouette failed: {e}")
            metrics["silhouette"] = -1.0

        # Dunn index using correlation distance
        try:
            metrics["dunn"] = float(
                _mu.pycrostates_dunn_score(latent_features, labels)
            )
        except Exception as e:
            self.logger.warning(f"Pairwise Dunn failed: {e}")
            metrics["dunn"] = -1.0

        # Davies-Bouldin using correlation distance
        try:
            metrics["davies_bouldin"] = float(
                _mu.pycrostates_davies_bouldin_score(latent_features, labels, centroids)
            )
        except Exception as e:
            self.logger.warning(f"Pairwise Davies-Bouldin failed: {e}")
            metrics["davies_bouldin"] = 999.0

        # Calinski-Harabasz with polarity alignment
        try:
            metrics["calinski_harabasz"] = float(
                _mu.pycrostates_calinski_harabasz_score(latent_features, labels, centroids)
            )
        except Exception as e:
            self.logger.warning(f"Pairwise Calinski-Harabasz failed: {e}")
            metrics["calinski_harabasz"] = 0.0

        return metrics

    def _extract_composite_scores_summary(
        self,
        train_metrics: Dict,
    ) -> Dict:
        """Extract composite scores for easy access (100% data mode)."""
        def get_composite(metrics, space="latent_space"):
            try:
                if space in metrics:
                    comp = metrics[space].get("composite_scores", {})
                else:
                    comp = metrics.get("composite_scores", {})
                return {
                    "gev": comp.get("gev", -1),
                    "sklearn_weighted": comp.get("sklearn", {}).get("weighted_average", -1),
                    "sklearn_geometric": comp.get("sklearn", {}).get("geometric_mean", -1),
                    "correlation_weighted": comp.get("correlation_based", {}).get("weighted_average", -1),
                    "correlation_geometric": comp.get("correlation_based", {}).get("geometric_mean", -1),
                    "recommended": comp.get("recommended", {}).get("score", -1),
                }
            except Exception:
                return {"error": "Could not extract composite scores"}

        return {
            "description": "Quick access to composite scores (100% data mode)",
            "train_latent": get_composite(train_metrics, "latent_space"),
            "train_raw": get_composite(train_metrics, "raw_space"),
            # In 100% data mode, test=train, so reuse train_metrics
            "test_latent": get_composite(train_metrics, "latent_space"),
            "test_raw": get_composite(train_metrics, "raw_space"),
        }

