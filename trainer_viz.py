"""Trainer Visualization Mixin — visualization methods for VAEClusteringTrainer."""
import glob
import json
import os
import numpy as np
import torch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class TrainerVizMixin:
    """Mixin providing visualization methods for VAEClusteringTrainer."""

    def _generate_comprehensive_latent_visualization(self, data_loader, save_prefix="final"):
        """
        Generate comprehensive t-SNE visualizations for unsupervised latent space analysis.

        Creates multiple visualizations:
        1. t-SNE colored by cluster assignments with density contours
        2. t-SNE colored by reconstruction error (identifies hard-to-reconstruct samples)
        3. Cluster centers projected onto t-SNE space
        4. Inter-cluster distance heatmap
        5. Latent dimension distributions per cluster
        6. Cluster separation metrics visualization
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.impute import SimpleImputer
        from scipy.stats import gaussian_kde
        from scipy.spatial.distance import pdist, squareform
        import matplotlib.gridspec as gridspec

        self.logger.info("=" * 60)
        self.logger.info("GENERATING COMPREHENSIVE LATENT SPACE VISUALIZATIONS")
        self.logger.info("=" * 60)

        self.model.set_eval_mode()

        # Reuse cached forward pass if available (from _compute_metrics_and_losses)
        cached = getattr(self, '_cached_forward_passes', {}).get(id(data_loader))
        if cached is not None:
            latents = cached["latents"]
            clusters = cached["clusters"]
            recon_errors = np.mean((cached["input"] - cached["decoded"]) ** 2, axis=1)
            self.logger.info(f"Reusing cached forward pass ({len(latents)} samples)")
        else:
            all_latents = []
            all_clusters = []
            all_recon_errors = []

            with torch.no_grad():
                for data, _ in tqdm(data_loader, desc="Collecting latent representations", leave=False):
                    data = data.to(self.device)
                    try:
                        mu, log_var = self.model.encode(data)
                        z = self.model.reparameterize(mu, log_var)
                        recon = self.model.decode(z)
                        recon_error = torch.mean((data - recon) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
                        clusters_batch = self.model.predict_from_latent(mu)
                        all_latents.append(mu.detach().cpu().numpy())
                        all_clusters.append(clusters_batch)
                        all_recon_errors.append(recon_error)
                    except Exception as e:
                        self.logger.warning(f"Error processing batch: {e}")
                        continue

            if not all_latents:
                self.logger.error("No valid data collected for visualization")
                self.model.train()
                return

            latents = np.concatenate(all_latents, axis=0)
            clusters = np.concatenate(all_clusters, axis=0)
            recon_errors = np.concatenate(all_recon_errors, axis=0)

        self.logger.info(f"Collected {len(latents)} samples for visualization")

        # Handle NaN values
        if np.isnan(latents).any():
            imputer = SimpleImputer(strategy="mean")
            latents = imputer.fit_transform(latents)
            self.logger.info("Imputed NaN values in latent representations")

        # Create output directory
        output_dir = self.output_dir / "latent_space_analysis"
        output_dir.mkdir(exist_ok=True)

        # Apply t-SNE (openTSNE if available, ~5-10x faster on large datasets)
        self.logger.info("Applying t-SNE dimensionality reduction...")
        perplexity = min(30, len(latents) - 1)
        try:
            from openTSNE import TSNE as OpenTSNE
            self.logger.info(f"Using openTSNE (FIt-SNE) on {len(latents)} samples")
            tsne = OpenTSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                n_iter=1000,
                initialization="pca",
                n_jobs=-1,
            )
            latents_2d = np.array(tsne.fit(latents))
        except ImportError:
            from sklearn.manifold import TSNE
            self.logger.info(f"Using sklearn TSNE on {len(latents)} samples (install openTSNE for ~5x speedup)")
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                n_iter=1000,
                learning_rate="auto",
                init="pca",
            )
            latents_2d = tsne.fit_transform(latents)

        # Get cluster centers in latent space
        cluster_centers = self.model.mu_c.detach().cpu().numpy()
        n_clusters = cluster_centers.shape[0]

        # =====================================================================
        # VISUALIZATION 1: t-SNE with Cluster Assignments and Density Contours
        # =====================================================================
        self.logger.info("Creating cluster assignment visualization with density...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Colors sized by n_clusters (total configured K) to handle dead clusters
        unique_clusters = np.unique(clusters)
        cmap = plt.colormaps["tab20"] if n_clusters > 10 else plt.colormaps["tab10"]
        colors = cmap(np.linspace(0, 1, n_clusters))

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            axes[0].scatter(
                latents_2d[mask, 0],
                latents_2d[mask, 1],
                c=[colors[cluster_id]],
                label=f"Cluster {cluster_id} (n={np.sum(mask)})",
                alpha=0.6,
                s=20,
            )

        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title("Latent Space: Cluster Assignments", fontsize=14)
        axes[0].legend(loc="best", fontsize=9)

        # Right: Density contours per cluster
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            if np.sum(mask) > 10:  # Need enough points for KDE
                try:
                    xy = latents_2d[mask].T
                    kde = gaussian_kde(xy)

                    # Create grid for contours
                    x_min, x_max = latents_2d[:, 0].min() - 1, latents_2d[:, 0].max() + 1
                    y_min, y_max = latents_2d[:, 1].min() - 1, latents_2d[:, 1].max() + 1
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)

                    axes[1].contour(xx, yy, density, levels=5, colors=[colors[cluster_id]], alpha=0.7)
                    axes[1].contourf(xx, yy, density, levels=5, colors=[colors[cluster_id]], alpha=0.2)
                except Exception:
                    pass

        # Add cluster centroids
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            centroid = latents_2d[mask].mean(axis=0)
            axes[1].scatter(centroid[0], centroid[1], c=[colors[cluster_id]], s=200,
                          marker="*", edgecolors="black", linewidths=1.5,
                          label=f"Cluster {cluster_id} centroid")

        axes[1].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[1].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[1].set_title("Latent Space: Cluster Density Contours", fontsize=14)
        axes[1].legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f"{save_prefix}_tsne_clusters_density.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 2: t-SNE Colored by Reconstruction Error
        # =====================================================================
        self.logger.info("Creating reconstruction error visualization...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: Reconstruction error heatmap
        scatter = axes[0].scatter(
            latents_2d[:, 0],
            latents_2d[:, 1],
            c=recon_errors,
            cmap="hot",
            alpha=0.7,
            s=20,
        )
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label("Reconstruction Error (MSE)", fontsize=10)
        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title("Latent Space: Reconstruction Error", fontsize=14)

        # Right: Box plot of reconstruction error per cluster
        recon_per_cluster = [recon_errors[clusters == c] for c in unique_clusters]
        bp = axes[1].boxplot(recon_per_cluster, labels=[f"C{c}" for c in unique_clusters], patch_artist=True)
        for patch, cluster_id in zip(bp["boxes"], unique_clusters):
            patch.set_facecolor(colors[cluster_id])
            patch.set_alpha(0.6)
        axes[1].set_xlabel("Cluster", fontsize=12)
        axes[1].set_ylabel("Reconstruction Error (MSE)", fontsize=12)
        axes[1].set_title("Reconstruction Error by Cluster", fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / f"{save_prefix}_tsne_recon_error.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 3: Inter-Cluster Distance Heatmap
        # =====================================================================
        self.logger.info("Creating inter-cluster distance heatmap...")

        # Compute pairwise distances between cluster centers
        center_distances = squareform(pdist(cluster_centers, metric="euclidean"))

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Distance matrix heatmap
        im = axes[0].imshow(center_distances, cmap="viridis", aspect="auto")
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label("Euclidean Distance", fontsize=10)
        axes[0].set_xticks(range(n_clusters))
        axes[0].set_yticks(range(n_clusters))
        axes[0].set_xticklabels([f"C{i}" for i in range(n_clusters)])
        axes[0].set_yticklabels([f"C{i}" for i in range(n_clusters)])
        axes[0].set_title("Inter-Cluster Center Distances (Latent Space)", fontsize=14)

        # Add distance values as text
        for i in range(n_clusters):
            for j in range(n_clusters):
                text_color = "white" if center_distances[i, j] > center_distances.max() / 2 else "black"
                axes[0].text(j, i, f"{center_distances[i, j]:.2f}", ha="center", va="center",
                           color=text_color, fontsize=8)

        # Right: Cluster sizes bar chart (includes dead clusters with 0 count)
        cluster_sizes = [np.sum(clusters == c) for c in range(n_clusters)]
        bars = axes[1].bar(range(n_clusters), cluster_sizes, color=colors, alpha=0.7)
        axes[1].set_xticks(range(n_clusters))
        axes[1].set_xticklabels([f"C{i}" for i in range(n_clusters)])
        axes[1].set_xlabel("Cluster", fontsize=12)
        axes[1].set_ylabel("Number of Samples", fontsize=12)
        axes[1].set_title("Cluster Size Distribution", fontsize=14)

        # Add count labels on bars
        for bar, size in zip(bars, cluster_sizes):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{size}", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / f"{save_prefix}_cluster_distances.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 4: Latent Dimension Distributions per Cluster
        # =====================================================================
        self.logger.info("Creating latent dimension distributions...")
        latent_dim = latents.shape[1]
        n_dims_to_show = min(8, latent_dim)  # Show up to 8 dimensions

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for dim_idx in range(n_dims_to_show):
            ax = axes[dim_idx]
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                ax.hist(latents[mask, dim_idx], bins=30, alpha=0.5,
                       label=f"C{cluster_id}", density=True,
                       color=colors[cluster_id])
            ax.set_xlabel(f"Latent Dim {dim_idx + 1}", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"Distribution: Latent Dim {dim_idx + 1}", fontsize=11)
            if dim_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        # Hide unused axes
        for idx in range(n_dims_to_show, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Latent Dimension Distributions by Cluster", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f"{save_prefix}_latent_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 5: Comprehensive Dashboard
        # =====================================================================
        self.logger.info("Creating comprehensive latent space dashboard...")

        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Top-left: t-SNE with clusters (large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            ax1.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                       c=[colors[cluster_id]], label=f"Cluster {cluster_id} (n={np.sum(mask)})",
                       alpha=0.6, s=15)
        ax1.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax1.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax1.set_title("Latent Space Clustering (t-SNE)", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)

        # Top-right: Reconstruction error
        ax2 = fig.add_subplot(gs[0, 2:4])
        scatter = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1], c=recon_errors,
                             cmap="hot", alpha=0.6, s=15)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Recon Error", fontsize=9)
        ax2.set_xlabel("t-SNE Dim 1", fontsize=10)
        ax2.set_ylabel("t-SNE Dim 2", fontsize=10)
        ax2.set_title("Reconstruction Error Map", fontsize=12, fontweight="bold")

        # Middle-right: Cluster sizes
        ax3 = fig.add_subplot(gs[1, 2])
        bars = ax3.bar(range(n_clusters), cluster_sizes, color=colors, alpha=0.7)
        ax3.set_xticks(range(n_clusters))
        ax3.set_xticklabels([f"C{i}" for i in range(n_clusters)])
        ax3.set_xlabel("Cluster", fontsize=10)
        ax3.set_ylabel("Count", fontsize=10)
        ax3.set_title("Cluster Sizes", fontsize=12, fontweight="bold")

        # Middle-right: Inter-cluster distances
        ax4 = fig.add_subplot(gs[1, 3])
        im = ax4.imshow(center_distances, cmap="viridis", aspect="auto")
        ax4.set_xticks(range(n_clusters))
        ax4.set_yticks(range(n_clusters))
        ax4.set_xticklabels([f"C{i}" for i in range(n_clusters)], fontsize=8)
        ax4.set_yticklabels([f"C{i}" for i in range(n_clusters)], fontsize=8)
        ax4.set_title("Cluster Distances", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax4, shrink=0.8)

        # Bottom: Latent dimension distributions (4 most important)
        for dim_idx in range(min(4, latent_dim)):
            ax = fig.add_subplot(gs[2, dim_idx])
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                ax.hist(latents[mask, dim_idx], bins=25, alpha=0.5,
                       density=True, color=colors[cluster_id])
            ax.set_xlabel(f"Latent Dim {dim_idx + 1}", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"Dim {dim_idx + 1} Distribution", fontsize=10)

        plt.suptitle(f"Comprehensive Latent Space Analysis (K={n_clusters})",
                    fontsize=16, fontweight="bold", y=0.98)
        plt.savefig(output_dir / f"{save_prefix}_comprehensive_dashboard.png",
                   dpi=300, bbox_inches="tight")
        plt.close()

        # =====================================================================
        # VISUALIZATION 6: Cluster Quality Metrics Summary
        # =====================================================================
        self.logger.info("Creating cluster quality summary...")

        # Compute metrics
        try:
            sil_score = silhouette_score(latents, clusters)
            db_score = davies_bouldin_score(latents, clusters)
            ch_score = calinski_harabasz_score(latents, clusters)
        except Exception as e:
            self.logger.warning(f"Could not compute clustering metrics: {e}")
            sil_score, db_score, ch_score = 0, 0, 0

        # Per-cluster statistics
        cluster_stats = []
        for c in unique_clusters:
            mask = clusters == c
            cluster_latents = latents[mask]
            cluster_stats.append({
                "cluster": c,
                "size": np.sum(mask),
                "mean_recon_error": np.mean(recon_errors[mask]),
                "std_recon_error": np.std(recon_errors[mask]),
                "latent_variance": np.mean(np.var(cluster_latents, axis=0)),
            })

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Global metrics
        ax1 = axes[0]
        metrics_names = ["Silhouette\n(higher=better)", "Davies-Bouldin\n(lower=better)", "Calinski-Harabasz\n(higher=better)"]
        metrics_values = [sil_score, db_score, ch_score / 100]  # Scale CH for visualization
        bars = ax1.bar(metrics_names, metrics_values, color=["#2ecc71", "#e74c3c", "#3498db"], alpha=0.7)
        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_title("Global Clustering Quality Metrics", fontsize=14, fontweight="bold")
        for bar, val in zip(bars, [sil_score, db_score, ch_score]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        # Per-cluster reconstruction error (only clusters with samples)
        ax2 = axes[1]
        cluster_recon = [s["mean_recon_error"] for s in cluster_stats]
        cluster_std = [s["std_recon_error"] for s in cluster_stats]
        bar_colors_active = [colors[c] for c in unique_clusters]
        ax2.bar(range(len(unique_clusters)), cluster_recon, yerr=cluster_std,
               color=bar_colors_active, alpha=0.7, capsize=5)
        ax2.set_xticks(range(len(unique_clusters)))
        ax2.set_xticklabels([f"C{c}" for c in unique_clusters])
        ax2.set_xlabel("Cluster", fontsize=12)
        ax2.set_ylabel("Mean Reconstruction Error", fontsize=12)
        ax2.set_title("Reconstruction Quality per Cluster", fontsize=14, fontweight="bold")

        # Per-cluster latent variance (only clusters with samples)
        ax3 = axes[2]
        cluster_var = [s["latent_variance"] for s in cluster_stats]
        ax3.bar(range(len(unique_clusters)), cluster_var, color=bar_colors_active, alpha=0.7)
        ax3.set_xticks(range(len(unique_clusters)))
        ax3.set_xticklabels([f"C{c}" for c in unique_clusters])
        ax3.set_xlabel("Cluster", fontsize=12)
        ax3.set_ylabel("Mean Latent Variance", fontsize=12)
        ax3.set_title("Cluster Compactness (Latent Space)", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_dir / f"{save_prefix}_cluster_quality_metrics.png",
                   dpi=300, bbox_inches="tight")
        plt.close()

        # Save cluster statistics to JSON
        stats_summary = {
            "global_metrics": {
                "silhouette_score": float(sil_score),
                "davies_bouldin_score": float(db_score),
                "calinski_harabasz_score": float(ch_score),
            },
            "cluster_statistics": [
                {
                    "cluster_id": int(s["cluster"]),
                    "size": int(s["size"]),
                    "mean_reconstruction_error": float(s["mean_recon_error"]),
                    "std_reconstruction_error": float(s["std_recon_error"]),
                    "mean_latent_variance": float(s["latent_variance"]),
                }
                for s in cluster_stats
            ],
            "n_samples": len(latents),
            "n_clusters": n_clusters,
            "latent_dim": latent_dim,
        }

        with open(output_dir / f"{save_prefix}_latent_analysis_stats.json", "w") as f:
            json.dump(stats_summary, f, indent=4)

        self.logger.info(f"Comprehensive latent space visualizations saved to: {output_dir}")
        self.logger.info(f"   - {save_prefix}_tsne_clusters_density.png")
        self.logger.info(f"   - {save_prefix}_tsne_recon_error.png")
        self.logger.info(f"   - {save_prefix}_cluster_distances.png")
        self.logger.info(f"   - {save_prefix}_latent_distributions.png")
        self.logger.info(f"   - {save_prefix}_comprehensive_dashboard.png")
        self.logger.info(f"   - {save_prefix}_cluster_quality_metrics.png")
        self.logger.info(f"   - {save_prefix}_latent_analysis_stats.json")

        self.model.train()
        return output_dir

    def _save_tsne_snapshot(self, epoch, subsample=5000):
        """Save a t-SNE snapshot of latent space for progression tracking."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.model.set_eval_mode()
        try:
            mus, labels_list = [], []
            with torch.no_grad():
                for batch in self.train_loader:
                    x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                    mu, _ = self.model.encode(x)
                    preds = self.model.predict_from_latent(mu)
                    mus.append(mu.cpu().numpy())
                    labels_list.append(preds)
            mus = np.concatenate(mus)
            labels = np.concatenate(labels_list)

            if len(mus) > subsample:
                idx = np.random.choice(len(mus), subsample, replace=False)
                mus, labels = mus[idx], labels[idx]

            perplexity = min(30, len(mus) - 1)
            try:
                from openTSNE import TSNE as OpenTSNE
                tsne = OpenTSNE(n_components=2, perplexity=perplexity, n_iter=500, random_state=42, initialization="pca", n_jobs=-1)
                emb = np.array(tsne.fit(mus))
            except ImportError:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=500, random_state=42)
                emb = tsne.fit_transform(mus)

            fig, ax = plt.subplots(figsize=(5, 5))
            for k in range(self.model.nClusters):
                mask = labels == k
                ax.scatter(emb[mask, 0], emb[mask, 1], s=3, alpha=0.4, label=chr(65 + k))
            ax.set_title(f"t-SNE Epoch {epoch}")
            ax.legend(fontsize=7, markerscale=3)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()

            snap_dir = os.path.join(self.output_dir, "tsne_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            fig.savefig(os.path.join(snap_dir, f"tsne_epoch_{epoch:04d}.png"), dpi=150)
            plt.close(fig)
        finally:
            self.model.train()

    def _compile_tsne_progression(self):
        """Compile t-SNE snapshots into a single progression grid."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        snap_dir = os.path.join(self.output_dir, "tsne_snapshots")
        if not os.path.isdir(snap_dir):
            return
        snaps = sorted(glob.glob(os.path.join(snap_dir, "tsne_epoch_*.png")))
        if len(snaps) < 2:
            return

        # Select up to 8 evenly spaced snapshots
        if len(snaps) > 8:
            indices = np.linspace(0, len(snaps) - 1, 8, dtype=int)
            snaps = [snaps[i] for i in indices]

        ncols = min(4, len(snaps))
        nrows = (len(snaps) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for idx, (ax, snap_path) in enumerate(zip(axes.flat, snaps)):
            img = Image.open(snap_path)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.flat[len(snaps):]:
            ax.set_visible(False)
        fig.suptitle("t-SNE Latent Space Progression", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "tsne_progression.png"), dpi=150)
        plt.close(fig)

