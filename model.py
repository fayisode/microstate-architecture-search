import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from tqdm import tqdm
from sklearn.cluster import KMeans
import logging
import warnings
from pathlib import Path


def _nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Backward-compatible torch.nan_to_num (added in PyTorch 1.8)."""
    if hasattr(torch, 'nan_to_num'):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    out = torch.where(torch.isnan(x), torch.full_like(x, nan), x)
    if posinf is not None:
        out = torch.where(torch.isinf(out) & (out > 0), torch.full_like(out, posinf), out)
    if neginf is not None:
        out = torch.where(torch.isinf(out) & (out < 0), torch.full_like(out, neginf), out)
    return out

# Visualization/EEG imports — lazy for CPU-only environments
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    plt = None
try:
    import mne
    from pycrostates.viz import (
        plot_cluster_centers,
        plot_raw_segmentation,
        plot_epoch_segmentation,
    )
    from pycrostates.io import ChData
except ImportError:
    mne = None


warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

EPSILON = 1e-8


class Encoder(nn.Module):
    def __init__(self, nc: int, ndf: int, latent_dim: int, n_conv_layers: int = 4):
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_dim = latent_dim
        self.n_conv_layers = n_conv_layers

        # Build conv layers dynamically
        layers = []
        for i in range(n_conv_layers):
            in_ch = nc if i == 0 else ndf * (2 ** min(i - 1, 3))
            out_ch = ndf * (2 ** min(i, 3))  # Cap at 8× ndf
            if i == 0:
                # First layer: no BatchNorm, has Dropout
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.2),
                ])
            else:
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
                if i % 2 == 0:  # Dropout every other layer
                    layers.append(nn.Dropout(0.2))

        # Bottleneck: adaptive pool → 1×1 → 1024 channels
        final_ch = ndf * (2 ** min(n_conv_layers - 1, 3))
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(final_ch, 1024, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.encoder = nn.Sequential(*layers)

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # mu
        self.fc22 = nn.Linear(512, latent_dim)  # logvar

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for _m in self.modules():
            if isinstance(_m, nn.Conv2d):
                # Kaiming (He) initialization for layers followed by LeakyReLU
                # Accounts for the activation function's effect on variance
                nn.init.kaiming_normal_(
                    _m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(_m, nn.Linear):
                # Xavier initialization for linear layers
                # Maintains variance across layers for symmetric activations
                nn.init.xavier_normal_(_m.weight)
                if _m.bias is not None:
                    nn.init.constant_(_m.bias, 0)
            elif isinstance(_m, nn.BatchNorm2d):
                # Standard BatchNorm initialization
                nn.init.constant_(_m.weight, 1)
                nn.init.constant_(_m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    def __init__(self, ngf: int, nc: int, latent_dim: int, n_conv_layers: int = 4, target_size: int = 40):
        super().__init__()
        self.ngf = ngf
        self.nc = nc
        self.latent_dim = latent_dim
        self.n_conv_layers = n_conv_layers
        self.target_size = target_size

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build mirror of encoder channel sequence in reverse
        # Channel sequence: [ngf*2^min(n-1,3), ..., ngf*2^min(1,3), ngf*2^0, nc]
        channel_seq = [ngf * (2 ** min(i, 3)) for i in range(n_conv_layers)]
        channel_seq.reverse()  # Reverse to go from deepest to shallowest

        layers = []
        # Layer 0: 1024 → first channel, 1×1 → 4×4
        layers.extend([
            nn.ConvTranspose2d(1024, channel_seq[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(channel_seq[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        ])

        # Intermediate layers: double spatial dim each step
        for i in range(1, len(channel_seq)):
            layers.extend([
                nn.ConvTranspose2d(channel_seq[i - 1], channel_seq[i], 4, 2, 1, bias=False),
                nn.BatchNorm2d(channel_seq[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Final: to output channels, no activation (linear for z-scored data)
        layers.append(
            nn.ConvTranspose2d(channel_seq[-1], nc, 4, 2, 1, bias=False)
        )
        self.decoder = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for _m in self.modules():
            if isinstance(_m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    _m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(_m, nn.Linear):
                nn.init.xavier_normal_(_m.weight)
                if _m.bias is not None:
                    nn.init.constant_(_m.bias, 0)
            elif isinstance(_m, nn.BatchNorm2d):
                nn.init.constant_(_m.weight, 1)
                nn.init.constant_(_m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 1024, 1, 1)
        out = self.decoder(decoder_input)
        # Safety net: resize to target if conv chain produces wrong spatial dim
        if out.shape[-1] != self.target_size or out.shape[-2] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
        return out


class LossBalancer:
    def __init__(
        self,
        init_beta: float = 0.1,
        min_beta: float = 0,
        max_beta: float = 2.0,
        beta_warmup_epochs: int = 30,
        adaptive_weight: bool = False,
        use_batch_cyclical: bool = False,
        n_cycles_per_epoch: int = 3,
        cycle_ratio: float = 0.5,
        gamma: float = 0.005,
    ):
        self.init_beta = init_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.adaptive_weight = adaptive_weight
        self.recon_avg: Optional[float] = None
        self.kld_avg: Optional[float] = None
        self.use_batch_cyclical = use_batch_cyclical
        self.n_cycles_per_epoch = n_cycles_per_epoch
        self.cycle_ratio = cycle_ratio
        self.gamma = gamma
        self.batch_schedule = None
        self.current_epoch = -1
        self.n_total_cycles = None  # Set to int to enable global schedule
        self.global_schedule = None

    def get_state(self):
        return {
            "recon_avg": self.recon_avg,
            "kld_avg": self.kld_avg,
            "current_epoch": self.current_epoch,
        }

    def set_state(self, state):
        self.recon_avg = state.get("recon_avg")
        self.kld_avg = state.get("kld_avg")
        # self.current_epoch = state.get("current_epoch", -1)

    def setup_batch_cyclical_schedule(self, n_batches_per_epoch: int) -> np.ndarray:
        n_iter = n_batches_per_epoch
        n_cycle = self.n_cycles_per_epoch
        ratio = self.cycle_ratio
        start = self.gamma
        stop = self.gamma + 1
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                if int(i + c * period) < n_iter:
                    L[int(i + c * period)] = v
                v += step
                i += 1
        L = np.clip(L, self.min_beta, self.max_beta)
        self.batch_schedule = L
        return L

    def setup_global_cyclical_schedule(self, n_batches_per_epoch: int, total_epochs: int):
        """Build ONE beta schedule over full training instead of per-epoch."""
        n_iter = n_batches_per_epoch * total_epochs
        n_cycle = self.n_total_cycles
        ratio = self.cycle_ratio
        start = self.gamma
        stop = self.gamma + 1
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        self.global_schedule = np.clip(L, self.min_beta, self.max_beta)

    def get_beta(
        self,
        epoch: int,
        recon_loss: Union[torch.Tensor, float],
        kld_loss: Union[torch.Tensor, float],
        batch_idx: Optional[int] = None,
        n_batches_per_epoch: Optional[int] = None,
    ) -> float:

        if (
            self.use_batch_cyclical
            and batch_idx is not None
            and n_batches_per_epoch is not None
        ):
            return self.get_batch_cyclical_beta(
                epoch, batch_idx, n_batches_per_epoch, recon_loss, kld_loss
            )
        if epoch < self.beta_warmup_epochs:
            return self.init_beta + (self.max_beta - self.init_beta) * (
                epoch / self.beta_warmup_epochs
            )
        if not self.adaptive_weight:
            return self.max_beta
        if self.recon_avg is None:
            self.recon_avg = (
                recon_loss.detach().item()
                if isinstance(recon_loss, torch.Tensor)
                else recon_loss
            )
            self.kld_avg = (
                kld_loss.detach().item()
                if isinstance(kld_loss, torch.Tensor)
                else kld_loss
            )
        else:
            recon_value = (
                recon_loss.detach().item()
                if isinstance(recon_loss, torch.Tensor)
                else recon_loss
            )
            kld_value = (
                kld_loss.detach().item()
                if isinstance(kld_loss, torch.Tensor)
                else kld_loss
            )
            self.recon_avg = 0.9 * self.recon_avg + 0.1 * recon_value
            self.kld_avg = 0.9 * self.kld_avg + 0.1 * kld_value
        ratio = min(self.recon_avg / (self.kld_avg + EPSILON), 4.0)
        dynamic_beta = min(self.max_beta, max(self.min_beta, ratio))
        return dynamic_beta

    def get_batch_cyclical_beta(
        self,
        epoch: int,
        batch_idx: int,
        n_batches_per_epoch: int,
        recon_loss: Union[torch.Tensor, float],
        kld_loss: Union[torch.Tensor, float],
    ) -> float:
        # Use global schedule when available (preferred)
        if self.global_schedule is not None:
            global_idx = epoch * n_batches_per_epoch + batch_idx
            safe_idx = min(global_idx, len(self.global_schedule) - 1)
            return float(self.global_schedule[safe_idx])
        # Fallback to per-epoch schedule
        if epoch != self.current_epoch:
            self.setup_batch_cyclical_schedule(n_batches_per_epoch)
            self.current_epoch = epoch
        safe_batch_idx = min(batch_idx, len(self.batch_schedule) - 1)
        base_beta = self.batch_schedule[safe_batch_idx]
        return float(base_beta)

    def cyclical_beta(
        self,
        epoch: int,
        total_epochs: int,
        n_cycles: int = 4,
        max_beta: Optional[float] = None,
    ) -> float:
        if max_beta is None:
            max_beta = self.max_beta
        cycle_length = max(1, total_epochs // n_cycles)
        cycle_progress = (epoch % cycle_length) / cycle_length
        beta = (
            self.min_beta
            + (max_beta - self.min_beta) * (1 - np.cos(np.pi * cycle_progress)) / 2
        )
        return beta


class MyModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        nClusters: int,
        batch_size: int,
        logger: any,
        device: torch.device,
        use_cyclical_annealing: bool = False,
        use_batch_cyclical: bool = True,
        nc: int = 1,
        ndf: int = 64,
        ngf: int = 64,
        info=None,
        n_conv_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.nClusters = nClusters
        self.nc = nc
        self.ndf = ndf
        self.ngf = ngf
        self.n_conv_layers = n_conv_layers
        self.use_cyclical_annealing = use_cyclical_annealing
        self.use_batch_cyclical = use_batch_cyclical
        self.batch_size = batch_size
        self.flat_size = None
        self.device = device
        self.logger = logger
        self.loss_balancer = LossBalancer()
        self.encoder = Encoder(nc, ndf, latent_dim, n_conv_layers=n_conv_layers)
        self.decoder = Decoder(ngf, nc, latent_dim, n_conv_layers=n_conv_layers)
        self.pi_ = nn.Parameter(
            torch.log(torch.FloatTensor(self.nClusters).fill_(1) / self.nClusters),
            requires_grad=True,
        )
        self.prior_frozen = False
        # FIX: Initialize cluster centers with appropriate scale
        # Use Xavier-like initialization: scale by sqrt(2 / latent_dim) for better spread
        init_scale = np.sqrt(2.0 / latent_dim)
        self.mu_c = nn.Parameter(torch.randn(nClusters, latent_dim) * init_scale)
        # Initialize log_var_c to small positive values (var ~ 0.5-1.0)
        self.log_var_c = nn.Parameter(torch.zeros(nClusters, latent_dim) - 0.5)
        self.min_log_var = -4.0
        self._needs_data_init = True
        self.logger.info(
            f"📊 Trainer initialized with batch size: {batch_size} and latent dim: {latent_dim} and nClusters: {nClusters}"
        )

        self._needs_data_init = True

        # MNE info for electrode positions (required for visualization)
        if info is not None:
            self.info = info
            self.logger.info(f"Using MNE info with {len(info.ch_names)} channels")
        else:
            # No info provided - visualization methods won't work
            self.info = None
            self.logger.warning("No MNE info provided - visualization may be limited")
        # ----------------------
        self.logger.info(f"📊 Trainer initialized...")

        # FIX: Add temperature parameter for softmax predictions
        self.prediction_temperature = 1.0

    def set_eval_mode(self):
        """Wrapper for PyTorch eval mode"""
        return nn.Module.eval(self)

    def set_train_mode(self):
        """Wrapper for PyTorch train mode"""
        return nn.Module.train(self)

    def set_prediction_temperature(self, temperature: float):
        """Set temperature for cluster assignment softmax (higher = more uniform)"""
        self.prediction_temperature = max(0.1, temperature)
        self.logger.info(f"Prediction temperature set to: {self.prediction_temperature}")

    def cluster_entropy_regularization(self) -> torch.Tensor:
        """
        FIX: Entropy regularization to encourage uniform cluster usage.
        Returns a loss term that penalizes non-uniform cluster weights.
        """
        # Get normalized cluster weights
        pi_normalized = torch.softmax(self.pi_, dim=0)
        # Compute entropy (higher = more uniform)
        entropy = -torch.sum(pi_normalized * torch.log(pi_normalized + EPSILON))
        # Max entropy for uniform distribution
        max_entropy = np.log(self.nClusters)
        # Return loss (0 when uniform, positive when imbalanced)
        entropy_loss = max_entropy - entropy
        return entropy_loss

    def cluster_tightening_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Augmented ELBO: KL(q(z|x) || N(mu_c*, var_c*)) for assigned cluster c*.

        Tightens each sample's posterior to match its most-probable cluster prior.
        Proven across VaDE, MFCVAE, DCGMM to improve cluster compactness.
        """
        log_var_c = torch.clamp(self.log_var_c, min=self.min_log_var, max=2.0)

        # Use mu directly for stable assignment (MAP estimate, no stochastic noise)
        log_pi = self.pi_.unsqueeze(0)
        log_gaussian = self.gaussian_pdfs_log(mu, self.mu_c, log_var_c)
        log_yita_c = log_pi + log_gaussian
        log_yita_c = log_yita_c - torch.logsumexp(log_yita_c, dim=1, keepdim=True)

        # Hard assignment: c* = argmax
        c_star = torch.argmax(log_yita_c, dim=1)  # (batch,)

        # Get assigned cluster params
        mu_c_star = self.mu_c[c_star]  # (batch, latent_dim)
        log_var_c_star = log_var_c[c_star]  # (batch, latent_dim)

        # KL(q(z|x) || N(mu_c*, var_c*))
        kl = 0.5 * torch.sum(
            log_var_c_star - log_var
            + (torch.exp(log_var) + (mu - mu_c_star).pow(2))
            / (torch.exp(log_var_c_star) + EPSILON)
            - 1,
            dim=1,
        )
        return kl.mean()

    def batch_cluster_entropy(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL(mean_batch_assignments || Uniform(1/K)).

        Encourages each mini-batch to use all clusters roughly equally,
        complementing the pi_-based entropy which only regularizes the prior.
        """
        log_var_c = torch.clamp(self.log_var_c, min=self.min_log_var, max=2.0)
        z = self.reparameterize(mu, log_var)
        log_pi = self.pi_.unsqueeze(0)
        log_gaussian = self.gaussian_pdfs_log(z, self.mu_c, log_var_c)
        log_yita_c = log_pi + log_gaussian
        log_yita_c = log_yita_c - torch.logsumexp(log_yita_c, dim=1, keepdim=True)
        yita_c = torch.exp(log_yita_c)
        mean_assignment = yita_c.mean(dim=0).clamp(min=1e-8)
        uniform = torch.ones_like(mean_assignment) / self.nClusters
        return torch.sum(mean_assignment * torch.log(mean_assignment / uniform))

    def _bisecting_kmeans(self, Z: np.ndarray, n_clusters: int, base_k: int = 4) -> np.ndarray:
        """Hierarchical bisecting KMeans for large K.

        For K > base_k: fits base_k clusters first, then iteratively splits
        the largest cluster until reaching target K. This prevents dead
        clusters that occur when flat KMeans tries to partition a space
        into too many clusters at once (e.g., 7/12 dead at K=12).

        Returns: centroids array of shape (n_clusters, latent_dim)
        """
        if n_clusters <= base_k:
            km = KMeans(n_clusters=n_clusters, n_init=20, max_iter=1000,
                        algorithm="full", random_state=42)
            km.fit(Z)
            self.logger.info(f"[BISECT] Standard KMeans for K={n_clusters}")
            return km.cluster_centers_

        # Phase 1: Fit base_k clusters
        self.logger.info(f"[BISECT] Phase 1: Fitting base K={base_k}...")
        km_base = KMeans(n_clusters=base_k, n_init=20, max_iter=1000,
                         algorithm="full", random_state=42)
        labels = km_base.fit_predict(Z)
        centroids = list(km_base.cluster_centers_)
        cluster_members = {i: Z[labels == i] for i in range(base_k)}

        counts = [len(cluster_members[i]) for i in range(len(centroids))]
        self.logger.info(f"[BISECT] Base clusters: {counts}")

        # Phase 2: Iteratively split largest cluster
        current_k = base_k
        while current_k < n_clusters:
            # Find the largest cluster to split
            sizes = [len(cluster_members[i]) for i in range(len(centroids))]
            largest_idx = int(np.argmax(sizes))
            largest_data = cluster_members[largest_idx]

            if len(largest_data) < 4:
                self.logger.warning(f"[BISECT] Cluster {largest_idx} too small to split ({len(largest_data)} samples)")
                break

            # Split into 2
            km_split = KMeans(n_clusters=2, n_init=10, max_iter=500, random_state=42)
            sub_labels = km_split.fit_predict(largest_data)

            # Replace the largest cluster with child 0, append child 1
            child0 = largest_data[sub_labels == 0]
            child1 = largest_data[sub_labels == 1]

            centroids[largest_idx] = km_split.cluster_centers_[0]
            cluster_members[largest_idx] = child0

            new_idx = len(centroids)
            centroids.append(km_split.cluster_centers_[1])
            cluster_members[new_idx] = child1

            current_k += 1
            self.logger.info(
                f"[BISECT] Split cluster {largest_idx} ({len(largest_data)}) → "
                f"{len(child0)} + {len(child1)}, now K={current_k}"
            )

        if len(centroids) < n_clusters:
            self.logger.warning(
                f"[BISECT] Could only produce {len(centroids)}/{n_clusters} centroids "
                f"due to early termination — falling back to flat KMeans"
            )
            km_fallback = KMeans(n_clusters=n_clusters, n_init=20, max_iter=1000,
                                 algorithm="full", random_state=42)
            km_fallback.fit(Z)
            return km_fallback.cluster_centers_

        centroids = np.array(centroids[:n_clusters])
        self.logger.info(f"[BISECT] Final: {n_clusters} centroids via bisecting KMeans")
        return centroids

    def reinitialize_dead_clusters(self, cluster_counts: np.ndarray, latents: np.ndarray, threshold: float = 0.01):
        """
        FIX: Re-initialize clusters that have very few assignments.
        Args:
            cluster_counts: Number of samples assigned to each cluster
            latents: Latent representations of samples
            threshold: Minimum fraction of samples per cluster (default 1%)
        """
        total_samples = np.sum(cluster_counts)
        min_samples = int(threshold * total_samples)

        dead_clusters = np.where(cluster_counts < min_samples)[0]
        if len(dead_clusters) == 0:
            return

        self.logger.warning(f"[REINIT] Found {len(dead_clusters)} dead clusters: {dead_clusters}")

        # Get device from existing parameters (more reliable than self.device)
        device = self.mu_c.device

        # Find the most populated cluster to split
        alive_clusters = np.where(cluster_counts >= min_samples)[0]
        if len(alive_clusters) == 0:
            self.logger.error("[REINIT] All clusters are dead - cannot reinitialize")
            return

        reinitialized_count = 0
        for dead_idx in dead_clusters:
            # Pick the largest cluster to split
            largest_cluster = alive_clusters[np.argmax(cluster_counts[alive_clusters])]

            # Get samples from largest cluster (approximate by distance to centroid)
            with torch.no_grad():
                centroid = self.mu_c[largest_cluster].detach().cpu().numpy()
                distances = np.linalg.norm(latents - centroid, axis=1)
                # Get indices of samples closest to this centroid
                closest_indices = np.argsort(distances)[:cluster_counts[largest_cluster]]

                if len(closest_indices) > 0:
                    # Reinitialize dead cluster to a random sample from largest cluster
                    random_sample_idx = np.random.choice(closest_indices)
                    new_centroid = latents[random_sample_idx]

                    # Add small noise to avoid exact overlap
                    noise = np.random.randn(self.latent_dim) * 0.1
                    new_centroid = new_centroid + noise

                    self.mu_c.data[dead_idx] = torch.tensor(
                        new_centroid, dtype=torch.float32, device=device
                    )
                    reinitialized_count += 1
                    self.logger.info(f"[REINIT] ✅ Reinitialized cluster {dead_idx} from cluster {largest_cluster}")
                else:
                    self.logger.warning(f"[REINIT] ⚠️ Could not reinitialize cluster {dead_idx} - no samples found")

        # Reset pi_ weights to uniform after reinitialization to give dead clusters a fair chance
        if reinitialized_count > 0:
            with torch.no_grad():
                uniform_pi = torch.zeros(self.nClusters, device=device)  # log(1/K) = 0 after softmax normalization
                self.pi_.data = uniform_pi
            self.logger.info(f"[REINIT] ✅ Reset pi_ weights to uniform after reinitializing {reinitialized_count} clusters")

    def _check_and_reinitialize_dead_clusters(self, data_loader, device):
        """
        FIX: Helper to check cluster assignments and reinitialize dead clusters during training.
        """
        self.set_eval_mode()
        all_latents = []
        all_predictions = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                mu, _ = self.encode(data)
                predictions = self.predict_from_latent(mu)
                all_latents.append(mu.detach().cpu().numpy())
                all_predictions.append(predictions)

        all_latents = np.concatenate(all_latents, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Count samples per cluster
        cluster_counts = np.bincount(all_predictions, minlength=self.nClusters)
        total_samples = len(all_predictions)

        # DEBUG: Detailed cluster distribution logging
        self.logger.info(f"[CLUSTER_CHECK] Total samples: {total_samples}")
        self.logger.info(f"[CLUSTER_CHECK] Cluster distribution: {cluster_counts.tolist()}")
        self.logger.info(f"[CLUSTER_CHECK] Cluster percentages: {[f'{c/total_samples*100:.1f}%' for c in cluster_counts]}")
        self.logger.info(f"[CLUSTER_CHECK] Active clusters (>1%): {np.sum(cluster_counts > total_samples * 0.01)}/{self.nClusters}")
        self.logger.info(f"[CLUSTER_CHECK] Dead clusters (0 samples): {np.sum(cluster_counts == 0)}")
        self.logger.info(f"[CLUSTER_CHECK] Max cluster: {cluster_counts.max()} ({cluster_counts.max()/total_samples*100:.1f}%)")
        self.logger.info(f"[CLUSTER_CHECK] Min cluster: {cluster_counts.min()} ({cluster_counts.min()/total_samples*100:.1f}%)")

        # Check current pi_ weights
        pi_weights = torch.exp(self.pi_).detach().cpu().numpy()
        self.logger.info(f"[CLUSTER_CHECK] Pi weights: {pi_weights}")

        # Check for dead clusters (less than 1% of samples)
        self.reinitialize_dead_clusters(cluster_counts, all_latents, threshold=0.01)

        self.set_train_mode()

    def monitor_kl_health(self, mu, log_var):
        with torch.no_grad():
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_per_dim = kl_per_dim.mean(0)
            inactive_dims = (kl_per_dim < 0.01).sum().item()
            self.logger.info("KL per dimension stats:")
            self.logger.info(f"  Mean: {kl_per_dim.mean():.6f}")
            self.logger.info(f"  Std: {kl_per_dim.std():.6f}")
            self.logger.info(f"  Min: {kl_per_dim.min():.6f}")
            self.logger.info(f"  Max: {kl_per_dim.max():.6f}")
            self.logger.info(
                f"  Inactive dimensions (KL < 0.01): {inactive_dims}/{len(kl_per_dim)}"
            )
            if inactive_dims > len(kl_per_dim) * 0.5:
                self.logger.warning(
                    "⚠️  More than 50% of latent dimensions are inactive!"
                )
                return False
            return True

    def detect_posterior_collapse(self, data_loader, threshold=0.1):
        self.eval()
        total_kl = 0
        total_mixture_kl = 0
        num_samples = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i > 10:
                    break
                data = data.to(self.device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                mu, log_var = self.encode(data)
                standard_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                total_kl += standard_kl.item()
                mixture_kl = self.KLD(mu, log_var, normalize=False)
                total_mixture_kl += mixture_kl.item()
                num_samples += data.size(0)
        avg_standard_kl = total_kl / num_samples
        avg_mixture_kl = total_mixture_kl / num_samples
        self.logger.info("Collapse Detection:")
        self.logger.info(f"  Standard KL per sample: {avg_standard_kl:.6f}")
        self.logger.info(f"  Mixture KL per sample: {avg_mixture_kl:.6f}")
        if avg_standard_kl < threshold or avg_mixture_kl < threshold:
            self.logger.warning("🚨 POSTERIOR COLLAPSE DETECTED!")
            self.logger.warning(f"   Standard KL: {avg_standard_kl:.6f} < {threshold}")
            self.logger.warning(f"   Mixture KL: {avg_mixture_kl:.6f} < {threshold}")
            self.train()
            return True
        self.train()
        return False

    def initialize_from_data(self, train_loader):
        if hasattr(self, "_needs_data_init") and self._needs_data_init:
            self.logger.info("Initializing clusters from FULL data distribution...")
            self.set_eval_mode()
            latents = []
            with torch.no_grad():
                # FIX: Use ALL data instead of just 100 batches
                for data, _ in train_loader:
                    data = data.to(self.device)
                    mu, _ = self.encode(data)
                    latents.append(mu.cpu())
            if latents:
                latents = torch.cat(latents, 0).numpy()
                self.logger.info(
                    f"[INIT] Collected {len(latents)} latent samples for initialization (FULL dataset)"
                )

                # Compute global latent statistics for adaptive variance floor
                global_latent_std = np.std(latents, axis=0)
                global_latent_mean = np.mean(latents, axis=0)
                computed_floor = 0.1 * np.mean(global_latent_std) ** 2
                # FIX: Ensure minimum floor of 0.1 to prevent collapse when encoder hasn't learned yet
                MIN_VAR_FLOOR = 0.1
                adaptive_var_floor = max(computed_floor, MIN_VAR_FLOOR)

                # DEBUG: Log latent space statistics
                self.logger.info(f"[INIT] Latent space stats:")
                self.logger.info(f"[INIT]   - Mean of means: {np.mean(global_latent_mean):.6f}")
                self.logger.info(f"[INIT]   - Mean of stds: {np.mean(global_latent_std):.6f}")
                self.logger.info(f"[INIT]   - Min std: {np.min(global_latent_std):.6f}")
                self.logger.info(f"[INIT]   - Max std: {np.max(global_latent_std):.6f}")
                self.logger.info(f"[INIT] Adaptive variance floor: {adaptive_var_floor:.6f} (computed: {computed_floor:.6f}, min: {MIN_VAR_FLOOR})")

                self.logger.info(f"[INIT] Running KMeans with {self.nClusters} clusters...")
                kmeans = KMeans(
                    n_clusters=self.nClusters, n_init=20, max_iter=500, random_state=42
                )
                cluster_labels = kmeans.fit_predict(latents)

                # DEBUG: Log cluster distribution from KMeans
                cluster_counts = np.bincount(cluster_labels, minlength=self.nClusters)
                self.logger.info(f"[INIT] KMeans cluster distribution: {cluster_counts.tolist()}")
                self.logger.info(f"[INIT] KMeans cluster sizes - min: {cluster_counts.min()}, max: {cluster_counts.max()}, mean: {cluster_counts.mean():.1f}")

                mu_c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                log_var_c = torch.zeros(self.nClusters, self.latent_dim)

                empty_clusters = []
                for k in range(self.nClusters):
                    cluster_data = latents[cluster_labels == k]
                    if len(cluster_data) > 1:
                        cluster_var = np.var(cluster_data, axis=0)
                        # FIX: Use adaptive variance floor instead of fixed 1.0
                        cluster_var = np.maximum(cluster_var, adaptive_var_floor)
                        log_var_c[k] = torch.log(
                            torch.tensor(cluster_var, dtype=torch.float32)
                        )
                    else:
                        # FIX: Use adaptive floor for empty clusters too
                        log_var_c[k] = torch.full((self.latent_dim,), np.log(adaptive_var_floor))
                        empty_clusters.append(k)

                if empty_clusters:
                    self.logger.warning(f"[INIT] Empty clusters at initialization: {empty_clusters}")

                self.mu_c.data = mu_c.to(self.device)
                self.log_var_c.data = log_var_c.to(self.device)

                # FIX: Keep pi_ uniform at initialization
                self.pi_.data = torch.log(
                    torch.ones(self.nClusters, device=self.device) / self.nClusters
                )

                # DEBUG: Log final cluster parameters
                self.logger.info(f"[INIT] Cluster centers (mu_c) stats - mean: {mu_c.mean():.6f}, std: {mu_c.std():.6f}")
                self.logger.info(f"[INIT] Cluster log_var stats - mean: {log_var_c.mean():.6f}, std: {log_var_c.std():.6f}")
                self.logger.info(f"[INIT] Pi weights (should be uniform): {torch.exp(self.pi_).detach().cpu().numpy()}")

                self._needs_data_init = False
                self.logger.info("✅ [INIT] Clusters initialized from data distribution with adaptive variance")
            self.set_train_mode()

    def check_prior_health(self):
        weights = torch.exp(self.pi_).detach().cpu().numpy()
        self.logger.info(f"Cluster weights: {weights}")
        self.logger.info(f"Cluster means std: {self.mu_c.std().item():.6f}")
        self.logger.info(
            f"Log var range: [{self.log_var_c.min().item():.3f}, {self.log_var_c.max().item():.3f}]"
        )
        if np.max(weights) > 0.95:
            self.logger.warning("One cluster dominates (weight > 95%)")
        if self.mu_c.std() < 0.01:
            self.logger.warning("Cluster means are too similar")

    def check_latent_space_health(self, data_loader):
        self.eval()
        all_mu = []
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i > 10:
                    break
                data = data.to(self.device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                mu, _ = self.encode(data)
                all_mu.append(mu.cpu())
        if all_mu:
            all_mu = torch.cat(all_mu, 0)
            latent_std = all_mu.std(0).mean().item()
            self.logger.info(f"Average latent dimension std: {latent_std:.6f}")
            if latent_std < 0.01:
                self.logger.warning("Latent space has collapsed (low variance)")
                return False
        return True

    def cluster_separation_loss(self):
        """
        Continuous centroid repulsion via mean absolute correlation.

        Always provides gradient signal proportional to centroid similarity.
        Unlike the previous log-barrier (which went to zero once centroids
        spread past a threshold), this maintains gentle pressure throughout
        training, preventing centroids from drifting back together.
        """
        try:
            decoded = self.decode(self.mu_c)  # (K, 1, 40, 40)
            decoded_flat = decoded.view(self.nClusters, -1)  # (K, 1600)

            # Center each decoded topomap (remove spatial mean)
            centered = decoded_flat - decoded_flat.mean(dim=1, keepdim=True)

            # Compute pairwise correlation matrix
            norms = torch.norm(centered, dim=1, keepdim=True).clamp(min=1e-8)
            normalized = centered / norms
            corr_matrix = torch.mm(normalized, normalized.t())  # (K, K)

            # Take absolute correlation (polarity-invariant)
            abs_corr = torch.abs(corr_matrix)

            # Mask diagonal
            mask = torch.eye(self.nClusters, device=self.device).byte()
            abs_corr = abs_corr.masked_fill(mask, 0.0)

            # Mean absolute correlation over unique pairs
            n_pairs = self.nClusters * (self.nClusters - 1) / 2
            separation_loss = abs_corr.sum() / (2 * max(n_pairs, 1))

            return separation_loss
        except Exception as e:
            self.logger.warning(f"Cluster separation loss failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        if torch.isnan(mu).any() or torch.isnan(std).any():
            self.logger.warning("NaN detected in reparameterize inputs")
            mu = _nan_to_num(mu, nan=0.0)
            std = _nan_to_num(std, nan=1.0)
        eps = torch.randn_like(std)
        z = eps * std + mu
        z = _nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        return z

    def gaussian_pdf_log(
        self, x: torch.Tensor, mu: torch.Tensor, log_sigma2: torch.Tensor
    ) -> torch.Tensor:
        log_sigma2 = torch.clamp(log_sigma2, min=-10, max=10)
        if (
            torch.isnan(x).any()
            or torch.isnan(mu).any()
            or torch.isnan(log_sigma2).any()
        ):
            self.logger.warning("NaN detected in gaussian_pdf_log inputs")
            x = _nan_to_num(x, nan=0.0)
            mu = _nan_to_num(mu, nan=0.0)
            log_sigma2 = _nan_to_num(log_sigma2, nan=-1.0)
        diff = x - mu
        sigma2 = torch.exp(log_sigma2) + EPSILON
        log_prob = -0.5 * (np.log(2 * np.pi) + log_sigma2 + (diff**2) / sigma2)
        log_prob = torch.sum(log_prob, dim=1)
        log_prob = _nan_to_num(log_prob, nan=-1000.0, posinf=0.0, neginf=-1000.0)
        log_prob = torch.clamp(log_prob, min=-1000.0, max=100.0)
        return log_prob

    def gaussian_pdfs_log(
        self, x: torch.Tensor, mus: torch.Tensor, log_sigma2s: torch.Tensor
    ) -> torch.Tensor:
        G = []
        for c in range(self.nClusters):
            G.append(
                self.gaussian_pdf_log(
                    x, mus[c : c + 1, :], log_sigma2s[c : c + 1, :]
                ).view(-1, 1)
            )
        return torch.cat(G, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            mu, logvar = self.encoder(x)
            if torch.isnan(mu).any() or torch.isnan(logvar).any():
                self.logger.warning("NaN detected in encoder output")
                mu = _nan_to_num(mu, nan=0.0)
                logvar = _nan_to_num(logvar, nan=-1.0)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            z = self.reparameterize(mu, logvar)
            z = _nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
            reconstructed_x = self.decoder(z)
            if torch.isnan(reconstructed_x).any():
                self.logger.warning("NaN detected in decoder output")
                reconstructed_x = _nan_to_num(reconstructed_x, nan=0.5)
            # reconstructed_x = torch.clamp(reconstructed_x, 0.0, 1.0)
            return reconstructed_x, mu, logvar
        except Exception as e:
            self.logger.warning(f"Forward pass failed: {e}")
            batch_size = x.size(0)
            mu = torch.zeros(batch_size, self.latent_dim, device=x.device)
            logvar = torch.full((batch_size, self.latent_dim), -1.0, device=x.device)
            reconstructed_x = torch.zeros_like(x)
            return reconstructed_x, mu, logvar

    def predict_robust(self, x: torch.Tensor) -> np.ndarray:
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                z_mu, z_log_var = self.encode(x)

                # Check for problematic values
                if torch.isnan(z_mu).any() or torch.isnan(z_log_var).any():
                    self.logger.warning(
                        "NaN in latent variables - using fallback prediction"
                    )
                    return np.random.randint(0, self.nClusters, size=x.size(0))

                # Use mean instead of sampling for more stable predictions
                z = z_mu  # Use mean directly instead of sampling

                log_pi = self.pi_
                log_sigma2_c = self.log_var_c
                mu_c = self.mu_c

                # Check cluster parameters
                if torch.isnan(mu_c).any() or torch.isnan(log_sigma2_c).any():
                    self.logger.warning("NaN in cluster parameters")
                    return np.random.randint(0, self.nClusters, size=x.size(0))

                log_probs = self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
                log_yita_c = log_pi.unsqueeze(0) + log_probs

                # FIX: Apply temperature scaling (higher temp = more uniform distribution)
                temperature = getattr(self, 'prediction_temperature', 1.0)
                log_yita_c = log_yita_c / temperature

                # More stable softmax
                log_yita_c = log_yita_c - torch.logsumexp(
                    log_yita_c, dim=1, keepdim=True
                )
                yita_c = torch.exp(log_yita_c)

                pred = torch.argmax(yita_c, dim=1)
                return pred.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.random.randint(0, self.nClusters, size=x.size(0))
        finally:
            if was_training:
                self.train()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        clusters = self.predict_robust(x)

        return clusters

    def predict_from_latent(self, mu: torch.Tensor) -> np.ndarray:
        """Assign cluster labels from pre-computed latent means (no re-encoding)."""
        with torch.no_grad():
            log_probs = self.gaussian_pdfs_log(mu, self.mu_c, self.log_var_c)
            log_yita_c = self.pi_.unsqueeze(0) + log_probs
            temperature = getattr(self, 'prediction_temperature', 1.0)
            log_yita_c = log_yita_c / temperature
            log_yita_c = log_yita_c - torch.logsumexp(log_yita_c, dim=1, keepdim=True)
            pred = torch.argmax(log_yita_c, dim=1)
            return pred.cpu().numpy()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def RE(
        self, recon_x: torch.Tensor, x: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        # No clamping — z-scored data can be negative and > 1
        batch_size = recon_x.size(0)
        flat_size = recon_x[0].numel()
        self.flat_size = flat_size

        recon_flat = recon_x.view(batch_size, -1)
        x_flat = x.view(batch_size, -1)

        # Polarity-invariant reconstruction: min(MSE(x, x̂), MSE(-x, x̂))
        # EEG microstates are polarity-invariant (map A and -A = same brain state).
        # Taking the per-sample minimum removes the incentive for the encoder to
        # preserve sign information, encouraging mu(x) ≈ mu(-x) in latent space.
        mse_pos = F.mse_loss(recon_flat, x_flat, reduction="none").mean(dim=1)
        mse_neg = F.mse_loss(recon_flat, -x_flat, reduction="none").mean(dim=1)
        mse_loss = torch.min(mse_pos, mse_neg).mean()

        # Scale to be comparable with original (multiply by flat_size for gradient magnitude)
        return mse_loss * flat_size

    def KLD(
        self, mu: torch.Tensor, log_var: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        log_var_c = torch.clamp(self.log_var_c, min=self.min_log_var, max=2.0)
        pi = self.pi_
        mu_c = self.mu_c
        z = self.reparameterize(mu, log_var)
        log_pi = pi.unsqueeze(0)
        log_gaussian = self.gaussian_pdfs_log(z, mu_c, log_var_c)
        log_yita_c = log_pi + log_gaussian
        log_sum = torch.logsumexp(log_yita_c, dim=1, keepdim=True)
        log_yita_c = log_yita_c - log_sum
        yita_c = torch.exp(log_yita_c)
        kl_first_term = 0.5 * torch.mean(
            torch.sum(
                yita_c
                * torch.sum(
                    log_var_c.unsqueeze(0)
                    + torch.exp(log_var.unsqueeze(1) - log_var_c.unsqueeze(0))
                    + (mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2)
                    / (torch.exp(log_var_c.unsqueeze(0)) + EPSILON),
                    2,
                ),
                1,
            )
        )
        kl_second_term = torch.mean(torch.sum(yita_c * (log_yita_c - log_pi), 1))
        entropy_term = 0.5 * torch.mean(torch.sum(1 + log_var, 1))
        loss = kl_first_term + kl_second_term - entropy_term
        loss = torch.max(loss, torch.tensor(EPSILON, device=loss.device))
        return loss

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        epoch: int = 0,
        total_epochs: int = 100,
        normalize: bool = True,
        batch_idx: int = 0,
        n_batches_per_epoch: int = 1000,
        is_pretraining: bool = False,
        beta_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalize = True
        try:
            reconst_loss = self.RE(recon_x, x, normalize=False)
            if torch.isnan(reconst_loss) or torch.isinf(reconst_loss):
                reconst_loss = torch.tensor(
                    1000.0, device=recon_x.device, requires_grad=True
                )
            kld_loss = self.KLD(mu, log_var, normalize=normalize)
            if torch.isnan(kld_loss) or torch.isinf(kld_loss):
                kld_loss = torch.tensor(100.0, device=mu.device, requires_grad=True)
            if beta_override is not None:
                beta = float(beta_override)
            elif self.use_batch_cyclical:
                beta = self.loss_balancer.get_beta(
                    epoch,
                    reconst_loss,
                    kld_loss,
                    batch_idx=batch_idx,
                    n_batches_per_epoch=n_batches_per_epoch,
                )
            elif self.use_cyclical_annealing:
                beta = self.loss_balancer.cyclical_beta(epoch, total_epochs, n_cycles=4)
            else:
                beta = self.loss_balancer.get_beta(epoch, reconst_loss, kld_loss)

            # Clustering regularization losses (disabled during pretrain — encoder
            # must learn features before clustering pressure is applied)
            _zero = torch.tensor(0.0, device=mu.device)
            if is_pretraining:
                entropy_loss = _zero
                entropy_weight = 0.0
                separation_loss = _zero
                separation_weight = 0.0
                batch_entropy_loss = _zero
                batch_entropy_weight = 0.0
                cluster_tightening = _zero
                cluster_tightening_weight = 0.0
            else:
                # Warmup: linearly ramp clustering weights over first N main-training epochs
                warmup_epochs = getattr(self, '_clustering_warmup_epochs', 10)
                warmup_delay = getattr(self, '_clustering_warmup_delay', 0)
                effective_epoch = max(0, epoch - warmup_delay)
                warmup_scale = min(1.0, effective_epoch / max(warmup_epochs, 1))

                # Log warmup state once per epoch (first batch only)
                if batch_idx == 0:
                    self.logger.info(
                        f"[WARMUP] epoch={epoch} delay={warmup_delay} "
                        f"effective_epoch={effective_epoch} scale={warmup_scale:.3f}"
                    )

                # 1. Entropy regularization - encourages uniform cluster usage
                entropy_loss = self.cluster_entropy_regularization()
                entropy_weight = 0.3 / max(np.log(float(self.nClusters)), 1.0) * warmup_scale

                # 2. Cluster separation loss - continuous centroid repulsion
                separation_loss = self.cluster_separation_loss()
                separation_weight = getattr(self, '_separation_weight', 50.0) * warmup_scale

                # 3. Batch-level cluster entropy — penalizes non-uniform usage within mini-batch
                batch_entropy_weight = getattr(self, '_batch_entropy_weight', 5.0) * warmup_scale
                batch_entropy_loss = self.batch_cluster_entropy(mu, log_var) if batch_entropy_weight > 0 else _zero

                # 4. Augmented ELBO — tighten each sample's posterior to its assigned cluster
                cluster_tightening_weight = getattr(self, '_cluster_tightening_weight', 0.2) * warmup_scale
                cluster_tightening = self.cluster_tightening_loss(mu, log_var) if cluster_tightening_weight > 0 else _zero

            total_loss = (
                reconst_loss
                + kld_loss * beta
                + entropy_loss * entropy_weight
                + separation_loss * separation_weight
                + batch_entropy_loss * batch_entropy_weight
                + cluster_tightening * cluster_tightening_weight
            )
            total_loss = torch.clamp(total_loss, min=EPSILON, max=10000.0)

            # DEBUG: Log loss components periodically (every 100 batches)
            if batch_idx % 100 == 0 and batch_idx > 0:
                _ws = f" warmup={warmup_scale:.2f}" if not is_pretraining else " PRETRAIN(no clustering)"
                self.logger.info(f"[LOSS] Epoch {epoch}, Batch {batch_idx}:{_ws}")
                self.logger.info(f"[LOSS]   - Recon: {reconst_loss.item():.4f}")
                self.logger.info(f"[LOSS]   - KLD: {kld_loss.item():.4f} (beta={beta:.4f}, weighted={kld_loss.item()*beta:.4f})")
                self.logger.info(f"[LOSS]   - Entropy: {entropy_loss.item():.4f} (weighted={entropy_loss.item()*entropy_weight:.4f})")
                self.logger.info(f"[LOSS]   - Separation: {separation_loss.item():.4f} (weighted={separation_loss.item()*separation_weight:.4f})")
                self.logger.info(f"[LOSS]   - BatchEntropy: {batch_entropy_loss.item():.4f} (weighted={batch_entropy_loss.item()*batch_entropy_weight:.4f})")
                self.logger.info(f"[LOSS]   - ClusterTight: {cluster_tightening.item():.4f} (weighted={cluster_tightening.item()*cluster_tightening_weight:.4f})")
                self.logger.info(f"[LOSS]   - Total: {total_loss.item():.4f}")
        except Exception as e:
            self.logger.warning(f"Loss computation failed: {e}")
            reconst_loss = torch.tensor(
                1000.0, device=recon_x.device, requires_grad=True
            )
            kld_loss = torch.tensor(100.0, device=mu.device, requires_grad=True)
            total_loss = torch.tensor(1100.0, device=recon_x.device, requires_grad=True)
        return reconst_loss, kld_loss, total_loss

    # Add these two new methods to your MyModel class

    def freeze_prior(self) -> None:
        for param in [self.pi_, self.mu_c, self.log_var_c]:
            param.requires_grad = False
        self.prior_frozen = True

    def unfreeze_prior(self) -> None:
        for param in [self.pi_, self.mu_c, self.log_var_c]:
            param.requires_grad = True
        self.prior_frozen = False

    def staged_unfreeze(self, relative_epoch: int) -> None:
        """Staged prior unfreeze: mu_c@+0, log_var_c@+2, pi_@+5."""
        if relative_epoch == 0:
            self.mu_c.requires_grad = True
            self.logger.info("[STAGED_UNFREEZE] mu_c unfrozen")
        if relative_epoch == 2:
            self.log_var_c.requires_grad = True
            self.logger.info("[STAGED_UNFREEZE] log_var_c unfrozen")
        if relative_epoch == 5:
            self.pi_.requires_grad = True
            self.prior_frozen = False
            self.logger.info("[STAGED_UNFREEZE] pi_ unfrozen — all GMM params active")

    def gradual_unfreeze(self, epoch: int) -> None:
        if epoch >= 2:
            self.pi_.requires_grad = True
        if epoch >= 3:
            self.mu_c.requires_grad = True
            self.log_var_c.requires_grad = True
            self.prior_frozen = False

    def _apply_circular_mask_single_torch(self, image, radius_factor=0.95):
        if image.device.type == "cuda":
            image = image.cpu()
        height, width = image.shape
        center_y, center_x = height // 2, width // 2
        radius = int(min(center_x, center_y) * radius_factor)
        y = (
            torch.arange(height, device="cpu")
            .float()
            .unsqueeze(1)
            .expand(height, width)
        )
        x = torch.arange(width, device="cpu").float().unsqueeze(0).expand(height, width)
        dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = (dist <= radius).float()
        return image * mask

    # =========================================================================
    # Visualization wrappers — delegate to model_viz.py standalone functions
    # =========================================================================

    def get_cluster_centroids_and_visualize(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.get_cluster_centroids_and_visualize(self, *args, **kwargs)

    def _visualize_centroids_mne_topomap(self, *args, **kwargs):
        import model_viz as _mv
        return _mv._visualize_centroids_mne_topomap(self, *args, **kwargs)

    def _visualize_centroid_spatial_correlations(self, *args, **kwargs):
        import model_viz as _mv
        return _mv._visualize_centroid_spatial_correlations(self, *args, **kwargs)

    def plot_segmentation(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.plot_segmentation(self, *args, **kwargs)

    def plot_microstate_statistics(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.plot_microstate_statistics(self, *args, **kwargs)

    def pycrostates_plot_cluster_centers(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.pycrostates_plot_cluster_centers(self, *args, **kwargs)

    def pycrostates_plot_raw_segmentation(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.pycrostates_plot_raw_segmentation(self, *args, **kwargs)

    def evaluate_on_raw(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.evaluate_on_raw(self, *args, **kwargs)

    def _plot_raw_statistics(self, *args, **kwargs):
        import model_viz as _mv
        return _mv._plot_raw_statistics(self, *args, **kwargs)

    def perform_research_analysis(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.perform_research_analysis(self, *args, **kwargs)

    def visualize_latent_space(self, *args, **kwargs):
        import model_viz as _mv
        return _mv.visualize_latent_space(self, *args, **kwargs)

    def pretrain(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        train_set: torch.utils.data.Dataset,
        epochs: int = 30,
        gamma_steps: int = 1000,
        initial_gamma: float = 0.1,
        freeze_prior_epochs: int = 0,
        evaluate_every: int = 5,
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None,
    ) -> dict:
        if device is None:
            device = next(self.parameters()).device

        pretrain_checkpoint_path = None
        if output_dir:
            pretrain_checkpoint_path = output_dir / "pretrain_checkpoint.pth"

        start_step = 0
        start_epoch = 0
        phase = "gamma"
        history = {
            "epoch_losses": [],
            "reconstruct_losses": [],
            "kld_losses": [],
            "nmi_scores": [],
            "ari_scores": [],
            "silhouette_scores": [],
            "db_scores": [],
            "ch_scores": [],
            "beta_values": [],
        }
        collapse_counter = 0

        # --- RESUME LOGIC ---
        if pretrain_checkpoint_path and pretrain_checkpoint_path.exists():
            self.logger.info(f"Resuming pretraining from {pretrain_checkpoint_path}")
            try:
                checkpoint = torch.load(
                    pretrain_checkpoint_path, map_location=device, weights_only=False
                )
            except TypeError:
                # PyTorch < 1.13 doesn't support weights_only parameter
                checkpoint = torch.load(
                    pretrain_checkpoint_path, map_location=device
                )
            self.load_state_dict(checkpoint["model_state_dict"])
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError):
                self.logger.warning(
                    "Pretrain optimizer state incompatible (param group change) — reinitializing"
                )
            phase = checkpoint.get("phase", "gamma")
            start_step = checkpoint.get("step_count", 0)
            start_epoch = checkpoint.get("epoch", 0)
            history = checkpoint.get("history", history)
            collapse_counter = checkpoint.get("collapse_counter", 0)
            if "loss_balancer_state" in checkpoint:
                self.loss_balancer.set_state(checkpoint["loss_balancer_state"])
            self.logger.info(
                f"Resuming from phase '{phase}', step {start_step}, epoch {start_epoch}"
            )

        self.train()
        # Use unaugmented train_set for cluster init to avoid polarity bias
        init_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.initialize_from_data(init_loader)

        if freeze_prior_epochs > 0 and start_epoch < freeze_prior_epochs:
            self.freeze_prior()
            self.logger.info(f"Prior frozen for {freeze_prior_epochs} epochs")
        else:
            self.logger.info("Prior NOT frozen - training with all parameters active")

        step_count = start_step if phase == "gamma" else gamma_steps
        n_batches_per_epoch = len(train_loader)

        # --- PHASE 1: GAMMA STEPS ---
        if phase == "gamma":
            self.logger.info(f"Starting/Resuming gamma steps from step {start_step}")
            gamma_progress_bar = tqdm(
                total=gamma_steps, initial=start_step, desc="Gamma Steps"
            )

            gamma_complete = False
            while not gamma_complete:
                for batch_idx, (data, _) in enumerate(train_loader):
                    if step_count < start_step:
                        if (batch_idx + 1) * data.size(0) > step_count:
                            pass
                        else:
                            continue

                    if step_count >= gamma_steps:
                        gamma_complete = True
                        break

                    data = data.to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = self(data)

                    # Simple pretraining loss
                    re_loss = self.RE(recon_batch, data)
                    kld_loss = self.KLD(mu, logvar)
                    loss = re_loss + (1e-3 * kld_loss)

                    loss.backward()
                    optimizer.step()

                    if step_count % 50 == 0:
                        kl_healthy = self.monitor_kl_health(mu, logvar)
                        if not kl_healthy:
                            collapse_counter += 1

                        current_lr = optimizer.param_groups[0]["lr"]
                        self.logger.info(
                            f"\n{'='*80}\n"
                            f"GAMMA STEP {step_count:4d} | LR: {current_lr:.2e}\n"
                            f"{'='*80}\n"
                            f"Loss: {loss.item():.4f} (RE: {re_loss.item():.4f}, KLD: {kld_loss.item():.4f})\n"
                        )

                    history["reconstruct_losses"].append(re_loss.item())
                    history["kld_losses"].append(kld_loss.item())
                    history["epoch_losses"].append(loss.item())

                    step_count += 1
                    gamma_progress_bar.update(1)

                    if pretrain_checkpoint_path and step_count % 50 == 0:
                        torch.save(
                            {
                                "phase": "gamma",
                                "step_count": step_count,
                                "epoch": 0,
                                "model_state_dict": self.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "history": history,
                                "collapse_counter": collapse_counter,
                                "loss_balancer_state": self.loss_balancer.get_state(),
                            },
                            pretrain_checkpoint_path,
                        )

                if not gamma_complete:
                    self.logger.info("Restarting dataloader to continue gamma steps...")

            gamma_progress_bar.close()
            phase = "epochs"
            start_epoch = 0

        # --- PHASE 2: EPOCH TRAINING ---
        if self.detect_posterior_collapse(train_loader, threshold=0.1):
            self.logger.warning(
                "🚨 Collapse detected after gamma steps - adjusting parameters"
            )
            self.loss_balancer.min_beta *= 0.1
            self.loss_balancer.max_beta *= 0.5
            self.loss_balancer.gamma *= 0.5
            collapse_counter += 1

        for epoch in range(start_epoch, freeze_prior_epochs):
            self.gradual_unfreeze(epoch)
            epoch_loss, reconstruct_loss, kld_loss_total = 0, 0, 0
            epoch_betas = []

            for batch_idx, (data, _) in enumerate(
                tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}")
            ):
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)

                re_loss, kld_loss, loss = self.loss_function(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    epoch=epoch + 1,
                    normalize=True,
                    total_epochs=epochs + 1,
                    batch_idx=batch_idx,
                    n_batches_per_epoch=n_batches_per_epoch,
                    is_pretraining=True,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                reconstruct_loss += re_loss.item()
                kld_loss_total += kld_loss.item()

                if self.use_batch_cyclical:
                    current_beta = self.loss_balancer.get_beta(
                        epoch + 1, re_loss, kld_loss, batch_idx, n_batches_per_epoch
                    )
                    epoch_betas.append(current_beta)

            avg_loss = epoch_loss / len(train_loader)
            avg_beta = np.mean(epoch_betas) if epoch_betas else 0.0

            history["epoch_losses"].append(avg_loss)
            history["beta_values"].append(avg_beta)

            self.logger.info(f"Epoch {epoch+1} Complete | Loss: {avg_loss:.4f}")

            # FIX: Check for dead clusters every 5 epochs and reinitialize them
            if (epoch + 1) % 5 == 0:
                self._check_and_reinitialize_dead_clusters(train_loader, device)

            if pretrain_checkpoint_path:
                torch.save(
                    {
                        "phase": "epochs",
                        "step_count": step_count,
                        "epoch": epoch + 1,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history,
                        "collapse_counter": collapse_counter,
                        "loss_balancer_state": self.loss_balancer.get_state(),
                    },
                    pretrain_checkpoint_path,
                )

        if freeze_prior_epochs > 0:
            self.gradual_unfreeze(freeze_prior_epochs)

        # ---------------------------------------------------------
        # HYBRID STRATEGY: Try 100%, Fallback to Safe Subset
        # Uses train_set (unaugmented) so KMeans/GMM sees clean data
        # without polarity-doubled pairs that bias centroids to origin.
        # ---------------------------------------------------------
        self.logger.info("Preparing data for GMM Initialization (unaugmented)...")
        self.set_eval_mode()

        # Create DataLoader from unaugmented train_set for clean GMM init
        gmm_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        try:
            # 1. Try to load 100% of unaugmented data
            Z, Y = [], []
            for data, y in tqdm(gmm_loader, desc="Extracting features (unaugmented)"):
                with torch.no_grad():
                    data = data.to(device)
                    mu, _ = self.encode(data)
                    Z.append(mu.cpu())
                    Y.append(y.cpu())

            Z_cat = torch.cat(Z, 0).numpy()
            self.logger.info(f"Extracted {len(Z_cat)} unaugmented samples for GMM init.")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning(
                    "OOM extracting 100% data. Switching to 20k subset."
                )
                torch.cuda.empty_cache()

                # Fallback: Safe Subset of 20,000 from unaugmented data
                subset_size = min(20000, len(train_set))
                subset_indices = np.random.choice(
                    len(train_set), size=subset_size, replace=False
                )
                subset_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train_set, subset_indices),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                Z, Y = [], []
                for data, y in tqdm(subset_loader, desc="Extracting SUBSET features"):
                    with torch.no_grad():
                        data = data.to(device)
                        mu, _ = self.encode(data)
                        Z.append(mu.cpu())
                        Y.append(y.cpu())
                Z_cat = torch.cat(Z, 0).numpy()
            else:
                raise e

        # 2. Fit K-Means (bisecting for K>6, flat otherwise)
        self.logger.info(f"Fitting KMeans (K={self.nClusters})...")
        kmeans_centers = self._bisecting_kmeans(Z_cat, self.nClusters, base_k=4)

        # 3. Compute cluster stats directly from KMeans assignments (no GMM — avoids EM collapse at high K)
        self.logger.info("Computing cluster stats from KMeans assignments...")
        dists = np.sum((Z_cat[:, None, :] - kmeans_centers[None, :, :]) ** 2, axis=2)
        labels = dists.argmin(axis=1)

        # Adaptive variance floor (same logic as before)
        global_latent_std = np.std(Z_cat, axis=0)
        computed_floor = 0.1 * np.mean(global_latent_std) ** 2
        MIN_VAR_FLOOR = 0.1
        adaptive_var_floor = max(computed_floor, MIN_VAR_FLOOR)

        for c in range(self.nClusters):
            mask = labels == c
            count = mask.sum()
            if count > 1:
                cluster_mean = Z_cat[mask].mean(axis=0)
                cluster_var = np.maximum(Z_cat[mask].var(axis=0), adaptive_var_floor)
            else:
                cluster_mean = kmeans_centers[c]
                cluster_var = np.full(self.latent_dim, adaptive_var_floor)
            self.mu_c.data[c] = torch.from_numpy(cluster_mean).float().to(device)
            self.log_var_c.data[c] = torch.from_numpy(np.log(cluster_var + EPSILON)).float().to(device)

        # Pi = cluster proportions (log-space) — aligns prior with data from the start
        counts = np.array([(labels == c).sum() for c in range(self.nClusters)])
        proportions = (counts / counts.sum()).clip(min=1e-6)
        self.pi_.data = torch.from_numpy(np.log(proportions)).float().to(device)

        cluster_sizes = [f"{c}:{(labels==c).sum()}" for c in range(self.nClusters)]
        self.logger.info(f"KMeans init complete (var_floor={adaptive_var_floor:.6f}): {', '.join(cluster_sizes)}")

        self.unfreeze_prior()
        self.train()
        return history

    def _evaluate_clustering(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        Y_cat: np.ndarray,
        Z_cat: np.ndarray,
        history: dict,
        current_epoch: int,
    ) -> None:
        self.eval()
        Z, Y, predictions = [], [], []
        with torch.no_grad():
            for data, y in tqdm(data_loader, desc="Evaluating clustering"):
                data = data.to(device)
                # data = (data - data.min()) / (data.max() - data.min() + EPSILON)
                batch_predictions = self.predict(data)
                predictions.append(batch_predictions)
                mu, _ = self.encode(data)
                Z.append(mu)
                Y.append(y)
        Z_cat = torch.cat(Z, 0).detach().cpu().numpy()
        Y_cat = torch.cat(Y, 0).detach().cpu().numpy()
        cluster_preds = np.concatenate(predictions, axis=0)
        # Define Y_cat_1d before conditional block to ensure it's always available
        Y_cat_1d = Y_cat.reshape(-1) if len(Y_cat.shape) > 1 else Y_cat
        if len(Y_cat) > 0:
            try:
                from sklearn.metrics import (
                    normalized_mutual_info_score,
                    adjusted_rand_score,
                    silhouette_score,
                    davies_bouldin_score,
                    calinski_harabasz_score,
                )

                nmi = normalized_mutual_info_score(Y_cat_1d, cluster_preds)
                ari = adjusted_rand_score(Y_cat_1d, cluster_preds)
                metrics = {}
                if len(np.unique(cluster_preds)) >= 2:
                    metrics["silhouette"] = silhouette_score(Z_cat, cluster_preds)
                    metrics["db_index"] = davies_bouldin_score(Z_cat, cluster_preds)
                    metrics["ch_index"] = calinski_harabasz_score(Z_cat, cluster_preds)
                else:
                    metrics["silhouette"], metrics["db_index"], metrics["ch_index"] = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                    self.logger.warning(
                        "Only one cluster found, clustering metrics cannot be computed"
                    )
                history["silhouette_scores"].append(metrics["silhouette"])
                history["db_scores"].append(metrics["db_index"])
                history["ch_scores"].append(metrics["ch_index"])
                history["nmi_scores"].append(nmi)
                history["ari_scores"].append(ari)
                silhouette_val = (
                    metrics["silhouette"]
                    if not np.isnan(metrics["silhouette"])
                    else float("nan")
                )

                db_str = f"{metrics['db_index']:.6f}" if not np.isnan(metrics['db_index']) else "N/A"
                ch_str = f"{metrics['ch_index']:.6f}" if not np.isnan(metrics['ch_index']) else "N/A"
                self.logger.info(
                    f"Epoch {current_epoch} - nmi: {nmi:.6f}, ari: {ari:.6f}, "
                    f"silhouette: {silhouette_val:.6f}, db: {db_str}, ch: {ch_str}"
                )
            except Exception as e:
                self.logger.error(f"Error computing clustering metrics: {e}")
                history["silhouette_scores"].append(float("nan"))
                history["db_scores"].append(float("nan"))
                history["ch_scores"].append(float("nan"))
                history["nmi_scores"].append(float("nan"))
                history["ari_scores"].append(float("nan"))
        try:
            self.visualize_latent_space(
                Z_cat, cluster_preds, Y_cat_1d, context="evaluation"
            )
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
        self.train()


def create_model_with_batch_cyclical(
    latent_dim: int = 10,
    nClusters: int = 8,
    batch_size: int = 128,
    logger=None,
    device=None,
    n_cycles_per_epoch: int = 5,
    cycle_ratio: float = 0.5,
    gamma: float = 0.01,
    info=None,
    n_total_cycles=None,
    max_beta: float = None,
    ndf: int = 64,
    ngf: int = 64,
    n_conv_layers: int = 4,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        import logging

        logger = logging.getLogger("vae_clustering")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    model = MyModel(
        latent_dim=latent_dim,
        nClusters=nClusters,
        batch_size=batch_size,
        logger=logger,
        device=device,
        use_cyclical_annealing=False,
        use_batch_cyclical=True,
        nc=1,
        ndf=ndf,
        ngf=ngf,
        info=info,
        n_conv_layers=n_conv_layers,
    )
    model.loss_balancer.n_cycles_per_epoch = n_cycles_per_epoch
    model.loss_balancer.cycle_ratio = cycle_ratio
    model.loss_balancer.gamma = gamma
    if n_total_cycles is not None:
        model.loss_balancer.n_total_cycles = n_total_cycles
    if max_beta is not None:
        model.loss_balancer.max_beta = max_beta
    # Global schedule is built lazily at first train_epoch (needs n_batches_per_epoch)
    model.loss_balancer.global_schedule = None
    logger.info("Created VAE model with batch-level cyclical annealing:")
    logger.info(f"  - Latent dimensions: {latent_dim}")
    logger.info(f"  - Number of clusters: {nClusters}")
    logger.info(f"  - Conv layers: {n_conv_layers}, Channel width (ndf): {ndf}")
    if n_total_cycles is not None:
        logger.info(f"  - Total cycles (global): {n_total_cycles}")
    else:
        logger.info(f"  - Cycles per epoch: {n_cycles_per_epoch}")
    logger.info(f"  - Cycle ratio: {cycle_ratio}")
    logger.info(f"  - Gamma (beta floor): {gamma}")
    logger.info(
        f"  - Beta range: [{model.loss_balancer.min_beta}, {model.loss_balancer.max_beta}]"
    )
    return model
