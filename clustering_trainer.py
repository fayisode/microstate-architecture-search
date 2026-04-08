import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, List, Union, Any, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import gc
import json
import time
from collections import defaultdict
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
import model as _m
import helper_function as _g
import metrics_utils as _mu
import centroid_metrics as _cm

# Lazy imports — not needed in fast-sweep mode (CPU-only, no viz/baseline)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import baseline as _b
except ImportError:
    _b = None
try:
    import centroid_analysis as _ca
except ImportError:
    _ca = None
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from trainer_metrics import TrainerMetricsMixin
from trainer_viz import TrainerVizMixin
from trainer_diagnostics import TrainerDiagnosticsMixin


def _fmt(val, fmt=".4f", default="N/A"):
    """Safely format a value that might be None."""
    if val is None:
        return default
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return default


def clear_gpu(logger=None):
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        if logger:
            logger.warning(f"Error clearing GPU memory: {e}")


def check_data_preprocessing(data_loader, device, logger=None):
    if logger:
        logger.info("🔍 CHECKING DATA PREPROCESSING")
        logger.info("=" * 40)
    sample_batch = next(iter(data_loader))[0].to(device)
    stats = {
        "shape": list(sample_batch.shape),
        "min": sample_batch.min().item(),
        "max": sample_batch.max().item(),
        "mean": sample_batch.mean().item(),
        "std": sample_batch.std().item(),
        "nan_count": torch.isnan(sample_batch).sum().item(),
        "inf_count": torch.isinf(sample_batch).sum().item(),
        "zero_count": (sample_batch == 0).sum().item(),
    }
    if logger:
        logger.info(f"Raw data shape: {stats['shape']}")
        logger.info(f"Raw data range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        logger.info(f"Raw data mean: {stats['mean']:.6f}")
        logger.info(f"Raw data std: {stats['std']:.6f}")
        logger.info(f"NaN values: {stats['nan_count']}")
        logger.info(f"Inf values: {stats['inf_count']}")
        logger.info(
            f"Zero values: {stats['zero_count']} ({100*stats['zero_count']/sample_batch.numel():.1f}%)"
        )
    return stats


def full_diagnosis(model, train_loader, device, logger=None):
    if logger:
        logger.info("🏥 COMPREHENSIVE CLUSTER DIAGNOSIS")
        logger.info("=" * 80)
    stats = check_data_preprocessing(train_loader, device, logger)
    if logger:
        logger.info("🏃 Quick feature extraction training (10 epochs)...")
    cluster_results = {}
    recommendations = []
    if cluster_results.get("optimal_k_silhouette"):
        recommendations.append(
            f"✅ Use {cluster_results['optimal_k_silhouette']} clusters (silhouette optimal)"
        )
    if logger:
        logger.info("💡 RECOMMENDATIONS:")
        logger.info("=" * 40)
        for rec in recommendations:
            logger.info(rec)
    return {
        "data_stats": stats,
        "cluster_analysis": cluster_results,
        "recommendations": recommendations,
    }


class ClusteringFailureError(Exception):
    pass


class VAEClusteringTrainer(TrainerMetricsMixin, TrainerVizMixin, TrainerDiagnosticsMixin):
    def __init__(
        self,
        model: _m.MyModel,
        train_loader: DataLoader,
        train_set: Any,
        config: Dict[str, Any],
        device: torch.device,
        cluster_id: int = 0,
        logger: Any = None,
        gfp_peaks=None,
        raw_mne=None,
        eval_loader: DataLoader = None,
        eval_set: Any = None,
        norm_params=None,
    ):
        self._validate_inputs(
            model, train_loader, config, device, logger
        )

        self.model = model
        self.train_loader = train_loader
        self.train_set = train_set
        self.eval_loader = eval_loader    # Held-out evaluation set
        self.eval_set = eval_set
        self.config = config
        self.device = device
        self.cluster_id = cluster_id
        self.logger = logger
        self.gfp_peaks = gfp_peaks  # For baseline ModKMeans
        self.raw_mne = raw_mne  # For segmentation plots
        self.norm_params = norm_params  # Z-score params for centroid denormalization
        self.logger.info(
            f"Trainer initialized with batch size: {self.train_loader.batch_size}"
        )
        self.polarity_weight = config.get("polarity_weight", 0.1)
        self.lr = config.get("learning_rate", 1e-3)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 40)
        self.pretrain_epochs = config.get("pretrain_epochs", 30)
        self.unfreeze_prior_epoch = config.get("unfreeze_prior_epoch", 1)
        # self.unfreeze_prior_epoch = config.get(
        #     "unfreeze_prior_epoch", max(15, int(self.epochs))
        # )

        # Dual parameter groups: lower LR for GMM prior params (canonical VaDE practice)
        gmm_lr_ratio = config.get("gmm_lr_ratio", 0.5)
        gmm_param_ids = {id(self.model.pi_), id(self.model.mu_c), id(self.model.log_var_c)}
        nn_params = [p for p in self.model.parameters() if id(p) not in gmm_param_ids]
        gmm_params = [self.model.pi_, self.model.mu_c, self.model.log_var_c]
        self.nn_params = nn_params
        self.gmm_params = gmm_params
        self.optimizer = optim.Adam([
            {"params": nn_params, "lr": self.lr, "weight_decay": 1e-5},
            {"params": gmm_params, "lr": self.lr * gmm_lr_ratio, "weight_decay": 0},
        ])
        self.logger.info(
            f"Dual optimizer: NN lr={self.lr}, GMM lr={self.lr * gmm_lr_ratio} "
            f"(ratio={gmm_lr_ratio})"
        )
        # Per-group min_lr: NN group can go lower (room to decay after unfreeze LR drop),
        # GMM group floor is higher (GMM params need meaningful updates)
        scheduler_patience = config.get("scheduler_patience", 10)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=0.5,
            min_lr=[1e-5, 1e-4],
            threshold=5e-4,
            threshold_mode="rel",
        )

        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.output_dir / "checkpoint.pth"
        self._setup_strategies()

        self.train_losses = []
        self.no_improvement_counter = 0
        self.current_epoch = 0
        self.best_train_loss = float("inf")
        self.best_train_epoch = 0
        self.best_gev = 0.0  # GEV-based early stopping (higher is better)
        self.gev_history = []  # Track GEV over epochs
        self.early_stopping_metric = config.get("early_stopping_metric", "gev")
        valid_metrics = {"gev", "composite", "silhouette"}
        if self.early_stopping_metric not in valid_metrics:
            self.logger.warning(
                f"Unknown early_stopping_metric '{self.early_stopping_metric}', "
                f"falling back to 'gev'. Valid: {valid_metrics}"
            )
            self.early_stopping_metric = "gev"
        # Initialize to metric's theoretical minimum so first valid epoch always saves
        self.best_stopping_value = -1.0 if self.early_stopping_metric == "silhouette" else 0.0

        # Centroid-based metrics tracker
        self.centroid_metrics = _cm.CentroidMetrics()
        self.centroid_metrics_history = []

        # Assignment stability tracking (DEC-style convergence signal)
        self._prev_labels = None  # Previous epoch's cluster assignments
        self.assignment_change_history = []  # Fraction of labels that changed per epoch
        self.centroid_metrics_log_interval = config.get("centroid_metrics_interval", 5)  # Log every N epochs

        # Eval reconstruction loss history (for overfitting monitoring)
        self.eval_recon_losses = []

        # SSIM data range: for z-scored data clipped at +/-clip_std, range = 2*clip_std
        # z_score_clip lives in [eeg] config, not [vae], so fall back to direct config read
        z_clip = config.get("z_score_clip", None)
        if z_clip is None:
            from config.config import config as _cfg
            z_clip = _cfg.get_eeg_config().get("z_score_clip", 5.0)
        self.ssim_data_range = 2.0 * float(z_clip)  # default 10.0 for +/-5 std

        # In-memory epoch metrics (flushed to JSON on checkpoint save and training end)
        self.epoch_metrics_history = []

        self.start_epoch = 1
        self.pretrain_epochs = config.get("pretrain_epochs", 30)
        self.gamma_steps = config.get("gamma_steps", len(self.train_loader))
        self.initial_gamma = config.get("initial_gamma", 0.1)
        self.freeze_prior_epochs = config.get("freeze_prior_epochs", 5)

        if self.checkpoint_path.exists():
            self.logger.info(
                f"Existing checkpoint found at {self.checkpoint_path}. Attempting to resume."
            )
            try:
                last_completed_epoch = self.load_checkpoint()
                self.start_epoch = last_completed_epoch + 1
                self.logger.info(
                    f"Successfully resumed. Will start training from epoch {self.start_epoch}."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load checkpoint: {e}. Starting from scratch."
                )

    def _setup_strategies(self):
        # Simplified: just track best model path
        self.best_model_path = self.output_dir / "best_model.pth"

    def _validate_inputs(
        self, model, train_loader, config, device, logger
    ):
        if model is None:
            raise ValueError("Model cannot be None")
        if not all(
            hasattr(model, m)
            for m in ["pretrain", "predict", "encode", "loss_function", "RE", "KLD"]
        ):
            raise ValueError("Model is missing required methods")
        if logger is None:
            raise ValueError("Logger cannot be None")
        if device is None:
            raise ValueError("Device cannot be None")

    def pretrain(self):
        try:
            self.logger.info(
                "=" * 60 + "\nSTEP 1: PRETRAINING & DIAGNOSTICS\n" + "=" * 60
            )
            # deepcopy model for diagnosis — temporarily remove logger (contains RLock, unpicklable)
            model_logger = getattr(self.model, 'logger', None)
            if model_logger is not None:
                self.model.logger = None
            model_copy = deepcopy(self.model)
            if model_logger is not None:
                self.model.logger = model_logger
                model_copy.logger = model_logger
            diagnosis_results = full_diagnosis(
                model_copy, self.train_loader, self.device, self.logger
            )
            del model_copy
            with open(self.output_dir / "cluster_diagnosis.json", "w") as f:
                json.dump(
                    self._convert_to_json_serializable(diagnosis_results), f, indent=4
                )

            self.logger.info(
                f"Diagnosis results saved to {self.output_dir / 'cluster_diagnosis.json'}"
            )

            self.logger.info("Starting pretraining...")
            self.model.pretrain(
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                train_set=self.train_set,
                epochs=self.pretrain_epochs,
                device=self.device,
                gamma_steps=self.gamma_steps,
                initial_gamma=self.initial_gamma,
                freeze_prior_epochs=self.freeze_prior_epochs,
                output_dir=self.output_dir,
            )
            self.logger.info("Pretraining completed successfully")
        except Exception as e:
            self.logger.error(f"Pretraining failed: {e}", exc_info=True)
            raise

    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        self.model.train()
        train_loss, valid_batches = 0.0, 0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch} Training", leave=False
        )
        n_batches = len(self.train_loader)
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            try:
                recon_batch, mu, logvar = self.model(data)
                # During safe-beta window (first 5 epochs after unfreeze),
                # cap beta but still include entropy + separation losses
                beta_ov = None
                if self.unfreeze_prior_epoch <= epoch < self.unfreeze_prior_epoch + 5:
                    beta_ov = max(0.3, self._get_safe_beta(epoch))
                re_loss, kld_loss, loss = self.model.loss_function(
                    recon_batch, data, mu, logvar, epoch, self.epochs,
                    batch_idx=batch_idx,
                    n_batches_per_epoch=n_batches,
                    beta_override=beta_ov,
                )
                # Encoder polarity invariance: mu(x) ≈ mu(-x)
                if self.polarity_weight > 0:
                    mu_neg, _ = self.model.encode(-data)
                    polarity_loss = F.mse_loss(mu, mu_neg)
                    loss = loss + self.polarity_weight * polarity_loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    self.logger.warning(
                        f"Collapse detected in training - loss: {loss.item()}"
                    )
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_params, max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.gmm_params, max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)
                valid_batches += data.size(0)
                progress_bar.set_postfix(
                    {
                        "Total": f"{loss.item():.4f}",
                        "Recon": f"{re_loss.item():.4f}",
                        "KLD": f"{kld_loss.item():.4f}",
                        "Beta": f"{self._get_current_beta(epoch):.4f}",
                    }
                )
            except Exception as e:
                self.logger.warning(f"Batch training error: {e}")
                continue

        if valid_batches == 0:
            return 50.0, {}

        avg_train_loss = train_loss / valid_batches

        # Calculate training metrics after training is complete
        self.logger.debug(f"Computing training metrics for epoch {epoch}...")
        clear_gpu(self.logger)
        train_metrics = self._compute_training_metrics(epoch, 0)
        clear_gpu(self.logger)

        return avg_train_loss, train_metrics

    def _get_safe_beta(self, epoch):
        if hasattr(self.model, "loss_balancer") and self.model.loss_balancer:
            beta = self.model.loss_balancer.get_beta(epoch, 0, 0)  # Dummy values
            # Cap beta very low during first 5 epochs after unfreezing
            if self.unfreeze_prior_epoch <= epoch < self.unfreeze_prior_epoch + 5:
                beta = min(beta, 0.001)
            return beta
        return 0.001

    def _get_current_beta(self, epoch):
        try:
            if hasattr(self.model, "loss_balancer") and self.model.loss_balancer:
                return self.model.loss_balancer.get_beta(epoch, 1.0, 1.0)
            return 1.0
        except Exception:
            return 1.0


    def _safe_beta_calculation(self, epoch, total_epochs, recon_loss, kld_loss):
        beta = 1.0
        if (
            hasattr(self.model, "loss_balancer")
            and self.model.loss_balancer is not None
        ):
            try:
                beta = self.model.loss_balancer.get_beta(epoch, recon_loss, kld_loss)
            except Exception as e:
                self.logger.debug(f"Beta calculation fallback: {e}")
        return float(beta)

    def train_and_evaluate(self):
        self.logger.info(
            "\n" + "=" * 60 + "\nSTEP 2: MAIN TRAINING (100% Data Mode)\n" + "=" * 60
        )
        self.logger.info("Starting training for %d epochs", self.epochs)
        # Always start with prior frozen; staged_unfreeze replays on resume
        self.model.freeze_prior()
        self.logger.info("Prior parameters frozen for initial training stability.")
        # Replay staged unfreeze milestones for resumed training
        if self.start_epoch >= self.unfreeze_prior_epoch and self.unfreeze_prior_epoch > 0:
            for milestone in (0, 2, 5):
                rel = self.start_epoch - self.unfreeze_prior_epoch
                if rel >= milestone:
                    self.model.staged_unfreeze(milestone)
            self.logger.info(f"Replayed staged unfreeze up to relative epoch {rel}")

        # Build global beta schedule once (covers entire training)
        if (hasattr(self.model, 'loss_balancer')
                and getattr(self.model.loss_balancer, 'n_total_cycles', None) is not None):
            self.model.loss_balancer.setup_global_cyclical_schedule(
                len(self.train_loader), self.epochs
            )
            self.logger.info(
                f"Global beta schedule: {self.model.loss_balancer.n_total_cycles} cycles "
                f"over {self.epochs} epochs ({len(self.train_loader)} batches/epoch)"
            )

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            # Staged prior unfreeze at milestones relative to unfreeze_prior_epoch
            if epoch >= self.unfreeze_prior_epoch:
                relative = epoch - self.unfreeze_prior_epoch
                if relative in (0, 2, 5):
                    self.model.staged_unfreeze(relative)
                # Reduce NN LR by 10x only at the first unfreeze epoch
                if relative == 0:
                    self.optimizer.param_groups[0]["lr"] *= 0.1
                    self.logger.info(
                        f"NN lr reduced to {self.optimizer.param_groups[0]['lr']:.2e}, "
                        f"GMM lr stays at {self.optimizer.param_groups[1]['lr']:.2e}"
                    )

            train_loss, train_metrics = self.train_epoch(epoch)

            # Per-dimension KL collapse warning
            collapse_ratio = train_metrics.get('kl_collapse_ratio', 0)
            if collapse_ratio > 0.5:
                latent_dim = self.model.latent_dim
                self.logger.warning(
                    f"POSTERIOR COLLAPSE WARNING: {train_metrics.get('kl_collapsed_dims', 0)}/{latent_dim} "
                    f"latent dims collapsed (KL < 0.05). Cyclical beta may recover this."
                )

            # Compute GEV on eval set for early stopping (or fall back to train)
            if self.eval_loader is not None:
                current_gev = self._compute_gev(self.eval_loader)
                # Train GEV is only for logging — compute every 5 epochs to save time
                if epoch % 5 == 0 or epoch == self.epochs:
                    train_gev = self._compute_gev(self.train_loader)
                    self._last_train_gev = train_gev
                else:
                    train_gev = getattr(self, '_last_train_gev', None)
            else:
                # 100% data mode: train GEV IS the early stopping criterion
                train_gev = self._compute_gev(self.train_loader)
                current_gev = train_gev

            self.gev_history.append(current_gev)

            # Assignment stability tracking (DEC-style convergence signal)
            current_silhouette = train_metrics.get('q1_latent_eucl_silhouette', -1)
            try:
                all_labels = []
                self.model.set_eval_mode()
                with torch.no_grad():
                    eval_source = self.eval_loader if self.eval_loader is not None else self.train_loader
                    for data, _ in eval_source:
                        data = data.to(self.device)
                        labels = self.model.predict(data)
                        all_labels.append(labels)
                current_labels = np.concatenate(all_labels, axis=0)
                if self._prev_labels is not None and len(current_labels) == len(self._prev_labels):
                    change_rate = float(np.mean(current_labels != self._prev_labels))
                    self.assignment_change_history.append(change_rate)
                    self.logger.info(f"[STABILITY] {change_rate:.1%} assignments changed vs prev epoch")
                else:
                    self.assignment_change_history.append(None)
                self._prev_labels = current_labels
            except Exception as e:
                self.logger.warning(f"Assignment stability tracking failed: {e}")
                self.assignment_change_history.append(None)
            finally:
                self.model.set_train_mode()

            # Track best loss for logging
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss

            # Compute the early stopping criterion value (all higher-is-better)
            if self.early_stopping_metric == "composite":
                stopping_value = self._compute_composite_score(current_silhouette, current_gev)
            elif self.early_stopping_metric == "silhouette":
                stopping_value = current_silhouette
            else:  # "gev" (default)
                stopping_value = current_gev

            # Always track best GEV for logging regardless of stopping metric
            if current_gev > self.best_gev:
                self.best_gev = current_gev

            snapped_this_epoch = False
            if stopping_value > self.best_stopping_value:
                self.best_stopping_value = stopping_value
                self.best_train_epoch = epoch
                self.no_improvement_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                gev_label = "eval" if self.eval_loader is not None else "train"
                self.logger.info(
                    f"New best [{self.early_stopping_metric}]: {_fmt(stopping_value)} "
                    f"({gev_label} GEV: {_fmt(current_gev)}, Sil: {_fmt(current_silhouette)}) - Model saved"
                )
                if not self.config.get("fast_sweep", False):
                    try:
                        self._save_tsne_snapshot(epoch)
                        snapped_this_epoch = True
                    except Exception as e:
                        self.logger.warning(f"t-SNE snapshot failed at epoch {epoch}: {e}")
            else:
                self.no_improvement_counter += 1

            self.train_losses.append(train_loss)

            # Lightweight eval reconstruction loss (overfitting monitor)
            eval_recon_loss = self._compute_eval_recon_loss()
            if eval_recon_loss is not None:
                self.eval_recon_losses.append(eval_recon_loss)

            self.logger.info(
                "\n" + "-" * 80 + f"\n[ EPOCH {epoch} / {self.epochs} ]\n" + "-" * 80
            )
            self._log_epoch_summary(train_loss, train_metrics,
                                    eval_gev=current_gev, train_gev=train_gev,
                                    eval_recon_loss=eval_recon_loss)

            assignment_change = self.assignment_change_history[-1] if self.assignment_change_history else None
            self._log_epoch_metrics_to_json(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_gev": train_gev,
                    "eval_gev": current_gev,
                    "eval_recon_loss": eval_recon_loss,
                    "assignment_change_rate": assignment_change,
                    "stopping_metric": self.early_stopping_metric,
                    "stopping_value": stopping_value,
                    **train_metrics,
                }
            )

            # Compute centroid-based metrics every N epochs
            if epoch % self.centroid_metrics_log_interval == 0 or epoch == self.epochs:
                self.logger.info(f"\n[CENTROID METRICS] Computing centroid-based metrics (epoch {epoch})...")
                try:
                    train_centroid_metrics = self._compute_centroid_based_metrics(
                        self.train_loader, split="train", save_to_file=False
                    )
                    # Log summary
                    latent_metrics = train_centroid_metrics["latent_space"]["centroid_based_metrics"]
                    latent_sil_sk = latent_metrics["sklearn"]["silhouette"]
                    latent_sil_corr = latent_metrics["correlation_based"]["silhouette"]
                    self.logger.info(f"[CENTROID METRICS] Latent space - Silhouette: sklearn={_fmt(latent_sil_sk)}, correlation={_fmt(latent_sil_corr)}")

                    # Log composite scores if available
                    history_entry = {
                        "epoch": epoch,
                        "gev": train_centroid_metrics.get("gev"),
                        "latent_silhouette_sklearn": latent_sil_sk,
                        "latent_silhouette_correlation": latent_sil_corr,
                        "latent_db_sklearn": latent_metrics["sklearn"]["davies_bouldin"],
                        "latent_db_correlation": latent_metrics["correlation_based"]["davies_bouldin"],
                    }

                    if "composite_scores" in train_centroid_metrics["latent_space"]:
                        comp = train_centroid_metrics["latent_space"]["composite_scores"]
                        history_entry["composite_sklearn_weighted"] = comp["sklearn"]["weighted_average"]
                        history_entry["composite_sklearn_geometric"] = comp["sklearn"]["geometric_mean"]
                        history_entry["composite_correlation_weighted"] = comp["correlation_based"]["weighted_average"]
                        history_entry["composite_correlation_geometric"] = comp["correlation_based"]["geometric_mean"]
                        history_entry["composite_recommended"] = comp["recommended"]["score"]

                        self.logger.info(f"[CENTROID METRICS] Composite scores:")
                        self.logger.info(f"    sklearn - weighted: {_fmt(comp['sklearn']['weighted_average'])}, geometric: {_fmt(comp['sklearn']['geometric_mean'])}")
                        self.logger.info(f"    correlation - weighted: {_fmt(comp['correlation_based']['weighted_average'])}, geometric: {_fmt(comp['correlation_based']['geometric_mean'])}")
                        self.logger.info(f"    RECOMMENDED (correlation geometric): {_fmt(comp['recommended']['score'])}")

                    # Store in history
                    self.centroid_metrics_history.append(history_entry)
                except Exception as e:
                    self.logger.warning(f"[CENTROID METRICS] Failed to compute: {e}")

            self.scheduler.step(-stopping_value)  # Negate: scheduler mode="min", stopping metric higher-is-better
            self.save_checkpoint(epoch)
            # Visualizations are generated at the end, not per-epoch (for efficiency)
            self.logger.info(
                f"Epoch {epoch} summary - [{self.early_stopping_metric}]: {_fmt(stopping_value)} "
                f"(best: {_fmt(self.best_stopping_value)}), GEV: {_fmt(current_gev)}, "
                f"No improvement: {self.no_improvement_counter}/{self.patience}\n"
            )

            # FIX: Check for dead clusters every 5 epochs and reinitialize them
            if epoch % 5 == 0:
                self._check_cluster_distribution(epoch)

            # t-SNE snapshot at ~10 evenly-spaced epochs (+ epoch 1), skip if best-GEV already snapped
            if not self.config.get("fast_sweep", False):
                snapshot_interval = max(1, self.epochs // 10)
                if not snapped_this_epoch and (epoch % snapshot_interval == 0 or epoch == 1):
                    try:
                        self._save_tsne_snapshot(epoch)
                    except Exception as e:
                        self.logger.warning(f"t-SNE snapshot failed at epoch {epoch}: {e}")

            if self.no_improvement_counter >= self.patience:
                self.logger.info(
                    f"Stopping early: No [{self.early_stopping_metric}] improvement for {self.patience} epochs. "
                    f"Best: {self.best_stopping_value:.4f} (GEV: {self.best_gev:.4f})"
                )
                break

        # Final flush of epoch metrics to disk
        self._flush_epoch_metrics_to_json()

    def _check_cluster_distribution(self, epoch):
        """Check cluster distribution and log statistics for debugging."""
        self.model.set_eval_mode()
        all_predictions = []
        all_latents = []
        all_log_vars = []

        # Single pass: encode + predict_from_latent (avoids double forward pass)
        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                mu, log_var = self.model.encode(data)
                predictions = self.model.predict_from_latent(mu)
                all_predictions.append(predictions)
                all_latents.append(mu.detach().cpu().numpy())
                all_log_vars.append(log_var.detach().cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_latents = np.concatenate(all_latents, axis=0)
        all_log_vars = np.concatenate(all_log_vars, axis=0)
        cluster_counts = np.bincount(all_predictions, minlength=self.model.nClusters)
        total_samples = len(all_predictions)

        # Detailed logging
        self.logger.info(f"[CLUSTER_CHECK] Epoch {epoch} - Total samples: {total_samples}")
        self.logger.info(f"[CLUSTER_CHECK] Cluster distribution: {cluster_counts.tolist()}")
        self.logger.info(f"[CLUSTER_CHECK] Cluster percentages: {[f'{c/total_samples*100:.1f}%' for c in cluster_counts]}")
        self.logger.info(f"[CLUSTER_CHECK] Active clusters (>1%): {np.sum(cluster_counts > total_samples * 0.01)}/{self.model.nClusters}")
        self.logger.info(f"[CLUSTER_CHECK] Dead clusters (0 samples): {np.sum(cluster_counts == 0)}")
        self.logger.info(f"[CLUSTER_CHECK] Max cluster: {cluster_counts.max()} ({cluster_counts.max()/total_samples*100:.1f}%)")
        self.logger.info(f"[CLUSTER_CHECK] Min cluster: {cluster_counts.min()} ({cluster_counts.min()/total_samples*100:.1f}%)")

        # Check pi_ weights
        pi_weights = torch.exp(self.model.pi_).detach().cpu().numpy()
        self.logger.info(f"[CLUSTER_CHECK] Pi weights (min={pi_weights.min():.4f}, max={pi_weights.max():.4f})")

        # ENCODER HEALTH CHECK
        latent_means = np.mean(all_latents, axis=0)
        latent_stds = np.std(all_latents, axis=0)
        log_var_means = np.mean(all_log_vars, axis=0)

        self.logger.info(f"[ENCODER_HEALTH] Epoch {epoch}:")
        self.logger.info(f"[ENCODER_HEALTH]   Latent mu - mean: {np.mean(latent_means):.6f}, std: {np.mean(latent_stds):.6f}")
        self.logger.info(f"[ENCODER_HEALTH]   Latent mu - min_std: {np.min(latent_stds):.6f}, max_std: {np.max(latent_stds):.6f}")
        self.logger.info(f"[ENCODER_HEALTH]   Latent log_var - mean: {np.mean(log_var_means):.6f}")

        # Check for posterior collapse (std too small)
        if np.mean(latent_stds) < 0.1:
            self.logger.warning(f"[ENCODER_HEALTH] ⚠️ POSTERIOR COLLAPSE DETECTED! Latent std={np.mean(latent_stds):.6f} < 0.1")
        elif np.mean(latent_stds) < 0.5:
            self.logger.warning(f"[ENCODER_HEALTH] ⚠️ Low latent variance. Encoder may not be learning diverse representations.")
        else:
            self.logger.info(f"[ENCODER_HEALTH] ✅ Encoder appears healthy (latent std={np.mean(latent_stds):.6f})")

        # Check for inactive dimensions (KL < 0.01)
        # KL per dim ≈ 0.5 * (mu^2 + exp(log_var) - log_var - 1)
        kl_per_dim = 0.5 * (latent_means**2 + np.exp(log_var_means) - log_var_means - 1)
        inactive_dims = np.sum(kl_per_dim < 0.01)
        self.logger.info(f"[ENCODER_HEALTH]   Inactive latent dims (KL<0.01): {inactive_dims}/{len(kl_per_dim)}")

        # Call model's reinitialization if needed
        if hasattr(self.model, 'reinitialize_dead_clusters'):
            # Count dead clusters before reinitialization
            min_samples = int(0.01 * total_samples)
            dead_before = np.sum(cluster_counts < min_samples)

            self.model.reinitialize_dead_clusters(cluster_counts, all_latents, threshold=0.01)

            # Log reinit but do NOT reset patience counter.
            # If GEV is genuinely improving, the counter resets naturally via the GEV improvement path.
            if dead_before > 0:
                self.logger.info(
                    f"[REINIT] Dead clusters reinitialized "
                    f"(patience: {self.no_improvement_counter}/{self.patience}, not reset)"
                )

        self.model.set_train_mode()


    def save_checkpoint(self, epoch):
        self._flush_epoch_metrics_to_json()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_train_loss": self.best_train_loss,
                "best_train_epoch": self.best_train_epoch,
                "no_improvement_counter": self.no_improvement_counter,
                "best_gev": self.best_gev,
                "gev_history": self.gev_history,
                "best_stopping_value": self.best_stopping_value,
                "early_stopping_metric": self.early_stopping_metric,
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # Handle backward compat: old checkpoints may have single param group
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError):
            self.logger.warning("Optimizer state incompatible (param group change) — reinitializing")
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, KeyError):
            self.logger.warning("Scheduler state incompatible — reinitializing")
        self.best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        self.best_train_epoch = checkpoint.get("best_train_epoch", 0)
        self.no_improvement_counter = checkpoint.get("no_improvement_counter", 0)
        self.best_gev = checkpoint.get("best_gev", 0.0)
        self.gev_history = checkpoint.get("gev_history", [])
        self.best_stopping_value = checkpoint.get(
            "best_stopping_value",
            checkpoint.get("best_composite_score", 0.0)  # backward compat
        )
        return checkpoint.get("epoch", 0)

    def run_final_comparison(self):
        self.logger.info(
            "\n" + "=" * 60 + "\nSTEP 3: FINAL EVALUATION (90/10 Split)\n" + "=" * 60
        )

        # Load best model
        if not self.best_model_path.exists():
            self.logger.error("No best model saved. Skipping final evaluation.")
            return {}

        self.logger.info(f"Loading best model from epoch {self.best_train_epoch}...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

        # Get train metrics (90% of data)
        train_loss, train_metrics = self._compute_metrics_and_losses(
            self.train_loader, self.best_train_epoch, is_test=False, strategy_name="VAE"
        )

        # Get eval metrics (10% of data) if available
        eval_loss, eval_metrics = None, None
        if self.eval_loader is not None:
            self.logger.info("\nComputing eval set metrics (10% held-out)...")
            eval_loss, eval_metrics = self._compute_metrics_and_losses(
                self.eval_loader, self.best_train_epoch, is_test=True, strategy_name="VAE"
            )

        # Cache 4Q results for reuse by _save_comprehensive_best_model_metrics
        self._cached_train_4q = train_metrics
        self._cached_eval_4q = eval_metrics
        self._cached_train_loss = train_loss
        self._cached_eval_loss = eval_loss

        # ================================================================
        # CENTROID-BASED METRICS (Final Evaluation)
        # ================================================================
        self.logger.info("\n" + "-" * 50)
        self.logger.info("CENTROID-BASED METRICS (Final Evaluation)")
        self.logger.info("-" * 50)

        train_centroid_metrics = self._compute_centroid_based_metrics(
            self.train_loader, split="train", save_to_file=True
        )
        self.centroid_metrics.print_summary(train_centroid_metrics["latent_space"])

        eval_centroid_metrics = None
        if self.eval_loader is not None:
            eval_centroid_metrics = self._compute_centroid_based_metrics(
                self.eval_loader, split="eval", save_to_file=True
            )

        # Save comprehensive centroid metrics report
        centroid_report = {
            "best_epoch": self.best_train_epoch,
            "train": train_centroid_metrics,
            "history": self.centroid_metrics_history,
        }
        if eval_centroid_metrics is not None:
            centroid_report["eval"] = eval_centroid_metrics

        centroid_report_path = self.output_dir / "centroid_metrics_report.json"
        with open(centroid_report_path, 'w') as f:
            json.dump(centroid_report, f, indent=2, default=str)
        self.logger.info(f"Centroid metrics report saved to: {centroid_report_path}")

        # Store centroid metrics for comprehensive report
        self._final_centroid_metrics = {
            "train": train_centroid_metrics,
        }
        if eval_centroid_metrics is not None:
            self._final_centroid_metrics["eval"] = eval_centroid_metrics

        # Load epoch history
        log_path = self.output_dir / "epoch_metrics_log.json"
        epoch_history = []
        if log_path.exists():
            with open(log_path, "r") as f:
                epoch_history = json.load(f)

        # Find training metrics at best epoch
        train_metrics_at_best = None
        for epoch_data in epoch_history:
            if epoch_data["epoch"] == self.best_train_epoch:
                train_metrics_at_best = {
                    "train_loss": epoch_data["train_loss"],
                    "train_metrics": epoch_data.get("train_metrics", {}),
                }
                break

        # Helper to extract 4-quadrant metrics from a metrics dict
        def _extract_quadrant_metrics(m):
            return {
                "reconstruction": {
                    "mse": m.get("recon_loss", -1),
                    "kld": m.get("kld_loss", -1),
                    "ssim": m.get("ssim_score", -1),
                    "spatial_corr": m.get("spatial_corr", -1),
                },
                "clustering": {
                    "latent_sklearn": {
                        "silhouette": m.get("q1_latent_eucl_silhouette", -1),
                        "db": m.get("q1_latent_eucl_db", 10),
                        "ch": m.get("q1_latent_eucl_ch", 0),
                        "dunn": m.get("q1_latent_eucl_dunn", 0),
                    },
                    "latent_pycrostates": {
                        "silhouette": m.get("q2_latent_corr_silhouette", -1),
                        "db": m.get("q2_latent_corr_db", 10),
                        "ch": m.get("q2_latent_corr_ch", 0),
                        "dunn": m.get("q2_latent_corr_dunn", 0),
                    },
                    "topomap_sklearn": {
                        "silhouette": m.get("q3_topo_eucl_silhouette", -1),
                        "db": m.get("q3_topo_eucl_db", 10),
                        "ch": m.get("q3_topo_eucl_ch", 0),
                        "dunn": m.get("q3_topo_eucl_dunn", 0),
                    },
                    "topomap_pycrostates": {
                        "silhouette": m.get("q4_topo_corr_silhouette", -1),
                        "db": m.get("q4_topo_corr_db", 10),
                        "ch": m.get("q4_topo_corr_ch", 0),
                        "dunn": m.get("q4_topo_corr_dunn", 0),
                    },
                },
            }

        # Build results structure with 4-quadrant metrics
        train_structured = _extract_quadrant_metrics(train_metrics)
        train_structured["total_loss"] = train_loss

        results = {
            "traditional": {
                "strategy_info": {
                    "strategy_name": "VAE",
                    "best_epoch": self.best_train_epoch,
                    "data_mode": "90/10 split",
                    "normalization": "zscore",
                },
                "train_performance": {
                    "train_loss": train_loss,
                    "recon_loss": train_metrics["recon_loss"],
                    "kld_loss": train_metrics["kld_loss"],
                    "beta": train_metrics["beta"],
                    "ssim_score": train_metrics["ssim_score"],
                    "spatial_corr": train_metrics.get("spatial_corr", -1),
                    # Legacy aliases (Q3 topomap euclidean)
                    "clustering_score": train_metrics["silhouette_scores"],
                    "silhouette_scores": train_metrics["silhouette_scores"],
                    "db_scores": train_metrics["db_scores"],
                    "ch_scores": train_metrics["ch_scores"],
                    # 4-quadrant structured
                    **{k: v for k, v in train_metrics.items() if k.startswith("q")},
                    "clustering_failed": train_metrics.get("clustering_failed", False),
                },
                "training_history": {
                    "train_loss_at_best_epoch": (
                        train_metrics_at_best["train_loss"] if train_metrics_at_best else None
                    ),
                },
                "model_config": {
                    "learning_rate": self.lr,
                    "total_epochs": self.epochs,
                    "patience": self.patience,
                    "pretrain_epochs": self.pretrain_epochs,
                    "unfreeze_prior_epoch": self.unfreeze_prior_epoch,
                },
            }
        }

        # Add eval performance if available
        if eval_metrics is not None:
            results["traditional"]["eval_performance"] = {
                "eval_loss": eval_loss,
                "recon_loss": eval_metrics["recon_loss"],
                "kld_loss": eval_metrics["kld_loss"],
                "beta": eval_metrics["beta"],
                "ssim_score": eval_metrics["ssim_score"],
                "spatial_corr": eval_metrics.get("spatial_corr", -1),
                "clustering_score": eval_metrics["silhouette_scores"],
                "silhouette_scores": eval_metrics["silhouette_scores"],
                "db_scores": eval_metrics["db_scores"],
                "ch_scores": eval_metrics["ch_scores"],
                **{k: v for k, v in eval_metrics.items() if k.startswith("q")},
                "clustering_failed": eval_metrics.get("clustering_failed", False),
            }

        # Backward-compatible keys: test_performance mirrors eval (or train if no eval)
        compat_src = eval_metrics if eval_metrics is not None else train_metrics
        compat_loss = eval_loss if eval_loss is not None else train_loss
        results["traditional"]["test_performance"] = {
            "test_loss": compat_loss,
            "recon_loss": compat_src["recon_loss"],
            "kld_loss": compat_src["kld_loss"],
            "beta": compat_src["beta"],
            "ssim_score": compat_src["ssim_score"],
            "clustering_score": compat_src["silhouette_scores"],
            "silhouette_scores": compat_src["silhouette_scores"],
            "db_scores": compat_src["db_scores"],
            "ch_scores": compat_src["ch_scores"],
            "clustering_failed": compat_src.get("clustering_failed", False),
        }

        results["traditional"]["validation_performance"] = {
            "val_loss": compat_loss,
            "beta": compat_src["beta"],
        }

        # Store results for later consolidation
        self._vae_results = results
        self._vae_simple_results = {
            "traditional": {
                "strategy_name": "VAE",
                "best_epoch": self.best_train_epoch,
                "train_loss": train_loss,
                "clustering_score": train_metrics["silhouette_scores"],
                "recon_loss": train_metrics["recon_loss"],
                "kld_loss": train_metrics["kld_loss"],
                "beta": train_metrics["beta"],
                "ssim_score": train_metrics["ssim_score"],
                "silhouette_scores": train_metrics["silhouette_scores"],
                "db_scores": train_metrics["db_scores"],
                "ch_scores": train_metrics["ch_scores"],
            }
        }

        self.logger.info(
            "\n" + "=" * 60 + "\nFINAL EVALUATION COMPLETE\n" + "=" * 60
        )

        # Generate all visualizations at the end (skip in fast-sweep mode)
        if self.config.get("fast_sweep", False):
            self.logger.info("FAST SWEEP: Skipping final visualizations")
        else:
            self.logger.info("Generating final visualizations...")

            # 1. Training curves (train loss over epochs - no val in 100% data mode)
            if self.train_losses:
                try:
                    _g.plot_epoch_results(
                        self.train_losses,
                        self.train_losses,  # Use train losses for both to maintain API compatibility
                        str(self.output_dir / "training_curves.png")
                    )
                except Exception as e:
                    self.logger.warning(f"Could not generate training curves: {e}")

            # 2. Detailed metrics plots from epoch history
            if epoch_history:
                # Build metrics dictionary for plotting
                metrics_for_plot = {
                    "epoch_losses": [h.get("train_loss", 0) for h in epoch_history],
                    "reconstruct_losses": [h.get("recon_loss", 0) for h in epoch_history],
                    "kld_losses": [h.get("kld_loss", 0) for h in epoch_history],
                    "ssim_scores": [h.get("ssim_score", 0) for h in epoch_history],
                    "silhouette_scores": [h.get("silhouette_scores", 0) for h in epoch_history],
                    "db_scores": [h.get("db_scores", 0) for h in epoch_history],
                    "ch_scores": [h.get("ch_scores", 0) for h in epoch_history],
                }
                _g.save_loss_plots(metrics_for_plot, str(self.output_dir / "training_metrics"))

                # 3. Comprehensive training analysis
                _g.create_comprehensive_training_plots(
                    epoch_history,
                    str(self.output_dir / "comprehensive_training_analysis.png"),
                )

                # 4. Detailed metrics report
                _g.create_detailed_metrics_report(
                    results, str(self.output_dir / "detailed_metrics_report.png")
                )

                # 5. Publication-ready plots
                _g.create_publication_ready_plots(
                    results, epoch_history, str(self.output_dir / "publication_ready")
                )

            # 6. Comprehensive latent space t-SNE visualizations (unsupervised analysis)
            try:
                self.logger.info("Generating comprehensive latent space visualizations...")
                self._generate_comprehensive_latent_visualization(
                    self.train_loader, save_prefix="full_data"
                )
            except Exception as e:
                self.logger.warning(f"Could not generate latent space visualizations: {e}")

            # 7. Compile t-SNE progression from epoch snapshots
            try:
                self._compile_tsne_progression()
            except Exception as e:
                self.logger.warning(f"Could not compile t-SNE progression: {e}")

            self.logger.info("All visualizations generated successfully!")

        return results


    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, (dict, defaultdict)):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def run_complete_pipeline(self, fast_sweep=False):
        try:
            self.pretrain()
            self.train_and_evaluate()

            # In fast-sweep mode, move model to CPU before final evaluation
            # to free GPU memory for other concurrent sweep processes
            if fast_sweep and self.device.type == "cuda":
                self.logger.info("FAST SWEEP: Moving model to CPU for final evaluation")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                self.model.device = self.device  # sync model's internal device ref
                gc.collect()
                torch.cuda.empty_cache()

            vae_results = self.run_final_comparison()

            if fast_sweep:
                # Tier 1 fast mode: skip viz/baseline/centroid — only save metrics
                self.logger.info("FAST SWEEP MODE: Skipping visualization and baseline pipeline")

                # Still save comprehensive best model metrics if available
                if hasattr(self, '_final_centroid_metrics'):
                    self._save_comprehensive_best_model_metrics(
                        train_centroid_metrics=self._final_centroid_metrics["train"],
                        baseline_metrics=None,
                    )

                return vae_results

            # --- Full pipeline (Tier 2) below ---
            topo_dir = str(self.output_dir / "final_cluster_topomaps")
            self.model.perform_research_analysis(
                data_loader=self.train_loader, output_dir=topo_dir,
                norm_params=self.norm_params,
                raw_mne=self.raw_mne,
            )
            self.model.plot_segmentation(
                data_loader=self.train_loader, output_dir=topo_dir
            )
            self.model.plot_microstate_statistics(
                data_loader=self.train_loader,
                output_dir=topo_dir,
                gfp_peaks=self.gfp_peaks,
            )
            # Backfit VAE to raw data for comparable statistics with baseline
            vae_raw_stats = None
            if self.raw_mne is not None and hasattr(self.model, 'evaluate_on_raw'):
                self.logger.info("Computing VAE statistics on raw data (backfitting)...")
                vae_raw_stats = self.model.evaluate_on_raw(
                    raw=self.raw_mne,
                    output_dir=topo_dir,
                    gfp_peaks=self.gfp_peaks,
                    norm_params=self.norm_params,
                )
                if vae_raw_stats:
                    self.logger.info("VAE backfitted statistics computed successfully")
                else:
                    self.logger.warning("VAE raw backfit returned None - check for errors above")
            elif self.raw_mne is not None:
                self.logger.info("Skipping VAE raw backfitting (method not implemented)")
            self.run_explainability()
            self.run_deep_diagnostics()

            # --- CENTROID REDUNDANCY ANALYSIS ---
            self.logger.info(
                "\n" + "=" * 60 + "\nSTEP 3.5: CENTROID REDUNDANCY ANALYSIS\n" + "=" * 60
            )
            try:
                centroid_results = _ca.run_centroid_analysis(
                    model=self.model,
                    data_loader=self.train_loader,  # 100% data mode
                    output_dir=str(self.output_dir),
                    logger=self.logger,
                    auto_merge=True,
                    raw_mne=self.raw_mne,  # Pass MNE for GEV computation
                    norm_params=self.norm_params,
                )
                self.logger.info(f"Centroid analysis complete. Results saved to: {centroid_results['output_dir']}")
                if centroid_results['redundancy_info']['total_redundant'] > 0:
                    self.logger.info(f"Found {centroid_results['redundancy_info']['total_redundant']} redundant centroid pairs")
                    if centroid_results['merge_info'].get('merged', False):
                        self.logger.info(f"Clusters reduced: {centroid_results['merge_info']['original_k']} -> {centroid_results['merge_info']['merged_k']}")
            except Exception as e:
                self.logger.warning(f"Centroid analysis failed: {e}")
                centroid_results = None

            # --- START BASELINE INTEGRATION ---
            self.logger.info(
                "\n" + "=" * 60 + "\nSTEP 4: RUNNING BASELINE (ModKMeans) - 100% Data\n" + "=" * 60
            )

            # Initialize Baseline Handler
            baseline_handler = _b.BaselineHandler(
                n_clusters=self.model.nClusters,
                device=self.device,
                logger=self.logger,
                output_dir=self.output_dir,
            )

            # Run Baseline (requires GFP peaks from EEG preprocessing)
            # In 100% data mode, fit on ALL GFP peaks
            # Note: In parallel training, gfp_peaks may not transfer properly via multiprocessing,
            # so baseline runs in the main process via _run_baseline_for_all_clusters() after workers finish
            gfp_peaks_valid = False
            try:
                if self.gfp_peaks is not None and hasattr(self.gfp_peaks, 'get_data'):
                    data = self.gfp_peaks.get_data()
                    gfp_peaks_valid = data is not None and data.shape[1] > 0
            except Exception as e:
                self.logger.warning(f"GFP peaks validation failed (may be in parallel worker): {e}")
            if gfp_peaks_valid:
                baseline_handler.fit(self.gfp_peaks)  # 100% data mode - no indices
                baseline_handler.plot()  # pycrostates native visualization
                baseline_handler.plot_merged_centroids()  # Publication-ready merged centroids
                try:
                    baseline_handler.plot_microstate_topomaps_named()
                except Exception as e:
                    self.logger.warning(f"Named topomap plot failed: {e}")

                # Centroid visualizations (correlation matrix, summary)
                baseline_handler.plot_centroid_spatial_correlations()
                baseline_handler.plot_centroid_summary()

                # Segmentation and statistics (requires raw MNE object)
                if self.raw_mne is not None:
                    baseline_handler.plot_segmentation(self.raw_mne)
                    baseline_handler.plot_microstate_statistics(self.raw_mne)
                    # MNE topomaps (requires channel info)
                    baseline_handler.plot_centroid_topomaps_mne(self.raw_mne.info)

                # Electrode space analysis (t-SNE visualizations) - similar to VAE latent space analysis
                try:
                    self.logger.info("Generating baseline electrode space visualizations...")
                    baseline_handler.generate_electrode_space_visualization(save_prefix="full_data")
                except Exception as e:
                    self.logger.warning(f"Could not generate electrode space visualizations: {e}")

                # Evaluate on GFP peaks (custom label assignment)
                baseline_metrics = baseline_handler.evaluate()

                # Compute pycrostates native cluster validation metrics (silhouette, CH, Dunn, DB)
                # with composite scores for fair comparison with VAE
                baseline_cluster_metrics = baseline_handler.compute_cluster_metrics()
                baseline_metrics["cluster_validation_metrics"] = baseline_cluster_metrics

                # Evaluate using pycrostates predict() on Raw data
                if self.raw_mne is not None:
                    baseline_raw_metrics = baseline_handler.evaluate_on_raw(self.raw_mne)
                    baseline_metrics["predict_on_raw"] = baseline_raw_metrics

                    # ================================================================
                    # VAE vs ModKMeans CENTROID COMPARISON
                    # ================================================================
                    try:
                        vae_electrode_values = self._extract_vae_electrode_values()
                        if vae_electrode_values is not None:
                            vae_gev = self.best_gev if hasattr(self, 'best_gev') else None
                            comparison_results = baseline_handler.plot_vae_comparison(
                                vae_electrode_values, vae_gev=vae_gev
                            )
                            baseline_metrics["vae_comparison"] = comparison_results
                    except Exception as e:
                        self.logger.warning(f"VAE vs ModKMeans comparison failed: {e}")

                    # ================================================================
                    # UNIFIED MICROSTATE STATISTICS COMPARISON (RAW DATA)
                    # ================================================================
                    self._save_unified_microstate_stats(
                        vae_raw_stats=vae_raw_stats,
                        baseline_handler=baseline_handler,
                    )
            else:
                self.logger.warning("GFP peaks not available. Skipping baseline ModKMeans analysis.")
                baseline_metrics = {"strategy_name": "Baseline (ModKMeans)", "gev": None, "n_clusters": self.model.nClusters, "note": "GFP peaks not available"}

            # Merge Results for Comparison Report (Legacy)
            vae_best_strategy = max(
                vae_results.keys(),
                key=lambda k: vae_results[k]["train_performance"]["clustering_score"],
            )
            vae_best = vae_results[vae_best_strategy]["train_performance"]

            self._generate_final_comparison_report(vae_best, baseline_metrics)

            # --- RUN FAIR COMPARISON ---
            self.logger.info(
                "\n" + "=" * 60 + "\nSTEP 5: FAIR COMPARISON (100% Data Mode)\n" + "=" * 60
            )
            fair_results = self._run_fair_comparison(
                baseline_handler,
                vae_test_metrics=vae_best,  # Now using train metrics (100% data)
                baseline_test_metrics=baseline_metrics
            )

            # ================================================================
            # COMPREHENSIVE BEST MODEL METRICS FILE (includes baseline)
            # ================================================================
            if hasattr(self, '_final_centroid_metrics'):
                comprehensive_report = self._save_comprehensive_best_model_metrics(
                    train_centroid_metrics=self._final_centroid_metrics["train"],
                    baseline_metrics=baseline_metrics,
                )
                if comprehensive_report is not None:
                    self._generate_docs_comparison_report(comprehensive_report)

            # Save SINGLE consolidated results.json (replaces multiple redundant files)
            combined_results = {
                "vae_results": vae_results,
                "baseline_metrics": baseline_metrics,
                "fair_comparison": fair_results if fair_results else None,
                "centroid_analysis": centroid_results if centroid_results else None,
                "training_config": {
                    "n_clusters": self.model.nClusters,
                    "latent_dim": self.model.latent_dim,
                    "batch_size": self.config.get("batch_size", 128),
                    "total_epochs": len(self.train_losses) if self.train_losses else 0,
                    "best_epoch": self.best_train_epoch,
                },
            }
            # Single JSON file with all results
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(self._convert_to_json_serializable(combined_results), f, indent=4)
            self.logger.info(f"✅ Consolidated results saved to: {self.output_dir / 'results.json'}")

            # Generate single comprehensive summary report
            self._generate_comprehensive_summary_report(combined_results)

            return vae_results
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


