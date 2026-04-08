import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

import model as _m
import clustering_trainer as _t


def _load_epoch_history(epoch_log_path: Path) -> List[Dict]:
    """Load epoch history from JSON file if it exists."""
    if not epoch_log_path.exists():
        return []
    with open(epoch_log_path, "r") as f:
        return json.load(f)


def _build_loss_history(epoch_history: List[Dict], trainer: Any) -> Dict[str, List]:
    """Build loss history from epoch history or fall back to trainer attributes.

    Maps epoch history keys to standardized loss history keys. When epoch history
    is unavailable, uses trainer's train_losses as fallback (100% data mode).
    """
    # Mapping from loss_history key to epoch_history key
    key_mapping = {
        "epoch_losses": "train_loss",
        "reconstruct_losses": "recon_loss",
        "kld_losses": "kld_loss",
        "beta_scores": "beta",
        "ssim_scores": "ssim_score",
        "silhouette_scores": "silhouette_scores",
        "db_scores": "db_scores",
        "ch_scores": "ch_scores",
        "silhouette_corr": "silhouette_corr",
        "db_corr": "db_corr",
        "ch_corr": "ch_corr",
        "nmi_scores": "nmi_scores",
        "ari_scores": "ari_scores",
    }

    if epoch_history:
        return {
            key: [h.get(epoch_key, 0) for h in epoch_history]
            for key, epoch_key in key_mapping.items()
        }

    # Fallback to trainer attributes (100% data mode)
    result = {key: [] for key in key_mapping}
    result["epoch_losses"] = trainer.train_losses
    return result


def train_cluster(
    n_clusters: int,
    config_dict: Dict,
    train_loader: DataLoader,
    train_set: Any,
    device: torch.device,
    batch_size: int,
    latent_dim: int,
    n_channels: int,
    logger: any = None,
    gfp_peaks=None,
    raw_mne=None,
    eval_loader: DataLoader = None,
    eval_set: Any = None,
    norm_params=None,
    ndf: int = 64,
    n_conv_layers: int = 4,
    fast_sweep: bool = False,
) -> Dict:
    """Train a VAE clustering model for given number of clusters (100% data mode)."""
    label = f"{n_clusters}_{batch_size}_{latent_dim}_{n_conv_layers}_{ndf}"
    base_output_dir = Path(config_dict.get("output_dir", "./outputs"))
    cluster_output_dir = base_output_dir / f"cluster_{label}"
    cluster_output_dir.mkdir(parents=True, exist_ok=True)

    worker_logger = logging.getLogger(f"worker_{label}")
    worker_logger.setLevel(logging.INFO)

    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_file_path = cluster_output_dir / "run.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    worker_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    worker_logger.addHandler(stream_handler)

    worker_logger.propagate = False

    try:
        worker_logger.info(f"Starting training for {label} on {device}. Logging to {log_file_path}")

        cluster_config = config_dict.copy()
        cluster_config["output_dir"] = str(cluster_output_dir)

        # Fast-sweep mode: all training params come from config.toml
        if fast_sweep:
            worker_logger.info(
                "FAST SWEEP: epochs={}, patience={}, warmup={}".format(
                    cluster_config.get("epochs", 100),
                    cluster_config.get("patience", 15),
                    cluster_config.get("clustering_warmup_epochs", 10),
                )
            )

        # Get MNE info from raw_mne for proper channel configuration
        mne_info = raw_mne.info if raw_mne is not None else None

        model = _m.create_model_with_batch_cyclical(
            latent_dim=latent_dim,
            nClusters=n_clusters,
            batch_size=batch_size,
            logger=worker_logger,
            device=device,
            n_cycles_per_epoch=cluster_config.get("n_cycles_per_epoch", 5),
            cycle_ratio=cluster_config.get("cycle_ratio", 0.5),
            gamma=cluster_config.get("beta_gamma", 0.1),
            info=mne_info,
            n_total_cycles=cluster_config.get("n_total_cycles", 4),
            max_beta=cluster_config.get("max_beta", 0.4),
            ndf=ndf,
            ngf=ndf,
            n_conv_layers=n_conv_layers,
        ).to(device)
        model._batch_entropy_weight = cluster_config.get("batch_entropy_weight", 5.0)
        model._cluster_tightening_weight = cluster_config.get("cluster_tightening_weight", 0.2)
        model._separation_weight = cluster_config.get("separation_weight", 50.0)
        model._clustering_warmup_epochs = cluster_config.get("clustering_warmup_epochs", 10)
        model._clustering_warmup_delay = cluster_config.get("clustering_warmup_delay", 0)

        trainer = _t.VAEClusteringTrainer(
            model=model,
            train_loader=train_loader,
            train_set=train_set,
            config=cluster_config,
            device=device,
            cluster_id=n_clusters,
            logger=worker_logger,
            gfp_peaks=gfp_peaks,
            raw_mne=raw_mne,
            eval_loader=eval_loader,
            eval_set=eval_set,
            norm_params=norm_params,
        )

        final_metrics = trainer.run_complete_pipeline(fast_sweep=fast_sweep)

        if not final_metrics:
            raise RuntimeError("Training pipeline did not return final metrics.")

        # Extract metrics from nested structure returned by run_final_comparison()
        # Structure: {"traditional": {"train_performance": {...}, "eval_performance": {...}}}
        _tp = final_metrics.get("traditional", {}).get("train_performance", {})
        _ep = final_metrics.get("traditional", {}).get("eval_performance", {})

        summary = {
            "n_clusters": n_clusters,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "ndf": ndf,
            "n_conv_layers": n_conv_layers,
            "label": label,
            "best_train_loss": trainer.best_train_loss,
            "best_train_epoch": trainer.best_train_epoch,
            "best_gev": trainer.best_gev,
            # Reconstruction metrics
            "ssim_score": _tp.get("ssim_score", -1),
            "spatial_corr": _tp.get("spatial_corr", -1),
            # Legacy aliases (backward compat)
            "silhouette": _tp.get("silhouette_scores", -1),
            "davies_bouldin": _tp.get("db_scores", -1),
            "calinski_harabasz": _tp.get("ch_scores", -1),
            # Q1: Latent + Euclidean (sklearn)
            "q1_latent_eucl_sil": _tp.get("q1_latent_eucl_silhouette", -1),
            "q1_latent_eucl_db": _tp.get("q1_latent_eucl_db", -1),
            "q1_latent_eucl_ch": _tp.get("q1_latent_eucl_ch", -1),
            "q1_latent_eucl_dunn": _tp.get("q1_latent_eucl_dunn", -1),
            # Q2: Latent + Correlation (pycrostates) — PRIMARY METRIC
            "q2_latent_corr_sil": _tp.get("q2_latent_corr_silhouette", -1),
            "q2_latent_corr_db": _tp.get("q2_latent_corr_db", -1),
            "q2_latent_corr_ch": _tp.get("q2_latent_corr_ch", -1),
            "q2_latent_corr_dunn": _tp.get("q2_latent_corr_dunn", -1),
            # Q3: Topomap + Euclidean (sklearn)
            "q3_topo_eucl_sil": _tp.get("q3_topo_eucl_silhouette", -1),
            "q3_topo_eucl_db": _tp.get("q3_topo_eucl_db", -1),
            "q3_topo_eucl_ch": _tp.get("q3_topo_eucl_ch", -1),
            "q3_topo_eucl_dunn": _tp.get("q3_topo_eucl_dunn", -1),
            # Q4: Topomap + Correlation (pycrostates)
            "q4_topo_corr_sil": _tp.get("q4_topo_corr_silhouette", -1),
            "q4_topo_corr_db": _tp.get("q4_topo_corr_db", -1),
            "q4_topo_corr_ch": _tp.get("q4_topo_corr_ch", -1),
            "q4_topo_corr_dunn": _tp.get("q4_topo_corr_dunn", -1),
            # Eval metrics (if available)
            "eval_silhouette": _ep.get("silhouette_scores", -1) if _ep else -1,
            "eval_ssim": _ep.get("ssim_score", -1) if _ep else -1,
            "eval_q2_latent_corr_sil": _ep.get("q2_latent_corr_silhouette", -1) if _ep else -1,
            "train_epochs_completed": len(trainer.train_losses),
            "total_epochs_configured": trainer.epochs,
            "patience": trainer.patience,
            "learning_rate": trainer.lr,
        }

        summary_file = cluster_output_dir / "summary_metrics.yaml"
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
        worker_logger.info(f"Summary metrics saved to {summary_file}")

        # Load epoch history from JSON for comprehensive metrics
        epoch_log_path = cluster_output_dir / "epoch_metrics_log.json"
        epoch_history = _load_epoch_history(epoch_log_path)
        loss_history = _build_loss_history(epoch_history, trainer)

        result = {
            "n_clusters": n_clusters,
            "label": label,
            "best_train_loss": trainer.best_train_loss,
            "best_train_epoch": trainer.best_train_epoch,
            "final_metrics": final_metrics,
            "output_dir": str(cluster_output_dir),
            "summary": summary,
            "success": True,
            "loss_history": loss_history,
            "epochs_completed": len(trainer.train_losses),
        }

        worker_logger.info(f"Completed training for {label} successfully.")
        return result

    except Exception as e:
        main_logger = logger or logging.getLogger("vae_clustering")
        main_logger.error(f"Error in training cluster {label}: {str(e)}", exc_info=True)
        return {
            "n_clusters": n_clusters,
            "label": label,
            "error": str(e),
            "success": False,
            "best_train_loss": float("inf"),
            "loss_history": None,
        }
    finally:
        if "worker_logger" in locals():
            for handler in worker_logger.handlers[:]:
                handler.close()
                worker_logger.removeHandler(handler)
