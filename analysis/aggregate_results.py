#!/usr/bin/env python3
"""Aggregate results from parallel cluster training runs.

This script collects results from all K-value training runs and generates
comparison reports including CSV summaries and YAML exports.

Usage:
    python aggregate_results.py --run_id eeg_lemon_010004_k3-20 --participant 010004
    python aggregate_results.py --run_id eeg_lemon_k3-20  # reads participant from config.toml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aggregate")


def load_cluster_summary(cluster_dir: Path) -> Optional[Dict[str, Any]]:
    """Load summary metrics from a cluster directory.

    Tries multiple file formats for compatibility.
    """
    summary_file = cluster_dir / "summary_metrics.yaml"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = yaml.safe_load(f)
        summary["output_dir"] = str(cluster_dir)
        return summary

    # Fallback: try JSON format
    json_file = cluster_dir / "summary_metrics.json"
    if json_file.exists():
        with open(json_file) as f:
            summary = json.load(f)
        summary["output_dir"] = str(cluster_dir)
        return summary

    return None


def load_baseline_metrics(cluster_dir: Path) -> Optional[Dict[str, Any]]:
    """Load baseline (ModKMeans) metrics if available."""
    baseline_file = cluster_dir / "baseline_metrics.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            return json.load(f)
    return None


def _parse_cluster_dir_name(dir_name: str) -> Dict[str, Any]:
    """Parse cluster directory name to extract architecture parameters.

    Supports both formats:
    - 3-segment (legacy): cluster_K_batch_ld
    - 5-segment (new):    cluster_K_batch_ld_depth_ndf
    """
    parts = dir_name.replace("cluster_", "").split("_")
    result = {}
    if len(parts) >= 3:
        result["n_clusters"] = int(parts[0])
        result["batch_size"] = int(parts[1])
        result["latent_dim"] = int(parts[2])
    if len(parts) >= 5:
        result["n_conv_layers"] = int(parts[3])
        result["ndf"] = int(parts[4])
    return result


def aggregate_results(
    run_dir: Path, participant_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Collect all cluster results and generate comparison reports.

    Args:
        run_dir: Path to the run directory containing cluster_* subdirectories
        participant_id: Optional participant ID for logging

    Returns:
        List of summary dictionaries for each cluster
    """
    # Find all cluster directories
    cluster_dirs = sorted(run_dir.glob("cluster_*"))
    logger.info(f"Found {len(cluster_dirs)} cluster directories in {run_dir}")

    summary_data = []
    baseline_data = []

    for cluster_dir in cluster_dirs:
        # Load VAE results
        summary = load_cluster_summary(cluster_dir)
        if summary:
            # Backfill architecture params from directory name if missing
            dir_params = _parse_cluster_dir_name(cluster_dir.name)
            if "ndf" not in summary and "ndf" in dir_params:
                summary["ndf"] = dir_params["ndf"]
            if "n_conv_layers" not in summary and "n_conv_layers" in dir_params:
                summary["n_conv_layers"] = dir_params["n_conv_layers"]
            summary_data.append(summary)
            logger.info(f"  Loaded VAE: {cluster_dir.name}")
        else:
            logger.warning(f"  Missing VAE summary: {cluster_dir.name}")

        # Load baseline results
        baseline = load_baseline_metrics(cluster_dir)
        if baseline:
            baseline["n_clusters"] = summary.get("n_clusters") if summary else None
            baseline["output_dir"] = str(cluster_dir)
            baseline_data.append(baseline)
            logger.info(f"  Loaded baseline: {cluster_dir.name}")

    if not summary_data:
        logger.error("No VAE results found!")
        return []

    # Create comparison directory
    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    # Save VAE comparison
    df_vae = pd.DataFrame(summary_data).sort_values("n_clusters")
    _save_comparison(df_vae, comparison_dir, "cluster_comparison", participant_id)

    # Save baseline comparison if available
    if baseline_data:
        df_baseline = pd.DataFrame(baseline_data).sort_values("n_clusters")
        _save_comparison(
            df_baseline, comparison_dir, "baseline_comparison", participant_id
        )

        # Create combined VAE vs Baseline comparison
        _create_combined_comparison(
            df_vae, df_baseline, comparison_dir, participant_id
        )

    # Print summary
    _print_summary(df_vae, baseline_data)

    return summary_data


def _save_comparison(
    df: pd.DataFrame,
    comparison_dir: Path,
    prefix: str,
    participant_id: Optional[str] = None,
) -> None:
    """Save comparison data to CSV and YAML."""
    suffix = f"_{participant_id}" if participant_id else ""

    csv_path = comparison_dir / f"{prefix}{suffix}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    yaml_path = comparison_dir / f"{prefix}{suffix}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(df.to_dict("records"), f, default_flow_style=False)
    logger.info(f"Saved: {yaml_path}")


def _create_combined_comparison(
    df_vae: pd.DataFrame,
    df_baseline: pd.DataFrame,
    comparison_dir: Path,
    participant_id: Optional[str] = None,
) -> None:
    """Create a combined VAE vs Baseline comparison table."""
    # Merge on n_clusters
    vae_cols = ["n_clusters", "best_val_loss", "silhouette", "nmi", "ari"]
    vae_subset = df_vae[[c for c in vae_cols if c in df_vae.columns]].copy()
    vae_subset = vae_subset.rename(
        columns={
            "best_val_loss": "vae_loss",
            "silhouette": "vae_silhouette",
            "nmi": "vae_nmi",
            "ari": "vae_ari",
        }
    )

    baseline_cols = ["n_clusters", "gev", "silhouette", "nmi", "ari"]
    baseline_subset = df_baseline[
        [c for c in baseline_cols if c in df_baseline.columns]
    ].copy()
    baseline_subset = baseline_subset.rename(
        columns={
            "silhouette": "baseline_silhouette",
            "nmi": "baseline_nmi",
            "ari": "baseline_ari",
            "gev": "baseline_gev",
        }
    )

    combined = pd.merge(vae_subset, baseline_subset, on="n_clusters", how="outer")
    combined = combined.sort_values("n_clusters")

    suffix = f"_{participant_id}" if participant_id else ""
    csv_path = comparison_dir / f"vae_vs_baseline{suffix}.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _print_summary(df_vae: pd.DataFrame, baseline_data: List[Dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("CLUSTER COMPARISON SUMMARY (VAE)")
    print("=" * 70)

    display_cols = [
        "n_clusters", "latent_dim", "ndf", "n_conv_layers",
        "best_train_loss", "best_gev", "silhouette",
        "q2_latent_corr_sil", "ssim_score",
    ]
    available_cols = [c for c in display_cols if c in df_vae.columns]
    print(df_vae[available_cols].to_string(index=False))

    if baseline_data:
        print("\n" + "-" * 70)
        print("BASELINE (ModKMeans)")
        print("-" * 70)
        df_baseline = pd.DataFrame(baseline_data).sort_values("n_clusters")
        baseline_cols = ["n_clusters", "gev", "silhouette"]
        available_baseline = [c for c in baseline_cols if c in df_baseline.columns]
        print(df_baseline[available_baseline].to_string(index=False))

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from parallel cluster training runs"
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="Run ID (directory name under outputs/participant/)",
    )
    parser.add_argument(
        "--participant",
        default=None,
        help="Participant ID (reads from config.toml if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Base output directory (default: ./outputs)",
    )
    args = parser.parse_args()

    # Read participant from config.toml if not specified via CLI
    if args.participant is None:
        from config.config import config as _cfg
        args.participant = _cfg.get_lemon_config().get("subject_id")

    run_dir = Path(args.output_dir) / args.participant / args.run_id

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1

    aggregate_results(run_dir, args.participant)
    return 0


if __name__ == "__main__":
    exit(main())
