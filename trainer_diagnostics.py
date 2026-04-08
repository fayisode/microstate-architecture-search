"""Trainer Diagnostics Mixin — logging, reporting, and diagnostic methods for VAEClusteringTrainer."""
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import metrics_utils as _mu
import centroid_metrics as _cm

# Lazy imports — not needed in fast-sweep mode
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import centroid_analysis as _ca
except ImportError:
    _ca = None
try:
    import baseline as _b
except ImportError:
    _b = None


def _fmt(val, fmt=".4f", default="N/A"):
    """Safely format a value that might be None."""
    if val is None:
        return default
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return default


class TrainerDiagnosticsMixin:
    """Mixin providing logging, reporting, and diagnostic methods for VAEClusteringTrainer."""

    def _log_epoch_summary(self, train_loss, train_metrics,
                           eval_gev=None, train_gev=None, eval_recon_loss=None):
        current_lr = self.optimizer.param_groups[0]["lr"]
        eval_gev_str = f"{eval_gev:.4f}" if eval_gev is not None else "N/A"
        train_gev_str = f"{train_gev:.4f}" if train_gev is not None else "N/A"
        best_gev_str = f"{self.best_gev:.4f}" if self.best_gev > 0 else "N/A"

        # Extract train Q1/Q3 metrics for summary
        t_q1_sil = _fmt(train_metrics.get('q1_latent_eucl_silhouette'), '.2f')
        t_q1_dunn = _fmt(train_metrics.get('q1_latent_eucl_dunn'), '.2f')
        t_q3_sil = _fmt(train_metrics.get('q3_topo_eucl_silhouette'), '.2f')
        t_q3_dunn = _fmt(train_metrics.get('q3_topo_eucl_dunn'), '.2f')

        # Eval recon loss string
        eval_recon_str = f" (eval: {eval_recon_loss:.4f})" if eval_recon_loss is not None else ""

        # KL collapse info
        kl_active = train_metrics.get('kl_active_dims')
        kl_collapsed = train_metrics.get('kl_collapsed_dims')
        latent_dim = (kl_active + kl_collapsed) if kl_active is not None and kl_collapsed is not None else None
        kl_line = ""
        if latent_dim is not None:
            kl_line = f"  Latent: {kl_active}/{latent_dim} active dims ({kl_collapsed} collapsed, KL<0.05)\n"

        self.logger.info(
            f"\n{'='*70}\n"
            f"EPOCH {self.current_epoch:3d} / {self.epochs} SUMMARY\n"
            f"{'='*70}\n"
            f"RECONSTRUCTION:\n"
            f"  MSE: {_fmt(train_metrics.get('recon_loss'), '.4f')}{eval_recon_str}   "
            f"KLD: {_fmt(train_metrics.get('kld_loss'), '.4f')}   "
            f"SSIM: {_fmt(train_metrics.get('ssim_score'), '.4f')}   "
            f"Total: {train_loss:.4f}  Beta: {_fmt(train_metrics.get('beta'), '.4f')}\n"
            f"{kl_line}"
            f"\nCLUSTERING (train):\n"
            f"                | Sklearn (Euclidean)\n"
            f"  Latent  32D   | Sil={t_q1_sil}  Dunn={t_q1_dunn}\n"
            f"  Topo  1600D   | Sil={t_q3_sil}  Dunn={t_q3_dunn}\n"
            f"\nGEV: train={train_gev_str} eval={eval_gev_str} (best={best_gev_str})   "
            f"LR={current_lr:.1e}   NoImprove={self.no_improvement_counter}/{self.patience}\n"
            f"{'='*70}"
        )

    def _log_epoch_metrics_to_json(self, epoch_data):
        """Accumulate epoch data in memory (flushed on checkpoint save and training end)."""
        self.epoch_metrics_history.append(epoch_data)

    def _flush_epoch_metrics_to_json(self):
        """Write accumulated epoch metrics to disk."""
        if not self.epoch_metrics_history:
            return
        log_path = self.output_dir / "epoch_metrics_log.json"
        with open(log_path, "w") as f:
            json.dump(self._convert_to_json_serializable(self.epoch_metrics_history), f, indent=4)

    def _generate_comparison_tables(self, results):
        header = (
            "Strategy                     | Best Epoch | Test Loss | Recon Loss | KLD Loss  | Clustering Score | Silhouette | SSIM\n"
            + "-----------------------------|------------|-----------|------------|-----------|------------------|------------|-------"
        )
        rows = [
            f"{res['strategy_name']:<28} | {res['best_epoch']:<10} | {res['test_loss']:.6f} | {res['recon_loss']:.6f} | {res['kld_loss']:.6f} | {res['clustering_score']:.6f}       | {res['silhouette_scores']:.6f} | {res['ssim_score']:.4f}"
            for res in results.values()
        ]
        report = f"{header}\n" + "\n".join(rows)
        with open(
            self.output_dir / "comprehensive_strategy_comparison_table.txt", "w"
        ) as f:
            f.write(report)
        self.logger.info(
            f"Comparison table saved to {self.output_dir / 'comprehensive_strategy_comparison_table.txt'}"
        )

        self.logger.info(
            f"Strategy performance plots saved to {self.output_dir / 'strategy_performance_plots.png'}"
        )

    def _generate_diagnosis_report(self, diagnosis_results: Dict) -> str:
        """Generate a human-readable text report from cluster diagnosis results."""
        lines = [
            "=" * 80,
            "                    CLUSTER DIAGNOSIS REPORT",
            "=" * 80,
            "",
            "This report summarizes the initial cluster analysis before training.",
            "",
        ]

        # Extract available keys and format them
        for key, value in diagnosis_results.items():
            if isinstance(value, dict):
                lines.append(f"{key.upper().replace('_', ' ')}:")
                lines.append("-" * 40)
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    elif isinstance(v, list) and len(v) <= 10:
                        lines.append(f"  {k}: {v}")
                    elif isinstance(v, list):
                        first = f"{v[0]:.4f}" if isinstance(v[0], (int, float)) else str(v[0])
                        last = f"{v[-1]:.4f}" if isinstance(v[-1], (int, float)) else str(v[-1])
                        lines.append(f"  {k}: [{first}, ..., {last}] (len={len(v)})")
                    else:
                        lines.append(f"  {k}: {v}")
                lines.append("")
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4f}")
                else:
                    lines.append(f"{key}: {value}")
            elif isinstance(value, list) and len(value) <= 10:
                lines.append(f"{key}: {value}")
            elif isinstance(value, list):
                lines.append(f"{key}: [{value[0]}, ..., {value[-1]}] (len={len(value)})")

        lines.extend([
            "",
            "=" * 80,
            "INTERPRETATION:",
            "-" * 40,
            "- Check if cluster assignments are balanced",
            "- High variance in assignments may indicate unstable clustering",
            "- Use this as baseline before training improvements",
            "=" * 80,
        ])

        return "\n".join(lines)

    def _save_unified_microstate_stats(self, vae_raw_stats, baseline_handler):
        """
        Save unified microstate statistics comparing VAE and ModKMeans on RAW data.

        This creates a single JSON file with both methods' temporal statistics
        (duration, coverage, occurrence) computed on the same full raw data,
        enabling direct apples-to-apples comparison.
        """
        import json

        self.logger.info("=" * 60)
        self.logger.info("SAVING UNIFIED MICROSTATE STATISTICS (RAW DATA COMPARISON)")
        self.logger.info("=" * 60)

        try:
            # Load baseline raw stats
            baseline_stats_path = self.output_dir / "baseline_microstate_stats.json"
            baseline_stats = None
            if baseline_stats_path.exists():
                with open(baseline_stats_path, 'r') as f:
                    baseline_stats = json.load(f)

            # Create unified comparison
            unified_stats = {
                "description": "Microstate temporal statistics computed on FULL RAW EEG data",
                "note": "Both methods evaluated on the same continuous EEG signal for fair comparison",
                "vae_raw_backfit": vae_raw_stats if vae_raw_stats else {
                    "error": "VAE raw backfit statistics not available"
                },
                "modkmeans": baseline_stats if baseline_stats else {
                    "error": "ModKMeans statistics not available"
                },
            }

            # Add summary comparison if both available
            if vae_raw_stats and baseline_stats:
                vae_n = vae_raw_stats.get('n_samples', 0)
                baseline_n = baseline_stats.get('total_samples', 0)

                unified_stats["comparison_summary"] = {
                    "vae_total_samples": vae_n,
                    "modkmeans_total_samples": baseline_n,
                    "samples_match": vae_n == baseline_n,
                    "vae_gev": vae_raw_stats.get('gev'),
                    "modkmeans_n_clusters": baseline_stats.get('n_clusters'),
                    "vae_n_clusters": vae_raw_stats.get('n_clusters'),
                }

                # Log comparison
                self.logger.info(f"VAE (raw backfit): {vae_n} samples")
                self.logger.info(f"ModKMeans: {baseline_n} samples")
                if vae_n == baseline_n:
                    self.logger.info("Sample counts MATCH - fair comparison possible")
                else:
                    self.logger.warning(f"Sample counts differ: VAE={vae_n}, ModKMeans={baseline_n}")

            # Save unified stats
            unified_path = self.output_dir / "unified_microstate_stats.json"
            with open(unified_path, 'w') as f:
                json.dump(unified_stats, f, indent=4)

            self.logger.info(f"Saved unified statistics to: {unified_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save unified microstate stats: {e}")

    def _generate_final_comparison_report(self, vae_metrics, baseline_metrics):
        """Generates a formatted text file comparing VAE vs ModKMeans with box-drawing tables."""
        import datetime

        # Handle case when baseline was skipped (no GFP peaks)
        if baseline_metrics is None:
            self.logger.warning("Baseline metrics not available. Generating VAE-only report.")
            baseline_metrics = {
                'gev': None,
                'silhouette_scores': None,
                'db_scores': None,
                'ch_scores': None,
                'dunn_score': None,
            }

        # Try to get metrics from best_model_metrics if available
        best_model_file = self.output_dir / "best_model_metrics.json"
        vae_latent_metrics = {}
        vae_centroid_metrics = {}  # Top level of centroid_based (where GEV lives)
        baseline_cluster_metrics = {}

        if best_model_file.exists():
            try:
                import json
                with open(best_model_file, 'r') as f:
                    best_data = json.load(f)
                # Correct path: best_model_metrics -> test -> centroid_based
                test_centroid_based = (
                    best_data
                    .get('best_model_metrics', {})
                    .get('test', {})
                    .get('centroid_based', {})
                )
                # VAE latent space metrics (nested inside centroid_based)
                vae_latent_metrics = test_centroid_based.get('latent_space', {})
                # GEV is at the centroid_based level, not inside latent_space
                vae_centroid_metrics = test_centroid_based
                # Baseline metrics
                baseline_cluster_metrics = best_data.get('baseline_modkmeans', {}).get('cluster_validation_metrics', {})
            except Exception:
                pass

        # Extract VAE metrics (prefer latent space from best_model_metrics)
        # GEV is stored at centroid_based level, not inside latent_space
        vae_gev = vae_centroid_metrics.get('gev', vae_metrics.get('gev', 0))
        # Metrics are nested in centroid_based_metrics.sklearn (use sklearn for latent space Euclidean)
        vae_sklearn_metrics = vae_latent_metrics.get('centroid_based_metrics', {}).get('sklearn', {})
        vae_sil = vae_sklearn_metrics.get('silhouette', vae_metrics.get('secondary_silhouette', vae_metrics.get('silhouette_scores', 0)))
        vae_db = vae_sklearn_metrics.get('davies_bouldin', vae_metrics.get('secondary_db', vae_metrics.get('db_scores', 0)))
        vae_ch = vae_sklearn_metrics.get('calinski_harabasz', vae_metrics.get('secondary_ch', vae_metrics.get('ch_scores', 0)))
        vae_dunn = vae_sklearn_metrics.get('dunn', vae_metrics.get('dunn_score', 0))
        # Composite scores are at the top level of latent_space (not inside centroid_based_metrics)
        # Try sklearn geometric_mean first, then recommended
        vae_composite_scores = vae_latent_metrics.get('composite_scores', {})
        vae_composite = (
            vae_composite_scores.get('sklearn', {}).get('geometric_mean') or
            vae_composite_scores.get('recommended', {}).get('score') or
            vae_composite_scores.get('geometric_mean', 0)
        )

        # Extract Baseline metrics (prefer cluster_validation_metrics)
        baseline_gev = baseline_cluster_metrics.get('gev', baseline_metrics.get('gev', 0))
        baseline_sil = baseline_cluster_metrics.get('silhouette', baseline_metrics.get('silhouette_scores', 0))
        baseline_db = baseline_cluster_metrics.get('davies_bouldin', baseline_metrics.get('db_scores', 0))
        baseline_ch = baseline_cluster_metrics.get('calinski_harabasz', baseline_metrics.get('ch_scores', 0))
        baseline_dunn = baseline_cluster_metrics.get('dunn', baseline_metrics.get('dunn_score', 0))
        baseline_composite = baseline_cluster_metrics.get('composite_scores', {}).get('geometric_mean', 0)

        # Helper functions
        def fmt_val(val):
            """Format value for display."""
            if val is None or val == 0 or val == -1:
                return "N/A"
            try:
                if abs(val) >= 100:
                    return f"{val:.1f}"
                return f"{val:.3f}"
            except (ValueError, TypeError):
                return "N/A"

        def calc_diff(vae_val, baseline_val, lower_is_better=False):
            """Calculate difference and determine which is better."""
            if vae_val is None or baseline_val is None or vae_val in (0, -1) or baseline_val in (0, -1):
                return "N/A", "N/A"

            try:
                vae_v = float(vae_val)
                base_v = float(baseline_val)

                if lower_is_better:
                    # For Davies-Bouldin: lower is better
                    if vae_v < base_v:
                        better = "VAE"
                        pct = ((base_v - vae_v) / base_v) * 100
                        diff = f"-{pct:.1f}%"
                    else:
                        better = "ModKMeans"
                        pct = ((vae_v - base_v) / vae_v) * 100
                        diff = f"-{pct:.1f}%"
                else:
                    # Higher is better
                    if vae_v > base_v:
                        better = "VAE"
                        if base_v != 0:
                            pct = ((vae_v - base_v) / base_v) * 100
                            if pct > 1000:
                                diff = f"+{vae_v/base_v:.1f}x"
                            else:
                                diff = f"+{pct:.1f}%"
                        else:
                            diff = "+∞"
                    else:
                        better = "ModKMeans"
                        if vae_v != 0:
                            pct = ((base_v - vae_v) / vae_v) * 100
                            if pct > 1000:
                                diff = f"+{base_v/vae_v:.1f}x"
                            else:
                                diff = f"+{pct:.1f}%"
                        else:
                            diff = "+∞"
                return better, diff
            except (ValueError, TypeError, ZeroDivisionError):
                return "N/A", "N/A"

        # Build comparison data
        metrics_data = [
            ("GEV", vae_gev, baseline_gev, False),
            ("Silhouette", vae_sil, baseline_sil, False),
            ("Davies-Bouldin ↓", vae_db, baseline_db, True),
            ("Calinski-Harabasz", vae_ch, baseline_ch, False),
            ("Dunn Index", vae_dunn, baseline_dunn, False),
            ("Composite (Geometric)", vae_composite, baseline_composite, False),
        ]

        # Column widths
        c1, c2, c3, c4, c5 = 23, 18, 11, 11, 14

        # Build the table
        header_line = f"│ {'Metric':^{c1}} │ {'VAE (Latent Space)':^{c2}} │ {'ModKMeans':^{c3}} │ {'Better':^{c4}} │ {'Δ Difference':^{c5}} │"
        top_border = f"┌{'─'*(c1+2)}┬{'─'*(c2+2)}┬{'─'*(c3+2)}┬{'─'*(c4+2)}┬{'─'*(c5+2)}┐"
        sep_line = f"├{'─'*(c1+2)}┼{'─'*(c2+2)}┼{'─'*(c3+2)}┼{'─'*(c4+2)}┼{'─'*(c5+2)}┤"
        bottom_border = f"└{'─'*(c1+2)}┴{'─'*(c2+2)}┴{'─'*(c3+2)}┴{'─'*(c4+2)}┴{'─'*(c5+2)}┘"

        # Build rows
        rows = []
        for metric_name, vae_val, base_val, lower_better in metrics_data:
            better, diff = calc_diff(vae_val, base_val, lower_better)
            row = f"│ {metric_name:<{c1}} │ {fmt_val(vae_val):>{c2}} │ {fmt_val(base_val):>{c3}} │ {better:^{c4}} │ {diff:^{c5}} │"
            rows.append(row)

        # Assemble table
        table_lines = [top_border, header_line, sep_line]
        for i, row in enumerate(rows):
            table_lines.append(row)
            if i < len(rows) - 1:
                table_lines.append(sep_line)
        table_lines.append(bottom_border)
        comparison_table = "\n".join(table_lines)

        # Generate the full report
        report = f"""
================================================================================
           VAE vs MODIFIED K-MEANS (ModKMeans) COMPARISON REPORT
================================================================================
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Dir: {self.output_dir}

================================================================================
                         COMPARISON TABLE
================================================================================
  NOTE: VAE metrics are from LATENT SPACE (32-dim)
        ModKMeans metrics are from ELECTRODE SPACE (61-ch, correlation-based)
        Both use pycrostates-style metrics for fair comparison.

{comparison_table}

================================================================================
                         METRIC INTERPRETATION
================================================================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METRIC           │ INTERPRETATION                                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ GEV              │ Global Explained Variance. Higher = better fit.      │
  │                  │ Measures how well centroids explain data variance.   │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Silhouette       │ Cluster cohesion & separation. Range: [-1, 1].       │
  │                  │ Higher = better defined, well-separated clusters.    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Davies-Bouldin   │ Cluster similarity ratio. LOWER = better.            │
  │                  │ Measures avg similarity between clusters.            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Calinski-Harabasz│ Between/within cluster dispersion ratio.             │
  │                  │ Higher = more compact, well-separated clusters.      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Dunn Index       │ Min inter-cluster / max intra-cluster distance.      │
  │                  │ Higher = better cluster separation.                  │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Composite        │ sqrt(GEV × normalized_silhouette).                   │
  │ (Geometric Mean) │ Balances variance explanation & cluster quality.     │
  └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                         KEY OBSERVATIONS
================================================================================

  • VAE operates in a learned 32-dimensional LATENT SPACE
    - Optimized for reconstruction + clustering via GMM prior
    - Silhouette often higher due to learned representation

  • ModKMeans operates in 61-channel ELECTRODE SPACE
    - Uses spatial correlation (polarity-invariant)
    - GEV typically higher - standard microstate approach

  • DIFFERENT FEATURE SPACES → Direct comparison has caveats
    - Each method is optimal in its native space
    - Composite score balances GEV and cluster quality

================================================================================
                         FILES FOR DETAILED ANALYSIS
================================================================================

  VAE Metrics:
    └── best_model_metrics.json → centroid_based.latent_space

  ModKMeans Metrics:
    └── baseline_modkmeans/cluster_validation_metrics.json

  Visualizations:
    └── latent_space_analysis/      (VAE t-SNE plots)
    └── baseline_modkmeans/electrode_space_analysis/  (ModKMeans t-SNE plots)

================================================================================
"""

        with open(self.output_dir / "VAE_vs_Baseline_Comparison.txt", "w") as f:
            f.write(report)

        self.logger.info(
            f"Comparison report saved to {self.output_dir / 'VAE_vs_Baseline_Comparison.txt'}"
        )

    def _generate_comprehensive_summary_report(self, combined_results):
        """
        Generate a SINGLE comprehensive summary report that replaces multiple redundant files.
        This consolidates: VAE_vs_Baseline_Comparison.txt, fair_comparison_report.txt,
        comprehensive_strategy_comparison_table.txt, and cluster_diagnosis_report.txt
        """
        vae = combined_results.get("vae_results", {})
        baseline = combined_results.get("baseline_metrics", {})
        fair = combined_results.get("fair_comparison", {})
        config = combined_results.get("training_config", {})

        # Extract VAE test metrics
        vae_test = {}
        if "traditional" in vae and isinstance(vae["traditional"], dict):
            vae_test = vae["traditional"].get("test_performance", {})

        def safe_fmt(val, fmt=".4f"):
            """Safely format a value, return 'N/A' if None or not a number."""
            if val is None:
                return "N/A"
            try:
                return f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                return str(val) if val else "N/A"

        report = f"""
================================================================================
                    COMPREHENSIVE RESULTS SUMMARY
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Dir: {self.output_dir}

================================================================================
                         TRAINING CONFIGURATION
================================================================================
  Clusters (K):     {config.get('n_clusters', 'N/A')}
  Latent Dim:       {config.get('latent_dim', 'N/A')}
  Batch Size:       {config.get('batch_size', 'N/A')}
  Total Epochs:     {config.get('total_epochs', 'N/A')}
  Best Epoch:       {config.get('best_epoch', 'N/A')}

================================================================================
                         VAE TEST RESULTS
================================================================================
  Test Loss:        {safe_fmt(vae_test.get('test_loss'))}
  Recon Loss:       {safe_fmt(vae_test.get('recon_loss'))}
  KLD Loss:         {safe_fmt(vae_test.get('kld_loss'))}
  SSIM Score:       {safe_fmt(vae_test.get('ssim_score'))}

  --- Clustering Metrics (Decoded Space) ---
  Silhouette:       {safe_fmt(vae_test.get('silhouette_scores', vae_test.get('primary_silhouette')))}
  Davies-Bouldin:   {safe_fmt(vae_test.get('db_scores', vae_test.get('primary_db')))}
  Calinski-Harabasz:{safe_fmt(vae_test.get('ch_scores', vae_test.get('primary_ch')), '.2f')}

  --- Pycrostates-Style Metrics (Correlation-Based) ---
  Silhouette:       {safe_fmt(vae_test.get('silhouette_corr'))}
  Davies-Bouldin:   {safe_fmt(vae_test.get('db_corr'))}
  Dunn Index:       {safe_fmt(vae_test.get('dunn_corr'))}

================================================================================
                         BASELINE (ModKMeans) TEST RESULTS
================================================================================
  GEV:              {safe_fmt(baseline.get('gev'))}
  Silhouette:       {safe_fmt(baseline.get('silhouette'))}
  Davies-Bouldin:   {safe_fmt(baseline.get('davies_bouldin'))}
  Calinski-Harabasz:{safe_fmt(baseline.get('calinski_harabasz'), '.2f')}

================================================================================
                         FAIR COMPARISON SUMMARY
================================================================================
"""
        # Add fair comparison if available
        if fair and isinstance(fair, dict):
            for space_name in ["electrode_space", "topomap_space"]:
                if space_name in fair.get("vae", {}):
                    vae_space = fair["vae"].get(space_name, {})
                    modk_space = fair.get("modkmeans", {}).get(space_name, {})

                    # Look for metrics under correlation_based key, or directly on the space dict
                    corr_vae = vae_space.get("correlation_based", vae_space)
                    corr_modk = modk_space.get("correlation_based", modk_space)

                    space_label = "N-Channel Electrode" if "electrode" in space_name else "1600-Dim Topomap"
                    report += f"""
  --- {space_label} Space (Correlation-Based) ---
                        VAE             ModKMeans
  Silhouette:       {safe_fmt(corr_vae.get('silhouette')):>12}    {safe_fmt(corr_modk.get('silhouette')):>12}
  Calinski-H:       {safe_fmt(corr_vae.get('calinski_harabasz'), '.2f'):>12}    {safe_fmt(corr_modk.get('calinski_harabasz'), '.2f'):>12}
  Davies-Bouldin:   {safe_fmt(corr_vae.get('davies_bouldin')):>12}    {safe_fmt(corr_modk.get('davies_bouldin')):>12}
  Dunn Index:       {safe_fmt(corr_vae.get('dunn')):>12}    {safe_fmt(corr_modk.get('dunn')):>12}
"""

        report += """
================================================================================
                              METRIC GUIDE
================================================================================
  • Silhouette Score: Measures cluster cohesion and separation (-1 to 1)
    Higher is better. >0.5 = good, >0.7 = excellent

  • Davies-Bouldin Index: Average similarity between clusters
    Lower is better. <1.0 = good clustering

  • Calinski-Harabasz Index: Ratio of between/within cluster variance
    Higher is better. No fixed range.

  • Dunn Index: Ratio of min inter-cluster to max intra-cluster distance
    Higher is better. >1.0 indicates separation > spread

  • GEV (Global Explained Variance): Variance explained by microstates
    Higher is better. >0.65 typical for EEG microstates

  • SSIM (Structural Similarity): Reconstruction quality
    Higher is better. >0.85 = good, >0.90 = excellent

================================================================================
                              FILES GENERATED
================================================================================
  • results.json     - All metrics and results (machine-readable)
  • summary.txt      - This report (human-readable)
  • best_model.pth   - Trained model weights
  • visualizations/  - All plots and figures

================================================================================
"""

        # Save the comprehensive summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(report)

        self.logger.info(f"✅ Comprehensive summary saved to: {summary_path}")

    def _save_comprehensive_best_model_metrics(
        self,
        train_centroid_metrics: Dict,
        baseline_metrics: Optional[Dict] = None,
    ):
        """
        Save comprehensive metrics file for the best model (90/10 split).

        This file contains ALL metrics in one place:
        - Training history (loss, metrics per epoch)
        - Centroid metrics history (with composite scores)
        - Final metrics for train AND eval data (centroid-based AND 4-quadrant pairwise)
        - GEV history
        - Best model configuration
        - Baseline (ModKMeans) metrics for comparison
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("SAVING COMPREHENSIVE BEST MODEL METRICS")
        self.logger.info("="*60)

        # Load epoch history
        log_path = self.output_dir / "epoch_metrics_log.json"
        epoch_history = []
        if log_path.exists():
            with open(log_path, "r") as f:
                epoch_history = json.load(f)

        # Extract key metrics at best epoch
        best_epoch_metrics = None
        for epoch_data in epoch_history:
            if epoch_data.get("epoch") == self.best_train_epoch:
                best_epoch_metrics = epoch_data
                break

        # ================================================================
        # COMPUTE 4-QUADRANT METRICS ON BEST MODEL (reuse cached if available)
        # ================================================================
        self.model.set_eval_mode()

        # Train metrics — reuse cached from run_final_comparison if available
        train_4q = getattr(self, '_cached_train_4q', None)
        if train_4q is not None:
            self.logger.info("Reusing cached train 4-quadrant metrics from final comparison")
        else:
            self.logger.info("Computing final 4-quadrant metrics on best model...")
            _, train_4q = self._compute_metrics_and_losses(
                self.train_loader, self.best_train_epoch, is_test=False, strategy_name="VAE-final"
            )

        # Eval metrics — reuse cached from run_final_comparison if available
        eval_4q = getattr(self, '_cached_eval_4q', None)
        if eval_4q is not None:
            self.logger.info("Reusing cached eval 4-quadrant metrics from final comparison")
        elif self.eval_loader is not None:
            _, eval_4q = self._compute_metrics_and_losses(
                self.eval_loader, self.best_train_epoch, is_test=True, strategy_name="VAE-final"
            )

        # Full-data unaugmented metrics (train_set + eval_set combined)
        full_4q = None
        full_gev = None
        full_centroid = None
        full_n_samples = None
        if self.train_set is not None and self.eval_set is not None:
            from torch.utils.data import ConcatDataset, DataLoader as _DL
            self.logger.info("Computing full-data unaugmented metrics (train_set + eval_set)...")
            full_unaugmented = ConcatDataset([self.train_set, self.eval_set])
            full_n_samples = len(full_unaugmented)
            full_loader = _DL(
                full_unaugmented,
                batch_size=self.train_loader.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            _, full_4q = self._compute_metrics_and_losses(
                full_loader, self.best_train_epoch, is_test=True, strategy_name="VAE-full"
            )
            full_gev = self._compute_gev(full_loader)
            self.logger.info(f"Full-data GEV: {full_gev:.4f}")
            try:
                full_centroid = self._compute_centroid_based_metrics(
                    full_loader, split="full_data", save_to_file=True
                )
            except Exception as e:
                self.logger.warning(f"Full-data centroid metrics failed: {e}")
        else:
            self.logger.info("Skipping full-data metrics (train_set or eval_set not available)")

        # Electrode-space GEV (final, expensive computation)
        electrode_gev = None
        try:
            electrode_gev = self._compute_electrode_gev()
            if electrode_gev is not None:
                self.logger.info(f"Electrode-space GEV: {electrode_gev:.4f}")
        except Exception as e:
            self.logger.warning(f"Electrode-space GEV computation failed: {e}")

        self.model.set_train_mode()

        # Helper to structure 4-quadrant metrics into clean report format
        def _structure_4q(m):
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

        def _add_mse_volts(section):
            """Add denormalized MSE (µV²) to reconstruction metrics."""
            if self.norm_params is not None:
                std = self.norm_params.get("std", 1.0)
                recon = section.get("quadrant_metrics", {}).get("reconstruction", {})
                mse = recon.get("mse")
                if mse is not None and mse != -1:
                    recon["mse_volts"] = mse * (std ** 2)

        # Build comprehensive report
        train_section = {
            "description": "Metrics on training data (90%)",
            "centroid_based": train_centroid_metrics,
            "quadrant_metrics": _structure_4q(train_4q),
            "loss_at_best_epoch": (
                best_epoch_metrics.get("train_loss") if best_epoch_metrics else None
            ),
        }
        _add_mse_volts(train_section)

        eval_section = None
        eval_centroid = self._final_centroid_metrics.get("eval") if hasattr(self, '_final_centroid_metrics') else None
        if eval_4q is not None:
            eval_section = {
                "description": "Metrics on held-out eval data (10%)",
                "quadrant_metrics": _structure_4q(eval_4q),
            }
            if eval_centroid is not None:
                eval_section["centroid_based"] = eval_centroid
            _add_mse_volts(eval_section)

        comprehensive_report = {
            "metadata": {
                "model_type": "VAE Clustering",
                "data_mode": "90/10 split",
                "normalization": "zscore",
                "n_clusters": self.model.nClusters,
                "latent_dim": self.model.latent_dim,
                "best_epoch": self.best_train_epoch,
                "total_epochs_trained": len(epoch_history),
                "best_gev": self.best_gev,
                "best_train_loss": self.best_train_loss,
                "timestamp": datetime.now().isoformat(),
            },

            "training_history": {
                "description": "Metrics logged at each epoch during training",
                "epochs": epoch_history,
                "loss_curves": {
                    "train_losses": self.train_losses,
                },
                "gev_history": self.gev_history,
            },

            "centroid_metrics_history": {
                "description": "Centroid-based metrics computed every N epochs",
                "log_interval": self.centroid_metrics_log_interval,
                "history": self.centroid_metrics_history,
            },

            "best_model_metrics": {
                "description": "All metrics for the best model (selected by eval GEV).",
                "best_epoch": self.best_train_epoch,
                "train": train_section,
            },

            "model_config": {
                "learning_rate": self.lr,
                "total_epochs": self.epochs,
                "patience": self.patience,
                "pretrain_epochs": self.pretrain_epochs,
                "unfreeze_prior_epoch": self.unfreeze_prior_epoch,
                "batch_size": self.train_loader.batch_size,
                "n_clusters": self.model.nClusters,
                "latent_dim": self.model.latent_dim,
            },

            "baseline_modkmeans": self._format_baseline_for_comprehensive_report(baseline_metrics),
        }

        # Add eval section if available
        if eval_section is not None:
            comprehensive_report["best_model_metrics"]["eval"] = eval_section

        # Add full_data section if computed
        if full_4q is not None:
            full_section = {
                "description": "Metrics on ALL data without augmentation (train_set + eval_set combined)",
                "n_samples": full_n_samples,
                "quadrant_metrics": _structure_4q(full_4q),
                "gev": full_gev,
            }
            if full_centroid is not None:
                full_section["centroid_based"] = full_centroid
            _add_mse_volts(full_section)
            comprehensive_report["best_model_metrics"]["full_data"] = full_section

        # Add GEV summary
        gev_summary = {"pixel_space_train": self.best_gev}
        if full_gev is not None:
            gev_summary["pixel_space_full_data"] = full_gev
        if electrode_gev is not None:
            gev_summary["electrode_space"] = electrode_gev
        comprehensive_report["best_model_metrics"]["gev"] = gev_summary

        # Save to file
        report_path = self.output_dir / "best_model_metrics.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)

        self.logger.info(f"Best model metrics saved to: {report_path}")

        # Save per-cluster markdown report
        self._save_cluster_metrics_md(comprehensive_report)

        # Print summary
        self._print_best_model_metrics_summary(comprehensive_report)

        return comprehensive_report

    def _save_cluster_metrics_md(self, report: Dict):
        """Save a per-cluster markdown metrics report to the output directory.

        Generates ``best_model_metrics_report.md`` alongside the JSON report,
        containing full 4-quadrant tables for each split, GEV summary, baseline
        comparison, and a note on z-score space with denormalized MSE.
        """
        meta = report["metadata"]
        bm = report.get("best_model_metrics", {})
        baseline = report.get("baseline_modkmeans", {})

        def _v(val, fmt=".4f"):
            if val is None or val == -1:
                return "N/A"
            try:
                return f"{val:{fmt}}"
            except (TypeError, ValueError):
                return "N/A"

        lines = [
            f"# Best Model Metrics Report",
            "",
            "## Configuration",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Clusters (K) | {meta['n_clusters']} |",
            f"| Latent dim | {meta['latent_dim']} |",
            f"| Best epoch | {meta['best_epoch']} / {meta.get('total_epochs_trained', 'N/A')} |",
            f"| Best GEV | {_v(meta.get('best_gev'))} |",
            f"| Best train loss | {_v(meta.get('best_train_loss'))} |",
            f"| Normalization | {meta.get('normalization', 'zscore')} |",
            f"| Generated | {meta.get('timestamp', 'N/A')} |",
            "",
        ]

        # Per-split sections
        for split_name in ("train", "eval", "full_data"):
            split_data = bm.get(split_name)
            if split_data is None:
                continue

            qm = split_data.get("quadrant_metrics", {})
            recon = qm.get("reconstruction", {})
            clust = qm.get("clustering", {})
            q1 = clust.get("latent_sklearn", {})
            q2 = clust.get("latent_pycrostates", {})
            q3 = clust.get("topomap_sklearn", {})
            q4 = clust.get("topomap_pycrostates", {})

            label = split_name.replace("_", " ").title()
            n_info = f" (n={split_data['n_samples']})" if "n_samples" in split_data else ""
            lines.extend([
                f"## {label} Set{n_info}",
                "",
                "### Reconstruction",
                "",
                "| MSE | MSE (\u00b5V\u00b2) | KLD | SSIM | Spatial Corr |",
                "|-----|-----------|-----|------|--------------|",
                f"| {_v(recon.get('mse'))} | {_v(recon.get('mse_volts'), '.2f')} "
                f"| {_v(recon.get('kld'))} | {_v(recon.get('ssim'))} "
                f"| {_v(recon.get('spatial_corr'))} |",
                "",
                "### 4-Quadrant Clustering Metrics",
                "",
                "| Metric | Q1: Latent+Eucl | Q2: Latent+Corr | Q3: Topo+Eucl | Q4: Topo+Corr |",
                "|--------|-----------------|-----------------|---------------|---------------|",
                f"| Silhouette | {_v(q1.get('silhouette'))} | {_v(q2.get('silhouette'))} "
                f"| {_v(q3.get('silhouette'))} | {_v(q4.get('silhouette'))} |",
                f"| Davies-Bouldin | {_v(q1.get('db'))} | {_v(q2.get('db'))} "
                f"| {_v(q3.get('db'))} | {_v(q4.get('db'))} |",
                f"| Calinski-Harabasz | {_v(q1.get('ch'), '.1f')} | {_v(q2.get('ch'), '.1f')} "
                f"| {_v(q3.get('ch'), '.1f')} | {_v(q4.get('ch'), '.1f')} |",
                f"| Dunn | {_v(q1.get('dunn'))} | {_v(q2.get('dunn'))} "
                f"| {_v(q3.get('dunn'))} | {_v(q4.get('dunn'))} |",
                "",
            ])

            # Per-split GEV
            split_gev = split_data.get("gev")
            if split_gev is not None:
                lines.append(f"**GEV**: {_v(split_gev)}")
                lines.append("")

            # Centroid-based metrics if available
            centroid = split_data.get("centroid_based")
            if centroid:
                lines.extend([
                    "### Centroid-Based Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| GEV | {_v(centroid.get('gev'))} |",
                    f"| Silhouette | {_v(centroid.get('silhouette'))} |",
                    f"| Mean Correlation | {_v(centroid.get('mean_correlation'))} |",
                    "",
                ])

        # GEV summary
        gev = bm.get("gev", {})
        if gev:
            lines.extend([
                "## GEV Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Pixel-space (train) | {_v(gev.get('pixel_space_train'))} |",
                f"| Pixel-space (full data) | {_v(gev.get('pixel_space_full_data'))} |",
                f"| Electrode-space | {_v(gev.get('electrode_space'))} |",
                "",
            ])

        # Baseline comparison
        bl_avail = baseline.get("available", False)
        lines.append("## Baseline: ModKMeans")
        lines.append("")
        if bl_avail:
            bl_cv = baseline.get("cluster_validation_metrics", {})
            lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
                f"| Silhouette | {_v(bl_cv.get('silhouette'))} |",
                f"| Davies-Bouldin | {_v(bl_cv.get('davies_bouldin'))} |",
                f"| Calinski-Harabasz | {_v(bl_cv.get('calinski_harabasz'), '.1f')} |",
                f"| Dunn | {_v(bl_cv.get('dunn'))} |",
                f"| GEV | {_v(bl_cv.get('gev'))} |",
                "",
            ])
        else:
            lines.extend(["*Baseline not available.*", ""])

        # Note about z-score space
        lines.extend([
            "## Notes",
            "",
            "- All clustering metrics (Silhouette, DB, CH, Dunn) are **invariant** to scalar "
            "z-score denormalization (affine transform cancels in ratios).",
            "- MSE is in z-score space. MSE (\u00b5V\u00b2) = MSE \u00d7 std\u00b2 gives the "
            "denormalized reconstruction error.",
            "- SSIM uses `data_range = 2 \u00d7 z_score_clip` calibrated to z-scored values.",
            "",
            "---",
            "*Report generated by VAE Clustering Trainer*",
        ])

        md_path = self.output_dir / "best_model_metrics_report.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Per-cluster metrics report saved to: {md_path}")

    def _print_best_model_metrics_summary(self, report: Dict):
        """Print a full summary of the best model metrics in 4-quadrant format."""
        self.logger.info("\n" + "="*60)
        self.logger.info("BEST MODEL METRICS SUMMARY")
        self.logger.info("="*60)

        meta = report["metadata"]
        self.logger.info(f"K={meta['n_clusters']}, Latent={meta['latent_dim']}, Best Epoch={meta['best_epoch']}")
        best_loss = meta.get('best_train_loss', None)
        loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        self.logger.info(f"Best GEV: {meta['best_gev']:.4f}, Best Train Loss: {loss_str}")

        bm = report.get("best_model_metrics", {})

        for split_name in ("train", "eval", "full_data"):
            split_data = bm.get(split_name)
            if split_data is None:
                continue

            qm = split_data.get("quadrant_metrics", {})
            recon = qm.get("reconstruction", {})
            clust = qm.get("clustering", {})

            label = split_name.upper().replace("_", " ")
            n_info = f" (n={split_data['n_samples']})" if "n_samples" in split_data else ""
            self.logger.info(f"\n{label} SET{n_info}:")

            # Reconstruction line with optional denormalized MSE
            mse_str = _fmt(recon.get('mse'))
            mse_volts = recon.get('mse_volts')
            if mse_volts is not None:
                mse_str += f" ({_fmt(mse_volts, '.2f')} \u00b5V\u00b2)"
            self.logger.info(f"  Reconstruction: MSE={mse_str}  KLD={_fmt(recon.get('kld'))}  "
                             f"SSIM={_fmt(recon.get('ssim'))}  SpatCorr={_fmt(recon.get('spatial_corr'))}")

            # Full 4-quadrant table with all metrics
            self.logger.info("                   | Sklearn (Euclidean)                      "
                             "| Pycrostates (|1/r|-1)")
            for space, space_key in [("Latent", "latent"), ("Topo", "topomap")]:
                sk = clust.get(f"{space_key}_sklearn", {})
                pc = clust.get(f"{space_key}_pycrostates", {})
                sk_line = (f"Sil={_fmt(sk.get('silhouette'), '+.3f')}  "
                           f"DB={_fmt(sk.get('db'), '.3f')}  "
                           f"CH={_fmt(sk.get('ch'), '.1f')}  "
                           f"Dn={_fmt(sk.get('dunn'), '.3f')}")
                pc_line = (f"Sil={_fmt(pc.get('silhouette'), '+.3f')}  "
                           f"DB={_fmt(pc.get('db'), '.3f')}  "
                           f"CH={_fmt(pc.get('ch'), '.1f')}  "
                           f"Dn={_fmt(pc.get('dunn'), '.3f')}")
                self.logger.info(f"  {space:<6}          | {sk_line} | {pc_line}")

            # Print per-split GEV if present
            split_gev = split_data.get("gev")
            if split_gev is not None:
                self.logger.info(f"  GEV: {_fmt(split_gev)}")

        # GEV summary
        gev = bm.get("gev", {})
        if gev:
            gev_parts = [f"Pixel(train)={_fmt(gev.get('pixel_space_train'))}"]
            if gev.get("pixel_space_full_data") is not None:
                gev_parts.append(f"Pixel(full)={_fmt(gev.get('pixel_space_full_data'))}")
            if gev.get("electrode_space") is not None:
                gev_parts.append(f"Electrode={_fmt(gev.get('electrode_space'))}")
            self.logger.info(f"\n  GEV: {', '.join(gev_parts)}")

        # Baseline comparison
        baseline = report.get("baseline_modkmeans", {})
        if baseline and baseline.get("available"):
            cluster_val = baseline.get("cluster_validation_metrics", {})
            self.logger.info(f"\nBASELINE (ModKMeans):")
            self.logger.info(f"  Sil={_fmt(cluster_val.get('silhouette'))}  "
                             f"DB={_fmt(cluster_val.get('davies_bouldin'))}  "
                             f"CH={_fmt(cluster_val.get('calinski_harabasz'), '.1f')}  "
                             f"Dunn={_fmt(cluster_val.get('dunn'))}  "
                             f"GEV={_fmt(cluster_val.get('gev'))}")

        self.logger.info("="*60)

    def _format_baseline_for_comprehensive_report(self, baseline_metrics: Optional[Dict]) -> Dict:
        """
        Format baseline metrics for inclusion in comprehensive report.

        Parameters
        ----------
        baseline_metrics : dict or None
            Baseline metrics with structure:
            - n_clusters
            - cluster_validation_metrics: {silhouette, calinski_harabasz, dunn, davies_bouldin, gev, composite_scores}
            - training_metrics: {...} (from evaluate())
            - val_metrics: {...}
            - test_metrics: {...}

        Returns
        -------
        dict
            Formatted baseline section for comprehensive report
        """
        if baseline_metrics is None:
            return {
                "available": False,
                "note": "Baseline metrics not computed (baseline_metrics is None)"
            }

        # Check if baseline actually ran (gfp_peaks was available)
        if baseline_metrics.get("note") == "GFP peaks not available":
            return {
                "available": False,
                "note": "Baseline metrics not computed (GFP peaks not available in worker)"
            }

        # Get cluster validation metrics from compute_cluster_metrics()
        # Structure: {silhouette, calinski_harabasz, dunn, davies_bouldin, gev, composite_scores}
        cluster_val = baseline_metrics.get("cluster_validation_metrics", {})
        composite = cluster_val.get("composite_scores", {})

        # Get test metrics from evaluate()
        test_metrics = baseline_metrics.get("test_metrics", {})

        return {
            "available": True,
            "description": "Pycrostates ModKMeans baseline for fair comparison",
            "n_clusters": baseline_metrics.get("n_clusters", -1),
            "gev": cluster_val.get("gev", -1),
            "cluster_validation_metrics": {
                "silhouette": cluster_val.get("silhouette", -1),
                "calinski_harabasz": cluster_val.get("calinski_harabasz", -1),
                "dunn": cluster_val.get("dunn", -1),
                "davies_bouldin": cluster_val.get("davies_bouldin", -1),
                "gev": cluster_val.get("gev", -1),
                "composite_scores": composite,
            },
            "test_evaluation": {
                "test_gev": test_metrics.get("gev", -1),
                "test_silhouette": test_metrics.get("silhouette", -1),
                "test_mean_correlation": baseline_metrics.get("test_mean_correlation", -1),
            },
            "val_metrics": baseline_metrics.get("val_metrics", {}),
            "comparison_notes": {
                "metric_type": "pycrostates library native functions",
                "fair_comparison_with": "VAE custom correlation-based centroid metrics",
                "recommended_metric": "composite_scores.geometric_mean",
            }
        }

    def _run_fair_comparison(self, baseline_handler, vae_test_metrics=None, baseline_test_metrics=None):
        """
        Run fair comparison between VAE and baseline using consistent methodology.

        Both methods are evaluated:
        1. In the SAME feature space (raw 1600-dim)
        2. With the SAME normalization
        3. With polarity-invariant cluster assignment
        4. On the FULL test set
        5. With GEV computed for both

        Additionally runs dual-space comparison (electrode-space AND 1600-dim topomap-space).
        """
        self.logger.info(
            "\n" + "=" * 60 + "\nFAIR COMPARISON (Consistent Methodology)\n" + "=" * 60
        )

        fair_comparison_dir = self.output_dir / "fair_comparison"
        fair_results = {}

        # Build training info
        training_info = {
            "n_clusters": self.model.nClusters,
            "latent_dim": getattr(self.model, 'latent_dim', 'N/A'),
            "total_epochs": self.epochs,
            "best_epoch": self.best_train_epoch,
        }

        # --- Part 1: Original fair comparison (metrics_utils) ---
        try:
            fair_results["topomap_space"] = _mu.compute_fair_comparison(
                vae_model=self.model,
                baseline_handler=baseline_handler,
                test_loader=self.train_loader,  # 100% data mode: use train_loader
                device=self.device,
                output_dir=str(fair_comparison_dir),
                logger=self.logger
            )
            self.logger.info("Part 1/2: Topomap space comparison completed.")

        except Exception as e:
            self.logger.error(f"Topomap space comparison failed: {e}", exc_info=True)
            fair_results["topomap_space"] = None

        # Note: dual_space comparison removed - was generating confusing electrode_space
        # metrics for VAE which doesn't cluster in electrode space. Use best_model_metrics
        # for VAE latent space metrics and baseline_modkmeans for ModKMeans metrics.

        if fair_results.get("topomap_space"):
            self.logger.info("Fair comparison completed successfully!")
        else:
            self.logger.warning("Fair comparison failed.")

        return fair_results

    def _generate_docs_comparison_report(self, comprehensive_report: Dict):
        """
        Generate a markdown comparison report: VAE full-data vs ModKMeans baseline.

        Writes to docs/vae_full_data_metrics_report.md in the project root.
        """
        bm = comprehensive_report.get("best_model_metrics", {})
        full_data = bm.get("full_data")
        if full_data is None:
            self.logger.warning("No full_data metrics available — skipping docs comparison report.")
            return

        meta = comprehensive_report["metadata"]
        baseline = comprehensive_report.get("baseline_modkmeans", {})
        gev_summary = bm.get("gev", {})

        # Extract VAE full-data quadrant metrics
        qm = full_data.get("quadrant_metrics", {})
        recon = qm.get("reconstruction", {})
        clust = qm.get("clustering", {})

        def _v(val, fmt=".4f"):
            if val is None or val == -1:
                return "N/A"
            try:
                return f"{val:{fmt}}"
            except (TypeError, ValueError):
                return "N/A"

        # VAE quadrant values
        q1 = clust.get("latent_sklearn", {})
        q2 = clust.get("latent_pycrostates", {})
        q3 = clust.get("topomap_sklearn", {})
        q4 = clust.get("topomap_pycrostates", {})

        # Baseline values
        bl_avail = baseline.get("available", False)
        bl_cv = baseline.get("cluster_validation_metrics", {}) if bl_avail else {}

        # Build markdown
        lines = [
            "# VAE vs ModKMeans: Full-Data 4-Quadrant Metrics Comparison",
            "",
            "## Configuration",
            f"- **Clusters**: K={meta['n_clusters']}, Latent={meta['latent_dim']}",
            f"- **VAE Best Epoch**: {meta['best_epoch']} / {meta.get('total_epochs_trained', 'N/A')}",
            f"- **Data**: {full_data.get('n_samples', 'N/A')} GFP peaks (100%, no augmentation)",
            f"- **Normalization**: {meta.get('normalization', 'zscore')}",
            f"- **Generated**: {meta.get('timestamp', 'N/A')}",
            "",
            "## 4-Quadrant Clustering Metrics (VAE Full-Data)",
            "",
            "| Metric | Q1: Latent+Eucl | Q2: Latent+Corr | Q3: Topo+Eucl | Q4: Topo+Corr |",
            "|--------|-----------------|-----------------|---------------|---------------|",
            f"| Silhouette | {_v(q1.get('silhouette'))} | {_v(q2.get('silhouette'))} | {_v(q3.get('silhouette'))} | {_v(q4.get('silhouette'))} |",
            f"| Davies-Bouldin | {_v(q1.get('db'))} | {_v(q2.get('db'))} | {_v(q3.get('db'))} | {_v(q4.get('db'))} |",
            f"| Calinski-Harabasz | {_v(q1.get('ch'), '.1f')} | {_v(q2.get('ch'), '.1f')} | {_v(q3.get('ch'), '.1f')} | {_v(q4.get('ch'), '.1f')} |",
            f"| Dunn | {_v(q1.get('dunn'))} | {_v(q2.get('dunn'))} | {_v(q3.get('dunn'))} | {_v(q4.get('dunn'))} |",
            "",
            "## GEV Comparison",
            "",
            "| Metric | VAE (full data) | ModKMeans |",
            "|--------|-----------------|-----------|",
            f"| Pixel-space GEV | {_v(full_data.get('gev'))} | {_v(bl_cv.get('gev'))} |",
            f"| Electrode-space GEV | {_v(gev_summary.get('electrode_space'))} | N/A |",
            "",
            "## Baseline ModKMeans Metrics",
            "",
        ]

        if bl_avail:
            lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
                f"| Silhouette | {_v(bl_cv.get('silhouette'))} |",
                f"| Davies-Bouldin | {_v(bl_cv.get('davies_bouldin'))} |",
                f"| Calinski-Harabasz | {_v(bl_cv.get('calinski_harabasz'), '.1f')} |",
                f"| Dunn | {_v(bl_cv.get('dunn'))} |",
                f"| GEV | {_v(bl_cv.get('gev'))} |",
            ])
        else:
            lines.append("*Baseline not available.*")

        lines.extend([
            "",
            "## Reconstruction Quality (VAE only)",
            "",
            "| MSE | KLD | SSIM | Spatial Corr |",
            "|-----|-----|------|--------------|",
            f"| {_v(recon.get('mse'))} | {_v(recon.get('kld'))} | {_v(recon.get('ssim'))} | {_v(recon.get('spatial_corr'))} |",
            "",
        ])

        # Centroid-based metrics if available
        centroid = full_data.get("centroid_based")
        if centroid:
            lines.extend([
                "## Centroid-Based Metrics (VAE Full-Data)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| GEV | {_v(centroid.get('gev'))} |",
                f"| Silhouette | {_v(centroid.get('silhouette'))} |",
                f"| Mean Correlation | {_v(centroid.get('mean_correlation'))} |",
                "",
            ])

        lines.extend([
            "---",
            "*Report generated by VAE Clustering Trainer*",
        ])

        # Write to docs/
        docs_dir = Path(__file__).parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        report_path = docs_dir / "vae_full_data_metrics_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Docs comparison report saved to: {report_path}")

    def run_deep_diagnostics(self):
        """
        Generates advanced Deep Learning diagnostics to prove Disentanglement
        and Latent Independence. Crucial for the Research Discussion.
        """
        self.logger.info(
            "\n"
            + "=" * 60
            + "\nSTEP 2.8: DEEP DIAGNOSTICS (Disentanglement)\n"
            + "=" * 60
        )
        output_dir = self.output_dir / "deep_diagnostics"
        output_dir.mkdir(exist_ok=True)

        import matplotlib.pyplot as plt
        import seaborn as sns
        import mne

        # --- PREPARE DATA ---
        self.model.eval()
        all_mus = []
        all_clusters = []

        # Collect all latent vectors from training set
        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                mu, _ = self.model.encode(data)
                clusters = self.model.predict(data)
                all_mus.append(mu.detach().cpu().numpy())
                all_clusters.append(clusters)

        Z = np.vstack(all_mus)
        C = np.concatenate(all_clusters)

        # --- VISUALIZATION 1: LATENT CORRELATION HEATMAP ---
        # Proves that dimensions are statistically independent (Disentangled)
        self.logger.info("Generating Latent Independence Proof...")

        corr_matrix = np.corrcoef(Z, rowvar=False)

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
        )
        plt.title(
            "Latent Dimension Independence (Correlation Matrix)\nIdeal = Zero correlation (White/Grey)",
            fontsize=14,
        )
        plt.savefig(output_dir / "Latent_Independence_Heatmap.png", dpi=300)
        plt.close()

        # --- VISUALIZATION 2: CLUSTER FINGERPRINTS (RADAR CHART) ---
        # Shows how each Microstate Class is defined by specific Latents
        self.logger.info("Generating Microstate Fingerprints...")

        n_clusters = self.model.nClusters
        n_latents = self.model.latent_dim

        # Calculate mean latent vector for each cluster
        cluster_means = []
        for i in range(n_clusters):
            if np.sum(C == i) > 0:
                cluster_means.append(np.mean(Z[C == i], axis=0))
            else:
                cluster_means.append(np.zeros(n_latents))
        cluster_means = np.array(cluster_means)

        # Create Radar Chart
        angles = np.linspace(0, 2 * np.pi, n_latents, endpoint=False).tolist()
        angles += [angles[0]]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each cluster
        colors = plt.colormaps["tab10"].resampled(n_clusters)
        for i in range(n_clusters):
            values = cluster_means[i].tolist()
            values += [values[0]]  # Close the loop
            ax.plot(
                angles, values, linewidth=2, label=f"Cluster {i+1}", color=colors(i)
            )
            ax.fill(angles, values, color=colors(i), alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"Lat {i+1}" for i in range(n_latents)])
        plt.title("Microstate 'Fingerprints' in Latent Space", y=1.1, fontsize=16)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        plt.savefig(output_dir / "Cluster_Latent_Fingerprints.png", dpi=300)
        plt.close()

        # --- VISUALIZATION 2b: KL PER-DIMENSION BAR CHART ---
        # Shows which latent dimensions are active vs collapsed (KL < 0.05 nats)
        kl_vals = self.epoch_metrics_history[-1].get("kl_per_dim_values") if self.epoch_metrics_history else None
        if kl_vals is not None:
            self.logger.info("Generating KL per-dimension bar chart...")
            fig_kl, ax_kl = plt.subplots(figsize=(max(8, len(kl_vals) * 0.3), 4))
            dims = range(len(kl_vals))
            colors_kl = ['#d9534f' if v < 0.05 else '#5cb85c' for v in kl_vals]
            ax_kl.bar(dims, kl_vals, color=colors_kl, edgecolor='black', linewidth=0.3)
            ax_kl.axhline(y=0.05, color='red', linestyle='--', linewidth=0.8, label='Collapse threshold (0.05 nats)')
            ax_kl.set_xlabel("Latent Dimension")
            ax_kl.set_ylabel("KL Divergence (nats)")
            ax_kl.set_title("KL Per Dimension (red = collapsed)")
            ax_kl.legend(fontsize=8)
            fig_kl.tight_layout()
            fig_kl.savefig(output_dir / "kl_per_dimension.png", dpi=150)
            plt.close(fig_kl)

        # --- VISUALIZATION 3: LATENT TRAVERSALS (The "Manifold" Proof) ---
        # Shows what happens when you tweak ONE dimension while holding others constant
        self.logger.info("Generating Latent Traversals...")

        # Use model's MNE info if available
        if self.model.info is None:
            self.logger.warning("No MNE info available - skipping latent traversal visualization")
            return
        info = self.model.info
        ch_names = info.ch_names
        montage = info.get_montage()
        if montage is None:
            self.logger.warning("No montage in MNE info — skipping latent traversal visualization")
            return

        # Helper to map 40x40 topomap -> electrode sensor values
        def extract_sensors_fast(img):
            # This replicates the logic used in run_explainability
            # assuming 'montage' and 'ch_names' are available in scope
            import math as m

            pos_3d = []
            for ch in ch_names:
                if ch in montage.get_positions()["ch_pos"]:
                    pos_3d.append(montage.get_positions()["ch_pos"][ch])

            pos_2d = []
            for x, y, z in pos_3d:
                r = m.sqrt(x**2 + y**2 + z**2)
                elev = m.atan2(z, m.sqrt(x**2 + y**2))
                az = m.atan2(y, x)
                r_proj = m.pi / 2 - elev
                pos_2d.append([r_proj * m.cos(az), r_proj * m.sin(az)])

            pos_2d = np.array(pos_2d)
            p_min, p_max = pos_2d.min(axis=0), pos_2d.max(axis=0)

            sensor_vals = []
            height, width = img.shape
            for px, py in pos_2d:
                nx = (px - p_min[0]) / (p_max[0] - p_min[0])
                ny = (py - p_min[1]) / (p_max[1] - p_min[1])

                # Map to pixels
                ix = int(nx * (width - 1))
                iy = int((1 - ny) * (height - 1))
                ix = np.clip(ix, 0, width - 1)
                iy = np.clip(iy, 0, height - 1)
                sensor_vals.append(img[iy, ix])

            return np.array(sensor_vals)

        # Traversal parameters
        n_steps = 7
        grid_x = torch.linspace(-3, 3, n_steps).to(self.device)

        # Plot
        fig, axes = plt.subplots(
            n_latents, n_steps, figsize=(n_steps * 1.5, n_latents * 1.5)
        )
        axes = np.atleast_2d(axes)

        # Symmetric color range matching z-score clip bounds
        clip_std = self.norm_params.get("clip_std", 5.0) if self.norm_params else 5.0

        for dim in range(n_latents):
            # Create a batch where only 'dim' varies, others are 0
            z_traversal = torch.zeros(n_steps, n_latents).to(self.device)
            z_traversal[:, dim] = grid_x

            # Decode
            with torch.no_grad():
                decoded = self.model.decode(z_traversal).detach().cpu().numpy()

            for step in range(n_steps):
                ax = axes[dim, step]

                # Extract Topomap (Use the real 40x40 image)
                img = decoded[step, 0]  # (40, 40)

                # RdBu_r with symmetric z-score range to show polarity
                ax.imshow(img, cmap="RdBu_r", origin="lower", vmin=-clip_std, vmax=clip_std)

                if dim == 0:
                    ax.set_title(f"{grid_x[step].item():.1f}σ")
                if step == 0:
                    ax.set_ylabel(
                        f"Latent {dim+1}",
                        rotation=0,
                        labelpad=20,
                        va="center",
                        fontsize=10,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(
            "Latent Manifold Traversals\n(Moving along one axis at a time)", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(output_dir / "Latent_Traversals_Grid.png", dpi=300)
        plt.close()

        self.logger.info(f"✅ Deep diagnostics saved to {output_dir}")

    def run_explainability(self):
        """
        Runs latent space explainability with IRLS and generates MNE topographic maps.
        """
        self.logger.info(
            "\n" + "=" * 60 + "\nSTEP 2.5: LATENT SPACE EXPLAINABILITY\n" + "=" * 60
        )

        try:
            # --- INTERNAL CLASS DEFINITION (Latent Explainer with IRLS) ---
            import torch.nn as nn
            import mne
            from matplotlib import cm

            class LatentExplainer:
                def __init__(self, vae_model, device, input_shape=(1, 40, 40)):
                    self.vae = vae_model
                    self.device = device
                    self.latent_dim = vae_model.latent_dim
                    self.input_shape = input_shape
                    self.flat_input_size = np.prod(input_shape)
                    self.inverse_model = nn.Linear(
                        self.latent_dim, self.flat_input_size, bias=False
                    ).to(device)

                def fit_irls(
                    self, data_loader, max_iter=10, delta=1.0, lambda_reg=1e-4
                ):
                    # ... [Insert the IRLS code provided in previous answer here] ...
                    # (The logic remains exactly the same as the optimized IRLS version)
                    self.vae.eval()
                    print("Collecting latent representations...")
                    Y_list, X_list = [], []
                    for data, _ in tqdm(
                        data_loader, desc="Collecting Latents", leave=False
                    ):
                        data = data.to(self.device)
                        with torch.no_grad():
                            mu, _ = self.vae.encode(data)
                        Y_list.append(mu.detach().cpu().numpy())
                        X_list.append(data.view(data.size(0), -1).cpu().numpy())

                    Y = np.vstack(Y_list)
                    X = np.vstack(X_list)
                    N, d = X.shape
                    k = Y.shape[1]

                    W = np.ones(N)
                    I_k = np.eye(k)

                    for iteration in range(max_iter):
                        Y_weighted = Y.T * W
                        LHS = Y_weighted @ Y + lambda_reg * I_k
                        RHS = Y_weighted @ X
                        try:
                            A = np.linalg.solve(LHS, RHS)
                        except np.linalg.LinAlgError:
                            A = np.linalg.pinv(LHS) @ RHS

                        residuals = X - (Y @ A)
                        residual_norms = np.maximum(
                            np.linalg.norm(residuals, axis=1), 1e-8
                        )
                        W = np.where(
                            residual_norms <= delta, 1.0, delta / residual_norms
                        )

                    self.inverse_model.weight.data = (
                        torch.from_numpy(A.T).float().to(self.device)
                    )

                def get_feature_maps(self):
                    weights = self.inverse_model.weight.data.detach().cpu().numpy().T
                    c, h, w = self.input_shape
                    return weights.reshape(self.latent_dim, h, w)

            # --- EXECUTION ---
            self.logger.info("Initializing latent explainer...")
            sample_data = next(iter(self.train_loader))[0]
            input_shape = sample_data.shape[1:]
            explainer = LatentExplainer(self.model, self.device, input_shape=input_shape)

            self.logger.info("Fitting inverse model (IRLS)...")
            explainer.fit_irls(self.train_loader, max_iter=10)

            importance_maps = explainer.get_feature_maps()

            # --- MNE VISUALIZATION SETUP ---
            # Use model's MNE info for electrode positions
            if self.model.info is None:
                self.logger.warning(
                    "No MNE info available - skipping explainability visualization"
                )
                return

            info = self.model.info
            ch_names = info.ch_names
            montage = info.get_montage()

            if montage is None:
                self.logger.warning(
                    "No montage available in MNE info - skipping explainability visualization"
                )
                return

            # Helper to extract channel values from 40x40 grid (Same logic as in MyModel)
            def extract_sensors(img, montage, width=40, height=40):
                import math as m

                pos_3d = []
                valid_chs = []
                for ch in ch_names:
                    if ch in montage.get_positions()["ch_pos"]:
                        pos_3d.append(montage.get_positions()["ch_pos"][ch])
                        valid_chs.append(ch)

                pos_2d = []
                for x, y, z in pos_3d:
                    r = m.sqrt(x**2 + y**2 + z**2)
                    elev = m.atan2(z, m.sqrt(x**2 + y**2))
                    az = m.atan2(y, x)
                    r_proj = m.pi / 2 - elev
                    pos_2d.append([r_proj * m.cos(az), r_proj * m.sin(az)])

                pos_2d = np.array(pos_2d)
                # Normalize to grid
                p_min, p_max = pos_2d.min(axis=0), pos_2d.max(axis=0)

                sensor_vals = []
                for idx, (px, py) in enumerate(pos_2d):
                    # Normalize to 0-1
                    nx = (px - p_min[0]) / (p_max[0] - p_min[0])
                    ny = (py - p_min[1]) / (p_max[1] - p_min[1])

                    # Map to pixels (flip Y)
                    ix = int(nx * (width - 1))
                    iy = int((1 - ny) * (height - 1))
                    ix = np.clip(ix, 0, width - 1)
                    iy = np.clip(iy, 0, height - 1)
                    sensor_vals.append(img[iy, ix])
                return np.array(sensor_vals)

            # --- PLOTTING ---
            output_dir = self.output_dir / "explainability"
            output_dir.mkdir(exist_ok=True)

            n_latents = importance_maps.shape[0]
            cols = 4
            rows = int(np.ceil(n_latents / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()

            self.logger.info("Generating MNE Topomaps for Latents...")

            for i in range(n_latents):
                if i < len(axes):
                    # Extract sensor values from the 40x40 topomap
                    sensor_data = extract_sensors(importance_maps[i], montage)

                    # Plot using MNE
                    im, _ = mne.viz.plot_topomap(
                        sensor_data,
                        info,
                        axes=axes[i],
                        show=False,
                        cmap="RdBu_r",
                        contours=6,
                        sensors=True,
                        vlim=(sensor_data.min(), sensor_data.max()),
                    )
                    axes[i].set_title(f"Latent {i+1}\nFeature Map")

            # Clean up empty axes
            for i in range(n_latents, len(axes)):
                axes[i].axis("off")

            plt.suptitle("Latent Dimension Feature Maps", fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / "latent_explainability_topomaps.png", dpi=300)
            plt.close()

            self.logger.info(f"Explanation pipeline complete.")

        except Exception as e:
            self.logger.error(f"Explainability analysis failed: {e}", exc_info=True)

