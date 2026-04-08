"""
Publication-quality analysis of a 4D hyperparameter sweep for EEG microstate VAE clustering.

Sweep dimensions: K (clusters) x latent_dim x n_conv_layers (depth) x ndf (width).

Results are read from directories matching:
    outputs/{participant}/{run_id}/cluster_{K}_{batch}_{latent}_{depth}_{ndf}/

Primary ranking metric: Q2 latent pycrostates silhouette.
Composite score: geometric mean of normalized Q2 silhouette and GEV.

Usage:
    python sweep_analysis.py --run_dir outputs/010004/run_20260304_120000
    python sweep_analysis.py --run_dir outputs/010004/run_20260304_120000 --top_n 15
    python sweep_analysis.py --participant 010004  # scans all runs under that participant
"""

import argparse
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_STYLE_CANDIDATES = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]
for _s in _STYLE_CANDIDATES:
    if _s in plt.style.available:
        plt.style.use(_s)
        break

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "pdf.fonttype": 42,       # TrueType in PDF (editable text)
    "ps.fonttype": 42,
})

logger = logging.getLogger("sweep_analysis")

# ---------------------------------------------------------------------------
# Sentinel / missing-value handling
# ---------------------------------------------------------------------------
_SENTINELS = {-1, -999, float("inf"), float("-inf")}


def _clean(val, default=np.nan):
    """Replace sentinel and None values with NaN."""
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if f in _SENTINELS or np.isnan(f):
        return default
    return f


# ---------------------------------------------------------------------------
# 1. collect_all_results
# ---------------------------------------------------------------------------
_DIR_PATTERN_5 = re.compile(
    r"^cluster_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$"
)
_DIR_PATTERN_3 = re.compile(
    r"^cluster_(\d+)_(\d+)_(\d+)$"
)


def _parse_cluster_dir_name(name: str) -> Optional[Dict]:
    """Parse a cluster directory name into config parameters.

    Handles two formats:
        5-segment (new): cluster_{K}_{batch}_{latent}_{depth}_{ndf}
        3-segment (old): cluster_{K}_{batch}_{latent}
    """
    m5 = _DIR_PATTERN_5.match(name)
    if m5:
        return {
            "K": int(m5.group(1)),
            "batch_size": int(m5.group(2)),
            "latent_dim": int(m5.group(3)),
            "n_conv_layers": int(m5.group(4)),
            "ndf": int(m5.group(5)),
        }
    m3 = _DIR_PATTERN_3.match(name)
    if m3:
        return {
            "K": int(m3.group(1)),
            "batch_size": int(m3.group(2)),
            "latent_dim": int(m3.group(3)),
            "n_conv_layers": np.nan,
            "ndf": np.nan,
        }
    return None


def _extract_4q_from_json(section: Dict) -> Dict:
    """Extract all 16 quadrant metrics from a best_model_metrics JSON section.

    The section is expected to be the train or eval dict inside
    ``best_model_metrics``, which contains ``quadrant_metrics.clustering``.
    """
    qm = section.get("quadrant_metrics", {}).get("clustering", {})
    recon = section.get("quadrant_metrics", {}).get("reconstruction", {})

    prefix_map = {
        "q1_latent_eucl": "latent_sklearn",
        "q2_latent_corr": "latent_pycrostates",
        "q3_topo_eucl": "topomap_sklearn",
        "q4_topo_corr": "topomap_pycrostates",
    }
    metrics = {}
    for prefix, json_key in prefix_map.items():
        block = qm.get(json_key, {})
        for metric in ("silhouette", "db", "ch", "dunn"):
            metrics[f"{prefix}_{metric}"] = _clean(block.get(metric))

    # Reconstruction
    for key in ("mse", "kld", "ssim", "spatial_corr"):
        metrics[f"recon_{key}"] = _clean(recon.get(key))
    return metrics


def _read_yaml_safe(path: Path) -> Optional[Dict]:
    """Read a YAML file, returning None on failure."""
    try:
        import yaml
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return None


def _read_json_safe(path: Path) -> Optional[Dict]:
    """Read a JSON file, returning None on failure."""
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return None


def collect_all_results(run_dir: Path) -> pd.DataFrame:
    """Walk ``cluster_*`` directories, parse YAML + JSON, return a flat DataFrame.

    Columns include:
        K, latent_dim, n_conv_layers, ndf, batch_size,
        best_gev, ssim_score, spatial_corr,
        train_q{1..4}_{metric}, eval_q{1..4}_{metric},
        train_recon_{mse,kld,ssim,spatial_corr},
        eval_recon_{mse,kld,ssim,spatial_corr},
        best_train_epoch, train_epochs_completed, dir_path

    Parameters
    ----------
    run_dir : Path
        Path to a run directory containing ``cluster_*`` sub-directories.

    Returns
    -------
    pd.DataFrame
        One row per cluster configuration.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    rows: List[Dict] = []
    cluster_dirs = sorted(run_dir.glob("cluster_*"))
    if not cluster_dirs:
        logger.warning("No cluster_* directories found in %s", run_dir)
        return pd.DataFrame()

    for cdir in cluster_dirs:
        if not cdir.is_dir():
            continue

        parsed = _parse_cluster_dir_name(cdir.name)
        if parsed is None:
            logger.debug("Skipping unrecognised directory: %s", cdir.name)
            continue

        row: Dict = {**parsed, "dir_path": str(cdir)}

        # ---- summary_metrics.yaml ----
        yaml_data = _read_yaml_safe(cdir / "summary_metrics.yaml")
        if yaml_data:
            row["best_gev"] = _clean(yaml_data.get("best_gev"))
            row["ssim_score"] = _clean(yaml_data.get("ssim_score"))
            row["spatial_corr"] = _clean(yaml_data.get("spatial_corr"))
            row["best_train_epoch"] = yaml_data.get("best_train_epoch")
            row["train_epochs_completed"] = yaml_data.get("train_epochs_completed")
            row["best_train_loss"] = _clean(yaml_data.get("best_train_loss"))
            # Grab any Q-metrics already in YAML as fallback
            for key in ("q1_latent_eucl_sil", "q3_topo_eucl_sil", "q4_topo_corr_sil"):
                val = yaml_data.get(key)
                if val is not None:
                    row[f"yaml_{key}"] = _clean(val)

        # ---- best_model_metrics.json ----
        json_data = _read_json_safe(cdir / "best_model_metrics.json")
        if json_data:
            bm = json_data.get("best_model_metrics", {})
            metadata = json_data.get("metadata", {})

            # GEV from JSON (may be more complete than YAML)
            gev_block = bm.get("gev", {})
            if "best_gev" not in row or np.isnan(row.get("best_gev", np.nan)):
                row["best_gev"] = _clean(
                    gev_block.get("pixel_space_train",
                                  metadata.get("best_gev"))
                )
            row["electrode_gev"] = _clean(gev_block.get("electrode_space"))

            # Train 4-quadrant
            train_sec = bm.get("train", {})
            if train_sec:
                for k, v in _extract_4q_from_json(train_sec).items():
                    row[f"train_{k}"] = v

            # Eval 4-quadrant
            eval_sec = bm.get("eval", {})
            if eval_sec:
                for k, v in _extract_4q_from_json(eval_sec).items():
                    row[f"eval_{k}"] = v

            # Full-data 4-quadrant
            full_sec = bm.get("full_data", {})
            if full_sec:
                for k, v in _extract_4q_from_json(full_sec).items():
                    row[f"full_{k}"] = v

            # Baseline ModKMeans for comparison
            baseline = json_data.get("baseline_modkmeans", {})
            bval = baseline.get("cluster_validation_metrics", {})
            if bval:
                row["baseline_gev"] = _clean(bval.get("gev"))
                row["baseline_silhouette"] = _clean(bval.get("silhouette"))

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure core columns are numeric
    for col in ("K", "latent_dim", "n_conv_layers", "ndf", "batch_size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_total = len(df)
    n_with_json = df["train_q2_latent_corr_silhouette"].notna().sum() if "train_q2_latent_corr_silhouette" in df.columns else 0
    logger.info(
        "Collected %d configurations (%d with full JSON metrics) from %s",
        n_total, n_with_json, run_dir,
    )
    return df


# ---------------------------------------------------------------------------
# Parameter count estimation
# ---------------------------------------------------------------------------

def estimate_param_count(latent_dim: int, n_conv_layers: int, ndf: int,
                         nc: int = 1, n_clusters: int = 4) -> int:
    """Estimate total trainable parameters for the VAE model.

    Mirrors the architecture in ``model.py``: Encoder + Decoder + GMM prior.

    Encoder conv layers:
        Layer 0: Conv2d(nc, ndf, 4, 2, 1, bias=False)
        Layer i: Conv2d(ndf*2^min(i-1,3), ndf*2^min(i,3), 4, 2, 1, bias=False)
        Bottleneck: Conv2d(final_ch, 1024, 1, 1, 0, bias=False)
        FC: 1024->512, 512->latent_dim (mu), 512->latent_dim (logvar)
        BatchNorm params: 2 * out_ch per BN layer

    Decoder mirrors the encoder with ConvTranspose2d.

    GMM prior: pi_(K), mu_c(K, ld), log_var_c(K, ld).
    """
    n_conv_layers = int(n_conv_layers)
    ndf = int(ndf)
    latent_dim = int(latent_dim)

    total = 0

    # --- Encoder ---
    for i in range(n_conv_layers):
        in_ch = nc if i == 0 else ndf * (2 ** min(i - 1, 3))
        out_ch = ndf * (2 ** min(i, 3))
        # Conv2d kernel=4x4, no bias
        total += in_ch * out_ch * 4 * 4
        # BatchNorm (weight + bias) for layers > 0
        if i > 0:
            total += 2 * out_ch

    # Bottleneck conv 1x1
    final_ch = ndf * (2 ** min(n_conv_layers - 1, 3))
    total += final_ch * 1024 * 1 * 1

    # FC head: 1024->512 (weight + bias)
    total += 1024 * 512 + 512
    # mu: 512->latent_dim
    total += 512 * latent_dim + latent_dim
    # logvar: 512->latent_dim
    total += 512 * latent_dim + latent_dim

    # --- Decoder ---
    # FC input: latent_dim->512->1024
    total += latent_dim * 512 + 512
    total += 512 * 1024 + 1024

    # Build channel sequence same as model.py
    channel_seq = [ndf * (2 ** min(i, 3)) for i in range(n_conv_layers)]
    channel_seq.reverse()

    # First deconv: 1024 -> channel_seq[0], kernel=4
    total += 1024 * channel_seq[0] * 4 * 4
    total += 2 * channel_seq[0]  # BN

    # Intermediate deconv layers
    for i in range(1, len(channel_seq)):
        total += channel_seq[i - 1] * channel_seq[i] * 4 * 4
        total += 2 * channel_seq[i]  # BN

    # Final deconv: channel_seq[-1] -> nc, kernel=4
    total += channel_seq[-1] * nc * 4 * 4

    # --- GMM prior ---
    total += n_clusters                          # pi_
    total += n_clusters * latent_dim             # mu_c
    total += n_clusters * latent_dim             # log_var_c

    return total


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1], handling constant series."""
    smin, smax = series.min(), series.max()
    if smax - smin < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - smin) / (smax - smin)


def _composite_score(df: pd.DataFrame,
                     sil_col: str = "train_q2_latent_corr_silhouette",
                     gev_col: str = "best_gev") -> pd.Series:
    """Compute composite score = geometric mean of normalised Q2 sil and GEV.

    Returns NaN where either input is NaN.
    """
    sil = df[sil_col].copy() if sil_col in df.columns else pd.Series(np.nan, index=df.index)
    gev = df[gev_col].copy() if gev_col in df.columns else pd.Series(np.nan, index=df.index)

    sil_norm = _minmax_norm(sil.fillna(sil.min() if sil.notna().any() else 0))
    gev_norm = _minmax_norm(gev.fillna(gev.min() if gev.notna().any() else 0))

    # Clip to avoid sqrt of negative due to floating point
    composite = np.sqrt(np.clip(sil_norm * gev_norm, 0, None))

    # Restore NaN where original was NaN
    composite[sil.isna() | gev.isna()] = np.nan
    return composite


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _savefig(fig, output_dir: Path, stem: str):
    """Save figure as both 300-DPI PNG and vector PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    logger.info("Saved %s.{png,pdf}", stem)


# ---------------------------------------------------------------------------
# 2. marginal_effect_plots
# ---------------------------------------------------------------------------

def marginal_effect_plots(df: pd.DataFrame, output_dir: Path):
    """For each sweep dimension, plot mean +/- 95 pct CI for key metrics.

    Creates one figure per sweep dimension (K, latent_dim, n_conv_layers, ndf)
    with subplots for Q2 sil, Q3 sil, GEV, SSIM, Dunn.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`collect_all_results`.
    output_dir : Path
        Directory to save figures into.
    """
    output_dir = Path(output_dir)

    dimensions = [
        ("K", "Number of Clusters (K)"),
        ("latent_dim", "Latent Dimension"),
        ("n_conv_layers", "Encoder Depth (conv layers)"),
        ("ndf", "Base Width (ndf)"),
    ]

    metric_specs = [
        ("train_q2_latent_corr_silhouette", "Q2 Silhouette (latent, corr)"),
        ("train_q3_topo_eucl_silhouette", "Q3 Silhouette (topo, eucl)"),
        ("best_gev", "GEV"),
        ("train_recon_ssim", "SSIM"),
        ("train_q2_latent_corr_dunn", "Q2 Dunn (latent, corr)"),
    ]

    # Filter to metrics that actually exist in the dataframe
    metric_specs = [(col, label) for col, label in metric_specs if col in df.columns]
    if not metric_specs:
        logger.warning("No metrics columns found in DataFrame -- skipping marginal_effect_plots")
        return

    for dim_col, dim_label in dimensions:
        if dim_col not in df.columns or df[dim_col].dropna().nunique() < 2:
            logger.info("Skipping marginal plot for %s (fewer than 2 unique values)", dim_col)
            continue

        n_metrics = len(metric_specs)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 3.8), squeeze=False)
        axes = axes[0]

        for ax, (metric_col, metric_label) in zip(axes, metric_specs):
            subset = df[[dim_col, metric_col]].dropna()
            if subset.empty:
                ax.set_title(metric_label)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            grouped = subset.groupby(dim_col)[metric_col]
            means = grouped.mean()
            sems = grouped.sem()
            counts = grouped.count()

            # 95% CI using t-distribution
            ci95 = sems * stats.t.ppf(0.975, np.maximum(counts - 1, 1))

            x = means.index.values
            ax.errorbar(x, means.values, yerr=ci95.values,
                        fmt="o-", capsize=4, capthick=1.2, linewidth=1.5,
                        markersize=5, color="#2c7bb6")

            # Show sample counts
            for xi, ni in zip(x, counts.values):
                ax.annotate(f"n={ni}", (xi, means.loc[xi]),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=7, ha="center", color="gray")

            ax.set_xlabel(dim_label)
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            if np.issubdtype(type(x[0]), np.integer) or (isinstance(x[0], float) and x[0] == int(x[0])):
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        fig.suptitle(f"Marginal Effect of {dim_label}", fontsize=13, y=1.02)
        fig.tight_layout()
        _savefig(fig, output_dir, f"marginal_{dim_col}")


# ---------------------------------------------------------------------------
# 3. interaction_heatmaps
# ---------------------------------------------------------------------------

def interaction_heatmaps(df: pd.DataFrame, output_dir: Path):
    """2D pivot heatmaps for pairs of sweep dimensions, one per quadrant silhouette.

    Pairs: K x latent_dim, depth x ndf, K x depth, latent_dim x ndf.
    For each pair, side-by-side heatmaps of Q1-Q4 silhouette.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`collect_all_results`.
    output_dir : Path
        Directory to save figures.
    """
    output_dir = Path(output_dir)

    pairs = [
        ("K", "latent_dim"),
        ("n_conv_layers", "ndf"),
        ("K", "n_conv_layers"),
        ("latent_dim", "ndf"),
    ]

    quadrant_cols = [
        ("train_q1_latent_eucl_silhouette", "Q1 (Latent, Eucl)"),
        ("train_q2_latent_corr_silhouette", "Q2 (Latent, Corr)"),
        ("train_q3_topo_eucl_silhouette", "Q3 (Topo, Eucl)"),
        ("train_q4_topo_corr_silhouette", "Q4 (Topo, Corr)"),
    ]
    quadrant_cols = [(c, l) for c, l in quadrant_cols if c in df.columns]
    if not quadrant_cols:
        logger.warning("No quadrant silhouette columns found -- skipping interaction_heatmaps")
        return

    cmap = plt.colormaps["RdYlGn"]

    for row_col, col_col in pairs:
        if row_col not in df.columns or col_col not in df.columns:
            continue
        sub = df[[row_col, col_col] + [c for c, _ in quadrant_cols]].dropna(
            subset=[row_col, col_col]
        )
        if sub.empty or sub[row_col].nunique() < 2 or sub[col_col].nunique() < 2:
            logger.info("Skipping heatmap %s x %s (insufficient data)", row_col, col_col)
            continue

        n_q = len(quadrant_cols)
        fig, axes = plt.subplots(1, n_q, figsize=(4.5 * n_q, 4.0), squeeze=False)
        axes = axes[0]

        for ax, (metric_col, metric_label) in zip(axes, quadrant_cols):
            pivot = sub.pivot_table(
                values=metric_col, index=row_col, columns=col_col, aggfunc="mean"
            )
            if pivot.empty:
                ax.set_visible(False)
                continue

            vmin = pivot.min().min()
            vmax = pivot.max().max()
            # Ensure symmetric range if negative values exist
            if vmin < 0 and vmax > 0:
                vlim = max(abs(vmin), abs(vmax))
                vmin, vmax = -vlim, vlim

            im = ax.imshow(
                pivot.values, aspect="auto", cmap=cmap,
                vmin=vmin, vmax=vmax, origin="lower",
            )

            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if np.isnan(val):
                        continue
                    text_color = "white" if abs(val - vmin) / max(vmax - vmin, 1e-9) < 0.3 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=text_color)

            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels([f"{v:g}" for v in pivot.index])
            ax.set_xlabel(col_col)
            ax.set_ylabel(row_col)
            ax.set_title(metric_label)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Interaction: {row_col} x {col_col}", fontsize=13, y=1.02)
        fig.tight_layout()
        _savefig(fig, output_dir, f"interaction_{row_col}_x_{col_col}")


# ---------------------------------------------------------------------------
# 4. quadrant_agreement_analysis
# ---------------------------------------------------------------------------

def quadrant_agreement_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyse agreement between Q1-Q4 silhouette scores.

    Produces:
        - 4x4 correlation matrix (Pearson) as annotated heatmap
        - Scatter plot of Q2 vs Q3 silhouette

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`collect_all_results`.
    output_dir : Path
        Directory to save figures.
    """
    output_dir = Path(output_dir)

    sil_cols = {
        "Q1 (Lat-Eucl)": "train_q1_latent_eucl_silhouette",
        "Q2 (Lat-Corr)": "train_q2_latent_corr_silhouette",
        "Q3 (Topo-Eucl)": "train_q3_topo_eucl_silhouette",
        "Q4 (Topo-Corr)": "train_q4_topo_corr_silhouette",
    }
    available = {label: col for label, col in sil_cols.items() if col in df.columns}
    if len(available) < 2:
        logger.warning("Fewer than 2 quadrant silhouette columns -- skipping agreement analysis")
        return

    sub = df[list(available.values())].dropna()
    sub.columns = list(available.keys())

    if len(sub) < 3:
        logger.warning("Too few rows with complete quadrant data (%d) -- skipping", len(sub))
        return

    # ---- Correlation matrix ----
    corr = sub.corr(method="pearson")

    fig_corr, ax_corr = plt.subplots(figsize=(5.5, 4.5))
    cmap_corr = plt.colormaps["coolwarm"]
    im = ax_corr.imshow(corr.values, cmap=cmap_corr, vmin=-1, vmax=1)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax_corr.text(j, i, f"{corr.values[i, j]:.2f}",
                         ha="center", va="center", fontsize=10,
                         fontweight="bold" if i != j else "normal")
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_corr.set_yticks(range(len(corr.index)))
    ax_corr.set_yticklabels(corr.index)
    ax_corr.set_title("Quadrant Silhouette Correlation Matrix")
    fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    fig_corr.tight_layout()
    _savefig(fig_corr, output_dir, "quadrant_correlation_matrix")

    # ---- Q2 vs Q3 scatter ----
    q2_label, q3_label = "Q2 (Lat-Corr)", "Q3 (Topo-Eucl)"
    if q2_label in sub.columns and q3_label in sub.columns:
        fig_sc, ax_sc = plt.subplots(figsize=(5.5, 5))
        scatter_data = sub[[q2_label, q3_label]].dropna()
        if "K" in df.columns:
            k_values = df.loc[scatter_data.index, "K"]
            unique_k = sorted(k_values.dropna().unique())
            cmap_k = plt.colormaps["tab10"].resampled(max(len(unique_k), 1))
            for idx, k in enumerate(unique_k):
                mask = k_values == k
                ax_sc.scatter(
                    scatter_data.loc[mask, q2_label],
                    scatter_data.loc[mask, q3_label],
                    c=[cmap_k(idx)], label=f"K={int(k)}", s=40, alpha=0.8, edgecolors="k", linewidths=0.3,
                )
            ax_sc.legend(title="K", fontsize=7, title_fontsize=8, ncol=2)
        else:
            ax_sc.scatter(scatter_data[q2_label], scatter_data[q3_label],
                          s=40, alpha=0.7, edgecolors="k", linewidths=0.3)

        # Regression line
        x_arr = scatter_data[q2_label].values
        y_arr = scatter_data[q3_label].values
        if len(x_arr) >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(x_arr, y_arr)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            ax_sc.plot(x_line, slope * x_line + intercept, "r--", linewidth=1,
                       label=f"r={r_value:.2f}, p={p_value:.2e}")
            ax_sc.legend(fontsize=7, loc="best")

        ax_sc.set_xlabel(q2_label + " Silhouette")
        ax_sc.set_ylabel(q3_label + " Silhouette")
        ax_sc.set_title("Latent-Corr vs Topo-Eucl Silhouette")
        fig_sc.tight_layout()
        _savefig(fig_sc, output_dir, "q2_vs_q3_scatter")


# ---------------------------------------------------------------------------
# 5. pareto_frontier
# ---------------------------------------------------------------------------

def _is_pareto_optimal(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal rows.

    Maximises all objectives (rows with higher values dominate).
    """
    n = costs.shape[0]
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j or not is_optimal[j]:
                continue
            # j dominates i if j >= i on all and j > i on at least one
            if np.all(costs[j] >= costs[i]) and np.any(costs[j] > costs[i]):
                is_optimal[i] = False
                break
    return is_optimal


def pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Scatter Q2 silhouette vs estimated parameter count; highlight Pareto-optimal configs.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`collect_all_results`.
    output_dir : Path
        Directory to save figures.
    """
    output_dir = Path(output_dir)

    sil_col = "train_q2_latent_corr_silhouette"
    if sil_col not in df.columns:
        logger.warning("Column %s not found -- skipping pareto_frontier", sil_col)
        return

    sub = df[["K", "latent_dim", "n_conv_layers", "ndf", sil_col]].dropna()
    if len(sub) < 2:
        logger.warning("Too few rows for Pareto analysis (%d)", len(sub))
        return

    # Estimate parameter counts
    param_counts = sub.apply(
        lambda r: estimate_param_count(
            latent_dim=int(r["latent_dim"]),
            n_conv_layers=int(r["n_conv_layers"]),
            ndf=int(r["ndf"]),
            n_clusters=int(r["K"]),
        ),
        axis=1,
    )
    sub = sub.copy()
    sub["param_count"] = param_counts
    sub["param_count_M"] = sub["param_count"] / 1e6

    # Pareto: maximise silhouette, minimise params -> maximise -params
    objectives = np.column_stack([
        sub[sil_col].values,
        -sub["param_count"].values,  # negate so higher = fewer params
    ])
    pareto_mask = _is_pareto_optimal(objectives)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Non-Pareto points
    ax.scatter(
        sub.loc[~pareto_mask, "param_count_M"],
        sub.loc[~pareto_mask, sil_col],
        c="#999999", s=30, alpha=0.5, edgecolors="k", linewidths=0.3,
        label="Sub-optimal",
    )

    # Pareto points
    pareto_df = sub.loc[pareto_mask].sort_values("param_count_M")
    ax.scatter(
        pareto_df["param_count_M"],
        pareto_df[sil_col],
        c="#d7191c", s=70, alpha=0.9, edgecolors="k", linewidths=0.8,
        label="Pareto-optimal", zorder=5,
    )
    # Connect Pareto frontier
    ax.plot(
        pareto_df["param_count_M"].values,
        pareto_df[sil_col].values,
        "r--", linewidth=1, alpha=0.6, zorder=4,
    )

    # Annotate Pareto points
    for _, row in pareto_df.iterrows():
        label_text = f"K={int(row['K'])} ld={int(row['latent_dim'])}"
        if not np.isnan(row.get("n_conv_layers", np.nan)):
            label_text += f"\nd={int(row['n_conv_layers'])} w={int(row['ndf'])}"
        ax.annotate(
            label_text, (row["param_count_M"], row[sil_col]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=6, color="#d7191c",
            arrowprops=dict(arrowstyle="-", color="#d7191c", lw=0.5),
        )

    ax.set_xlabel("Estimated Parameters (millions)")
    ax.set_ylabel("Q2 Silhouette (latent, correlation)")
    ax.set_title("Pareto Frontier: Quality vs Model Size")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, output_dir, "pareto_frontier")


# ---------------------------------------------------------------------------
# 6. rank_configs
# ---------------------------------------------------------------------------

def rank_configs(df: pd.DataFrame, top_n: int = 10) -> Tuple[pd.DataFrame, str]:
    """Rank configurations by composite score and generate a LaTeX table.

    Composite = geometric mean of min-max-normalised Q2 silhouette and GEV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`collect_all_results`.
    top_n : int
        Number of top configurations to return.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (top_df sorted by composite descending, LaTeX table string)
    """
    df = df.copy()
    df["composite"] = _composite_score(df)

    # Sort by composite descending, drop NaN composites
    ranked = df.dropna(subset=["composite"]).sort_values("composite", ascending=False)
    top = ranked.head(top_n).copy()
    top["rank"] = range(1, len(top) + 1)

    # Select display columns
    display_cols = ["rank", "K", "latent_dim"]
    if "n_conv_layers" in top.columns and top["n_conv_layers"].notna().any():
        display_cols.append("n_conv_layers")
    if "ndf" in top.columns and top["ndf"].notna().any():
        display_cols.append("ndf")

    metric_cols = []
    for col in [
        "train_q2_latent_corr_silhouette",
        "train_q3_topo_eucl_silhouette",
        "best_gev",
        "train_recon_ssim",
        "composite",
    ]:
        if col in top.columns:
            metric_cols.append(col)
    display_cols.extend(metric_cols)

    top_display = top[display_cols].copy()

    # Rename for readability
    rename_map = {
        "n_conv_layers": "Depth",
        "ndf": "Width",
        "latent_dim": "LD",
        "train_q2_latent_corr_silhouette": "Q2 Sil",
        "train_q3_topo_eucl_silhouette": "Q3 Sil",
        "best_gev": "GEV",
        "train_recon_ssim": "SSIM",
        "composite": "Composite",
        "rank": "Rank",
    }
    top_display = top_display.rename(columns=rename_map)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Top-" + str(top_n) + r" Hyperparameter Configurations (ranked by composite score)}",
        r"\label{tab:sweep_top_configs}",
        r"\begin{tabular}{" + "c" * len(top_display.columns) + "}",
        r"\toprule",
        " & ".join(top_display.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in top_display.iterrows():
        cells = []
        for col in top_display.columns:
            val = row[col]
            if pd.isna(val):
                cells.append("--")
            elif isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(int(val)) if isinstance(val, (int, np.integer)) else str(val))
        latex_lines.append(" & ".join(cells) + r" \\")
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    latex_str = "\n".join(latex_lines)

    logger.info("Top-%d configurations:\n%s", top_n, top_display.to_string(index=False))
    return top, latex_str


# ---------------------------------------------------------------------------
# 7. cross_subject_consistency
# ---------------------------------------------------------------------------

def cross_subject_consistency(
    subject_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    top_n: int = 10,
):
    """Assess consistency of top configurations across subjects using ICC(3,1).

    For each subject, ranks configs by composite score. Then for the union of
    top-N configs, computes ICC(3,1) -- two-way mixed, single measures,
    consistency -- across subjects.

    Parameters
    ----------
    subject_dfs : Dict[str, pd.DataFrame]
        Mapping subject_id -> DataFrame from :func:`collect_all_results`.
    output_dir : Path
        Directory to save figures.
    top_n : int
        Number of top configs per subject to include.
    """
    output_dir = Path(output_dir)
    if len(subject_dfs) < 2:
        logger.warning("Need at least 2 subjects for cross-subject analysis -- skipping")
        return

    metric_col = "train_q2_latent_corr_silhouette"

    # Build a config key for matching across subjects
    def _config_key(row):
        parts = [f"K{int(row['K'])}"]
        if pd.notna(row.get("latent_dim")):
            parts.append(f"ld{int(row['latent_dim'])}")
        if pd.notna(row.get("n_conv_layers")):
            parts.append(f"d{int(row['n_conv_layers'])}")
        if pd.notna(row.get("ndf")):
            parts.append(f"w{int(row['ndf'])}")
        return "_".join(parts)

    # Gather top configs per subject
    all_top_keys = set()
    subject_scores = {}  # config_key -> {subject_id: score}

    for subj_id, sdf in subject_dfs.items():
        sdf = sdf.copy()
        sdf["composite"] = _composite_score(sdf)
        sdf = sdf.dropna(subset=["composite"]).sort_values("composite", ascending=False)
        top_configs = sdf.head(top_n)

        for _, row in top_configs.iterrows():
            key = _config_key(row)
            all_top_keys.add(key)

        # Record all scores for this subject (not just top)
        for _, row in sdf.iterrows():
            key = _config_key(row)
            if key not in subject_scores:
                subject_scores[key] = {}
            subject_scores[key][subj_id] = row.get(metric_col, np.nan)

    # Build rating matrix: configs (rows) x subjects (cols)
    subjects = sorted(subject_dfs.keys())
    configs = sorted(all_top_keys)
    rating_matrix = np.full((len(configs), len(subjects)), np.nan)

    for i, cfg in enumerate(configs):
        for j, subj in enumerate(subjects):
            rating_matrix[i, j] = subject_scores.get(cfg, {}).get(subj, np.nan)

    # Drop rows with any NaN (config must appear in all subjects)
    complete_mask = ~np.any(np.isnan(rating_matrix), axis=1)
    rating_complete = rating_matrix[complete_mask]
    configs_complete = [c for c, m in zip(configs, complete_mask) if m]

    if rating_complete.shape[0] < 3:
        logger.warning(
            "Only %d configs overlap across all subjects -- ICC unreliable, skipping",
            rating_complete.shape[0],
        )
        return

    # ICC(3,1) computation -- two-way mixed, single measures, consistency
    n_targets, n_raters = rating_complete.shape
    mean_total = rating_complete.mean()
    ss_total = np.sum((rating_complete - mean_total) ** 2)
    row_means = rating_complete.mean(axis=1)
    col_means = rating_complete.mean(axis=0)
    ss_rows = n_raters * np.sum((row_means - mean_total) ** 2)
    ss_cols = n_targets * np.sum((col_means - mean_total) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / max(n_targets - 1, 1)
    ms_error = ss_error / max((n_targets - 1) * (n_raters - 1), 1)

    icc = (ms_rows - ms_error) / (ms_rows + (n_raters - 1) * ms_error) if (ms_rows + (n_raters - 1) * ms_error) != 0 else np.nan

    logger.info(
        "ICC(3,1) = %.3f across %d subjects, %d shared configs",
        icc, n_raters, len(configs_complete),
    )

    # ---- Heatmap of scores ----
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(subjects)), max(4, 0.35 * len(configs_complete))))
    cmap_heat = plt.colormaps["YlGnBu"]
    im = ax.imshow(rating_complete, aspect="auto", cmap=cmap_heat)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(configs_complete)))
    ax.set_yticklabels(configs_complete, fontsize=7)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Config")
    ax.set_title(f"Q2 Silhouette Across Subjects (ICC(3,1) = {icc:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Q2 Sil")

    # Annotate
    for i in range(rating_complete.shape[0]):
        for j in range(rating_complete.shape[1]):
            val = rating_complete[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    _savefig(fig, output_dir, "cross_subject_consistency")

    # Save ICC result
    icc_result = {
        "icc_3_1": float(icc),
        "n_subjects": len(subjects),
        "n_shared_configs": len(configs_complete),
        "subjects": subjects,
        "configs": configs_complete,
    }
    icc_path = output_dir / "icc_results.json"
    with open(icc_path, "w") as f:
        json.dump(icc_result, f, indent=2)
    logger.info("ICC results saved to %s", icc_path)


# ---------------------------------------------------------------------------
# 8. generate_paper_figures
# ---------------------------------------------------------------------------

def generate_paper_figures(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """Generate all publication figures and tables from sweep data.

    Calls all analysis functions and saves outputs under
    ``output_dir/paper_figures/``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`collect_all_results`.
    output_dir : Path
        Base output directory. A ``paper_figures/`` subdirectory is created.
    top_n : int
        Number of top configurations to include in rankings.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating paper figures in %s", fig_dir)
    logger.info("DataFrame shape: %s", df.shape)

    # Summary statistics
    _print_sweep_summary(df)

    # All analyses
    marginal_effect_plots(df, fig_dir)
    interaction_heatmaps(df, fig_dir)
    quadrant_agreement_analysis(df, fig_dir)
    pareto_frontier(df, fig_dir)

    top_df, latex_str = rank_configs(df, top_n=top_n)

    # Save LaTeX table
    latex_path = fig_dir / "top_configs_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_str)
    logger.info("LaTeX table saved to %s", latex_path)

    # Save full results CSV
    csv_path = fig_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Full results saved to %s", csv_path)

    # Save top configs CSV
    top_csv_path = fig_dir / "top_configs.csv"
    top_df.to_csv(top_csv_path, index=False)
    logger.info("Top configs saved to %s", top_csv_path)

    logger.info("Paper figure generation complete.")
    return top_df


def _print_sweep_summary(df: pd.DataFrame):
    """Log a human-readable summary of the sweep DataFrame."""
    logger.info("=" * 60)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 60)
    logger.info("Total configurations: %d", len(df))

    for col in ("K", "latent_dim", "n_conv_layers", "ndf"):
        if col in df.columns:
            vals = df[col].dropna().unique()
            logger.info("  %s: %s", col, sorted(vals))

    for col, label in [
        ("train_q2_latent_corr_silhouette", "Q2 Sil (train)"),
        ("best_gev", "GEV"),
        ("train_recon_ssim", "SSIM"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                logger.info(
                    "  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (n=%d)",
                    label, s.mean(), s.std(), s.min(), s.max(), len(s),
                )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Multi-subject convenience
# ---------------------------------------------------------------------------

def collect_multi_subject(base_dir: Path, run_id: str,
                          participants: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Collect results across multiple subjects sharing the same run_id.

    Parameters
    ----------
    base_dir : Path
        Outputs root (e.g. ``outputs/``).
    run_id : str
        Run subdirectory name to look for under each participant.
    participants : list of str, optional
        Specific participant IDs to include. If None, scans all subdirs.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping participant_id -> DataFrame.
    """
    base_dir = Path(base_dir)
    if participants is None:
        participants = [
            d.name for d in sorted(base_dir.iterdir())
            if d.is_dir() and (d / run_id).is_dir()
        ]

    result = {}
    for pid in participants:
        run_dir = base_dir / pid / run_id
        if run_dir.is_dir():
            try:
                df = collect_all_results(run_dir)
                if not df.empty:
                    result[pid] = df
                    logger.info("Loaded %d configs for participant %s", len(df), pid)
            except Exception as exc:
                logger.warning("Failed to load results for %s: %s", pid, exc)
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for sweep analysis."""
    parser = argparse.ArgumentParser(
        description="Analyse hyperparameter sweep results for EEG microstate VAE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Path to a single run directory containing cluster_* subdirs.",
    )
    parser.add_argument(
        "--participant", type=str, default=None,
        help="Participant ID. If given without --run_dir, scans all runs under outputs/{participant}/.",
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Run ID for multi-subject analysis (used with --multi_subject).",
    )
    parser.add_argument(
        "--multi_subject", action="store_true",
        help="Run cross-subject consistency analysis across all participants.",
    )
    parser.add_argument(
        "--outputs_dir", type=str, default="./outputs",
        help="Base outputs directory (default: ./outputs).",
    )
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="Number of top configurations to display (default: 10).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for analysis outputs. Defaults to {run_dir}/sweep_analysis/.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    outputs_base = Path(args.outputs_dir)

    # --- Multi-subject mode ---
    if args.multi_subject:
        if not args.run_id:
            parser.error("--run_id is required with --multi_subject")
        subject_dfs = collect_multi_subject(outputs_base, args.run_id)
        if not subject_dfs:
            logger.error("No subject data found.")
            return

        out_dir = Path(args.output_dir) if args.output_dir else outputs_base / f"multi_subject_{args.run_id}"

        # Generate per-subject figures + combined
        combined_df = pd.concat(
            [sdf.assign(subject=sid) for sid, sdf in subject_dfs.items()],
            ignore_index=True,
        )
        generate_paper_figures(combined_df, out_dir, top_n=args.top_n)
        cross_subject_consistency(subject_dfs, out_dir / "paper_figures", top_n=args.top_n)
        return

    # --- Single run mode ---
    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.participant:
        participant_dir = outputs_base / args.participant
        if not participant_dir.is_dir():
            logger.error("Participant directory not found: %s", participant_dir)
            return
        # Find the most recent run
        run_dirs = sorted(
            [d for d in participant_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if args.run_id:
            candidates = [d for d in run_dirs if d.name == args.run_id]
            if candidates:
                run_dir = candidates[0]
            else:
                logger.error("Run '%s' not found under %s", args.run_id, participant_dir)
                return
        elif run_dirs:
            run_dir = run_dirs[0]
            logger.info("Using most recent run: %s", run_dir.name)
        else:
            logger.error("No run directories found under %s", participant_dir)
            return
    else:
        parser.error("Provide either --run_dir or --participant.")

    if not run_dir.is_dir():
        logger.error("Run directory not found: %s", run_dir)
        return

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "sweep_analysis"

    df = collect_all_results(run_dir)
    if df.empty:
        logger.error("No results found in %s", run_dir)
        return

    generate_paper_figures(df, out_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
