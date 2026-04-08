"""Model Visualization — standalone visualization functions extracted from MyModel.

Each function takes a model instance as its first argument.
Wrapper methods on MyModel delegate to these functions.
"""
import os
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Optional
from tqdm import tqdm
import warnings

# Visualization/EEG imports — lazy for CPU-only environments
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
except ImportError:
    plt = None
try:
    import mne
    from pycrostates.io import ChData
    from pycrostates.viz import (
        plot_cluster_centers,
        plot_raw_segmentation,
    )
except ImportError:
    mne = None

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

EPSILON = 1e-8


def get_cluster_centroids_and_visualize(model, output_dir: str = "images/clusters", norm_params=None):
    """
    Visualize VAE cluster centroids as publication-ready microstate maps.

    Creates a single high-quality visualization with:
    - Circular head mask for each microstate
    - Consistent color scale across all maps
    - Clean, publication-ready styling
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.colors import TwoSlopeNorm
    import os
    import math

    os.makedirs(output_dir, exist_ok=True)
    model.logger.info(f"Visualizing {model.nClusters} VAE cluster centroids...")

    model.eval()
    with torch.no_grad():
        decoded_centroids_raw = model.decode(model.mu_c).detach().cpu().numpy()

    decoded_centroids_raw = decoded_centroids_raw.squeeze(1)

    # Denormalize from z-score back to microvolts if norm_params available
    is_denormalized = False
    if norm_params is not None:
        z_mean = norm_params.get("mean", 0.0)
        z_std = norm_params.get("std", 1.0)
        decoded_centroids_raw = decoded_centroids_raw * z_std + z_mean
        is_denormalized = True
        model.logger.info(f"Denormalized centroids to μV (mean={z_mean:.4f}, std={z_std:.4f})")

    # Create centered version for proper polarity visualization
    decoded_centroids_centered = decoded_centroids_raw.copy()
    for i in range(model.nClusters):
        decoded_centroids_centered[i] = decoded_centroids_centered[i] - decoded_centroids_centered[i].mean()

    # Apply circular mask to each centroid (head shape)
    h, w = decoded_centroids_centered[0].shape
    y, x = np.ogrid[:h, :w]
    center = (h / 2, w / 2)
    radius = min(h, w) / 2 * 0.95
    mask = ((x - center[1])**2 + (y - center[0])**2) > radius**2

    # Mask outside the circle
    broadcast_mask = np.broadcast_to(mask, decoded_centroids_centered.shape)
    decoded_centroids_masked = np.ma.array(decoded_centroids_centered, mask=broadcast_mask)

    # Layout
    n_cols = min(4, model.nClusters)
    n_rows = math.ceil(model.nClusters / n_cols)

    # Consistent color scale (symmetric around zero)
    abs_max = np.percentile(np.abs(decoded_centroids_centered[~mask.reshape(1, h, w).repeat(model.nClusters, axis=0)]), 99)

    # Create figure with dark background for better contrast
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              facecolor='white')
    if model.nClusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Microstate labels (A, B, C, D...)
    labels = [chr(65 + i) for i in range(model.nClusters)]

    for i in range(model.nClusters):
        ax = axes[i]
        ax.set_facecolor('white')

        # Plot the masked centroid
        im = ax.imshow(
            decoded_centroids_masked[i],
            cmap="RdBu_r",
            origin="lower",
            vmin=-abs_max,
            vmax=abs_max,
            interpolation="bilinear",
        )

        # Add head outline circle
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
        # Left ear
        ax.plot([w/2 - radius - 2, w/2 - radius - 5, w/2 - radius - 2],
               [ear_y - 4, ear_y, ear_y + 4], 'k-', linewidth=1.5)
        # Right ear
        ax.plot([w/2 + radius + 2, w/2 + radius + 5, w/2 + radius + 2],
               [ear_y - 4, ear_y, ear_y + 4], 'k-', linewidth=1.5)

        ax.set_title(f"Microstate {labels[i]}", fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
        ax.set_xlim(-5, w + 5)
        ax.set_ylim(-5, h + 8)

    # Hide empty subplots
    for i in range(model.nClusters, len(axes)):
        axes[i].axis("off")

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_label = "Amplitude (μV, centered)" if is_denormalized else "Amplitude (z-score, centered)"
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle(
        f"VAE Microstate Templates (K={model.nClusters})",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    grid_path = os.path.join(output_dir, "vae_merged_centroids.png")
    plt.savefig(grid_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    model.logger.info(f"Saved VAE centroid visualization: {grid_path}")

    # --- PYCROSTATES-STYLE MNE TOPOMAP VISUALIZATION ---
    model._visualize_centroids_mne_topomap(decoded_centroids_centered, output_dir, norm_params=norm_params)


def _visualize_centroids_mne_topomap(model, decoded_centroids, output_dir: str, norm_params=None):
    """
    Visualize cluster centers using MNE's plot_topomap (pycrostates-style).

    This extracts electrode values from the 40x40 topomap at the electrode
    positions and creates proper EEG topographic maps.
    """
    import math

    model.logger.info(" Creating pycrostates-style MNE topomaps...")

    try:
        # Check if MNE info is available
        if model.info is None:
            model.logger.warning("No MNE info available. Skipping MNE topomap visualization.")
            return

        # Get electrode positions from MNE info
        montage = model.info.get_montage()
        if montage is None:
            model.logger.warning("No montage found in info. Skipping MNE topomap.")
            return

        # Get 2D positions for electrodes
        pos_dict = montage.get_positions()
        ch_pos = pos_dict["ch_pos"]

        # Extract x, y positions (ignore z for 2D topomap)
        positions_3d = np.array([ch_pos[ch] for ch in model.info.ch_names])
        positions_2d = positions_3d[:, :2]  # Take only x, y

        # Normalize positions to [0, 1] range for mapping to 40x40 image
        pos_min = positions_2d.min(axis=0)
        pos_max = positions_2d.max(axis=0)
        pos_normalized = (positions_2d - pos_min) / (pos_max - pos_min + 1e-8)

        # Map to image coordinates (40x40)
        img_size = decoded_centroids.shape[1]  # Should be 40
        # Add margin to avoid edge effects
        margin = 0.1
        pos_img = pos_normalized * (1 - 2 * margin) + margin
        pos_img = (pos_img * (img_size - 1)).astype(int)
        pos_img = np.clip(pos_img, 0, img_size - 1)

        # Extract electrode values from each centroid
        n_channels = len(model.info.ch_names)
        electrode_values_raw = np.zeros((model.nClusters, n_channels))

        for k in range(model.nClusters):
            centroid_img = decoded_centroids[k]
            for ch_idx in range(n_channels):
                x, y = pos_img[ch_idx]
                electrode_values_raw[k, ch_idx] = centroid_img[y, x]

        # Create centered version for symmetric colormap
        electrode_values_centered = electrode_values_raw.copy()
        for k in range(model.nClusters):
            electrode_values_centered[k] = electrode_values_centered[k] - electrode_values_centered[k].mean()

        n_cols = min(4, model.nClusters)
        n_rows = math.ceil(model.nClusters / n_cols)

        # ===== VERSION 1: Centered (symmetric around zero) =====
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows)
        )
        if model.nClusters == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        vmax = np.percentile(np.abs(electrode_values_centered), 99)

        for k in range(model.nClusters):
            ax = axes[k]
            im, _ = mne.viz.plot_topomap(
                electrode_values_centered[k],
                model.info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                contours=6,
                sensors=True,
                outlines="head",
            )
            ax.set_title(f"Microstate {k+1}", fontsize=12, fontweight="bold")

        for k in range(model.nClusters, len(axes)):
            axes[k].axis("off")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        topo_cbar_label = "Amplitude (μV, centered)" if norm_params is not None else "Amplitude (z-score, centered)"
        fig.colorbar(im, cax=cbar_ax, label=topo_cbar_label)

        plt.suptitle(
            f"VAE Microstate Topomaps - Centered (K={model.nClusters})\n(Zero=White, Symmetric Scale)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        save_path = os.path.join(output_dir, "centroids_mne_topomap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        model.logger.info(f"Saved MNE topomap: {save_path}")

        # Save electrode values for potential pycrostates integration
        np.save(
            os.path.join(output_dir, "centroid_electrode_values.npy"),
            electrode_values_centered,
        )

        # Generate spatial correlation matrix between centroids
        model._visualize_centroid_spatial_correlations(electrode_values_centered, output_dir)

    except Exception as e:
        model.logger.warning(f"MNE topomap visualization failed: {e}")
        model.logger.warning("Falling back to image-based visualization only.")


def _visualize_centroid_spatial_correlations(model, electrode_values: np.ndarray, output_dir: str):
    """
    Compute and visualize spatial correlation matrix between microstate centroids.

    This helps identify polarity-inverted centroids (correlation ≈ -1) which are
    treated as the same microstate in EEG literature due to polarity invariance.

    Parameters
    ----------
    electrode_values : np.ndarray
        Shape (n_clusters, n_channels) - electrode values for each centroid.
    output_dir : str
        Directory to save outputs.
    """
    import pandas as pd
    from scipy.stats import pearsonr

    model.logger.info("Computing spatial correlation matrix between centroids...")

    n_clusters = electrode_values.shape[0]

    # Compute full correlation matrix
    corr_matrix = np.zeros((n_clusters, n_clusters))
    pval_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            r, p = pearsonr(electrode_values[i], electrode_values[j])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    # Create labels for centroids
    labels = [f"C{i+1}" for i in range(n_clusters)]

    # ===== VISUALIZATION 1: Correlation Matrix Heatmap =====
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use diverging colormap centered at 0
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spatial Correlation (Pearson r)', fontsize=12)

    # Set ticks and labels
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Add correlation values as text annotations
    for i in range(n_clusters):
        for j in range(n_clusters):
            r = corr_matrix[i, j]
            # Use white text for extreme values, black for middle
            text_color = 'white' if abs(r) > 0.7 else 'black'
            ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                   fontsize=10, color=text_color, fontweight='bold')

    ax.set_xlabel('Centroid', fontsize=12, fontweight='bold')
    ax.set_ylabel('Centroid', fontsize=12, fontweight='bold')
    ax.set_title(f'Spatial Correlation Matrix Between Microstate Centroids (K={n_clusters})\n'
                'Values ≈ -1 indicate polarity-inverted pairs (treated as same microstate)',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "centroid_spatial_correlation_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ===== VISUALIZATION 2: Absolute Correlation (Polarity-Invariant) =====
    abs_corr_matrix = np.abs(corr_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(abs_corr_matrix, cmap='Reds', vmin=0, vmax=1, aspect='equal')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Absolute Spatial Correlation |r|', fontsize=12)

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

    ax.set_xlabel('Centroid', fontsize=12, fontweight='bold')
    ax.set_ylabel('Centroid', fontsize=12, fontweight='bold')
    ax.set_title(f'Absolute Spatial Correlation Matrix (Polarity-Invariant)\n'
                'High values (≈1) indicate same/inverted topographies',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path_abs = os.path.join(output_dir, "centroid_spatial_correlation_absolute.png")
    plt.savefig(save_path_abs, dpi=300, bbox_inches='tight')
    plt.close()

    # ===== Identify polarity-inverted pairs =====
    polarity_inverted_pairs = []
    similar_pairs = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            r = corr_matrix[i, j]
            if r < -0.7:  # Strong negative correlation = polarity inverted
                polarity_inverted_pairs.append((labels[i], labels[j], r))
            elif r > 0.7:  # Strong positive correlation = similar
                similar_pairs.append((labels[i], labels[j], r))

    # ===== Save results to CSV =====
    df_corr = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    df_corr.to_csv(os.path.join(output_dir, "centroid_spatial_correlation_matrix.csv"))

    # ===== Save summary report to TXT =====
    summary_lines = [
        "=" * 70,
        "SPATIAL CORRELATION ANALYSIS BETWEEN MICROSTATE CENTROIDS",
        "=" * 70,
        "",
        f"Number of centroids: {n_clusters}",
        "",
        "FULL CORRELATION MATRIX:",
        "-" * 40,
    ]

    # Add formatted correlation matrix
    header = "        " + "  ".join([f"{l:>6}" for l in labels])
    summary_lines.append(header)
    for i, row_label in enumerate(labels):
        row_vals = "  ".join([f"{corr_matrix[i, j]:>6.3f}" for j in range(n_clusters)])
        summary_lines.append(f"{row_label:>6}  {row_vals}")

    summary_lines.extend([
        "",
        "POLARITY-INVERTED PAIRS (r < -0.7):",
        "-" * 40,
    ])

    if polarity_inverted_pairs:
        for c1, c2, r in polarity_inverted_pairs:
            summary_lines.append(f"  {c1} <-> {c2}: r = {r:.4f}  (POLARITY INVERTED)")
    else:
        summary_lines.append("  None found")

    summary_lines.extend([
        "",
        "HIGHLY SIMILAR PAIRS (r > 0.7):",
        "-" * 40,
    ])

    if similar_pairs:
        for c1, c2, r in similar_pairs:
            summary_lines.append(f"  {c1} <-> {c2}: r = {r:.4f}  (SIMILAR)")
    else:
        summary_lines.append("  None found")

    summary_lines.extend([
        "",
        "INTERPRETATION:",
        "-" * 40,
        "- Polarity-inverted pairs (r ≈ -1) represent the same underlying",
        "  brain state but measured at opposite phases of the oscillation.",
        "- In microstate literature, these are typically treated as ONE state",
        "  by using |r| (absolute correlation) for assignment.",
        "- This is why polarity-invariant clustering is used.",
        "",
        "=" * 70,
    ])

    summary_text = "\n".join(summary_lines)

    # Save summary to txt file
    txt_path = os.path.join(output_dir, "centroid_spatial_correlation_report.txt")
    with open(txt_path, 'w') as f:
        f.write(summary_text)

    # Print to console
    model.logger.info("\n" + summary_text)

    model.logger.info(f"Saved spatial correlation analysis:")
    model.logger.info(f"   - centroid_spatial_correlation_matrix.png")
    model.logger.info(f"   - centroid_spatial_correlation_absolute.png")
    model.logger.info(f"   - centroid_spatial_correlation_matrix.csv")
    model.logger.info(f"   - centroid_spatial_correlation_report.txt")


def plot_segmentation(
    model,
    data_loader: torch.utils.data.DataLoader,
    output_dir: str = "images/analysis",
    n_samples: int = 500,
    sfreq: float = 250,
):
    """
    Plot temporal segmentation of VAE-predicted microstates.

    Similar to pycrostates' plot_raw_segmentation, shows how microstate
    labels change over time with GFP overlay.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader with data.
    output_dir : str
        Directory to save outputs.
    n_samples : int
        Number of samples to visualize.
    sfreq : float
        Sampling frequency for time axis.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.logger.info("Generating VAE microstate segmentation plot...")

    model.eval()
    X_list, labels_list = [], []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(model.device)
            preds = model.predict(data)
            X_list.append(data.cpu().numpy().reshape(data.size(0), -1))
            labels_list.append(preds)

    X = np.concatenate(X_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Limit samples for visualization
    n_samples = min(n_samples, len(labels))
    labels_plot = labels[:n_samples]
    time = np.arange(n_samples) / sfreq

    # Calculate GFP
    gfp = np.std(X[:n_samples], axis=1)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Top: GFP with colored background by microstate
    ax1 = axes[0]
    ax1.plot(time, gfp, color="black", linewidth=0.8, label="GFP")

    # Color background by microstate
    colors = plt.cm.Set1(np.linspace(0, 1, model.nClusters))
    for i in range(len(labels_plot) - 1):
        ax1.axvspan(
            time[i],
            time[i + 1],
            alpha=0.3,
            color=colors[labels_plot[i]],
            linewidth=0,
        )

    ax1.set_ylabel("GFP (a.u.)", fontweight="bold")
    ax1.set_title(
        f"Microstate Segmentation (VAE Clustering, K={model.nClusters})",
        fontweight="bold",
        fontsize=14,
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: Microstate labels as step plot
    ax2 = axes[1]
    ax2.step(time, labels_plot, where="mid", color="black", linewidth=1.5)
    ax2.set_ylim(-0.5, model.nClusters - 0.5)
    ax2.set_yticks(range(model.nClusters))
    ax2.set_yticklabels([f"MS {i+1}" for i in range(model.nClusters)])
    ax2.set_xlabel("Time (s)", fontweight="bold")
    ax2.set_ylabel("Microstate", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # Add legend for microstate colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[i], alpha=0.5, label=f"MS {i+1}")
        for i in range(model.nClusters)
    ]
    ax2.legend(
        handles=legend_elements,
        loc="upper right",
        ncol=min(model.nClusters, 4),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "vae_segmentation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    model.logger.info(
        f"✅ Saved VAE segmentation plot to: {output_dir}/vae_segmentation.png"
    )

    return labels


def plot_microstate_statistics(
    model,
    data_loader: torch.utils.data.DataLoader,
    output_dir: str = "images/analysis",
    sfreq: float = 250,
    gfp_peaks=None,
):
    """
    Plot microstate temporal statistics for VAE clustering (100% data mode).

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader with data (100% of data).
    output_dir : str
        Directory to save outputs.
    sfreq : float
        Sampling frequency.
    gfp_peaks : ChData, optional
        GFP peaks structure from pycrostates.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    model.logger.info("Computing VAE microstate statistics...")

    model.set_eval_mode()
    labels_list = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(model.device)
            preds = model.predict(data)
            labels_list.append(preds)

    labels = np.concatenate(labels_list, axis=0)
    n_samples = len(labels)

    # ChData from pycrostates is atemporal (no .times attribute)
    # Use sample-count-based calculation with sampling frequency
    total_time = n_samples / sfreq
    model.logger.info(f"Computing statistics for {n_samples} samples at {sfreq} Hz (100% data mode)")

    # Duration: length of continuous segments (in ms)
    # GFP peaks are discrete samples, so duration = sample_count / sfreq
    durations = {i: [] for i in range(model.nClusters)}
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
    coverage = np.zeros(model.nClusters)
    for u, c in zip(unique, counts):
        coverage[u] = c / n_samples * 100

    # Occurrence: number of times each state appears per second
    occurrence = np.zeros(model.nClusters)
    for i in range(model.nClusters):
        occurrence[i] = len(durations[i]) / total_time

    # Mean duration
    mean_duration = [
        np.mean(durations[i]) if durations[i] else 0 for i in range(model.nClusters)
    ]
    std_duration = [
        np.std(durations[i]) if len(durations[i]) > 1 else 0
        for i in range(model.nClusters)
    ]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(model.nClusters)
    colors = plt.cm.Set1(np.linspace(0, 1, model.nClusters))

    # Duration
    axes[0].bar(
        x, mean_duration, yerr=std_duration, color=colors, alpha=0.8, capsize=5
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"MS {i+1}" for i in range(model.nClusters)])
    axes[0].set_ylabel("Duration (ms)", fontweight="bold")
    axes[0].set_title("Mean Duration", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Coverage
    axes[1].bar(x, coverage, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"MS {i+1}" for i in range(model.nClusters)])
    axes[1].set_ylabel("Coverage (%)", fontweight="bold")
    axes[1].set_title("Time Coverage", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Occurrence
    axes[2].bar(x, occurrence, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"MS {i+1}" for i in range(model.nClusters)])
    axes[2].set_ylabel("Occurrence (per second)", fontweight="bold")
    axes[2].set_title("Occurrence Rate", fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Microstate Statistics (VAE Clustering, K={model.nClusters})",
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "vae_microstate_statistics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save statistics to JSON
    # NOTE: These stats are computed on GFP peaks only, NOT full continuous data.
    # For comparison with ModKMeans (which uses full raw data), use evaluate_on_raw()
    # which saves to vae_raw_microstate_stats.json
    stats = {
        "method": "VAE (GFP peaks only)",
        "WARNING": "These statistics are computed on GFP peaks only, not full continuous data. "
                   "For fair comparison with ModKMeans, see vae_raw_microstate_stats.json",
        "mean_duration_ms": mean_duration,
        "std_duration_ms": std_duration,
        "coverage_percent": coverage.tolist(),
        "occurrence_per_sec": occurrence.tolist(),
        "n_clusters": model.nClusters,
        "total_samples": n_samples,
        "total_samples_note": "This is number of GFP peaks, not raw EEG samples",
        "sfreq": sfreq,
        "total_time_seconds": total_time,
    }
    with open(os.path.join(output_dir, "vae_microstate_stats_peaks.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Generate text report
    timing_method = f"assumed {sfreq} Hz sampling"
    report_lines = [
        "=" * 80,
        "          VAE MICROSTATE STATISTICS REPORT (GFP PEAKS ONLY)",
        "=" * 80,
        "",
        "WARNING: These statistics are computed on GFP peaks only, not full",
        "         continuous EEG data. For fair comparison with ModKMeans,",
        "         see vae_raw_microstate_stats.json (computed via evaluate_on_raw).",
        "",
        f"METHOD: Convolutional VAE with Clustering",
        f"DATA: GFP peaks (sparse samples, not continuous)",
        f"NUMBER OF CLUSTERS: {model.nClusters}",
        f"TOTAL GFP PEAKS: {n_samples}",
        f"TOTAL TIME: {total_time:.2f} seconds",
        f"TIMING METHOD: {timing_method}",
        "",
        "DURATION STATISTICS (milliseconds):",
        "-" * 40,
    ]

    for i in range(model.nClusters):
        report_lines.append(f"  Microstate {i+1}: {mean_duration[i]:.2f} +/- {std_duration[i]:.2f} ms")

    report_lines.extend([
        "",
        "TIME COVERAGE (%):",
        "-" * 40,
    ])
    for i in range(model.nClusters):
        report_lines.append(f"  Microstate {i+1}: {coverage[i]:.2f}%")

    report_lines.extend([
        "",
        "OCCURRENCE RATE (per second):",
        "-" * 40,
    ])
    for i in range(model.nClusters):
        report_lines.append(f"  Microstate {i+1}: {occurrence[i]:.2f}/s")

    report_lines.extend([
        "",
        "=" * 80,
    ])

    with open(os.path.join(output_dir, "vae_microstate_stats_peaks_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    model.logger.info(
        f"Saved VAE GFP peak statistics to: {output_dir}/vae_microstate_statistics.png"
    )
    model.logger.info(
        "NOTE: These are GFP peak stats only. For raw data stats comparable to ModKMeans, "
        "evaluate_on_raw() will be called next."
    )

    return stats


def pycrostates_plot_cluster_centers(model, output_dir: str = "images/analysis", norm_params=None):
    """
    Plot VAE cluster centers using pycrostates' plot_cluster_centers function.

    This extracts electrode values from the decoded VAE centroids and uses
    pycrostates' native visualization for LEMON's 61-channel data.

    Parameters
    ----------
    output_dir : str
        Directory to save the output figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.logger.info("Plotting cluster centers using pycrostates.viz.plot_cluster_centers...")

    model.eval()
    try:
        with torch.no_grad():
            # Decode cluster centers from latent space
            decoded_centroids = model.decode(model.mu_c).detach().cpu().numpy()

        decoded_centroids = decoded_centroids.squeeze(1)  # (n_clusters, 40, 40)

        # Check if MNE info is available
        if model.info is None:
            model.logger.warning("No MNE info available. Skipping pycrostates cluster centers plot.")
            return

        # Extract electrode values from 40x40 topomaps
        # Get electrode positions from MNE info
        montage = model.info.get_montage()
        if montage is None:
            model.logger.warning("No montage found. Cannot use pycrostates plot.")
            return

        pos_dict = montage.get_positions()
        ch_pos = pos_dict["ch_pos"]

        # Get 2D positions
        positions_3d = np.array([ch_pos[ch] for ch in model.info.ch_names])
        positions_2d = positions_3d[:, :2]

        # Normalize to image coordinates
        pos_min = positions_2d.min(axis=0)
        pos_max = positions_2d.max(axis=0)
        pos_normalized = (positions_2d - pos_min) / (pos_max - pos_min + 1e-8)

        img_size = decoded_centroids.shape[1]
        margin = 0.1
        pos_img = pos_normalized * (1 - 2 * margin) + margin
        pos_img = (pos_img * (img_size - 1)).astype(int)
        pos_img = np.clip(pos_img, 0, img_size - 1)

        # Extract electrode values for each cluster center
        n_channels = len(model.info.ch_names)
        cluster_centers = np.zeros((model.nClusters, n_channels))

        for k in range(model.nClusters):
            for ch_idx in range(n_channels):
                x, y = pos_img[ch_idx]
                cluster_centers[k, ch_idx] = decoded_centroids[k, y, x]

        # Denormalize from z-score space to original scale (Volts)
        if norm_params is not None:
            z_mean = norm_params.get("mean", 0.0)
            z_std = norm_params.get("std", 1.0)
            cluster_centers = cluster_centers * z_std + z_mean

        # Call pycrostates plot_cluster_centers
        fig = plot_cluster_centers(cluster_centers, model.info)

        if fig is not None:
            # Add title with GEV if available
            fig.suptitle(
                f"VAE Microstate Topomaps (K={model.nClusters})",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            save_path = os.path.join(output_dir, "vae_microstate_topomaps.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            model.logger.info(f"Saved VAE microstate topomaps to: {save_path}")

    except Exception as e:
        model.logger.warning(f"pycrostates plot_cluster_centers failed: {e}")


def pycrostates_plot_raw_segmentation(
    model,
    raw: mne.io.BaseRaw,
    output_dir: str = "images/analysis",
    norm_params=None,
):
    """
    Plot raw segmentation using pycrostates' plot_raw_segmentation function.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing the EEG data.
    output_dir : str
        Directory to save the output figure.
    norm_params : dict, optional
        Normalization parameters {mean, std} for denormalizing centroids.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.logger.info("Plotting segmentation using pycrostates.viz.plot_raw_segmentation...")

    try:
        # Get data from raw object
        data = raw.get_data()  # (n_channels, n_times)

        # Process through VAE to get labels
        model.eval()
        labels_list = []

        with torch.no_grad():
            # Need to convert raw EEG to topomaps for VAE
            # This requires the same preprocessing as training
            model.logger.warning(
                "Direct Raw segmentation requires topomap conversion. "
                "Using simplified approach..."
            )

            # For now, use the data loader approach if available
            # or create ChData for pycrostates native prediction

        # Create ChData for pycrostates
        ch_data = ChData(data, info=raw.info)

        # Use stored cluster centers to predict
        # Extract electrode values from VAE centroids
        decoded_centroids = model.decode(model.mu_c).detach().cpu().numpy().squeeze(1)

        # Check if MNE info is available
        if model.info is None:
            model.logger.warning("No MNE info available. Skipping raw segmentation plot.")
            return

        montage = model.info.get_montage()
        pos_dict = montage.get_positions()
        ch_pos = pos_dict["ch_pos"]
        positions_3d = np.array([ch_pos[ch] for ch in model.info.ch_names])
        positions_2d = positions_3d[:, :2]
        pos_min = positions_2d.min(axis=0)
        pos_max = positions_2d.max(axis=0)
        pos_normalized = (positions_2d - pos_min) / (pos_max - pos_min + 1e-8)
        img_size = decoded_centroids.shape[1]
        margin = 0.1
        pos_img = pos_normalized * (1 - 2 * margin) + margin
        pos_img = (pos_img * (img_size - 1)).astype(int)
        pos_img = np.clip(pos_img, 0, img_size - 1)

        n_channels = len(model.info.ch_names)
        cluster_centers = np.zeros((model.nClusters, n_channels))
        for k in range(model.nClusters):
            for ch_idx in range(n_channels):
                x, y = pos_img[ch_idx]
                cluster_centers[k, ch_idx] = decoded_centroids[k, y, x]

        # Denormalize centroids from z-score to original scale
        if norm_params is not None:
            z_mean = norm_params.get("mean", 0.0)
            z_std = norm_params.get("std", 1.0)
            cluster_centers = cluster_centers * z_std + z_mean

        # Compute labels using correlation-based assignment
        data_norm = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-8)
        centers_norm = cluster_centers / (
            np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8
        )
        correlation = np.abs(np.dot(centers_norm, data_norm))
        labels = np.argmax(correlation, axis=0)

        # Call pycrostates plot_raw_segmentation
        fig = plot_raw_segmentation(
            labels=labels,
            raw=raw,
            n_clusters=model.nClusters,
        )

        if fig is not None:
            save_path = os.path.join(output_dir, "pycrostates_vae_raw_segmentation.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            model.logger.info(f"✅ Saved pycrostates raw segmentation to: {save_path}")

    except Exception as e:
        model.logger.warning(f"pycrostates plot_raw_segmentation failed: {e}")


def evaluate_on_raw(
    model,
    raw: mne.io.BaseRaw,
    output_dir: str = "images/analysis",
    gfp_peaks=None,
    norm_params=None,
):
    """
    Evaluate VAE microstate statistics on continuous raw EEG data (100% data mode).

    This provides a fair comparison with ModKMeans by:
    1. Backfitting VAE centroids to continuous raw data (not just GFP peaks)
    2. Computing temporal statistics (duration, coverage, occurrence) on full segmentation

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object containing continuous EEG data.
    output_dir : str
        Directory to save outputs.
    gfp_peaks : ChData, optional
        GFP peaks structure from pycrostates.

    Returns
    -------
    dict
        Dictionary containing microstate statistics.
    """
    import json
    os.makedirs(output_dir, exist_ok=True)
    model.logger.info("=" * 60)
    model.logger.info("Computing VAE microstate statistics on RAW continuous data...")
    model.logger.info("=" * 60)

    try:
        # Check if MNE info is available
        if model.info is None:
            model.logger.warning("No MNE info available. Cannot evaluate on raw.")
            return None

        # Get raw data
        sfreq = raw.info['sfreq']
        data = raw.get_data()  # (n_channels, n_times)
        n_channels, n_times = data.shape

        # Note: ChData from pycrostates is atemporal (no .times attribute)
        # We evaluate on the full raw data duration
        model.logger.info("Evaluating VAE on full raw recording (ChData is atemporal)")

        model.logger.info(f"Raw data shape: {n_channels} channels x {n_times} samples")
        model.logger.info(f"Sampling frequency: {sfreq} Hz")
        model.logger.info(f"Total duration: {n_times / sfreq:.2f} seconds")

        # ===== Step 1: Extract VAE centroids in electrode space =====
        model.set_eval_mode()
        with torch.no_grad():
            decoded_centroids = model.decode(model.mu_c).detach().cpu().numpy()
            if decoded_centroids.ndim == 4:
                decoded_centroids = decoded_centroids.squeeze(1)  # (n_clusters, H, W)

        # Get electrode positions from montage
        montage = model.info.get_montage()
        if montage is None:
            model.logger.error("No montage found in MNE info. Cannot extract electrode positions.")
            return None

        pos_dict = montage.get_positions()
        ch_pos = pos_dict["ch_pos"]

        # Get 2D positions and normalize to image coordinates
        positions_3d = np.array([ch_pos[ch] for ch in model.info.ch_names])
        positions_2d = positions_3d[:, :2]
        pos_min = positions_2d.min(axis=0)
        pos_max = positions_2d.max(axis=0)
        pos_normalized = (positions_2d - pos_min) / (pos_max - pos_min + 1e-8)

        img_size = decoded_centroids.shape[1]
        margin = 0.1
        pos_img = pos_normalized * (1 - 2 * margin) + margin
        pos_img = (pos_img * (img_size - 1)).astype(int)
        pos_img = np.clip(pos_img, 0, img_size - 1)

        # Extract electrode values for each centroid
        n_info_channels = len(model.info.ch_names)
        cluster_centers = np.zeros((model.nClusters, n_info_channels))
        for k in range(model.nClusters):
            for ch_idx in range(n_info_channels):
                x, y = pos_img[ch_idx]
                cluster_centers[k, ch_idx] = decoded_centroids[k, y, x]

        model.logger.info(f"Extracted {model.nClusters} cluster centers with {n_info_channels} electrode values each")

        # Denormalize centroids from z-score space to microvolts
        if norm_params is not None:
            z_mean = norm_params.get("mean", 0.0)
            z_std = norm_params.get("std", 1.0)
            cluster_centers = cluster_centers * z_std + z_mean
            model.logger.info(f"Denormalized centroids to microvolts (mean={z_mean:.4f}, std={z_std:.4f})")
        else:
            model.logger.warning("norm_params not provided — centroids remain in z-score space")

        # ===== Step 2: Assign each timepoint using correlation (polarity-invariant) =====
        model.logger.info("Computing correlation-based assignments for all timepoints...")

        # Normalize data and centers for correlation
        data_norm = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-8)
        centers_norm = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)

        # Compute absolute correlation (polarity invariant)
        correlation = np.abs(np.dot(centers_norm, data_norm))  # (n_clusters, n_times)
        labels = np.argmax(correlation, axis=0)  # (n_times,)

        model.logger.info(f"Assigned {n_times} timepoints to {model.nClusters} clusters")

        # ===== Step 3: Compute temporal statistics =====
        model.logger.info("Computing temporal statistics...")

        total_time = n_times / sfreq

        # Duration: length of continuous segments
        durations = {i: [] for i in range(model.nClusters)}
        current_label = labels[0]
        current_duration = 1

        for i in range(1, len(labels)):
            if labels[i] == current_label:
                current_duration += 1
            else:
                durations[current_label].append(current_duration / sfreq * 1000)  # Convert to ms
                current_label = labels[i]
                current_duration = 1
        durations[current_label].append(current_duration / sfreq * 1000)  # Last segment

        # Calculate mean and std duration per cluster
        mean_duration = []
        std_duration = []
        for i in range(model.nClusters):
            if len(durations[i]) > 0:
                mean_duration.append(np.mean(durations[i]))
                std_duration.append(np.std(durations[i]))
            else:
                mean_duration.append(0.0)
                std_duration.append(0.0)

        # Coverage: percentage of time in each state
        unique, counts = np.unique(labels, return_counts=True)
        coverage = np.zeros(model.nClusters)
        for u, c in zip(unique, counts):
            if u < model.nClusters:
                coverage[u] = c / n_times * 100

        # Occurrence: number of segments per second
        occurrence = np.array([len(durations[i]) / total_time for i in range(model.nClusters)])

        # ===== Step 4: Compute GEV on raw data =====
        model.logger.info("Computing GEV on raw data...")
        total_gfp_squared = np.sum(np.var(data, axis=0))
        explained_variance = 0.0
        for k in range(model.nClusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_data = data[:, mask]
                centroid = cluster_centers[k:k+1].T  # (n_channels, 1)
                # Project data onto centroid direction
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                projections = np.dot(centroid_norm.T, cluster_data)  # (1, n_samples)
                reconstructed = centroid_norm * projections  # (n_channels, n_samples)
                explained_variance += np.sum(np.var(reconstructed, axis=0))

        gev = explained_variance / (total_gfp_squared + 1e-8)

        # ===== Step 5: Generate report =====
        report_lines = [
            "=" * 80,
            "              VAE MICROSTATE STATISTICS REPORT (RAW BACKFIT)",
            "=" * 80,
            "",
            f"METHOD: Convolutional VAE with Clustering (backfitted to raw)",
            f"NUMBER OF CLUSTERS: {model.nClusters}",
            f"TOTAL SAMPLES: {n_times}",
            f"SAMPLING FREQUENCY: {sfreq} Hz",
            f"TOTAL DURATION: {total_time:.2f} seconds",
            f"GLOBAL EXPLAINED VARIANCE (GEV): {gev:.4f}",
            "",
            "DURATION STATISTICS (milliseconds):",
            "-" * 40,
        ]

        for i in range(model.nClusters):
            report_lines.append(f"  Microstate {i+1}: {mean_duration[i]:.2f} +/- {std_duration[i]:.2f} ms")

        report_lines.extend([
            "",
            "TIME COVERAGE (%):",
            "-" * 40,
        ])
        for i in range(model.nClusters):
            report_lines.append(f"  Microstate {i+1}: {coverage[i]:.2f}%")

        report_lines.extend([
            "",
            "OCCURRENCE RATE (per second):",
            "-" * 40,
        ])
        for i in range(model.nClusters):
            report_lines.append(f"  Microstate {i+1}: {occurrence[i]:.2f}/s")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)
        model.logger.info("\n" + report_text)

        # Save report to file
        report_path = os.path.join(output_dir, "vae_raw_microstate_stats.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        model.logger.info(f"Saved report to: {report_path}")

        # Save statistics to JSON
        stats = {
            "method": "VAE (raw backfit)",
            "n_clusters": model.nClusters,
            "n_samples": n_times,
            "sfreq": sfreq,
            "total_time_seconds": total_time,
            "gev": float(gev),
            "mean_duration_ms": mean_duration,
            "std_duration_ms": std_duration,
            "coverage_percent": coverage.tolist(),
            "occurrence_per_sec": occurrence.tolist(),
        }

        json_path = os.path.join(output_dir, "vae_raw_microstate_stats.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)
        model.logger.info(f"Saved statistics to: {json_path}")

        # ===== Step 6: Generate visualization =====
        model._plot_raw_statistics(
            mean_duration, std_duration, coverage, occurrence, gev, output_dir
        )

        return stats

    except Exception as e:
        model.logger.error(f"evaluate_on_raw failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _plot_raw_statistics(
    model, mean_duration, std_duration, coverage, occurrence, gev, output_dir
):
    """Helper to plot raw backfit statistics."""
    try:
        colors = plt.cm.Set2(np.linspace(0, 1, model.nClusters))
        x = np.arange(model.nClusters)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Duration
        axes[0].bar(x, mean_duration, yerr=std_duration, color=colors, alpha=0.8, capsize=5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"MS {i+1}" for i in range(model.nClusters)])
        axes[0].set_ylabel("Duration (ms)", fontweight="bold")
        axes[0].set_title("Mean Duration", fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Coverage
        axes[1].bar(x, coverage, color=colors, alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"MS {i+1}" for i in range(model.nClusters)])
        axes[1].set_ylabel("Coverage (%)", fontweight="bold")
        axes[1].set_title("Time Coverage", fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Occurrence
        axes[2].bar(x, occurrence, color=colors, alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"MS {i+1}" for i in range(model.nClusters)])
        axes[2].set_ylabel("Occurrence (per second)", fontweight="bold")
        axes[2].set_title("Occurrence Rate", fontweight="bold")
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"VAE Microstate Statistics (Raw Backfit, K={model.nClusters}, GEV={gev:.3f})",
            fontweight="bold",
            fontsize=14,
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, "vae_raw_microstate_statistics.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        model.logger.info(f"Saved statistics plot to: {save_path}")

    except Exception as e:
        model.logger.warning(f"Could not plot raw statistics: {e}")


def perform_research_analysis(
    model,
    data_loader: torch.utils.data.DataLoader,
    output_dir: str = "images/analysis",
    norm_params=None,
    raw_mne=None,
):
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    model.logger.info("🧪 Starting Research Analysis & Validation...")
    model.get_cluster_centroids_and_visualize(output_dir=output_dir, norm_params=norm_params)
    model.pycrostates_plot_cluster_centers(output_dir=output_dir, norm_params=norm_params)  # Pycrostates-style topomaps

    model.eval()
    X_list, Z_list, preds_list = [], [], []

    with torch.no_grad():
        centroids_decoded = model.decode(model.mu_c).detach().cpu().numpy()
        # Ensure centroids are flattened to match X (e.g., [K, 32] or [K, 1024])
        centroids_flat = centroids_decoded.reshape(model.nClusters, -1)

        for data, _ in tqdm(data_loader, desc="Aggregating Statistics"):
            data = data.to(model.device)
            mu, _ = model.encode(data)
            preds = model.predict(data)

            X_list.append(data.cpu().numpy().reshape(data.size(0), -1))
            Z_list.append(mu.detach().cpu().numpy())
            preds_list.append(preds)

    X = np.concatenate(X_list, axis=0)
    Z = np.concatenate(Z_list, axis=0)
    labels = np.concatenate(preds_list, axis=0)

    # --- 1. Global Explained Variance (GEV) [Vectorized] ---
    model.logger.info("Calculating GEV...")

    # GFP (Global Field Power) = Standard deviation across channels (spatial)
    gfp = np.std(X, axis=1)
    gfp_squared_sum = np.sum(gfp**2)

    # Get the template map assigned to each timepoint
    assigned_maps = centroids_flat[labels]  # Shape: (N_samples, N_features)

    # Row-wise Pearson Correlation (Vectorized)
    # Center the data
    X_centered = X - X.mean(axis=1, keepdims=True)
    maps_centered = assigned_maps - assigned_maps.mean(axis=1, keepdims=True)

    # Calculate cosine similarity of centered data (Correlation)
    num = np.sum(X_centered * maps_centered, axis=1)
    den = np.sqrt(np.sum(X_centered**2, axis=1) * np.sum(maps_centered**2, axis=1))
    correlations = num / (den + EPSILON)

    numerator = np.sum((gfp * correlations) ** 2)
    gev = numerator / (gfp_squared_sum + EPSILON)

    model.logger.info(f"📊 Global Explained Variance (GEV): {gev:.4f}")

    # --- 2. Temporal Dynamics ---

    # A. Mean Duration
    fs = getattr(model.info, "sfreq", 250)
    durations = {i: [] for i in range(model.nClusters)}

    curr_label = labels[0]
    curr_dur = 0
    for lab in labels:
        if lab == curr_label:
            curr_dur += 1
        else:
            durations[curr_label].append(curr_dur / fs * 1000)
            curr_label = lab
            curr_dur = 1
    durations[curr_label].append(curr_dur / fs * 1000)

    mean_durs = [
        np.mean(durations[i]) if durations[i] else 0 for i in range(model.nClusters)
    ]

    # B. Transition Probability Matrix
    trans_matrix = np.zeros((model.nClusters, model.nClusters))
    for i in range(len(labels) - 1):
        trans_matrix[labels[i], labels[i + 1]] += 1

    # Normalize rows to get probabilities
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_probs = np.divide(
        trans_matrix, row_sums, out=np.zeros_like(trans_matrix), where=row_sums != 0
    )

    # --- Visualization ---

    unique, counts = np.unique(labels, return_counts=True)
    # Ensure all clusters are represented in counts
    counts_full = np.zeros(model.nClusters)
    for u, c in zip(unique, counts):
        counts_full[u] = c

    occupancy = counts_full / len(labels) * 100

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(
        x=[f"C{i+1}" for i in range(model.nClusters)], y=occupancy, palette="viridis"
    )
    plt.title(f"Occupancy (GEV={gev:.2f})")
    plt.ylabel("% Time Active")

    plt.subplot(1, 2, 2)
    sns.barplot(
        x=[f"C{i+1}" for i in range(model.nClusters)], y=mean_durs, palette="magma"
    )
    plt.title("Mean Duration (ms)")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_stats.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        trans_probs,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"C{i+1}" for i in range(model.nClusters)],
        yticklabels=[f"C{i+1}" for i in range(model.nClusters)],
    )
    plt.title("Transition Probability Matrix")
    plt.savefig(os.path.join(output_dir, "transition_matrix.png"), dpi=300)
    plt.close()

    # --- Per-class GEV bar chart ---
    class_gev = []
    for k in range(model.nClusters):
        mask = labels == k
        if mask.sum() > 0:
            class_gev.append(float((gfp[mask] ** 2 * correlations[mask] ** 2).sum() / gfp_squared_sum))
        else:
            class_gev.append(0.0)

    fig_gev, ax_gev = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, model.nClusters))
    bars = ax_gev.bar(range(model.nClusters), class_gev, color=colors, edgecolor='black', linewidth=0.5)
    ax_gev.set_xlabel("Microstate Class")
    ax_gev.set_ylabel("GEV")
    ax_gev.set_title(f"Per-Class GEV (Total={sum(class_gev):.3f})")
    ax_gev.set_xticks(range(model.nClusters))
    ax_gev.set_xticklabels([chr(65 + k) for k in range(model.nClusters)])
    for bar, val in zip(bars, class_gev):
        ax_gev.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    fig_gev.tight_layout()
    fig_gev.savefig(os.path.join(output_dir, "gev_per_class.png"), dpi=200)
    plt.close(fig_gev)

    # --- Raw segmentation ribbon (pycrostates) ---
    if raw_mne is not None:
        try:
            pycrostates_plot_raw_segmentation(model, raw_mne, output_dir, norm_params=norm_params)
        except Exception as e:
            model.logger.warning(f"Raw segmentation plot failed: {e}")

    model.logger.info(f"Research analysis complete. GEV: {gev:.4f}")


def visualize_latent_space(
    model,
    Z: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    context: str = "general",
) -> None:
    if np.isnan(Z).any():
        model.logger.warning(
            f"Found {np.isnan(Z).sum()} NaN values in latent space."
        )
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="mean")
        Z_clean = imputer.fit_transform(Z)
        model.logger.info("Imputed NaN values with mean values.")
    else:
        Z_clean = Z
    try:
        model.logger.info("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(Z_clean) - 1),
            random_state=42,
            n_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        Z_embedded = tsne.fit_transform(Z_clean)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            Z_embedded[:, 0],
            Z_embedded[:, 1],
            c=pred_labels,
            cmap="viridis",
            alpha=0.7,
            s=30,
            label="Predicted Clusters",
        )
        if true_labels is not None:
            plt.scatter(
                Z_embedded[:, 0],
                Z_embedded[:, 1],
                c=true_labels,
                cmap="Set1",
                marker="x",
                s=100,
                alpha=0.5,
                label="True Labels",
            )
        plt.colorbar(scatter, label="Cluster")
        plt.title("Latent Space Visualization (t-SNE)", fontsize=14)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        if true_labels is not None:
            plt.legend(loc="upper right")
        plt.tight_layout()
        output_dir = os.path.join("images", "clustering")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"cluster_latent_space_{context}_k{model.nClusters}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        model.logger.info(f"Visualization saved to {output_path}")
        plt.close()
    except Exception as e:
        model.logger.error(f"Error during visualization: {e}")

