#!/usr/bin/env python3
"""
EEG Processor for Microstate Analysis (LEMON Dataset)
-------------------------------------------------------
"""

import sys
import os
import json
import logging
import warnings
import math as m
from pathlib import Path

# Visualization imports — lazy for CPU-only / headless environments
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
except ImportError:
    plt = None

# Data Science
import numpy as np
import scipy.signal
import scipy.linalg
from scipy.interpolate import griddata

# EEG — lazy for CPU-only environments without MNE/pycrostates
try:
    import mne
    from mne.io import read_raw_eeglab
    from pycrostates.preprocessing import extract_gfp_peaks
    from pycrostates.datasets import lemon
except ImportError:
    mne = None

# Deep Learning
import torch as T
from torch.utils.data import DataLoader, TensorDataset, Subset

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eeg_processor")

warnings.filterwarnings("ignore")
if mne is not None:
    logging.getLogger("mne").setLevel(logging.WARNING)

# Visualization Style Settings
if plt is not None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "image.cmap": "RdBu_r",
        }
    )


class VaeDatasets:
    """Container for dataset and EEG processing artifacts.

    Holds train/eval DataLoaders, split indices, normalization params,
    and GFP peaks (full set, used by ModKMeans baseline).
    """

    def __init__(
        self, train, train_set=None, gfp_peaks=None, raw_mne=None,
        eval=None, eval_set=None, train_indices=None, eval_indices=None,
        norm_params=None,
    ):
        self.train = train          # DataLoader for training split
        self.train_set = train_set  # TensorDataset/Subset for training split
        self.eval = eval            # DataLoader for eval split
        self.eval_set = eval_set    # TensorDataset/Subset for eval split
        self.train_indices = train_indices
        self.eval_indices = eval_indices
        self.norm_params = norm_params  # {mean, std, clip_std, method}
        self.gfp_peaks = gfp_peaks  # pycrostates ChData for baseline ModKMeans (ALL peaks)
        self.raw_mne = raw_mne      # MNE Raw object for segmentation plots


class EEGProcessor:
    """
    EEG processing pipeline for microstate analysis using LEMON dataset.

    Features:
    - LEMON dataset loading via pycrostates
    - Average referencing
    - GFP peak extraction
    - Topographic map generation
    - Visualization suite
    """

    def __init__(self, config_dict=None, logger=None, subject_id=None):
        """
        Initialize EEG Processor for LEMON dataset.

        Parameters
        ----------
        config_dict : dict, optional
            Configuration dictionary with LEMON parameters.
        logger : logging.Logger, optional
            Logger instance.
        subject_id : str, optional
            LEMON subject ID (overrides config if provided).
        """
        self.logger = logger or logging.getLogger("eeg_processor")

        # LEMON configuration
        self.config = {
            "data_dir": "./data",
            "output_path": "./Topomaps",
            "figure_dir": "./Figure",
            "sample_freq": 250,  # LEMON native frequency
            "num_eeg_channels": 61,  # LEMON has 61 EEG channels
            "topo_map_size": 40,
            "interpolation_method": "cubic",
            "batch_size": 64,
            "max_topo_samples": 500000,
            "lemon_condition": "EC",  # Eyes closed
        }

        if config_dict:
            self.config.update(config_dict)

        # Override subject_id if provided
        if subject_id:
            self.config["lemon_subject_id"] = subject_id

        # Read subject ID from config.toml if not set by config_dict or subject_id param
        if not self.config.get("lemon_subject_id"):
            try:
                from config.config import config as _cfg
                self.config["lemon_subject_id"] = _cfg.get_lemon_config().get("subject_id")
            except Exception:
                pass

        # Core parameters
        self.sfreq = self.config.get("sample_freq", 250)
        self.topo_map_size = self.config.get("topo_map_size", 40)
        self.ch_names = None  # Will be set when loading data

        # Create output directories
        self._create_directories()

        # Data containers
        self.info = None
        self.pos_3d = None
        self.pos_2d = None
        self.sampling_indices = None
        self.gfp_curve = None
        self.gfp_peaks = None  # Store GFP peaks (ChData) for standard microstate analysis

    def _create_directories(self):
        """Create necessary output directories."""
        for key in ["figure_dir", "output_path", "data_dir"]:
            Path(self.config.get(key)).mkdir(parents=True, exist_ok=True)

    def _ensure_directory_exists(self, file_path):
        directory = (
            os.path.dirname(file_path) if os.path.dirname(file_path) else file_path
        )
        os.makedirs(directory, exist_ok=True)

    # =========================================================================
    # Caching Methods (Two-Phase Execution Support)
    # =========================================================================

    def get_cache_dir(self) -> Path:
        """Get the cache directory for preprocessed data."""
        subject_id = self.config.get("lemon_subject_id")
        if not subject_id:
            raise ValueError("lemon_subject_id not set in config. Set it in config.toml [lemon] subject_id.")
        cache_dir = Path(self.config.get("data_dir", "./data")) / "cache" / subject_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def is_cached(self) -> bool:
        """Check if preprocessed data exists in cache (including norm params and split)."""
        cache_dir = self.get_cache_dir()
        required_files = [
            "topo_maps.npy",
            "all_data.npy",
            "raw_preprocessed.fif",
            "norm_params.npz",
            "split_indices.npz",
        ]
        return all((cache_dir / f).exists() for f in required_files)

    def save_to_cache(self, topo_maps, X_all, raw_mne, norm_params, train_indices, eval_indices):
        """Save all preprocessed artifacts to cache for parallel training jobs."""
        cache_dir = self.get_cache_dir()
        self.logger.info(f"Saving preprocessed data to cache: {cache_dir}")

        # Save raw topographic maps (un-normalized, for reference)
        np.save(cache_dir / "topo_maps.npy", topo_maps)

        # Save z-scored data
        np.save(cache_dir / "all_data.npy", X_all)

        # Save preprocessed MNE Raw (for GFP peaks regeneration and visualizations)
        raw_mne.save(cache_dir / "raw_preprocessed.fif", overwrite=True)

        # Save channel info for topomap generation
        np.save(cache_dir / "pos_2d.npy", self.pos_2d)

        # Save normalization parameters
        np.savez(cache_dir / "norm_params.npz", **norm_params)

        # Save split indices
        np.savez(cache_dir / "split_indices.npz",
                 train_indices=train_indices, eval_indices=eval_indices)

        self.logger.info(
            f"Cache saved: {len(topo_maps)} topomaps, "
            f"split={len(train_indices)}/{len(eval_indices)} train/eval"
        )

    def load_from_cache(self):
        """Load preprocessed data from cache. Returns VaeDatasets."""
        cache_dir = self.get_cache_dir()
        self.logger.info(f"Loading preprocessed data from cache: {cache_dir}")

        # Load z-scored data
        X_all = np.load(cache_dir / "all_data.npy")

        # Load normalization parameters (saved as individual numpy arrays)
        norm_npz = np.load(cache_dir / "norm_params.npz")
        norm_params = {
            "mean": float(norm_npz["mean"]),
            "std": float(norm_npz["std"]),
            "clip_std": float(norm_npz["clip_std"]),
            "method": "zscore",
        }

        # Load split indices
        split_data = np.load(cache_dir / "split_indices.npz")
        train_indices = split_data["train_indices"]
        eval_indices = split_data["eval_indices"]

        # Load preprocessed MNE Raw
        raw_mne = mne.io.read_raw_fif(cache_dir / "raw_preprocessed.fif", preload=True, verbose=False)
        self.info = raw_mne.info
        self.ch_names = raw_mne.ch_names
        self.sfreq = raw_mne.info["sfreq"]

        # Load 2D positions for topomaps
        self.pos_2d = np.load(cache_dir / "pos_2d.npy")

        # Regenerate GFP peaks from cached raw (needed for baseline)
        self.logger.info("Regenerating GFP peaks from cached raw...")
        self.gfp_peaks = extract_gfp_peaks(raw_mne, min_peak_distance=3, reject_by_annotation=True)

        self.logger.info(
            f"Cache loaded: {len(X_all)} samples, "
            f"split={len(train_indices)}/{len(eval_indices)} train/eval"
        )

        # Create dataloaders from cached data
        return self.create_dataloaders(
            X_all,
            train_indices=train_indices,
            eval_indices=eval_indices,
            norm_params=norm_params,
            gfp_peaks=self.gfp_peaks,
            raw_mne=raw_mne,
        )

    def load_from_cache_fast(self):
        """Load preprocessed data from cache using numpy only — no MNE/pycrostates.

        Returns VaeDatasets with gfp_peaks=None and raw_mne=None.
        Designed for CPU-only sweep environments (e.g., Python 3.6 IBM server).
        """
        cache_dir = self.get_cache_dir()
        self.logger.info(f"FAST cache load (numpy-only): {cache_dir}")

        X_all = np.load(str(cache_dir / "all_data.npy"))

        norm_npz = np.load(str(cache_dir / "norm_params.npz"))
        norm_params = {
            "mean": float(norm_npz["mean"]),
            "std": float(norm_npz["std"]),
            "clip_std": float(norm_npz["clip_std"]),
            "method": "zscore",
        }

        split_data = np.load(str(cache_dir / "split_indices.npz"))
        train_indices = split_data["train_indices"]
        eval_indices = split_data["eval_indices"]

        self.logger.info(
            f"Fast cache loaded: {len(X_all)} samples, "
            f"split={len(train_indices)}/{len(eval_indices)} train/eval"
        )

        return self.create_dataloaders(
            X_all,
            train_indices=train_indices,
            eval_indices=eval_indices,
            norm_params=norm_params,
            gfp_peaks=None,
            raw_mne=None,
        )

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_lemon_data(self, subject_id=None, condition=None):
        """
        Load LEMON dataset from pycrostates.

        Parameters
        ----------
        subject_id : str, optional
            LEMON subject ID (e.g., "010004"). Uses config if not provided.
        condition : str, optional
            Recording condition: "EC" (eyes closed) or "EO" (eyes open).
            Uses config if not provided.

        Returns
        -------
        raw : mne.io.Raw
            MNE Raw object with preprocessed LEMON data.
        """
        subject_id = subject_id or self.config.get("lemon_subject_id")
        condition = condition or self.config.get("lemon_condition", "EC")

        self.logger.info(f"Loading LEMON data: subject={subject_id}, condition={condition}")

        # Download and get path to LEMON data
        raw_fname = lemon.data_path(subject_id=subject_id, condition=condition)
        self.logger.info(f"LEMON data path: {raw_fname}")

        # Load with MNE's EEGLAB reader
        raw = read_raw_eeglab(raw_fname, preload=True, verbose=False)

        # Pick only EEG channels
        raw.pick("eeg")
        self.logger.info(f"Loaded {len(raw.ch_names)} EEG channels")

        # Set average reference
        raw.set_eeg_reference("average", verbose=False)

        # Store channel names from the loaded data
        self.ch_names = raw.ch_names
        self.sfreq = raw.info["sfreq"]
        self.config["sample_freq"] = self.sfreq
        self.config["num_eeg_channels"] = len(self.ch_names)

        # Apply bandpass filter (2-20 Hz for microstate analysis)
        self.logger.info("Applying 2-20 Hz bandpass filter...")
        raw.filter(l_freq=2.0, h_freq=20.0, method="fir", phase="zero", verbose=False)

        # Store info for later use
        self.info = raw.info

        self.logger.info(
            f"LEMON data loaded: {len(self.ch_names)} channels, "
            f"{raw.n_times} samples, {self.sfreq} Hz"
        )

        return raw

    # =========================================================================
    # Geometry & Coordinate Transformations
    # =========================================================================

    def cart2sph(self, x, y, z):
        """Convert Cartesian to spherical coordinates."""
        x2_y2 = x**2 + y**2
        r = m.sqrt(x2_y2 + z**2)
        elev = m.atan2(z, m.sqrt(x2_y2))
        az = m.atan2(y, x)
        return r, elev, az

    def pol2cart(self, theta, rho):
        """Convert polar to Cartesian coordinates."""
        return rho * m.cos(theta), rho * m.sin(theta)

    def azim_proj(self, pos):
        """Azimuthal equidistant projection."""
        r, elev, az = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    def get_3d_coordinates(self, montage_channel_location):
        """Extract 3D electrode coordinates from montage."""
        location = []
        n_channels = self.config.get("num_eeg_channels", len(self.ch_names) if self.ch_names else 32)

        # Get electrode locations for the number of channels we have
        locs = montage_channel_location[-n_channels:]

        for i in range(len(locs)):
            vals = list(locs[i].values())
            location.append(vals[1] * 1000)

        return np.array(location)

    # =========================================================================
    # Topographic Map Generation
    # =========================================================================

    def create_topographic_map(self, channel_values, pos_2d):
        """Create interpolated topographic map."""
        grid_x, grid_y = np.mgrid[
            min(pos_2d[:, 0]) : max(pos_2d[:, 0]) : self.topo_map_size * 1j,
            min(pos_2d[:, 1]) : max(pos_2d[:, 1]) : self.topo_map_size * 1j,
        ]

        interpolated = griddata(
            pos_2d,
            channel_values,
            (grid_x, grid_y),
            method=self.config["interpolation_method"],
            fill_value=0,
        )

        return interpolated

    def generate_topographic_maps(self, raw_mne):
        """
        Generate topographic maps using pycrostates for LEMON continuous data.
        """
        # Geometry Setup
        self.pos_3d = self.get_3d_coordinates(raw_mne.info["dig"])
        self.pos_2d = np.array([self.azim_proj(p) for p in self.pos_3d])

        #  Extract GFP Peaks using the Library
        self.logger.info("Extracting GFP peaks using pycrostates...")

        # reject_by_annotation=True (default) ensures artifacts are skipped
        gfp_peaks_structure = extract_gfp_peaks(
            raw_mne, min_peak_distance=3, reject_by_annotation=True
        )

        # Store GFP peaks for standard microstate analysis (native electrode space)
        self.gfp_peaks = gfp_peaks_structure

        #  Get Data and Downsample
        peak_maps_data = gfp_peaks_structure.get_data()
        n_peaks = peak_maps_data.shape[1]
        self.logger.info(f"Found {n_peaks} clean GFP peaks")

        max_samples = self.config.get("max_topo_samples", 500000)
        if n_peaks > max_samples:
            selected_indices = np.random.choice(n_peaks, max_samples, replace=False)
            selected_indices.sort()
            selected_data = peak_maps_data[:, selected_indices]
        else:
            selected_data = peak_maps_data

        #  Generate Maps (Interpolation)
        self.logger.info(f"Interpolating {selected_data.shape[1]} maps...")
        maps = []
        for i in range(selected_data.shape[1]):
            vals = selected_data[:, i]
            img = self.create_topographic_map(vals, self.pos_2d)
            maps.append(img)

        #  Store GFP Curve for Visualization
        # We calculate this manually just for the Figure 2 plot
        full_data = raw_mne.get_data()
        self.gfp_curve = np.std(full_data, axis=0)

        # We leave this empty as mapping exact indices back from
        # the cleaned structure to the raw plot is complex and purely cosmetic
        self.sampling_indices = []

        return np.array(maps)

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def make_circular_mask(self, ax, img_size):
        """Create circular mask and head outline on axis."""
        center = img_size / 2 - 0.5
        radius = img_size / 2

        # Create circle for clipping
        circle = patches.Circle((center, center), radius, transform=ax.transData)

        # Draw head outline
        head_circle = patches.Circle(
            (center, center),
            radius,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(head_circle)

        # Draw nose (triangle at top)
        nose_len = radius * 0.15
        nose_wid = radius * 0.1
        nose_x = [center - nose_wid, center, center + nose_wid]
        nose_y = [img_size - 1, img_size - 1 + nose_len, img_size - 1]
        ax.plot(nose_x, nose_y, color="black", linewidth=2, zorder=10)

        return circle

    def _save_figure(self, fig_name, dpi=300):
        """Save figure and close to free memory."""
        fig_dir = self.config["figure_dir"]
        plt.savefig(f"{fig_dir}/{fig_name}", dpi=dpi, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved {fig_name}")

    # =========================================================================
    # Visualizations (17 Figures)
    # =========================================================================

    def generate_figures(self, topo_maps, raw_dataset,
                         X_all=None, norm_params=None,
                         train_indices=None, eval_indices=None):
        """Generate visualizations for LEMON continuous data.

        Parameters
        ----------
        topo_maps : ndarray
            Topographic maps in µV (pre-normalization).
        raw_dataset : ndarray
            Raw EEG data (channels × time) from MNE.
        X_all : ndarray, optional
            Z-score normalized topomaps (for diagnostic figure).
        norm_params : dict, optional
            Normalization parameters {mean, std, clip_std, method}.
        train_indices : ndarray, optional
            Indices of training split samples.
        eval_indices : ndarray, optional
            Indices of evaluation split samples.
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating LEMON Visualizations")
        self.logger.info("=" * 60)

        self._figure_02_gfp_peak_extraction()
        self._figure_04_topomap_grid(topo_maps)
        self._figure_05_gfp_distribution()
        self._figure_06_channel_correlation_matrix(raw_dataset)
        self._figure_07_sensor_layout()
        self._figure_08_spatial_variance(topo_maps)
        self._figure_09_butterfly_plot(raw_dataset)
        self._figure_11_split_half_reliability(topo_maps)

        # Pipeline diagnostic figure (requires normalization data)
        if X_all is not None and norm_params is not None:
            self._figure_pipeline_diagnostic(
                topo_maps=topo_maps,
                X_all=X_all,
                norm_params=norm_params,
                train_indices=train_indices,
                eval_indices=eval_indices,
            )

        self.logger.info("=" * 60)
        self.logger.info("All visualizations complete!")
        self.logger.info("=" * 60)

    def _figure_02_gfp_peak_extraction(self):
        """Figure 2: GFP curve with extracted peaks."""
        try:
            plt.figure(figsize=(14, 5))
            start, end = 1000, 1500
            if len(self.gfp_curve) < end:
                end = len(self.gfp_curve)

            t_axis = np.arange(0, end - start) / self.sfreq
            gfp_segment = self.gfp_curve[start:end]

            plt.fill_between(
                t_axis, gfp_segment, color="#3498db", alpha=0.3, label="GFP Envelope"
            )
            plt.plot(
                t_axis,
                gfp_segment,
                color="#2980b9",
                linewidth=2,
                label="Global Field Power",
            )

            idx_in_window = [
                i - start for i in self.sampling_indices if start <= i < end
            ]
            if idx_in_window:
                peaks_y = gfp_segment[idx_in_window]
                plt.scatter(
                    np.array(idx_in_window) / self.sfreq,
                    peaks_y,
                    c="#e74c3c",
                    s=50,
                    zorder=5,
                    marker="o",
                    edgecolors="white",
                    linewidths=1.5,
                    label=f"Extracted Peaks (n={len(idx_in_window)})",
                )

            plt.xlabel("Time (s)", fontweight="bold")
            plt.ylabel("Global Field Power (μV)", fontweight="bold")
            plt.title(
                "Figure 2: GFP Peak Extraction Strategy", fontweight="bold", fontsize=16
            )
            plt.legend(frameon=True, fancybox=True, shadow=True)
            sns.despine()

            self._save_figure("fig02_gfp_peak_extraction.png")

        except Exception as e:
            self.logger.error(f"Figure 2 Error: {e}")

    def _figure_04_topomap_grid(self, topo_maps):
        """Figure 4: Grid of topographic maps."""
        try:
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            axes = axes.flatten()

            # Select evenly spaced maps
            idx = np.linspace(0, len(topo_maps) - 1, 32, dtype=int)
            vmin = np.percentile(topo_maps[idx], 5)
            vmax = np.percentile(topo_maps[idx], 95)

            for i in range(32):
                img = topo_maps[idx[i]]
                im = axes[i].imshow(
                    img, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax
                )
                clip_path = self.make_circular_mask(axes[i], self.topo_map_size)
                im.set_clip_path(clip_path)
                axes[i].axis("off")
                axes[i].set_title(f"#{idx[i]}", fontsize=8)

            fig.colorbar(
                im,
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.05,
                label="Amplitude (μV)",
            )

            plt.suptitle(
                "Figure 4: Representative Topographic Maps (n=32)",
                fontweight="bold",
                fontsize=16,
            )

            self._save_figure("fig04_topomap_grid.png")

        except Exception as e:
            self.logger.error(f"Figure 4 Error: {e}")

    def _figure_05_gfp_distribution(self):
        """Figure 5: Distribution of GFP values."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(
                self.gfp_curve, bins=100, color="#3498db", alpha=0.7, edgecolor="black"
            )
            axes[0].axvline(
                np.median(self.gfp_curve),
                color="#e74c3c",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(self.gfp_curve):.2f} μV",
            )
            axes[0].set_xlabel("GFP Amplitude (μV)", fontweight="bold")
            axes[0].set_ylabel("Frequency", fontweight="bold")
            axes[0].set_title("GFP Distribution", fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            sns.despine(ax=axes[0])

            sorted_gfp = np.sort(self.gfp_curve)
            cumulative = np.arange(1, len(sorted_gfp) + 1) / len(sorted_gfp)

            axes[1].plot(sorted_gfp, cumulative, color="#2ecc71", linewidth=2)
            axes[1].axhline(0.95, color="#e74c3c", linestyle="--", alpha=0.5)
            axes[1].axhline(0.50, color="#f39c12", linestyle="--", alpha=0.5)
            axes[1].set_xlabel("GFP Amplitude (μV)", fontweight="bold")
            axes[1].set_ylabel("Cumulative Probability", fontweight="bold")
            axes[1].set_title("Cumulative Distribution Function", fontweight="bold")
            axes[1].grid(True, alpha=0.3)
            sns.despine(ax=axes[1])

            plt.suptitle(
                "Figure 5: Global Field Power Statistics",
                fontweight="bold",
                fontsize=16,
            )
            plt.tight_layout()

            self._save_figure("fig05_gfp_distribution.png")

        except Exception as e:
            self.logger.error(f"Figure 5 Error: {e}")

    def _figure_06_channel_correlation_matrix(self, raw_dataset):
        """Figure 6: Spatial correlation between channels."""
        try:
            # Need to reshape flat dataset back to channels x time for correlation
            # raw_dataset from load_preprocessed is (Channels, Time)
            corr_matrix = np.corrcoef(raw_dataset)

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation Coefficient", fontweight="bold")

            tick_positions = np.arange(0, 32, 4)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([self.ch_names[i] for i in tick_positions], rotation=45)
            ax.set_yticklabels([self.ch_names[i] for i in tick_positions])

            ax.set_title(
                "Figure 6: Spatial Correlation Matrix",
                fontweight="bold",
                fontsize=16,
                pad=20,
            )

            plt.tight_layout()
            self._save_figure("fig06_channel_correlation.png")

        except Exception as e:
            self.logger.error(f"Figure 6 Error: {e}")

    def _figure_07_sensor_layout(self):
        """Figure 7: EEG sensor layout."""
        try:
            fig = plt.figure(figsize=(8, 8))
            if self.info is not None:
                mne.viz.plot_sensors(
                    self.info,
                    kind="topomap",
                    show_names=True,
                    axes=plt.gca(),
                    title="",
                    show=False,
                )
                plt.title(
                    "Figure 7: EEG Sensor Layout",
                    fontweight="bold",
                    fontsize=16,
                    pad=20,
                )
            self._save_figure("fig07_sensor_layout.png")
        except Exception as e:
            self.logger.error(f"Figure 7 Error: {e}")

    def _figure_08_spatial_variance(self, topo_maps):
        """Figure 8: Spatial variance map showing active regions."""
        try:
            spatial_var = np.var(topo_maps, axis=0)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(spatial_var, cmap="hot", origin="lower")

            clip_path = self.make_circular_mask(ax, self.topo_map_size)
            im.set_clip_path(clip_path)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Variance (μV²)", fontweight="bold")
            plt.title(
                "Figure 8: Spatial Variance Map (Active Regions)",
                fontweight="bold",
                fontsize=16,
            )
            plt.axis("off")
            self._save_figure("fig08_spatial_variance.png")
        except Exception as e:
            self.logger.error(f"Figure 8 Error: {e}")

    def _figure_09_butterfly_plot(self, raw_dataset):
        """Figure 9: Butterfly plot with GFP overlay (Entire Duration)."""
        try:
            # Get dimensions for the full dataset
            # raw_dataset shape is (Channels, Time)
            n_channels, n_samples = raw_dataset.shape

            # Create full time axis
            times = np.arange(n_samples) / self.sfreq

            #  Transpose data for plotting: (Time, Channels)
            channel_data = raw_dataset.T

            #  Get full GFP curve
            # Ensure GFP matches raw_dataset length (handling potential edge cases)
            gfp_data = self.gfp_curve[:n_samples]

            fig, ax = plt.subplots(figsize=(14, 8))

            #  Plot All Channels (Butterfly Wings)
            # Use very thin lines (0.05) and low alpha to handle high density
            ax.plot(
                times,
                channel_data,
                color="#95a5a6",
                alpha=0.4,
                linewidth=0.05,
                zorder=1,
            )

            #  Plot Global Field Power (Body)
            # Make this prominent to see the envelope over the dense data
            ax.plot(
                times,
                gfp_data,
                color="#e74c3c",
                linewidth=1.0,  # Thinner than before, but thicker than EEG
                label="Global Field Power",
                zorder=10,
            )

            #  Formatting
            ax.set_xlabel("Time (s)", fontweight="bold")
            ax.set_ylabel("Amplitude (μV)", fontweight="bold")
            ax.set_xlim(times[0], times[-1])  # Ensure tight fit

            ax.set_title(
                f"Figure 9: Butterfly Plot (Full Duration: {times[-1]:.1f}s)",
                fontweight="bold",
                fontsize=16,
            )

            # Only add one legend entry for GFP
            # (The channel plots don't need individual legends here)
            ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

            ax.grid(True, alpha=0.3, linestyle=":")
            sns.despine()

            self._save_figure("fig09_butterfly_plot.png")

        except Exception as e:
            self.logger.error(f"Figure 9 Error: {e}")

    def _figure_11_split_half_reliability(self, topo_maps):
        """Figure 11: Split-half reliability analysis."""
        try:
            mid = len(topo_maps) // 2
            half1 = np.mean(topo_maps[:mid], axis=0)
            half2 = np.mean(topo_maps[mid:], axis=0)
            corr = np.corrcoef(half1.flatten(), half2.flatten())[0, 1]

            fig, axes = plt.subplots(1, 3, figsize=(16, 6))

            im1 = axes[0].imshow(half1, cmap="RdBu_r", origin="lower")
            clip1 = self.make_circular_mask(axes[0], self.topo_map_size)
            im1.set_clip_path(clip1)
            axes[0].set_title(f"First Half\n(n={mid} maps)", fontweight="bold")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="µV")

            im2 = axes[1].imshow(half2, cmap="RdBu_r", origin="lower")
            clip2 = self.make_circular_mask(axes[1], self.topo_map_size)
            im2.set_clip_path(clip2)
            axes[1].set_title(
                f"Second Half\n(n={len(topo_maps)-mid} maps)", fontweight="bold"
            )
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="µV")

            sns.regplot(
                x=half1.flatten(),
                y=half2.flatten(),
                ax=axes[2],
                scatter_kws={"alpha": 0.2, "color": "#34495e", "s": 10},
                line_kws={"color": "#e74c3c", "linewidth": 2},
            )
            axes[2].set_title(
                f"Reliability: r = {corr:.4f}", fontweight="bold", fontsize=14
            )
            axes[2].set_xlabel("Pixel Intensity (First Half)", fontweight="bold")
            axes[2].set_ylabel("Pixel Intensity (Second Half)", fontweight="bold")
            axes[2].grid(True, alpha=0.3)
            sns.despine(ax=axes[2])

            plt.suptitle(
                "Figure 11: Split-Half Reliability Analysis",
                fontweight="bold",
                fontsize=16,
            )
            plt.tight_layout()
            self._save_figure("fig11_split_half_reliability.png")

        except Exception as e:
            self.logger.error(f"Figure 11 Error: {e}")

    def _figure_pipeline_diagnostic(self, topo_maps, X_all, norm_params,
                                     train_indices=None, eval_indices=None):
        """Pipeline diagnostic: 3×4 grid showing each preprocessing stage.

        Row 1: Electrode-space → image-space progression
        Row 2: Normalization round-trip verification
        Row 3: Population-level diagnostics (train/eval split, polarity)
        """
        try:
            # Create output directory
            diag_dir = Path(self.config["figure_dir"]) / "pipeline_diagnostic"
            diag_dir.mkdir(parents=True, exist_ok=True)

            # Select representative sample: GFP peak with median amplitude
            peak_amplitudes = np.max(np.abs(topo_maps.reshape(len(topo_maps), -1)), axis=1)
            idx = int(np.argsort(peak_amplitudes)[len(peak_amplitudes) // 2])

            mean = norm_params["mean"]
            std = norm_params["std"]
            clip_std = norm_params.get("clip_std", 5.0)
            img_size = self.topo_map_size

            # --- Prepare data for Row 1 ---
            # Get GFP peak electrode values for the representative sample
            peak_data = self.gfp_peaks.get_data()  # (channels, n_peaks)
            if idx < peak_data.shape[1]:
                electrode_vals = peak_data[:, idx]
            else:
                electrode_vals = peak_data[:, 0]

            # --- Create combined 3×4 figure ---
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle("Pipeline Diagnostic: Preprocessing Stage Verification",
                         fontweight="bold", fontsize=18, y=0.98)

            # ====== ROW 1: Electrode-Space → Image-Space ======

            # P(0,0): Electrode space topomap at GFP peak
            electrode_topo = self.create_topographic_map(electrode_vals, self.pos_2d)
            vmax_r1 = max(np.abs(electrode_topo).max(), np.abs(topo_maps[idx]).max())
            im00 = axes[0, 0].imshow(electrode_topo, cmap="RdBu_r", origin="lower",
                                      vmin=-vmax_r1, vmax=vmax_r1)
            clip00 = self.make_circular_mask(axes[0, 0], img_size)
            im00.set_clip_path(clip00)
            axes[0, 0].set_title("Electrode Space at GFP Peak", fontweight="bold", fontsize=10)
            axes[0, 0].axis("off")
            plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04, label="µV")

            # P(0,1): unused (ICA comparison removed)
            axes[0, 1].axis("off")

            # P(0,2): Interpolated 40×40 topomap (µV)
            im02 = axes[0, 2].imshow(topo_maps[idx], cmap="RdBu_r", origin="lower",
                                      vmin=-vmax_r1, vmax=vmax_r1)
            clip02 = self.make_circular_mask(axes[0, 2], img_size)
            im02.set_clip_path(clip02)
            axes[0, 2].set_title("Interpolated 40×40 Topomap", fontweight="bold", fontsize=10)
            axes[0, 2].axis("off")
            plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04, label="µV")

            # P(0,3): Electrode overlay on grid
            im03 = axes[0, 3].imshow(topo_maps[idx], cmap="RdBu_r", origin="lower",
                                      vmin=-vmax_r1, vmax=vmax_r1)
            clip03 = self.make_circular_mask(axes[0, 3], img_size)
            im03.set_clip_path(clip03)
            # Map electrode positions to grid coordinates
            pos = self.pos_2d
            pos_min = pos.min(axis=0)
            pos_max = pos.max(axis=0)
            pos_norm = (pos - pos_min) / (pos_max - pos_min) * (img_size - 1)
            axes[0, 3].scatter(pos_norm[:, 0], pos_norm[:, 1], c="black", s=15,
                               zorder=5, edgecolors="white", linewidths=0.5)
            axes[0, 3].set_title("Electrode Overlay on Grid", fontweight="bold", fontsize=10)
            axes[0, 3].axis("off")
            plt.colorbar(im03, ax=axes[0, 3], fraction=0.046, pad=0.04, label="µV")

            # ====== ROW 2: Normalization Round-Trip ======

            # P(1,0): After z-score + clip
            zscore_map = X_all[idx]
            vmax_z = clip_std
            im10 = axes[1, 0].imshow(zscore_map, cmap="RdBu_r", origin="lower",
                                      vmin=-vmax_z, vmax=vmax_z)
            clip10 = self.make_circular_mask(axes[1, 0], img_size)
            im10.set_clip_path(clip10)
            axes[1, 0].set_title("After Z-Score + Clip", fontweight="bold", fontsize=10)
            axes[1, 0].axis("off")
            plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04, label="z-score")

            # P(1,1): Denormalized (round-trip)
            denorm_map = self.denormalize(zscore_map, mean, std)
            im11 = axes[1, 1].imshow(denorm_map, cmap="RdBu_r", origin="lower",
                                      vmin=-vmax_r1, vmax=vmax_r1)
            clip11 = self.make_circular_mask(axes[1, 1], img_size)
            im11.set_clip_path(clip11)
            roundtrip_corr = np.corrcoef(topo_maps[idx].ravel(), denorm_map.ravel())[0, 1]
            axes[1, 1].set_title(f"Denormalized (r={roundtrip_corr:.4f})",
                                  fontweight="bold", fontsize=10)
            axes[1, 1].axis("off")
            plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04, label="µV")

            # P(1,2): Pixel histogram pre-norm (µV)
            axes[1, 2].hist(topo_maps[idx].ravel(), bins=50, color="#3498db",
                            alpha=0.7, edgecolor="black", linewidth=0.5)
            axes[1, 2].set_xlabel("Amplitude (µV)", fontweight="bold")
            axes[1, 2].set_ylabel("Pixel Count", fontweight="bold")
            axes[1, 2].set_title("Pixel Histogram: Pre-Norm", fontweight="bold", fontsize=10)
            axes[1, 2].grid(True, alpha=0.3)
            sns.despine(ax=axes[1, 2])

            # P(1,3): Pixel histogram post-norm (z-score) with clip lines
            axes[1, 3].hist(X_all[idx].ravel(), bins=50, color="#e74c3c",
                            alpha=0.7, edgecolor="black", linewidth=0.5)
            axes[1, 3].axvline(-clip_std, color="black", linestyle="--", linewidth=1.5,
                               label=f"Clip ±{clip_std}")
            axes[1, 3].axvline(clip_std, color="black", linestyle="--", linewidth=1.5)
            axes[1, 3].set_xlabel("Amplitude (z-score)", fontweight="bold")
            axes[1, 3].set_ylabel("Pixel Count", fontweight="bold")
            axes[1, 3].set_title("Pixel Histogram: Post-Norm", fontweight="bold", fontsize=10)
            axes[1, 3].legend(fontsize=9)
            axes[1, 3].grid(True, alpha=0.3)
            sns.despine(ax=axes[1, 3])

            # ====== ROW 3: Population-Level Diagnostics ======

            if train_indices is not None and eval_indices is not None:
                # P(2,0): Train mean topomap
                train_mean = np.mean(topo_maps[train_indices], axis=0)
                eval_mean = np.mean(topo_maps[eval_indices], axis=0)
                vmax_pop = max(np.abs(train_mean).max(), np.abs(eval_mean).max())

                im20 = axes[2, 0].imshow(train_mean, cmap="RdBu_r", origin="lower",
                                          vmin=-vmax_pop, vmax=vmax_pop)
                clip20 = self.make_circular_mask(axes[2, 0], img_size)
                im20.set_clip_path(clip20)
                axes[2, 0].set_title(f"Train Mean (n={len(train_indices)})",
                                      fontweight="bold", fontsize=10)
                axes[2, 0].axis("off")
                plt.colorbar(im20, ax=axes[2, 0], fraction=0.046, pad=0.04, label="µV")

                # P(2,1): Eval mean topomap
                im21 = axes[2, 1].imshow(eval_mean, cmap="RdBu_r", origin="lower",
                                          vmin=-vmax_pop, vmax=vmax_pop)
                clip21 = self.make_circular_mask(axes[2, 1], img_size)
                im21.set_clip_path(clip21)
                axes[2, 1].set_title(f"Eval Mean (n={len(eval_indices)})",
                                      fontweight="bold", fontsize=10)
                axes[2, 1].axis("off")
                plt.colorbar(im21, ax=axes[2, 1], fraction=0.046, pad=0.04, label="µV")

                # P(2,2): Train − Eval difference
                diff_map = train_mean - eval_mean
                vmax_diff = np.abs(diff_map).max()
                if vmax_diff < 1e-10:
                    vmax_diff = 1.0
                im22 = axes[2, 2].imshow(diff_map, cmap="RdBu_r", origin="lower",
                                          vmin=-vmax_diff, vmax=vmax_diff)
                clip22 = self.make_circular_mask(axes[2, 2], img_size)
                im22.set_clip_path(clip22)
                axes[2, 2].set_title(f"Train − Eval (max |diff|={vmax_diff:.4f}µV)",
                                      fontweight="bold", fontsize=10)
                axes[2, 2].axis("off")
                plt.colorbar(im22, ax=axes[2, 2], fraction=0.046, pad=0.04, label="µV")
            else:
                for col in range(3):
                    axes[2, col].text(0.5, 0.5, "No split data", ha="center", va="center",
                                       transform=axes[2, col].transAxes, fontsize=12, color="gray")
                    axes[2, col].axis("off")

            # P(2,3): Polarity augmentation (sign-flip mirror)
            flipped = -topo_maps[idx]
            vmax_flip = np.abs(topo_maps[idx]).max()
            im23 = axes[2, 3].imshow(flipped, cmap="RdBu_r", origin="lower",
                                      vmin=-vmax_flip, vmax=vmax_flip)
            clip23 = self.make_circular_mask(axes[2, 3], img_size)
            im23.set_clip_path(clip23)
            axes[2, 3].set_title("Polarity Augmentation (−map)", fontweight="bold", fontsize=10)
            axes[2, 3].axis("off")
            plt.colorbar(im23, ax=axes[2, 3], fraction=0.046, pad=0.04, label="µV")

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save combined grid
            fig.savefig(diag_dir / "pipeline_diagnostic.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            self.logger.info("Saved pipeline_diagnostic/pipeline_diagnostic.png")

            # --- Save individual panels ---
            panel_configs = [
                ("01_electrode_space", electrode_topo, "Electrode Space at GFP Peak", "µV", -vmax_r1, vmax_r1, True),
                ("03_interpolated_topomap_uv", topo_maps[idx], "Interpolated 40×40 Topomap", "µV", -vmax_r1, vmax_r1, True),
                ("05_zscore_normalized", zscore_map, "After Z-Score + Clip", "z-score", -vmax_z, vmax_z, True),
                ("06_denormalized_roundtrip", denorm_map, f"Denormalized (r={roundtrip_corr:.4f})", "µV", -vmax_r1, vmax_r1, True),
                ("12_polarity_augmentation", flipped, "Polarity Augmentation (−map)", "µV", -vmax_flip, vmax_flip, True),
            ]

            # Topomap panels (with circular mask)
            for fname, data, title, cbar_label, vmin, vmax, use_mask in panel_configs:
                fig_i, ax_i = plt.subplots(figsize=(6, 5))
                im_i = ax_i.imshow(data, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
                if use_mask:
                    clip_i = self.make_circular_mask(ax_i, img_size)
                    im_i.set_clip_path(clip_i)
                ax_i.set_title(title, fontweight="bold", fontsize=12)
                ax_i.axis("off")
                plt.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04, label=cbar_label)
                fig_i.savefig(diag_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
                plt.close(fig_i)

            # P04: Electrode overlay (special — needs scatter)
            fig_04, ax_04 = plt.subplots(figsize=(6, 5))
            im_04 = ax_04.imshow(topo_maps[idx], cmap="RdBu_r", origin="lower",
                                  vmin=-vmax_r1, vmax=vmax_r1)
            clip_04 = self.make_circular_mask(ax_04, img_size)
            im_04.set_clip_path(clip_04)
            ax_04.scatter(pos_norm[:, 0], pos_norm[:, 1], c="black", s=15,
                          zorder=5, edgecolors="white", linewidths=0.5)
            ax_04.set_title("Electrode Overlay on Grid", fontweight="bold", fontsize=12)
            ax_04.axis("off")
            plt.colorbar(im_04, ax=ax_04, fraction=0.046, pad=0.04, label="µV")
            fig_04.savefig(diag_dir / "04_electrode_overlay.png", dpi=300, bbox_inches="tight")
            plt.close(fig_04)

            # P07: Histogram pre-norm
            fig_07, ax_07 = plt.subplots(figsize=(6, 5))
            ax_07.hist(topo_maps[idx].ravel(), bins=50, color="#3498db",
                       alpha=0.7, edgecolor="black", linewidth=0.5)
            ax_07.set_xlabel("Amplitude (µV)", fontweight="bold")
            ax_07.set_ylabel("Pixel Count", fontweight="bold")
            ax_07.set_title("Pixel Histogram: Pre-Norm", fontweight="bold", fontsize=12)
            ax_07.grid(True, alpha=0.3)
            sns.despine(ax=ax_07)
            fig_07.savefig(diag_dir / "07_histogram_pre_norm.png", dpi=300, bbox_inches="tight")
            plt.close(fig_07)

            # P08: Histogram post-norm
            fig_08, ax_08 = plt.subplots(figsize=(6, 5))
            ax_08.hist(X_all[idx].ravel(), bins=50, color="#e74c3c",
                       alpha=0.7, edgecolor="black", linewidth=0.5)
            ax_08.axvline(-clip_std, color="black", linestyle="--", linewidth=1.5,
                          label=f"Clip ±{clip_std}")
            ax_08.axvline(clip_std, color="black", linestyle="--", linewidth=1.5)
            ax_08.set_xlabel("Amplitude (z-score)", fontweight="bold")
            ax_08.set_ylabel("Pixel Count", fontweight="bold")
            ax_08.set_title("Pixel Histogram: Post-Norm", fontweight="bold", fontsize=12)
            ax_08.legend(fontsize=9)
            ax_08.grid(True, alpha=0.3)
            sns.despine(ax=ax_08)
            fig_08.savefig(diag_dir / "08_histogram_post_norm.png", dpi=300, bbox_inches="tight")
            plt.close(fig_08)

            # Population panels (train/eval)
            if train_indices is not None and eval_indices is not None:
                for fname, data, title, vmin, vmax in [
                    ("09_train_mean_topomap", train_mean, f"Train Mean (n={len(train_indices)})", -vmax_pop, vmax_pop),
                    ("10_eval_mean_topomap", eval_mean, f"Eval Mean (n={len(eval_indices)})", -vmax_pop, vmax_pop),
                    ("11_train_eval_difference", diff_map, f"Train − Eval (max |diff|={vmax_diff:.4f}µV)", -vmax_diff, vmax_diff),
                ]:
                    fig_p, ax_p = plt.subplots(figsize=(6, 5))
                    im_p = ax_p.imshow(data, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
                    clip_p = self.make_circular_mask(ax_p, img_size)
                    im_p.set_clip_path(clip_p)
                    ax_p.set_title(title, fontweight="bold", fontsize=12)
                    ax_p.axis("off")
                    plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04, label="µV")
                    fig_p.savefig(diag_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig_p)

            self.logger.info(f"Pipeline diagnostic: 13 files saved to {diag_dir}")

        except Exception as e:
            self.logger.error(f"Pipeline diagnostic figure error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    # =========================================================================
    # Data Preparation for Deep Learning
    # =========================================================================

    def prepare_and_normalize_data(self, topo_maps, train_indices):
        """
        Z-score normalize topographic maps.

        Computes mean and std on the TRAINING split only to prevent data leakage,
        then applies to ALL data. Optionally clips at +/- z_score_clip std devs.

        Returns:
            (data_norm, norm_params) where norm_params is a dict with
            {mean, std, clip_std, method}.
        """
        clip_std = self.config.get("z_score_clip", 5.0)
        total_samples = len(topo_maps)
        self.logger.info(
            f"Z-score normalizing {total_samples} samples "
            f"(stats from {len(train_indices)} train samples, clip=+/-{clip_std} std)"
        )

        # Compute stats on train split only (global over all pixels)
        train_data = topo_maps[train_indices]
        mean = float(np.mean(train_data))
        std = float(np.std(train_data))
        if std < 1e-8:
            std = 1e-8

        # Apply z-score to ALL data
        data_norm = (topo_maps - mean) / std

        # Optional clipping at +/- clip_std
        if clip_std > 0:
            data_norm = np.clip(data_norm, -clip_std, clip_std)

        norm_params = {
            "mean": mean,
            "std": std,
            "clip_std": clip_std,
            "method": "zscore",
        }

        self.logger.info(
            f"Z-score stats (train): mean={mean:.4f}, std={std:.4f}, "
            f"result range=[{data_norm.min():.2f}, {data_norm.max():.2f}]"
        )

        return data_norm, norm_params

    @staticmethod
    def denormalize(data, mean, std):
        """Reverse z-score normalization: data * std + mean."""
        return data * std + mean

    def create_dataloaders(self, X_all, train_indices, eval_indices,
                           norm_params=None, gfp_peaks=None, raw_mne=None):
        """Create PyTorch train and eval DataLoaders from z-scored data.

        When polarity_augmentation is enabled (default), sign-flipped copies of
        train topomaps are appended to the training set. EEG microstates are
        polarity-invariant (map A and -A represent the same brain state), so
        augmentation teaches the encoder to place both polarities near the same
        latent position. The eval set is NOT augmented.
        """
        # Reshape for CNN/VAE: (N, 1, 40, 40)
        data_torch = X_all.reshape(-1, 1, self.topo_map_size, self.topo_map_size)
        tensor_x = T.tensor(data_torch, dtype=T.float32)
        tensor_y = T.zeros(len(tensor_x))
        full_dataset = TensorDataset(tensor_x, tensor_y)

        # Create subsets for train and eval
        train_subset = Subset(full_dataset, train_indices.tolist())
        eval_subset = Subset(full_dataset, eval_indices.tolist())

        # Polarity augmentation: append sign-flipped copies to train set only.
        # train_subset (unaugmented) is always kept for GMM initialization —
        # GMM must see clean data, not polarity-doubled pairs that would bias
        # cluster means toward the origin before the encoder learns invariance.
        polarity_aug = self.config.get("polarity_augmentation", True)
        if polarity_aug:
            train_data = T.stack([train_subset[i][0] for i in range(len(train_subset))])
            train_labels = T.zeros(len(train_data))
            flipped_data = -train_data  # sign-flip for polarity invariance
            aug_x = T.cat([train_data, flipped_data], dim=0)
            aug_y = T.cat([train_labels, train_labels], dim=0)
            train_dataset = TensorDataset(aug_x, aug_y)
            self.logger.info(
                f"Polarity augmentation: {len(train_data)} -> {len(aug_x)} train samples "
                f"(sign-flipped copies added)"
            )
        else:
            train_dataset = train_subset

        batch_size = self.config["batch_size"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        eval_loader = DataLoader(
            eval_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.logger.info(
            f"DataLoaders: train={len(train_dataset)} samples "
            f"({len(train_loader)} batches), eval={len(eval_subset)} samples "
            f"({len(eval_loader)} batches)"
        )

        return VaeDatasets(
            train=train_loader,
            train_set=train_subset,  # Unaugmented — used by model.pretrain() for GMM init
            eval=eval_loader,
            eval_set=eval_subset,
            train_indices=train_indices,
            eval_indices=eval_indices,
            norm_params=norm_params,
            gfp_peaks=gfp_peaks,
            raw_mne=raw_mne,
        )

    # =========================================================================
    # Main Processing Pipeline
    # =========================================================================

    def process(self, use_cached: bool = False, save_cache: bool = False, generate_figures: bool = True):
        """Execute complete EEG processing pipeline for LEMON dataset (100% data mode).

        Args:
            use_cached: If True, load preprocessed data from cache (Phase 2).
            save_cache: If True, save preprocessed data to cache (Phase 1).
            generate_figures: If True, generate visualization figures.
        """
        self.logger.info("=" * 60)
        self.logger.info("Processing LEMON Dataset (100% Data Mode)")
        self.logger.info(f"Subject: {self.config.get('lemon_subject_id', 'unknown')}")

        # Phase 2: Load from cache if requested
        if use_cached:
            if self.is_cached():
                self.logger.info("Using CACHED preprocessed data (Phase 2)")
                self.logger.info("=" * 60)
                return self.load_from_cache()
            else:
                self.logger.warning("Cache not found! Falling back to full processing...")

        self.logger.info("Running FULL preprocessing (Phase 1)" if save_cache else "Running standard processing")
        self.logger.info("=" * 60)

        # Step 1: Load LEMON data
        self.logger.info("Step 1/4: Loading LEMON data...")
        raw_mne = self.load_lemon_data()
        raw_data = raw_mne.get_data()

        # Step 2: Generate topographic maps
        self.logger.info("Step 2/5: Generating topographic maps...")
        topo_maps = self.generate_topographic_maps(raw_mne)

        # Save raw maps
        output_path = Path(self.config["output_path"])
        np.save(output_path / "topo_maps.npy", topo_maps)
        self.logger.info(f"Saved {len(topo_maps)} topographic maps")

        # Step 3: Create deterministic 90/10 split
        val_split = self.config.get("val_split", 0.10)
        self.logger.info(f"Step 3/5: Creating {1-val_split:.0%}/{val_split:.0%} train/eval split...")
        n_samples = len(topo_maps)
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_samples)
        split_point = int(n_samples * (1 - val_split))
        train_indices = np.sort(indices[:split_point])
        eval_indices = np.sort(indices[split_point:])
        self.logger.info(
            f"Split: {len(train_indices)} train, {len(eval_indices)} eval"
        )

        # Step 4: Z-score normalize (stats from train split only)
        self.logger.info("Step 4/5: Z-score normalizing data...")
        X_all, norm_params = self.prepare_and_normalize_data(topo_maps, train_indices)

        # Step 5: Generate visualizations (skip in Phase 2 parallel jobs)
        if generate_figures:
            self.logger.info("Step 5/5: Generating visualizations...")
            self.generate_figures(
                topo_maps=topo_maps,
                raw_dataset=raw_data,
                X_all=X_all,
                norm_params=norm_params,
                train_indices=train_indices,
                eval_indices=eval_indices,
            )
        else:
            self.logger.info("Step 5/5: Skipping visualizations (cached mode)")

        # Save to cache for parallel training jobs (Phase 1)
        if save_cache:
            self.save_to_cache(topo_maps, X_all, raw_mne, norm_params,
                               train_indices, eval_indices)

        # Create dataloaders (include gfp_peaks and raw_mne for baseline)
        vae_datasets = self.create_dataloaders(
            X_all,
            train_indices=train_indices,
            eval_indices=eval_indices,
            norm_params=norm_params,
            gfp_peaks=self.gfp_peaks,
            raw_mne=raw_mne,
        )

        self.logger.info("=" * 60)
        self.logger.info("Processing Complete!")
        if generate_figures:
            self.logger.info(f"Figures saved to: {self.config['figure_dir']}")
        self.logger.info(f"Data saved to: {self.config['output_path']}")
        self.logger.info("=" * 60)

        return vae_datasets


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Configure processor for LEMON dataset
    # Read subject ID from config.toml (single source of truth)
    from config.config import config as _cfg
    lemon_cfg = _cfg.get_lemon_config()

    config = {
        "data_dir": "./data",
        "batch_size": 64,
        "max_topo_samples": 50000,
        "lemon_subject_id": lemon_cfg.get("subject_id"),
        "lemon_condition": lemon_cfg.get("condition", "EC"),
    }

    processor = EEGProcessor(config)

    try:
        vae_datasets = processor.process()
        print("\n" + "=" * 60)
        print("SUCCESS! Processing complete.")
        print(f"Check the '{config['data_dir']}/Figure/' directory for results.")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
