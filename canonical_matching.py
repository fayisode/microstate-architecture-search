"""
Canonical EEG Microstate Template Matching (A-B-C-D)
=====================================================

Matches learned VAE cluster centroids against canonical microstate templates
defined by Koenig et al. (2002) and Michel & Koenig (2018).

Canonical microstate classes:
    A: Right-anterior (+) / left-posterior (-) diagonal
    B: Left-anterior (+) / right-posterior (-) diagonal (mirror of A)
    C: Frontal (+) / occipital (-) anterior-posterior gradient
    D: Frontal-midline focal (+) / diffuse (-)

Templates are generated from MNE montage electrode positions using spatial
weighting rules, NOT hardcoded arrays. Matching uses polarity-invariant
|Pearson r| correlation with Hungarian algorithm for optimal assignment.

References:
    - Koenig, T., et al. (2002). Millisecond by millisecond, year by year:
      normative EEG microstates and developmental stages. NeuroImage.
    - Michel, C.M. & Koenig, T. (2018). EEG microstates as a tool for
      studying the temporal dynamics of whole-brain neuronal networks.
      NeuroImage.

Author: Microstate EEG Project
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import mne
except ImportError:
    mne = None
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

EPSILON = 1e-8


class CanonicalMatcher:
    """Match learned EEG microstate centroids to canonical A-B-C-D templates.

    Templates are generated from an MNE montage by applying spatial weighting
    rules that capture the well-established topographic patterns of the four
    canonical microstate classes.

    Parameters
    ----------
    montage_name : str
        Name of the MNE standard montage to use for electrode positions.
        Default is "standard_1020".

    Examples
    --------
    >>> matcher = CanonicalMatcher()
    >>> results = matcher.match(centroids, channel_names)
    >>> print(results["assignments"])
    {'A': 2, 'B': 0, 'C': 3, 'D': 1}
    """

    # Canonical class labels in standard order
    CANONICAL_LABELS = ["A", "B", "C", "D"]

    def __init__(self, montage_name: str = "standard_1020"):
        self.montage_name = montage_name
        self.montage = mne.channels.make_standard_montage(montage_name)
        self._montage_positions = self.montage.get_positions()
        self._ch_pos = self._montage_positions["ch_pos"]
        self._all_channel_names = list(self._ch_pos.keys())
        logger.info(
            f"CanonicalMatcher initialized with {montage_name} montage "
            f"({len(self._all_channel_names)} channels)"
        )

    # =========================================================================
    # Template Generation
    # =========================================================================

    def _get_2d_positions(self, channel_names: List[str]) -> np.ndarray:
        """Extract and normalize 2D electrode positions for given channels.

        Uses azimuthal equidistant projection (standard EEG convention) to
        convert 3D positions to 2D, then centers and normalizes so that x
        and y each span approximately [-1, 1].

        Parameters
        ----------
        channel_names : list of str
            Channel names to extract positions for. Must exist in the montage.

        Returns
        -------
        positions_2d : np.ndarray, shape (n_channels, 2)
            Normalized 2D positions with x (left-right, +right) and
            y (posterior-anterior, +anterior).

        Raises
        ------
        ValueError
            If any channel name is not found in the montage.
        """
        missing = [ch for ch in channel_names if ch not in self._ch_pos]
        if missing:
            raise ValueError(
                f"Channels not found in {self.montage_name} montage: {missing}. "
                f"Available channels: {sorted(self._all_channel_names)[:20]}..."
            )

        positions_3d = np.array([self._ch_pos[ch] for ch in channel_names])

        # Azimuthal equidistant projection (x, y from 3D)
        # MNE convention: x=right, y=anterior, z=superior
        x_3d = positions_3d[:, 0]
        y_3d = positions_3d[:, 1]
        z_3d = positions_3d[:, 2]

        # Project to 2D using standard EEG projection
        r = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
        r = np.maximum(r, EPSILON)
        # Elevation angle from z-axis
        elev = np.arctan2(z_3d, np.sqrt(x_3d**2 + y_3d**2))
        # Azimuth angle
        az = np.arctan2(y_3d, x_3d)
        # Projected radius (inversely related to elevation)
        r_proj = np.pi / 2 - elev

        x_2d = r_proj * np.cos(az)
        y_2d = r_proj * np.sin(az)

        positions_2d = np.column_stack([x_2d, y_2d])

        # Center and normalize to [-1, 1] range
        center = positions_2d.mean(axis=0)
        positions_2d = positions_2d - center
        max_extent = np.max(np.abs(positions_2d))
        if max_extent > EPSILON:
            positions_2d = positions_2d / max_extent

        return positions_2d

    def _load_canonical_templates(
        self, channel_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Generate canonical A-B-C-D microstate templates from electrode positions.

        Templates are constructed by applying spatial weighting rules to the
        2D electrode positions derived from the MNE montage. Each template
        captures the characteristic topographic pattern of its microstate class.

        The spatial rules follow the established literature (Koenig et al. 2002,
        Michel & Koenig 2018):
            A: Diagonal right-anterior to left-posterior gradient
            B: Diagonal left-anterior to right-posterior gradient (mirror of A)
            C: Anterior-posterior gradient (frontal positive, occipital negative)
            D: Focal frontal-midline positivity with diffuse negative background

        Parameters
        ----------
        channel_names : list of str
            Channel names matching the learned centroids.

        Returns
        -------
        templates : dict
            Mapping from canonical label ("A", "B", "C", "D") to template
            array of shape (n_channels,), each normalized to unit norm.
        """
        pos = self._get_2d_positions(channel_names)
        x = pos[:, 0]  # Left(-) to Right(+)
        y = pos[:, 1]  # Posterior(-) to Anterior(+)

        templates = {}

        # --- Template A: Right-anterior(+) / Left-posterior(-) ---
        # Weight = x * (1 - y_norm) where y_norm shifts y so anterior is low
        # This creates a diagonal gradient from left-posterior(-) to right-anterior(+)
        # Shift y to [0, 1] range where 0=posterior, 1=anterior, then invert
        y_norm = (y - y.min()) / (y.max() - y.min() + EPSILON)
        template_a = x * (1 - y_norm)
        templates["A"] = self._normalize_template(template_a)

        # --- Template B: Left-anterior(+) / Right-posterior(-) ---
        # Mirror of A across the midline (negate x component)
        template_b = -x * (1 - y_norm)
        templates["B"] = self._normalize_template(template_b)

        # --- Template C: Frontal(+) / Occipital(-) ---
        # Pure anterior-posterior gradient. In MNE coordinates, y is positive
        # for anterior electrodes. Since we want frontal=positive and
        # occipital=negative, template_c = y directly captures this.
        # Note: sign convention does not affect matching (|Pearson r| is used).
        template_c = y
        templates["C"] = self._normalize_template(template_c)

        # --- Template D: Frontal-midline focal(+) / Diffuse(-) ---
        # Gaussian centered on frontal midline (x=0, y=max)
        frontal_midline_x = 0.0
        frontal_midline_y = y.max()

        # Gaussian with sigma proportional to head radius
        sigma = 0.4
        dist_sq = (x - frontal_midline_x) ** 2 + (y - frontal_midline_y) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma**2))

        # Subtract uniform background to create focal-vs-diffuse contrast
        template_d = gaussian - gaussian.mean()
        templates["D"] = self._normalize_template(template_d)

        logger.info(
            f"Generated canonical templates for {len(channel_names)} channels "
            f"from {self.montage_name} montage"
        )

        return templates

    @staticmethod
    def _normalize_template(template: np.ndarray) -> np.ndarray:
        """Normalize template to unit L2 norm after zero-centering.

        Parameters
        ----------
        template : np.ndarray
            Raw template weights.

        Returns
        -------
        normalized : np.ndarray
            Zero-mean, unit-norm template.
        """
        template = template - template.mean()
        norm = np.linalg.norm(template)
        if norm < EPSILON:
            logger.warning("Template has near-zero norm; returning zeros")
            return template
        return template / norm

    # =========================================================================
    # Matching
    # =========================================================================

    def match(
        self,
        centroids: np.ndarray,
        channel_names: List[str],
    ) -> Dict:
        """Match learned centroids to canonical A-B-C-D templates.

        Uses polarity-invariant absolute Pearson correlation and the Hungarian
        algorithm for optimal assignment. Handles any number of centroids:
        - If n_centroids == 4: one-to-one assignment
        - If n_centroids < 4: best subset of canonical templates
        - If n_centroids > 4: best 4 centroids matched, rest labeled "unmatched"

        Parameters
        ----------
        centroids : np.ndarray, shape (n_centroids, n_channels)
            Learned centroid topographies. Each row is one centroid with values
            at each electrode position.
        channel_names : list of str
            Electrode names corresponding to the columns of ``centroids``.
            Must exist in the montage.

        Returns
        -------
        result : dict
            Keys:
            - "assignments": dict mapping canonical label -> centroid index
            - "correlation_matrix": (4, n_centroids) array of |Pearson r| values
            - "correlation_matrix_signed": (4, n_centroids) array of signed r
            - "matched_correlations": dict of canonical label -> matched |r|
            - "confidence_scores": dict of canonical label -> confidence float
            - "mean_correlation": average |r| across matched pairs
            - "unmatched_centroids": list of centroid indices not assigned

        Raises
        ------
        ValueError
            If centroids shape is inconsistent with channel_names.
        """
        n_centroids, n_channels = centroids.shape

        if n_channels != len(channel_names):
            raise ValueError(
                f"Centroid dimension ({n_channels}) does not match number of "
                f"channel names ({len(channel_names)})"
            )

        if n_centroids == 0:
            raise ValueError("Cannot match zero centroids")

        # Generate canonical templates for the given electrode layout
        templates = self._load_canonical_templates(channel_names)
        n_templates = len(templates)
        template_labels = list(templates.keys())
        template_array = np.array([templates[label] for label in template_labels])

        # Compute correlation matrix: (n_templates, n_centroids)
        corr_matrix_signed = np.zeros((n_templates, n_centroids))
        corr_matrix_abs = np.zeros((n_templates, n_centroids))

        for i in range(n_templates):
            for j in range(n_centroids):
                r, _ = pearsonr(template_array[i], centroids[j])
                corr_matrix_signed[i, j] = r
                corr_matrix_abs[i, j] = abs(r)

        # Hungarian assignment on the matchable subset
        # Build cost matrix for the matchable dimensions
        if n_centroids >= n_templates:
            # More centroids than templates: match all templates
            cost = 1.0 - corr_matrix_abs
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            # Fewer centroids than templates: match best subset of templates
            # Transpose so rows=centroids, cols=templates
            cost = 1.0 - corr_matrix_abs.T
            col_ind_t, row_ind_t = linear_sum_assignment(cost)
            # row_ind_t are centroid indices, col_ind_t are template indices
            row_ind = row_ind_t  # template indices
            col_ind = col_ind_t  # centroid indices

        # Build assignment dict and matched correlations
        assignments = {}
        matched_correlations = {}
        confidence_scores = {}
        matched_centroid_indices = set()

        for t_idx, c_idx in zip(row_ind, col_ind):
            label = template_labels[t_idx]
            assignments[label] = int(c_idx)
            matched_correlations[label] = float(corr_matrix_abs[t_idx, c_idx])
            matched_centroid_indices.add(int(c_idx))

            # Confidence: difference between best and second-best correlation
            # for this template (how uniquely it matches)
            sorted_corrs = np.sort(corr_matrix_abs[t_idx])[::-1]
            if len(sorted_corrs) >= 2:
                confidence = float(sorted_corrs[0] - sorted_corrs[1])
            else:
                confidence = float(sorted_corrs[0])
            confidence_scores[label] = confidence

        # Identify unmatched centroids
        all_centroid_indices = set(range(n_centroids))
        unmatched = sorted(all_centroid_indices - matched_centroid_indices)

        # Compute mean matched correlation
        mean_corr = float(np.mean(list(matched_correlations.values())))

        # Log results
        logger.info(f"Canonical matching results (n_centroids={n_centroids}):")
        for label in template_labels:
            if label in assignments:
                c_idx = assignments[label]
                r_val = matched_correlations[label]
                conf = confidence_scores[label]
                polarity = "same" if corr_matrix_signed[
                    template_labels.index(label), c_idx
                ] > 0 else "inverted"
                logger.info(
                    f"  {label} -> centroid {c_idx}: |r|={r_val:.3f}, "
                    f"confidence={conf:.3f}, polarity={polarity}"
                )
            else:
                logger.info(f"  {label} -> unmatched")
        logger.info(f"  Mean |r|: {mean_corr:.3f}")
        if unmatched:
            logger.info(f"  Unmatched centroids: {unmatched}")

        return {
            "assignments": assignments,
            "correlation_matrix": corr_matrix_abs,
            "correlation_matrix_signed": corr_matrix_signed,
            "matched_correlations": matched_correlations,
            "confidence_scores": confidence_scores,
            "mean_correlation": mean_corr,
            "unmatched_centroids": unmatched,
            "n_centroids": n_centroids,
            "n_templates": n_templates,
            "template_labels": template_labels,
        }

    # =========================================================================
    # Visualization
    # =========================================================================

    def visualize_matching(
        self,
        centroids: np.ndarray,
        info=None,
        output_dir: Union[str, Path],
        norm_params: Optional[Dict] = None,
    ) -> Optional[Path]:
        """Create side-by-side visualization of learned vs canonical centroids.

        For each matched pair, shows the learned centroid topomap next to the
        canonical template topomap with the correlation score. Also saves a
        JSON matching report.

        Parameters
        ----------
        centroids : np.ndarray, shape (n_centroids, n_channels)
            Learned centroid topographies (electrode-space values).
        info : mne.Info
            MNE Info object with channel names and montage.
        output_dir : str or Path
            Directory to save outputs. A ``canonical_matching/`` subdirectory
            will be created.
        norm_params : dict, optional
            If provided, denormalize centroids from z-score to microvolts
            using ``data * std + mean``. Expected keys: "mean", "std".

        Returns
        -------
        output_path : Path or None
            Path to the saved figure, or None if visualization failed.
        """
        output_dir = Path(output_dir)
        match_dir = output_dir / "canonical_matching"
        match_dir.mkdir(parents=True, exist_ok=True)

        try:
            channel_names = info.ch_names

            # Validate info has a montage
            montage = info.get_montage()
            if montage is None:
                logger.warning(
                    "No montage in MNE info -- skipping canonical matching "
                    "visualization"
                )
                return None

            n_centroids, n_channels = centroids.shape
            if n_channels != len(channel_names):
                logger.error(
                    f"Centroid channels ({n_channels}) != info channels "
                    f"({len(channel_names)})"
                )
                return None

            # Denormalize if norm_params provided
            centroids_display = centroids.copy()
            is_denormalized = False
            if norm_params is not None:
                z_mean = norm_params.get("mean", 0.0)
                z_std = norm_params.get("std", 1.0)
                centroids_display = centroids_display * z_std + z_mean
                is_denormalized = True
                logger.info(
                    f"Denormalized centroids to uV "
                    f"(mean={z_mean:.4f}, std={z_std:.4f})"
                )

            # Run matching
            match_result = self.match(centroids, channel_names)
            assignments = match_result["assignments"]
            corr_matrix_signed = match_result["correlation_matrix_signed"]
            template_labels = match_result["template_labels"]

            # Generate templates for visualization
            templates = self._load_canonical_templates(channel_names)

            # Determine matched pairs for plotting
            matched_pairs = []
            for label in self.CANONICAL_LABELS:
                if label in assignments:
                    c_idx = assignments[label]
                    r_abs = match_result["matched_correlations"][label]
                    t_idx = template_labels.index(label)
                    r_signed = corr_matrix_signed[t_idx, c_idx]
                    matched_pairs.append((label, c_idx, r_abs, r_signed))

            # Include unmatched centroids
            unmatched = match_result["unmatched_centroids"]

            # Total rows: matched pairs + unmatched (if any)
            n_matched = len(matched_pairs)
            n_unmatched = len(unmatched)
            n_rows = n_matched + (1 if n_unmatched > 0 else 0)

            if n_rows == 0:
                logger.warning("No pairs to visualize")
                return None

            # --- Figure: Side-by-side topomaps ---
            fig_width = 10
            fig_height = 3.5 * n_rows
            fig, axes = plt.subplots(
                n_rows, 2, figsize=(fig_width, fig_height),
                facecolor="white",
            )
            axes = np.atleast_2d(axes)

            cmap = plt.colormaps["RdBu_r"]
            amplitude_label = (
                "Amplitude (uV)" if is_denormalized else "Amplitude (z-score)"
            )

            for row, (label, c_idx, r_abs, r_signed) in enumerate(matched_pairs):
                # Left: learned centroid
                ax_learned = axes[row, 0]
                centroid_vals = centroids_display[c_idx]
                centroid_centered = centroid_vals - centroid_vals.mean()
                vmax_l = np.percentile(np.abs(centroid_centered), 99)
                if vmax_l < EPSILON:
                    vmax_l = 1.0

                im_l, _ = mne.viz.plot_topomap(
                    centroid_centered,
                    info,
                    axes=ax_learned,
                    show=False,
                    cmap=cmap,
                    vlim=(-vmax_l, vmax_l),
                    contours=6,
                    sensors=True,
                    outlines="head",
                )
                ax_learned.set_title(
                    f"Learned centroid {c_idx}",
                    fontsize=11,
                    fontweight="bold",
                )

                # Right: canonical template
                ax_canon = axes[row, 1]
                template_vals = templates[label]
                vmax_t = np.percentile(np.abs(template_vals), 99)
                if vmax_t < EPSILON:
                    vmax_t = 1.0

                im_t, _ = mne.viz.plot_topomap(
                    template_vals,
                    info,
                    axes=ax_canon,
                    show=False,
                    cmap=cmap,
                    vlim=(-vmax_t, vmax_t),
                    contours=6,
                    sensors=True,
                    outlines="head",
                )

                polarity_str = "same" if r_signed > 0 else "inv."
                confidence = match_result["confidence_scores"][label]
                ax_canon.set_title(
                    f"Canonical {label}  |  |r|={r_abs:.3f} ({polarity_str})\n"
                    f"confidence={confidence:.3f}",
                    fontsize=11,
                    fontweight="bold",
                )

            # Plot unmatched centroids in a final row (if any)
            if n_unmatched > 0:
                ax_unmatched_left = axes[n_matched, 0]
                ax_unmatched_right = axes[n_matched, 1]

                # Show first unmatched centroid on the left
                first_unmatched = unmatched[0]
                um_vals = centroids_display[first_unmatched]
                um_centered = um_vals - um_vals.mean()
                vmax_um = np.percentile(np.abs(um_centered), 99)
                if vmax_um < EPSILON:
                    vmax_um = 1.0

                mne.viz.plot_topomap(
                    um_centered,
                    info,
                    axes=ax_unmatched_left,
                    show=False,
                    cmap=cmap,
                    vlim=(-vmax_um, vmax_um),
                    contours=6,
                    sensors=True,
                    outlines="head",
                )
                unmatched_str = ", ".join(str(u) for u in unmatched)
                ax_unmatched_left.set_title(
                    f"Unmatched centroids: [{unmatched_str}]",
                    fontsize=10,
                    fontweight="bold",
                )
                ax_unmatched_right.axis("off")
                ax_unmatched_right.text(
                    0.5, 0.5,
                    f"{n_unmatched} centroid(s) not matched\n"
                    f"to canonical A-B-C-D templates",
                    ha="center", va="center",
                    fontsize=12, style="italic",
                    transform=ax_unmatched_right.transAxes,
                )

            plt.suptitle(
                f"Canonical Microstate Matching (K={n_centroids})\n"
                f"Mean |r| = {match_result['mean_correlation']:.3f}",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()

            fig_path = match_dir / "centroid_vs_canonical.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info(f"Saved canonical matching figure: {fig_path}")

            # --- Correlation matrix heatmap ---
            self._plot_correlation_matrix(
                match_result, match_dir, n_centroids
            )

            # --- Save JSON report ---
            report = self._build_report(match_result, channel_names, norm_params)
            report_path = match_dir / "matching_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved matching report: {report_path}")

            return fig_path

        except Exception as e:
            logger.error(f"Canonical matching visualization failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _plot_correlation_matrix(
        self,
        match_result: Dict,
        output_dir: Path,
        n_centroids: int,
    ) -> None:
        """Plot the full correlation matrix as a heatmap.

        Parameters
        ----------
        match_result : dict
            Output from ``self.match()``.
        output_dir : Path
            Directory to save the figure.
        n_centroids : int
            Number of learned centroids.
        """
        corr_abs = match_result["correlation_matrix"]
        template_labels = match_result["template_labels"]
        assignments = match_result["assignments"]

        fig, ax = plt.subplots(figsize=(max(6, n_centroids * 1.2), 5), facecolor="white")

        im = ax.imshow(corr_abs, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="|Pearson r|")

        ax.set_xticks(range(n_centroids))
        ax.set_xticklabels([f"C{i}" for i in range(n_centroids)], fontsize=10)
        ax.set_yticks(range(len(template_labels)))
        ax.set_yticklabels(
            [f"Template {l}" for l in template_labels], fontsize=10
        )
        ax.set_xlabel("Learned Centroids", fontsize=12, fontweight="bold")
        ax.set_ylabel("Canonical Templates", fontsize=12, fontweight="bold")
        ax.set_title(
            "Canonical Template Correlation Matrix",
            fontsize=13,
            fontweight="bold",
        )

        # Annotate cells with correlation values
        for i in range(len(template_labels)):
            for j in range(n_centroids):
                val = corr_abs[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold",
                )

        # Highlight matched pairs with a box
        for label, c_idx in assignments.items():
            t_idx = template_labels.index(label)
            rect = plt.Rectangle(
                (c_idx - 0.5, t_idx - 0.5), 1, 1,
                linewidth=2.5, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)

        plt.tight_layout()
        fig_path = output_dir / "correlation_matrix.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved correlation matrix: {fig_path}")

    def _build_report(
        self,
        match_result: Dict,
        channel_names: List[str],
        norm_params: Optional[Dict],
    ) -> Dict:
        """Build a JSON-serializable matching report.

        Parameters
        ----------
        match_result : dict
            Output from ``self.match()``.
        channel_names : list of str
            Electrode channel names used.
        norm_params : dict or None
            Normalization parameters if denormalization was applied.

        Returns
        -------
        report : dict
            Complete matching report with metadata.
        """
        # Build polarity info
        polarity_info = {}
        for label in self.CANONICAL_LABELS:
            if label in match_result["assignments"]:
                c_idx = match_result["assignments"][label]
                t_idx = match_result["template_labels"].index(label)
                r_signed = match_result["correlation_matrix_signed"][t_idx, c_idx]
                polarity_info[label] = "same" if r_signed > 0 else "inverted"

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "montage": self.montage_name,
                "n_channels": len(channel_names),
                "n_centroids": match_result["n_centroids"],
                "denormalized": norm_params is not None,
                "method": "polarity-invariant |Pearson r| + Hungarian algorithm",
            },
            "assignments": match_result["assignments"],
            "matched_correlations": match_result["matched_correlations"],
            "confidence_scores": match_result["confidence_scores"],
            "polarity": polarity_info,
            "mean_correlation": match_result["mean_correlation"],
            "unmatched_centroids": match_result["unmatched_centroids"],
            "interpretation": self._interpret_quality(match_result),
        }

        return report

    @staticmethod
    def _interpret_quality(match_result: Dict) -> Dict[str, str]:
        """Provide human-readable interpretation of matching quality.

        Parameters
        ----------
        match_result : dict
            Output from ``self.match()``.

        Returns
        -------
        interpretation : dict
            Keys: "overall", "per_class" with text descriptions.
        """
        mean_r = match_result["mean_correlation"]

        if mean_r >= 0.8:
            overall = "Excellent match to canonical templates"
        elif mean_r >= 0.6:
            overall = "Good match to canonical templates"
        elif mean_r >= 0.4:
            overall = "Moderate match -- learned states partially align with canonical patterns"
        elif mean_r >= 0.2:
            overall = "Weak match -- learned states may represent non-canonical patterns"
        else:
            overall = "Poor match -- learned states do not resemble canonical microstates"

        per_class = {}
        for label, r_val in match_result["matched_correlations"].items():
            confidence = match_result["confidence_scores"][label]
            if r_val >= 0.7 and confidence >= 0.15:
                quality = "strong, confident"
            elif r_val >= 0.5:
                quality = "moderate"
            elif r_val >= 0.3:
                quality = "weak"
            else:
                quality = "very weak / ambiguous"
            per_class[label] = (
                f"|r|={r_val:.3f}, confidence={confidence:.3f} -- {quality}"
            )

        return {"overall": overall, "per_class": per_class}


# =============================================================================
# Convenience Function
# =============================================================================


def match_centroids_to_canonical(
    centroids: np.ndarray,
    channel_names: List[str],
    info=None,
    output_dir: Optional[Union[str, Path]] = None,
    norm_params: Optional[Dict] = None,
    montage_name: str = "standard_1020",
) -> Dict:
    """Convenience function to match centroids and optionally visualize.

    Parameters
    ----------
    centroids : np.ndarray, shape (n_centroids, n_channels)
        Learned centroid topographies (electrode-space values).
    channel_names : list of str
        Electrode names for each column of centroids.
    info : mne.Info, optional
        MNE Info object. Required for visualization.
    output_dir : str or Path, optional
        If provided, save visualizations and report here.
    norm_params : dict, optional
        If provided, denormalize centroids before visualization.
    montage_name : str
        MNE montage name (default "standard_1020").

    Returns
    -------
    result : dict
        Matching results from ``CanonicalMatcher.match()``, with an added
        "figure_path" key if visualization was generated.
    """
    matcher = CanonicalMatcher(montage_name=montage_name)
    result = matcher.match(centroids, channel_names)

    if output_dir is not None and info is not None:
        fig_path = matcher.visualize_matching(
            centroids, info, output_dir, norm_params=norm_params,
        )
        result["figure_path"] = str(fig_path) if fig_path else None

    return result
