# Interpretable EEG Microstate Discovery via Variational Deep Embedding: A Systematic Architecture Search with Multi-Quadrant Evaluation

## Overview

This repository provides the code, sweep results, and analysis tools for a systematic architecture search of Variational Deep Embedding (VaDE) models applied to EEG microstate discovery. The study evaluates 486 architecture configurations per participant across 10 LEMON dataset subjects, spanning a 4-dimensional hyperparameter grid:

- **K** (number of clusters): 3--20
- **Latent dimension**: 16, 32, 64
- **Encoder depth**: 2, 3, 4 convolutional layers
- **Channel width (ndf)**: 32, 64, 128

Models are evaluated using a novel **multi-quadrant framework** that measures clustering quality in two representation spaces (latent embeddings and decoded topographic maps) with two distance metrics (Euclidean and correlation-based), yielding four complementary perspectives on cluster validity.

The sweep comprises **4,832 trained models** with results stored as comprehensive CSV files (110 metrics per configuration), enabling fine-grained analysis of how architectural choices interact with clustering quality, reconstruction fidelity, and agreement with traditional ModKMeans baselines.

## Architecture

The model follows a **VaDE (Variational Deep Embedding)** architecture combining a convolutional VAE with a Gaussian Mixture Model prior in latent space:

```
Input (40x40 topomap) --> Encoder (Conv2D x depth) --> AdaptivePool --> FC --> mu, log_var
                                                                                |
                                                               Reparameterize z ~ N(mu, var)
                                                                                |
                                                                     GMM Prior (K components)
                                                                     pi_, mu_c, log_var_c
                                                                                |
                                                        FC --> ConvTranspose2D x depth --> Reconstruction
```

Key training features:
- **Polarity-invariant** data augmentation (sign-flipped topomaps)
- **Dual optimizer** with separate learning rates for neural network and GMM parameters
- **Cyclical beta-annealing** to prevent posterior collapse
- **Composite early stopping** on geometric mean of silhouette score and GEV

## Multi-Quadrant Evaluation

All clustering metrics are computed in a 2x2 evaluation grid:

| Quadrant | Space | Distance | Use Case |
|----------|-------|----------|----------|
| **Q1** | Latent (32D) | Euclidean (sklearn) | Standard latent clustering |
| **Q2** | Latent (32D) | \|1/corr\|-1 (pycrostates) | EEG-native latent metric |
| **Q3** | Topomap (1600D) | Euclidean (sklearn) | Decoded reconstruction quality |
| **Q4** | Topomap (1600D) | \|1/corr\|-1 (pycrostates) | Gold-standard EEG comparison |

Each quadrant reports: Silhouette, Davies-Bouldin, Calinski-Harabasz, and Dunn index.

## Key Results

- **4,832 models** trained across 10 LEMON participants
- **GEV** range: 0.214--0.913 (mean 0.724)
- **Best composite scores** (geometric mean of normalized silhouette and GEV): up to 0.77
- Lower K values (3--5) consistently achieve highest composite scores
- Latent dimension and depth show interaction effects that vary by K

## Quick Start

### Installation (Modern Python 3.10+)

```bash
git clone https://github.com/saheedfaremi/microstate-architecture-search.git
cd microstate-architecture-search
pip install -e .
```

### Installation (IBM Power / Python 3.6 CPU)

```bash
bash scripts/setup_cpu_env.sh
pip install -r requirements-py36.txt
```

### Train a Single Configuration

```bash
python train.py --participant 010012 --k 4 --latent-dim 32 --depth 3 --ndf 64
```

### Run the Full 4D Sweep

```bash
bash scripts/run_cube_sweep.sh
```

### Reproduce Paper Figures from Included CSVs

```bash
python analysis/sweep_analysis.py data/sweep_results/all_participants_merged.csv
```

## Repository Structure

```
microstate-architecture-search/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── pyproject.toml                 # Modern Python dependencies
├── requirements-py36.txt          # Python 3.6 CPU-only dependencies
│
├── config/                        # Model and data configuration
│   ├── config.py                  # TOML config loader
│   └── config.toml                # Default hyperparameters
│
├── model.py                       # VAE-GMM architecture (Encoder, Decoder, MyModel)
├── clustering_trainer.py          # Training loop with early stopping (mixin-based)
├── trainer_metrics.py             # 4-quadrant metric computation mixin
├── trainer_viz.py                 # t-SNE visualization mixin
├── trainer_diagnostics.py         # Logging and diagnostics mixin
├── train.py                       # Main entry point (parallel sweep orchestration)
├── train_cluster.py               # Single-configuration training wrapper
├── process_eeg_signals.py         # LEMON data loading, z-score normalization, GFP peaks
├── metrics_utils.py               # Polarity alignment, GEV, pycrostates-style metrics
├── centroid_metrics.py            # Centroid-based clustering evaluation
├── centroid_analysis.py           # Centroid redundancy and merging analysis
├── canonical_matching.py          # Cross-configuration cluster correspondence
├── baseline.py                    # ModKMeans baseline (pycrostates)
├── helper_function.py             # Utilities
├── model_viz.py                   # Model visualization functions
├── seeding.py                     # Reproducibility seed setup
├── parse_args.py                  # CLI argument definitions
│
├── scripts/                       # Sweep orchestration
│   ├── run_cube_sweep.sh          # GPU/CPU 4D sweep (SLURM-compatible)
│   ├── run_sweep_pool.sh          # Continuous slot-filling scheduler (IBM Power)
│   ├── setup_cpu_env.sh           # Conda environment setup for CPU training
│   └── extract_sweep_csv.py       # Extract metrics CSV from training outputs
│
├── analysis/                      # Publication analysis
│   ├── sweep_analysis.py          # Publication-quality figures and tables
│   └── aggregate_results.py       # Result aggregation across runs
│
├── data/
│   ├── README.md                  # LEMON download instructions + CSV column schema
│   └── sweep_results/             # Pre-computed sweep metrics (~16 MB)
│       ├── {subject_id}_sweep.csv # Per-participant results (486 configs each)
│       └── all_participants_merged.csv  # All 4,832 configurations
│
└── docs/
    ├── TECHNICAL_REPORT.md        # Detailed methods and results
    └── RESEARCH_METHODOLOGY.md    # Theory and methodology
```

## Data

### EEG Data (LEMON)

Raw EEG data from the MPI-Leipzig LEMON dataset must be downloaded separately. See `data/README.md` for instructions.

### Sweep Results (Included)

The `data/sweep_results/` directory contains pre-computed metrics for all 4,832 trained models across 10 participants. Each CSV has 110 columns covering:

- Architecture identity (K, latent_dim, depth, ndf)
- Training metadata (best epoch, loss, GEV)
- Reconstruction quality (MSE, KLD, SSIM, spatial correlation) for train/eval/full-data splits
- 4-quadrant clustering metrics (silhouette, DB, CH, Dunn) for each split
- Centroid-based metrics in 4 space/distance combinations
- KL collapse diagnostics
- ModKMeans baseline comparison
- Composite score

See `data/README.md` for the complete column schema.

## Sweep Parameters

| Dimension | Values | Count |
|-----------|--------|-------|
| K (clusters) | 3, 4, 5, ..., 20 | 18 |
| Latent dim | 16, 32, 64 | 3 |
| Depth | 2, 3, 4 | 3 |
| NDF | 32, 64, 128 | 3 |
| **Total per participant** | | **486** |
| **Participants** | | **10** |
| **Total configurations** | | **4,832** |

Some configurations were skipped due to OOM constraints (large K with high ndf/depth), resulting in slightly fewer than 486 per participant.

## Citation

```bibtex
@inproceedings{faremi2026microstate,
  title={Interpretable {EEG} Microstate Discovery via Variational Deep Embedding: A Systematic Architecture Search with Multi-Quadrant Evaluation},
  author={Faremi, Saheed},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
