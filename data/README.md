# Data

## LEMON Dataset

This project uses the MPI-Leipzig Mind-Brain-Body LEMON dataset. Raw EEG data is not included in this repository due to size (~6 GB).

### Download Instructions

1. Visit the LEMON dataset page: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/
2. Download the EEG resting-state data for the desired participants
3. Alternatively, the `pycrostates` library provides direct access to sample LEMON data:

```python
from pycrostates.datasets import lemon
raw = lemon.load_raw_lemon(subject_id="010012", condition="EO")
```

## Sweep Results

The `sweep_results/` directory contains pre-computed metrics from the 4D architecture sweep. These CSVs are the primary data for the paper's analysis.

### Files

- `{subject_id}_sweep.csv` — Per-participant sweep results (one row per architecture configuration)
- `all_participants_merged.csv` — All participants combined

### Column Schema (110 columns)

#### Identity (7 columns)
| Column | Description |
|--------|-------------|
| `subject` | LEMON participant ID (e.g., "010012") |
| `config` | Configuration folder name |
| `K` | Number of clusters (3-20) |
| `batch_size` | Training batch size |
| `latent_dim` | VAE latent dimension (16, 32, or 64) |
| `depth` | Number of convolutional layers (2, 3, or 4) |
| `ndf` | Base channel width (32, 64, or 128) |

#### Training Metadata (7 columns)
| Column | Description |
|--------|-------------|
| `best_epoch` | Epoch with best stopping metric |
| `total_epochs` | Total epochs trained |
| `train_gev` | Training GEV (Global Explained Variance) |
| `full_data_gev` | GEV on all data (train + eval) |
| `electrode_gev` | GEV in electrode space |
| `best_train_loss` | Best training loss |
| `loss_at_best_epoch` | Loss at the best epoch |

#### Eval Reconstruction (5 columns)
`eval_mse`, `eval_mse_volts`, `eval_kld`, `eval_ssim`, `eval_spatial_corr`

#### Eval 4-Quadrant Clustering (16 columns)
Each quadrant reports silhouette, Davies-Bouldin, Calinski-Harabasz, and Dunn index:

| Quadrant | Space | Distance | Columns |
|----------|-------|----------|---------|
| Q1 | Latent 32D | Euclidean | `eval_q1_sil`, `eval_q1_db`, `eval_q1_ch`, `eval_q1_dunn` |
| Q2 | Latent 32D | \|1/corr\|-1 | `eval_q2_sil`, `eval_q2_db`, `eval_q2_ch`, `eval_q2_dunn` |
| Q3 | Topomap 1600D | Euclidean | `eval_q3_sil`, `eval_q3_db`, `eval_q3_ch`, `eval_q3_dunn` |
| Q4 | Topomap 1600D | \|1/corr\|-1 | `eval_q4_sil`, `eval_q4_db`, `eval_q4_ch`, `eval_q4_dunn` |

#### Train Reconstruction (5 columns)
`train_mse`, `train_mse_volts`, `train_kld`, `train_ssim`, `train_spatial_corr`

#### Train 4-Quadrant Clustering (16 columns)
Same structure as eval quadrants with `train_` prefix.

#### Full-Data Metrics (22 columns)
`full_n_samples`, `full_gev`, `full_mse`, `full_mse_volts`, `full_kld`, `full_ssim`, `full_spatial_corr`, plus 16 quadrant metrics with `full_` prefix.

#### Centroid-Based Metrics (16 columns)
Evaluated using centroid assignments in two spaces with two distance metrics:
- `cent_lat_eucl_{sil,db,ch,dunn}` — Latent space, Euclidean
- `cent_lat_corr_{sil,db,ch,dunn}` — Latent space, correlation
- `cent_raw_eucl_{sil,db,ch,dunn}` — Raw/topomap space, Euclidean
- `cent_raw_corr_{sil,db,ch,dunn}` — Raw/topomap space, correlation

#### KL Diagnostics (3 columns)
| Column | Description |
|--------|-------------|
| `kl_active_dims` | Number of latent dimensions with KL > 0.05 nats |
| `kl_collapsed_dims` | Number of collapsed dimensions (KL < 0.05 nats) |
| `kl_collapse_ratio` | Fraction of collapsed dimensions |

#### Baseline ModKMeans (7 columns)
| Column | Description |
|--------|-------------|
| `baseline_available` | Whether ModKMeans baseline was computed |
| `baseline_gev` | ModKMeans GEV |
| `baseline_sil` | ModKMeans silhouette score |
| `baseline_db` | ModKMeans Davies-Bouldin index |
| `baseline_ch` | ModKMeans Calinski-Harabasz index |
| `baseline_dunn` | ModKMeans Dunn index |
| `baseline_composite` | ModKMeans composite score |

#### Model Config (4 columns)
`lr`, `config_patience`, `pretrain_epochs`, `config_total_epochs`

#### Composite Score (1 column)
| Column | Description |
|--------|-------------|
| `composite` | Geometric mean: `sqrt(((silhouette + 1) / 2) * gev)` |

### Regenerating CSVs

To regenerate CSVs from raw training outputs:

```bash
python scripts/extract_sweep_csv.py outputs/{participant}/{run_id}/ --output data/sweep_results/{participant}_sweep.csv
```

To merge multiple participant CSVs:

```bash
python scripts/extract_sweep_csv.py --merge data/sweep_results/*_sweep.csv --output data/sweep_results/all_participants_merged.csv
```
