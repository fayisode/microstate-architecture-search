# Technical Report: EEG Microstate Analysis via Variational Deep Embedding

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Preprocessing Pipeline](#2-data-preprocessing-pipeline)
   - 2.1 [Bandpass Filtering](#21-bandpass-filtering)
   - 2.2 [GFP Peak Extraction & Topographic Maps](#22-gfp-peak-extraction--topographic-maps)
   - 2.3 [Z-Score Normalization](#23-z-score-normalization)
   - 2.4 [Train/Evaluation Split](#24-trainevaluation-split)
3. [Polarity Invariance](#3-polarity-invariance)
   - 3.1 [The Polarity Problem in EEG Microstates](#31-the-polarity-problem-in-eeg-microstates)
   - 3.2 [Data Augmentation Strategy](#32-data-augmentation-strategy)
   - 3.3 [Polarity-Invariant Reconstruction Loss](#33-polarity-invariant-reconstruction-loss)
   - 3.4 [Polarity-Invariant Cluster Separation](#34-polarity-invariant-cluster-separation)
4. [Model Architecture](#4-model-architecture)
   - 4.1 [Encoder](#41-encoder)
   - 4.2 [Decoder](#42-decoder)
   - 4.3 [Gaussian Mixture Prior (VaDE)](#43-gaussian-mixture-prior-vade)
5. [The Augmented ELBO: 6-Term Loss Function](#5-the-augmented-elbo-6-term-loss-function)
   - 5.1 [Term 1: Reconstruction Loss](#51-term-1-reconstruction-loss)
   - 5.2 [Term 2: KL Divergence (Mixture-Weighted)](#52-term-2-kl-divergence-mixture-weighted)
   - 5.3 [Term 3: Cluster Entropy Regularization](#53-term-3-cluster-entropy-regularization)
   - 5.4 [Term 4: Log-Barrier Cluster Separation](#54-term-4-log-barrier-cluster-separation)
   - 5.5 [Term 5: Batch-Level Cluster Entropy](#55-term-5-batch-level-cluster-entropy)
   - 5.6 [Term 6: Cluster Tightening (Augmented ELBO)](#56-term-6-cluster-tightening-augmented-elbo)
   - 5.7 [Loss Balance Analysis](#57-loss-balance-analysis)
6. [Beta Annealing Schedule](#6-beta-annealing-schedule)
   - 6.1 [Cyclical Annealing](#61-cyclical-annealing)
   - 6.2 [Beta Ceiling Tuning](#62-beta-ceiling-tuning)
7. [Training Procedure](#7-training-procedure)
   - 7.1 [Two-Phase Pretraining](#71-two-phase-pretraining)
   - 7.2 [Cluster Initialization: Bisecting KMeans](#72-cluster-initialization-bisecting-kmeans)
   - 7.3 [Prior Freeze & Staged Unfreeze](#73-prior-freeze--staged-unfreeze)
   - 7.4 [Dual Optimizer (Canonical VaDE)](#74-dual-optimizer-canonical-vade)
   - 7.5 [GEV-Based Early Stopping](#75-gev-based-early-stopping)
8. [Evaluation Framework](#8-evaluation-framework)
   - 8.1 [4-Quadrant Metric Structure](#81-4-quadrant-metric-structure)
   - 8.2 [Global Explained Variance (GEV)](#82-global-explained-variance-gev)
   - 8.3 [Reconstruction Quality Metrics](#83-reconstruction-quality-metrics)
   - 8.4 [Centroid-Based Metrics](#84-centroid-based-metrics)
   - 8.5 [Baseline Comparison: ModKMeans](#85-baseline-comparison-modkmeans)
9. [Codebase Architecture](#9-codebase-architecture)
   - 9.1 [File Structure & Responsibilities](#91-file-structure--responsibilities)
   - 9.2 [Trainer Mixin Pattern](#92-trainer-mixin-pattern)
   - 9.3 [Configuration Routing](#93-configuration-routing)
   - 9.4 [Parallel Training & Caching](#94-parallel-training--caching)
10. [Iterative Improvements & Rationale](#10-iterative-improvements--rationale)
    - 10.1 [Phase 1: Pipeline Overhaul (Feb 2026)](#101-phase-1-pipeline-overhaul-feb-2026)
    - 10.2 [Phase 2: Research-Backed Model Improvements (Feb 2026)](#102-phase-2-research-backed-model-improvements-feb-2026)
    - 10.3 [Phase 3: Clustering Quality Fixes (Mar 2026)](#103-phase-3-clustering-quality-fixes-mar-2026)
    - 10.4 [Phase 4: Post-Sweep Tuning (Mar 2026)](#104-phase-4-post-sweep-tuning-mar-2026)
11. [Hyperparameter Summary](#11-hyperparameter-summary)
12. [References](#12-references)

---

## 1. Project Overview

This project implements EEG microstate analysis using **Variational Deep Embedding (VaDE)** — a deep generative model that jointly learns a low-dimensional latent representation of EEG topographic maps and clusters them using a Gaussian Mixture Model (GMM) prior. The VAE-GMM approach is compared against the traditional **Modified K-Means (ModKMeans)** baseline from the pycrostates library.

**Dataset**: LEMON (Leipzig Study for Mind-Body-Emotion Interactions), subject 010004, eyes-closed (EC) condition, sampled at 250 Hz.

**Core pipeline**:
1. Preprocess raw EEG (bandpass filter, GFP peak extraction)
2. Extract 40x40 topographic maps at GFP peaks
3. Z-score normalize (train-only statistics)
4. Train VAE-GMM with augmented ELBO for K=3 to K=20 clusters
5. Compare against ModKMeans baseline on standardized metrics

---

## 2. Data Preprocessing Pipeline

### 2.1 Bandpass Filtering

Raw EEG is first bandpass-filtered to **2-20 Hz**, the standard frequency band for microstate analysis (Khanna et al., 2015). This isolates the quasi-stable topographic patterns while removing slow drifts (<2 Hz) and high-frequency muscle artifacts (>20 Hz).

### 2.2 GFP Peak Extraction & Topographic Maps

Global Field Power (GFP) is computed as the standard deviation across all electrodes at each time point:

$$\text{GFP}(t) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (V_i(t) - \bar{V}(t))^2}$$

Topographic maps are extracted only at **GFP peaks** — time points of maximum map stability. This is the standard approach in microstate analysis (Pascual-Marqui et al., 1995), as maps at GFP peaks have the highest signal-to-noise ratio. Each map is interpolated onto a regular 40x40 grid for CNN processing, producing grayscale "images" of the scalp voltage distribution.

### 2.3 Z-Score Normalization

A critical methodological decision was the choice of **z-score normalization** over the initially implemented percentile-clip + min-max approach. The change was motivated by a research audit that identified the following issues with min-max normalization:

| Issue | Min-Max [0,1] | Z-Score |
|-------|---------------|---------|
| Spatial amplitude structure | Destroyed (all maps scaled to same range) | Preserved (relative amplitudes maintained) |
| Decoder output activation | Requires Sigmoid (bounded [0,1]) | Linear (unbounded, matches z-scored range) |
| Data leakage risk | High (global stats from all data) | Low (stats from train split only) |
| Outlier sensitivity | Extreme (p1/p99 clip distorts distribution) | Moderate (clip at ±5σ) |

**Implementation**:

```
1. Compute mean and std from TRAIN split only (prevents data leakage)
2. Apply z-score to ALL data: x_norm = (x - mean) / std
3. Clip at ±5 standard deviations: x_clipped = clip(x_norm, -5, +5)
4. Save normalization parameters for denormalization
```

Statistics are computed exclusively on the training split and then applied to both train and evaluation data. This prevents information about the evaluation set from leaking into the model during training — a methodological concern highlighted by Frontiers in Neuroscience (2024) guidelines on data leakage in neuroimaging.

### 2.4 Train/Evaluation Split

Data is split **90/10** (train/evaluation) using a deterministic random permutation (seed=42):

- **Train set (90%)**: Used for VAE training, with polarity augmentation (see Section 3)
- **Evaluation set (10%)**: Held-out for GEV-based early stopping and overfitting detection; never augmented
- **ModKMeans baseline**: Uses ALL GFP peaks (no split) — this is a fair comparison since ModKMeans has no learned parameters and cannot overfit in the traditional sense

The `VaeDatasets` container maintains both augmented DataLoaders (for training) and unaugmented Subsets (for GMM initialization), ensuring clean separation of data usage throughout the pipeline.

---

## 3. Polarity Invariance

### 3.1 The Polarity Problem in EEG Microstates

EEG microstates are **polarity-invariant**: a topographic map and its sign-inverted counterpart represent the same underlying brain state. This is because the sign of the scalp potential depends on the orientation of the cortical source relative to the scalp surface — a dipole pointing inward vs. outward produces opposite polarity maps but reflects the same neural generator (Murray et al., 2008).

Traditional ModKMeans handles this naturally by using **spatial correlation** as its distance metric: `d(x, c) = 1 - |corr(x, c)|`, where the absolute value makes the metric polarity-invariant.

For VAE-GMM, polarity invariance is non-trivial because:
1. The encoder maps x and -x to different latent positions
2. The GMM prior may learn separate clusters for x and -x (cluster splitting)
3. At high K, this manifests as duplicate clusters with opposite polarities

### 3.2 Data Augmentation Strategy

The primary polarity invariance mechanism is **data augmentation** — sign-flipped copies of all training topomaps are appended to the training set:

```
Original train set:  {x_1, x_2, ..., x_N}
Augmented train set: {x_1, x_2, ..., x_N, -x_1, -x_2, ..., -x_N}
```

This teaches the encoder to map x and -x to nearby latent positions, since both appear with the same pseudo-label. Key design decisions:

- **Train set only**: The evaluation set is never augmented — it represents the true data distribution for unbiased metric computation
- **GMM initialization uses unaugmented data**: The `VaeDatasets.train_set` field provides the original (non-doubled) training subset. If GMM/KMeans initialization used augmented data, cluster means would be biased toward the origin (since x and -x average to zero)
- **Shuffle enabled**: The augmented DataLoader shuffles data so that x and -x from the same sample rarely appear in the same mini-batch

This approach follows Chen et al. (2019) but with the caveat that augmentation alone does not guarantee encoder invariance — the encoder may still encode polarity information. The augmented ELBO terms (Section 5.6) provide additional pressure toward polarity-invariant representations.

### 3.3 Polarity-Invariant Reconstruction Loss

The reconstruction loss uses a **per-sample minimum** over both polarities:

$$\mathcal{L}_{\text{recon}}(x, \hat{x}) = \min\left(\text{MSE}(x, \hat{x}),\; \text{MSE}(-x, \hat{x})\right) \times D$$

where D = 1600 (the flattened 40x40 map size). This removes any incentive for the decoder to preserve the sign of the input — it is rewarded equally for reconstructing either x or -x.

### 3.4 Polarity-Invariant Cluster Separation

The cluster separation loss (Section 5.4) operates on **absolute correlations** between decoded centroids. By taking `|corr(c_i, c_j)|`, the loss penalizes centroids that are similar regardless of polarity — a centroid and its sign-inverse are treated as identical, preventing the model from allocating separate clusters for opposite-polarity versions of the same brain state.

---

## 4. Model Architecture

### 4.1 Encoder

A 6-layer convolutional encoder maps 40x40 topographic maps to a 32-dimensional latent space:

```
Input: (B, 1, 40, 40)
├─ Conv2d(1→32,  k=4, s=2, p=1) + LeakyReLU(0.2) + Dropout(0.2)
├─ Conv2d(32→64, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
├─ Conv2d(64→128, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2) + Dropout(0.2)
├─ Conv2d(128→256, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2)
├─ AdaptiveAvgPool2d(1)
├─ Conv2d(256→1024, k=1, s=1) + LeakyReLU(0.2)
├─ Flatten → FC(1024→512) + LeakyReLU(0.2)
└─ Split → FC(512→32) [mu], FC(512→32) [log_var]
Output: mu ∈ R^32, log_var ∈ R^32
```

The encoder outputs the mean (μ) and log-variance (log σ²) of the approximate posterior q(z|x) = N(μ, diag(σ²)). The latent dimension of 32 was selected based on a sweep over {16, 32, 64} across K=3-20 — 32 dimensions provided the best clustering metrics across all K values.

### 4.2 Decoder

A 5-layer transposed convolutional decoder reconstructs topographic maps from latent vectors:

```
Input: z ∈ R^32
├─ FC(32→512→1024) + LeakyReLU(0.2)
├─ Reshape → (1024, 1, 1)
├─ ConvTranspose2d(1024→256, k=5, s=1) + BatchNorm2d + LeakyReLU(0.2) + Dropout(0.2)
├─ ConvTranspose2d(256→128, k=3, s=1, p=1) + BatchNorm2d + LeakyReLU(0.2)
├─ ConvTranspose2d(128→64, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
├─ ConvTranspose2d(64→32, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
└─ ConvTranspose2d(32→1, k=4, s=2, p=1) → LINEAR OUTPUT (no activation)
Output: (B, 1, 40, 40)
```

The decoder has **no final activation function** (linear output). This is essential because z-scored data has an unbounded range (approximately [-5, +5] after clipping). A Sigmoid activation would compress the output to [0, 1], destroying the amplitude information that z-score normalization preserves.

### 4.3 Gaussian Mixture Prior (VaDE)

Following the Variational Deep Embedding (VaDE) framework (Jiang et al., 2017), the prior over the latent space is a **Gaussian Mixture Model** rather than the standard N(0, I):

$$p(z) = \sum_{c=1}^{K} \pi_c \cdot \mathcal{N}(z \mid \mu_c, \text{diag}(\sigma_c^2))$$

Three sets of learnable parameters define the GMM prior:

| Parameter | Shape | Description | Initialization |
|-----------|-------|-------------|----------------|
| `pi_` | (K,) | Mixture weights in log-space | Log of KMeans cluster proportions |
| `mu_c` | (K, 32) | Cluster means | KMeans centroids in latent space |
| `log_var_c` | (K, 32) | Cluster log-variances | Log of within-cluster variances (floored) |

Cluster assignment for a sample z uses the MAP estimate:

$$c^* = \arg\max_c \left[\log \pi_c + \log \mathcal{N}(z \mid \mu_c, \sigma_c^2)\right]$$

---

## 5. The Augmented ELBO: 6-Term Loss Function

The total loss extends the standard VAE ELBO with four clustering-specific regularization terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} + w_e \cdot \mathcal{L}_{\text{entropy}} + w_s \cdot \mathcal{L}_{\text{sep}} + w_b \cdot \mathcal{L}_{\text{batch}} + w_t \cdot \mathcal{L}_{\text{tight}}$$

### 5.1 Term 1: Reconstruction Loss

$$\mathcal{L}_{\text{recon}} = \min\left(\|x - \hat{x}\|^2,\; \|-x - \hat{x}\|^2\right) \times 1600$$

Mean squared error with polarity-invariant minimum (see Section 3.3). Scaled by the flattened input dimensionality (1600) for gradient magnitude consistency. No clamping is applied — z-scored data naturally spans a bounded range.

### 5.2 Term 2: KL Divergence (Mixture-Weighted)

The KL divergence for VaDE involves the mixture posterior:

$$\mathcal{L}_{\text{KL}} = \sum_c \gamma_c \left[\frac{1}{2}\sum_d \left(\log\frac{\sigma_{c,d}^2}{\sigma_d^2} + \frac{\sigma_d^2 + (\mu_d - \mu_{c,d})^2}{\sigma_{c,d}^2} - 1\right)\right] - H[\gamma]$$

where γ_c = p(c|z) is the soft cluster assignment (posterior responsibility) and H[γ] is the entropy of the assignment distribution. This encourages the approximate posterior to match the mixture prior while maintaining uncertainty in cluster assignments.

### 5.3 Term 3: Cluster Entropy Regularization

$$\mathcal{L}_{\text{entropy}} = \log K - H[\text{softmax}(\pi_)]$$

where H[p] = -Σ p_k log p_k is the Shannon entropy.

**Purpose**: Prevents the GMM from collapsing to a single dominant cluster by encouraging uniform mixture weights. The maximum entropy distribution over K clusters is uniform (H = log K), so this loss is zero when all clusters are equally weighted and positive otherwise.

**Weight**: `w_e = 0.3 / max(log(K), 1.0)` — scales inversely with K to prevent over-regularization at high cluster counts.

### 5.4 Term 4: Log-Barrier Cluster Separation

This term prevents cluster centroids from collapsing to the same topographic pattern. It operates in **decoded (pixel) space** using **absolute spatial correlation** for polarity invariance:

```
Algorithm:
1. Decode all K centroids: c_i = decode(mu_c_i) → (K, 1, 40, 40)
2. Flatten and center each: c_i = c_i - mean(c_i)
3. Compute pairwise absolute correlations: r_ij = |corr(c_i, c_j)|
4. For each unique pair (i,j):
   a. violation = max(0, r_ij - 0.5) / 0.5     (normalized to [0, 1))
   b. violation = clamp(violation, max=0.99)
   c. barrier = -log(1 - violation + 1e-6)       (log-barrier penalty)
5. Loss = mean(barrier) over all unique pairs
```

**Why log-barrier instead of ReLU**: The original implementation used a ReLU penalty: `max(0, r - threshold)`. With typical correlation values of 0.5-0.8 and a threshold of 0.5, this produced raw loss values of ~0.08 — negligible compared to reconstruction loss of ~400. The log-barrier grows to infinity as correlations approach 1.0, creating a strong repulsive force between similar centroids. At `r = 0.9`, the log-barrier produces loss ≈ 2.3, while ReLU gives only 0.4.

**Weight**: `w_s = 50.0`

### 5.5 Term 5: Batch-Level Cluster Entropy

$$\mathcal{L}_{\text{batch}} = \text{KL}\left(\bar{\gamma}_{\text{batch}} \;\|\; \text{Uniform}(1/K)\right)$$

where γ̄_batch is the mean cluster assignment distribution across the mini-batch.

**Purpose**: Complements the prior-level entropy (Term 3) by enforcing uniform cluster usage at the **mini-batch level**. While Term 3 regularizes the global mixture weights π, this term ensures that each mini-batch contains samples from all clusters — preventing training dynamics where certain clusters dominate specific batches and starve others of gradient signal.

**Weight**: `w_b = 5.0`

### 5.6 Term 6: Cluster Tightening (Augmented ELBO)

$$\mathcal{L}_{\text{tight}} = \frac{1}{N}\sum_{i=1}^{N} \text{KL}\left(q(z|x_i) \;\|\; \mathcal{N}(\mu_{c_i^*}, \sigma_{c_i^*}^2)\right)$$

where c*_i = argmax_c p(c|z_i) is the hard cluster assignment using the MAP estimate of z (not stochastic sampling — this is more stable for assignment).

**Purpose**: The standard VaDE ELBO regularizes the aggregate posterior against the mixture prior, but does not explicitly encourage each sample's posterior to be "tight" around its assigned cluster. This term adds that pressure — it is a KL divergence between the encoder output q(z|x) and the specific cluster component N(μ_c*, σ_c*²) that the sample is assigned to.

**Effect**: Samples cluster more compactly around their assigned centroids in latent space, reducing overlap between clusters and improving downstream clustering metrics (silhouette, Davies-Bouldin).

**Weight**: `w_t = 1.0`

### 5.7 Loss Balance Analysis

A key challenge in multi-objective loss functions is ensuring each term contributes meaningful gradient signal. The weight tuning was guided by empirical analysis of loss magnitudes at training steady-state (~epoch 30):

| Term | Raw Magnitude | Weight | Weighted Contribution | % of Total |
|------|--------------|--------|----------------------|------------|
| Reconstruction | ~220 | 1.0 | ~220 | ~85% |
| KL × beta | ~20 × 0.4 | 1.0 | ~8 | ~3% |
| Entropy | ~0.5 | ~0.27 | ~0.14 | <1% |
| Separation | ~0.2 | 50.0 | ~10 | ~4% |
| Batch Entropy | ~0.1 | 5.0 | ~0.5 | <1% |
| Cluster Tightening | ~20 | 1.0 | ~20 | ~8% |
| **Total** | | | **~258** | **100%** |

The clustering terms (separation + tightening + batch entropy + entropy) collectively contribute approximately **12%** of the total loss. This was calibrated through iterative sweeps — an earlier configuration with default weights (0.1 for all clustering terms) resulted in clustering contributing only 1-2% of total gradient, effectively making the model a pure reconstruction autoencoder.

---

## 6. Beta Annealing Schedule

### 6.1 Cyclical Annealing

The β coefficient that weights the KL divergence follows a **cyclical cosine annealing** schedule (Fu et al., 2019), configured with 4 complete cycles over the full training run:

$$\beta(t) = \gamma + \frac{\beta_{\max} - \gamma}{2}\left(1 - \cos\left(\pi \cdot \frac{t \bmod T_c}{T_c}\right)\right)$$

where:
- γ = 0.1 (cycle floor — KL is never fully suppressed)
- β_max = 0.4 (ceiling)
- T_c = total_epochs / 4 (cycle length, ~50 epochs per cycle)

Each cycle ramps β from the floor (0.1) to the ceiling (0.4) following a cosine curve. At the floor, the model focuses on reconstruction; at the ceiling, KL pressure pushes the latent space toward the GMM prior.

The system also supports **batch-level cyclical annealing**, where β varies within each epoch across mini-batches. This provides finer-grained control over the reconstruction-regularization balance within a single pass through the data.

### 6.2 Beta Ceiling Tuning

The maximum beta value was reduced through two iterations:

| Version | max_beta | Rationale |
|---------|----------|-----------|
| Initial | 1.0 | Standard VAE default |
| R1 (Mar 2026) | 0.6 | Lower beta preserves cluster-relevant information (iVAE, 2025) |
| R2 (Mar 2026) | 0.4 | K=3-10 sweep showed best GEV always at beta < 0.5 |

**Why lower beta improves clustering**: High KL pressure forces the posterior q(z|x) toward the prior p(z), which in VaDE is the GMM. But early in training, the GMM prior is poorly calibrated — strong KL pressure pushes encodings toward incorrect cluster centers, destroying useful structure. A lower beta ceiling allows the encoder to maintain discriminative features that are useful for clustering even as the prior adapts.

---

## 7. Training Procedure

### 7.1 Two-Phase Pretraining

Following the S3VDC pattern (Li et al., 2020) rather than vanilla VaDE's pure autoencoder pretraining, training begins with a **gamma-step phase**:

**Phase 1: Gamma Steps**
- Fixed number of gradient steps (= number of batches in one epoch)
- Simplified loss: `L = L_recon + 0.001 × L_KL`
- Very low KL weight (0.001) allows the encoder to learn basic features without KL pressure
- Checkpoint saved every 50 steps for resumability

**Phase 2: Epoch-Based Training with Prior Freeze**
- Full 6-term loss function with beta annealing
- GMM prior parameters frozen for first 5 epochs (see Section 7.3)
- Dead cluster reinitialization checked every 5 epochs
- Transitions into main training after freeze period

### 7.2 Cluster Initialization: Bisecting KMeans

After pretraining the encoder, the GMM prior parameters (μ_c, log_var_c, π) must be initialized. The pipeline uses a **bisecting KMeans** strategy that avoids the mode collapse observed with GMM EM fitting:

```
Bisecting KMeans Algorithm:
1. For K ≤ 4: Standard KMeans with n_init=20
2. For K > 4:
   a. Fit KMeans with K_base=4 clusters
   b. Repeat until target K reached:
      - Find the cluster with most members
      - Split it into 2 using KMeans(n_clusters=2)
      - Replace parent cluster with 2 children
3. Compute cluster statistics from hard KMeans assignments:
   - mu_c[k] = mean of latent vectors assigned to cluster k
   - var_c[k] = max(variance of cluster k, adaptive_floor)
   - pi_[k]  = log(count_k / total_count)
```

**Why not GMM EM initialization**: The original pipeline used sklearn's `GaussianMixture` with KMeans-seeded means. However, at K ≥ 6, EM consistently collapsed to 1-2 dominant modes, leaving 50-60% of clusters dead (zero or near-zero membership) immediately after initialization. This occurred because EM's soft assignments allow a single broad Gaussian to "absorb" multiple nearby clusters. KMeans hard assignments guarantee every cluster retains its members.

**Proportional π instead of uniform**: Earlier versions forced π to uniform (1/K for all clusters). This creates a mismatch when cluster sizes are naturally imbalanced — the model wastes early training epochs fighting this prior. Setting π to the actual cluster proportions aligns the prior with data from the start, resulting in faster convergence.

**Adaptive variance floor**: Within-cluster variances are floored at `max(0.1 × mean(global_std²), 0.1)` to prevent degenerate clusters with near-zero variance (which would produce infinite log-likelihood and numerical instability).

### 7.3 Prior Freeze & Staged Unfreeze

The GMM prior parameters (π, μ_c, log_var_c) are **frozen** during the first 5 epochs of pretraining. This is a canonical VaDE practice that allows the encoder to stabilize its latent representations before the GMM prior starts adapting:

```
Epoch 0-4: Prior FROZEN
  - Only encoder/decoder parameters receive gradients
  - GMM params (pi_, mu_c, log_var_c) have requires_grad=False
  - Encoder learns to produce useful latent features without moving prior targets

Epoch 5+: Staged UNFREEZE
  - Epoch 5: mu_c unfrozen (centroids can adapt)
  - Epoch 7: log_var_c unfrozen (cluster widths can adapt)
  - Epoch 10: pi_ unfrozen (mixture weights can adapt)
```

At the unfreeze epoch in main training, the **NN learning rate is reduced by 10x** while the GMM learning rate is maintained. This prevents the encoder from destabilizing the GMM parameters with large gradients immediately after unfreeze.

### 7.4 Dual Optimizer (Canonical VaDE)

Following canonical VaDE practice, a **single Adam optimizer with two parameter groups** maintains separate learning rates and regularization for neural network vs. GMM parameters:

| Group | Parameters | Learning Rate | Weight Decay | Min LR |
|-------|-----------|---------------|--------------|--------|
| Group 0 (NN) | Encoder + Decoder | 1e-3 | 1e-5 | 1e-5 |
| Group 1 (GMM) | π, μ_c, log_var_c | 5e-4 (= 1e-3 × 0.5) | 0 | 1e-4 |

**Rationale**:
- **Lower GMM LR**: GMM parameters are low-dimensional (3 × K × 32 ≈ 300 params for K=3) compared to the NN (~2M params). Equal learning rates would cause GMM parameters to oscillate wildly.
- **No weight decay on GMM**: μ_c and log_var_c should not be regularized toward zero — their magnitude is determined by the data distribution in latent space.
- **Higher GMM min_lr floor**: GMM parameters need meaningful updates throughout training to track the evolving latent space. The NN floor is lower (1e-5) to allow fine-tuning in late training.

A **ReduceLROnPlateau** scheduler monitors training loss with patience=10 and factor=0.5, providing per-group minimum LR floors.

### 7.5 GEV-Based Early Stopping

Early stopping is based on **Global Explained Variance (GEV)** computed on the held-out evaluation set:

```
Early Stopping Logic:
1. After each training epoch, compute GEV on eval set (10%)
2. If GEV improves: reset patience counter, save model checkpoint
3. If no improvement: increment patience counter
4. If patience counter reaches 20: stop training
5. Restore best model from checkpoint for final evaluation
```

**Why GEV instead of training loss**: Training loss can continue to decrease as the model improves reconstruction, even as clustering quality degrades (the loss terms compete). GEV directly measures how well the learned cluster centroids explain the data variance — the actual objective of microstate analysis.

**Scheduler vs. early stopping patience**: The LR scheduler patience (10) is deliberately set lower than early stopping patience (20). This creates a cascade: the learning rate drops twice before training stops, giving the model a chance to find a better optimum at a lower learning rate before giving up entirely.

---

## 8. Evaluation Framework

### 8.1 4-Quadrant Metric Structure

All clustering metrics are organized in a 2×2 grid crossing **representation space** with **distance metric**:

|  | Euclidean (sklearn) | Correlation \|1/r\|-1 (pycrostates) |
|---|---|---|
| **Latent 32D** | Q1: `q1_latent_eucl_{metric}` | Q2: `q2_latent_corr_{metric}` |
| **Decoded 1600D** | Q3: `q3_topo_eucl_{metric}` | Q4: `q4_topo_corr_{metric}` |

Each quadrant reports four clustering quality metrics:

| Metric | Optimal | Interpretation |
|--------|---------|----------------|
| **Silhouette** | Higher (max 1.0) | Cluster cohesion vs. separation |
| **Davies-Bouldin** | Lower (min 0.0) | Ratio of within-cluster to between-cluster distance |
| **Calinski-Harabasz** | Higher | Ratio of between-cluster to within-cluster dispersion |
| **Dunn** | Higher | Minimum inter-cluster distance / maximum intra-cluster diameter |

**Why 4 quadrants**: Different spaces and distance metrics capture different aspects of clustering quality. Q1 (latent + Euclidean) measures how well the encoder separates clusters geometrically. Q4 (decoded + correlation) is the most directly comparable to the ModKMeans baseline, which operates on correlation distance in electrode space. Cross-referencing all four quadrants reveals whether good latent-space clustering translates to good topographic-space clustering.

### 8.2 Global Explained Variance (GEV)

GEV measures how well the cluster centroids explain the variance in the data, weighted by each time point's Global Field Power:

$$\text{GEV} = \frac{\sum_{t} \left(\text{GFP}(t) \cdot |r(x_t, c_{k_t^*})|\right)^2}{\sum_{t} \text{GFP}(t)^2}$$

where r(x_t, c_k*) is the Pearson correlation between sample x_t and its assigned centroid c_k*, and |·| provides polarity invariance. GFP weighting ensures that time points with strong, reliable signals contribute more to the metric.

GEV is the standard evaluation metric in microstate analysis (Pascual-Marqui et al., 1995) and enables direct comparison between VAE-GMM and ModKMeans. The implementation computes GEV on **decoded centroids** (pixel space) to maintain consistency with the topomap representation.

### 8.3 Reconstruction Quality Metrics

| Metric | Description | Data Range |
|--------|-------------|------------|
| **MSE** (recon_loss) | Mean squared error between input and reconstruction | Lower is better |
| **KLD** (kld_loss) | KL divergence from posterior to mixture prior | Lower is better |
| **SSIM** | Structural similarity index (window-based) | Higher is better (max 1.0), data_range = 10.0 |
| **Spatial Correlation** | Mean \|Pearson r\| between input and reconstruction | Higher is better (max 1.0) |
| **Eval Recon Loss** | MSE on evaluation set (overfitting detector) | Should track train MSE |

### 8.4 Centroid-Based Metrics

In addition to sample-level clustering metrics, **centroid-based analysis** provides a complementary view:

- **Centroid distance matrix**: Using `d = 1 - |corr(c_i, c_j)|` between all pairs of decoded centroids
- **Centroid-to-sample assignment**: Each sample assigned to nearest centroid using correlation distance
- **Per-cluster statistics**: Size, mean distance to centroid, variance
- **History tracking**: Centroid evolution across training epochs

### 8.5 Baseline Comparison: ModKMeans

The Modified K-Means (ModKMeans) baseline uses the **pycrostates** library, which implements the standard microstate analysis pipeline:

1. Extract GFP peaks as ChData objects (61 electrodes, not interpolated to 40x40)
2. Fit ModKMeans with polarity-invariant distance: `d = 1 - |corr|`
3. Multiple random initializations for robustness
4. Report GEV, cluster maps, and segmentation statistics

**Key differences from VAE-GMM**:
- ModKMeans operates in **61D electrode space** (raw channel data)
- VAE-GMM operates in **1600D pixel space** (40x40 interpolated topomaps) mapped to **32D latent space**
- ModKMeans uses ALL data (no train/eval split needed)
- ModKMeans has no learned parameters — results are deterministic given the initialization

---

## 9. Codebase Architecture

### 9.1 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `train.py` | Top-level orchestrator — parallel/sequential dispatch, results aggregation |
| `train_cluster.py` | Per-cluster training — model creation, trainer setup, config wiring |
| `clustering_trainer.py` | VAEClusteringTrainer — main training loop, early stopping, pipeline orchestration |
| `trainer_metrics.py` | TrainerMetricsMixin — GEV computation, 4-quadrant metrics, centroid metrics |
| `trainer_viz.py` | TrainerVizMixin — t-SNE snapshots, strategy visualizations, latent space plots |
| `trainer_diagnostics.py` | TrainerDiagnosticsMixin — logging, reports, `best_model_metrics.json` |
| `model.py` | VAE architecture, GMM clustering, all loss functions, pretrain logic |
| `model_viz.py` | Standalone visualization functions (extracted from model.py) |
| `process_eeg_signals.py` | EEG loading, z-score normalization, 90/10 split, caching |
| `baseline.py` | ModKMeans baseline using pycrostates |
| `centroid_metrics.py` | Centroid-based metrics (`d = 1 - \|corr\|`) |
| `metrics_utils.py` | Pairwise metrics (`d = \|1/corr\| - 1`, pycrostates formula) |
| `evaluate_polarity_merge.py` | Polarity analysis and multi-strategy comparison |
| `config/config.toml` | All hyperparameters |
| `config/config.py` | Config loading and section routing |

### 9.2 Trainer Mixin Pattern

The trainer was refactored from a monolithic 3890-line class into a **mixin-based architecture** for maintainability:

```
VAEClusteringTrainer(TrainerMetricsMixin, TrainerVizMixin, TrainerDiagnosticsMixin)
│
├── TrainerMetricsMixin (trainer_metrics.py)
│   ├── _compute_gev()
│   ├── _compute_metrics_and_losses()
│   ├── _compute_4quadrant_metrics()
│   └── _compute_centroid_based_metrics()
│
├── TrainerVizMixin (trainer_viz.py)
│   ├── generate_tsne_snapshot()
│   ├── plot_strategy_comparison()
│   └── plot_latent_space()
│
└── TrainerDiagnosticsMixin (trainer_diagnostics.py)
    ├── generate_epoch_summary()
    ├── generate_final_report()
    └── save_best_model_metrics()
```

Method Resolution Order: `VAEClusteringTrainer → MetricsMixin → VizMixin → DiagnosticsMixin → object`. No method overrides across mixins — all cross-mixin calls resolve via `self` at runtime.

### 9.3 Configuration Routing

All hyperparameters are centralized in `config/config.toml`:

```
get_model_config()      → [vae] section (training, loss, architecture)
get_eeg_config()        → [eeg] section + [lemon] keys (preprocessing)
get_lemon_config()      → [lemon] section (dataset-specific)
get_clustering_config() → [clustering] section (merge thresholds)
```

### 9.4 Parallel Training & Caching

For K-sweep experiments (e.g., K=3 to K=20), the pipeline supports:

**Two-phase execution**:
1. `--preprocess-only`: Full EEG preprocessing, cached to `data/cache/{subject_id}/`
2. `--use-cached`: Load from cache, spawn one training process per K value

**Parallel dispatch**: Multiple K values train simultaneously on different GPUs. DataLoaders use `num_workers=0` inside parallel workers to avoid nested-fork deadlocks (a fix applied after observing process hangs during final evaluation).

**Cache files**: `all_data.npy`, `raw_preprocessed.fif`, `norm_params.npz`, `split_indices.npz`, `pos_2d.npy`

---

## 10. Iterative Improvements & Rationale

### 10.1 Phase 1: Pipeline Overhaul (Feb 2026)

**Motivation**: Research audit identified methodological gaps — normalization destroying spatial structure, missing denormalization parameters, GEV computed in wrong space, and fundamental space mismatch between VAE and baseline.

**Changes**:
1. Z-score normalization (train-only stats) replacing percentile clip + min-max
2. 90/10 train/eval split with deterministic seed
3. Linear decoder output (removed Sigmoid activation)
5. Removed clamping from reconstruction loss
6. 4-quadrant metrics framework
7. GEV-based early stopping on held-out eval set

### 10.2 Phase 2: Research-Backed Model Improvements (Feb 2026)

**Motivation**: Literature audit of VaDE (2017), S3VDC (2020), and EEG microstate papers identified standard practices not yet implemented.

**Changes**:
1. Polarity invariance via data augmentation (sign-flipped training copies)
2. Polarity-invariant reconstruction loss (per-sample minimum)
3. Canonical dual optimizer (separate NN/GMM learning rates)
4. Prior freeze for first 5 pretraining epochs
5. Per-group minimum LR floors
6. Staged prior unfreeze

### 10.3 Phase 3: Clustering Quality Fixes (Mar 2026)

**Motivation**: Latent dimension sweep (K=3-20 × ld={16,32,64}) revealed systematic clustering issues — dead clusters, collapsed centroids, and poor metrics at K ≥ 6.

**Changes**:
1. **max_beta reduced** 1.0 → 0.6: Lower beta preserves cluster-relevant information
2. **Bisecting KMeans**: Hierarchical initialization prevents dead clusters at high K
3. **Log-barrier separation loss**: Replaces ReLU penalty (was 5000x too small)
4. **Cluster tightening loss**: New augmented ELBO term for tighter cluster ownership
5. **Batch-level cluster entropy**: Enforces uniform usage within mini-batches
6. **GMM n_init=1**: Fixes sklearn gotcha where means_init is ignored for restarts > 1

### 10.4 Phase 4: Post-Sweep Tuning (Mar 2026)

**Motivation**: K=3-10 sweep with Phase 3 changes revealed that clustering terms still contributed only 1-2% of total gradient (dominated by reconstruction), GMM EM still collapsed at K ≥ 6, and patience=40 wasted 30+ epochs after GEV peaked.

**Changes**:
1. **Loss weight rebalancing**: separation 5→50, tightening 0.1→1.0, batch entropy 0.1→5.0 (clustering now ~12% of total)
2. **max_beta 0.6 → 0.4**: Sweep confirmed best GEV always at beta < 0.5
3. **GMM replaced with KMeans-only init**: Eliminates EM mode collapse entirely; proportional π instead of forced uniform
4. **Early stopping patience 40 → 20**: Prevents 30+ wasted epochs after GEV peak
5. **Configurable scheduler patience**: 10 epochs (< early stopping patience of 20)
6. **DataLoader fix**: Disabled nested workers in parallel mode to prevent deadlocks

---

## 11. Hyperparameter Summary

### Model Architecture
| Parameter | Value | Notes |
|-----------|-------|-------|
| Latent dimensions | 32 | Optimal across K=3-20 (sweep-validated) |
| Encoder channels | [32, 64, 128, 256, 512, 1024] | 6-layer CNN |
| Decoder output | Linear (no activation) | Matches z-scored data range |
| Batch size | 128 | |

### Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| Max epochs | 200 | Rarely reached (early stopping) |
| Early stopping patience | 20 | Based on eval GEV |
| Scheduler patience | 10 | ReduceLROnPlateau, per-group min_lr |
| NN learning rate | 1e-3 | Weight decay 1e-5, min_lr 1e-5 |
| GMM learning rate | 5e-4 | No weight decay, min_lr 1e-4 |
| Prior freeze epochs | 5 | Encoder stabilizes before GMM adapts |

### Loss Function
| Parameter | Value | Notes |
|-----------|-------|-------|
| max_beta | 0.4 | Cyclical cosine, 4 cycles, floor=0.1 |
| separation_weight | 50.0 | Log-barrier on decoded centroid correlations |
| cluster_tightening_weight | 1.0 | KL(q(z|x) ‖ N(μ_c*, σ_c*²)) |
| batch_entropy_weight | 5.0 | KL(batch_assignments ‖ Uniform) |
| polarity_weight | 0.1 | MSE(μ(x), μ(-x)) |

### Data Preprocessing
| Parameter | Value | Notes |
|-----------|-------|-------|
| Bandpass filter | 2-20 Hz | Standard for microstates |
| Z-score clip | ±5σ | Train-only statistics |
| Train/eval split | 90/10 | Deterministic seed=42 |
| Polarity augmentation | Enabled | Train only; eval never augmented |

---

## 12. References

1. Jiang, Z., Zheng, Y., Tan, H., Tang, B., & Zhou, H. (2017). Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering. *IJCAI*.

2. Li, X., et al. (2020). S3VDC: Semi-supervised Deep Clustering with Self-supervision. *arXiv:2009.00578*.

3. Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks. *NeuroImage*, 180, 577-593.

4. Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of brain electrical activity into microstates. *IEEE Transactions on Biomedical Engineering*, 42(7), 658-665.

5. Murray, M. M., Brunet, D., & Michel, C. M. (2008). Topographic ERP analyses: A step-by-step tutorial review. *Brain Topography*, 20(4), 249-264.

6. Khanna, A., Pascual-Leone, A., Michel, C. M., & Farzan, F. (2015). Microstates in resting-state EEG. *Neuroscience & Biobehavioral Reviews*, 49, 135-150.

7. Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., & Carin, L. (2019). Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing. *NAACL*.

8. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.

9. Pycrostates: A Python library for EEG microstate analysis. *Journal of Open Source Software*.
