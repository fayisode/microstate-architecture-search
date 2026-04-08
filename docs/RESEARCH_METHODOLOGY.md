================================================================================
    EXPLAINABLE EEG MICROSTATE DISCOVERY USING CONVOLUTIONAL VAE-GMM
                    Research Methodology Documentation
================================================================================

Author: Generated from codebase analysis
Date: 2026-02-05
Codebase: /Users/saheedfaremi/PythonProject/microstate_eeg

================================================================================
                        TABLE OF CONTENTS
================================================================================

1. EXECUTIVE SUMMARY
2. ELECTRODE SPACE ANALYSIS (ModKMeans Baseline)
   2.1 Algorithm Overview
   2.2 GFP Peak Extraction
   2.3 Cluster Centroids (Microstates)
   2.4 Distance Metric: Correlation-Based
   2.5 Global Explained Variance (GEV)
   2.6 Clustering Metrics
   2.7 Polarity Invariance Handling
3. LATENT SPACE ANALYSIS (VAE-GMM)
   3.1 VAE Architecture
   3.2 EEG Topomap Encoding Process
   3.3 Latent Space Clustering (GMM)
   3.4 Decoded Output for Fair Comparison
   3.5 Loss Functions
   3.6 Metrics: Latent vs Decoded Space
   3.7 GEV-Based Early Stopping
4. VISUALIZATIONS FOR RESEARCH SUPPORT
   4.1 Microstate Topomap Visualizations
   4.2 Latent Space Visualizations
   4.3 Training Curves and Loss Plots
   4.4 Clustering Quality Plots
   4.5 VAE vs ModKMeans Comparison Plots
   4.6 Polarity and Redundancy Analysis
   4.7 GEV Plots
   4.8 Temporal Dynamics Visualizations
   4.9 Deep Diagnostic Visualizations (Explainability)
5. OUTPUT DIRECTORY STRUCTURE
6. KEY FILES REFERENCE

================================================================================
                        1. EXECUTIVE SUMMARY
================================================================================

This research implements a novel approach to EEG microstate discovery using a
Convolutional Variational Autoencoder with Gaussian Mixture Model clustering
(VAE-GMM). The approach is compared against the traditional Modified K-Means
(ModKMeans) baseline from the pycrostates library.

KEY CONTRIBUTIONS:
- Learned 32-dimensional latent representation of 1600D EEG topomaps
- Probabilistic clustering via Mixture of Gaussians (not hard K-means)
- Dual-space evaluation: latent (32D) and decoded (1600D) metrics
- Polarity-invariant correlation-based distance metrics
- GEV-based early stopping aligned with microstate literature
- Comprehensive visualization suite for explainability

================================================================================
            2. ELECTRODE SPACE ANALYSIS (ModKMeans Baseline)
================================================================================

------------------------------------------------------------------------------
2.1 ALGORITHM OVERVIEW
------------------------------------------------------------------------------

The Modified K-Means (ModKMeans) algorithm operates directly on raw EEG
topographies in electrode space. It is implemented via the pycrostates library.

Configuration (baseline.py, lines 36-40):
    n_clusters: K (typically 4-8)
    n_init: 10 (restarts to find global optimum)
    max_iter: 100 (per restart)
    random_state: 42 (reproducibility)

Input Data:
    - Shape: (n_channels, n_peaks) where n_channels = 32 electrodes
    - Source: GFP peak topographies extracted from continuous EEG
    - Format: pycrostates ChData objects

Output:
    - Cluster centroids (microstates): (K, n_channels)
    - Cluster assignments for each GFP peak
    - Global Explained Variance (GEV)

------------------------------------------------------------------------------
2.2 GFP PEAK EXTRACTION
------------------------------------------------------------------------------

Global Field Power (GFP) represents moments of maximum brain synchronization
where microstate identification is most reliable.

Process (process_eeg_signals.py, lines 345-360):
    1. Compute GFP = std(electrode_values) across all electrodes
    2. Find local maxima in GFP curve (peaks)
    3. Apply minimum peak distance constraint (default: 3 samples)
    4. Reject peaks in artifact-annotated segments
    5. Extract topomap at each peak: (n_channels,)

Output:
    - peak_maps_data: (n_channels, n_peaks)
    - Typically yields thousands to tens of thousands of peaks

WHY GFP PEAKS?
    - High signal-to-noise ratio at peaks
    - Standard practice in microstate literature
    - Reduces computational burden
    - Most stable topographic patterns

------------------------------------------------------------------------------
2.3 CLUSTER CENTROIDS (MICROSTATES)
------------------------------------------------------------------------------

After fitting, ModKMeans produces K centroids representing microstate templates.

Centroid Structure:
    centroids = modk.cluster_centers_  # Shape: (K, 32)
    - K = number of microstates (typically 4)
    - 32 = number of electrode channels

Each centroid represents the SPATIAL TOPOGRAPHY of one microstate - the
characteristic electrode voltage pattern for that brain state.

Centroid Retrieval (baseline.py, lines 584-615):
    {
        "centroids": centroids,          # Raw electrode values (K, 32)
        "n_clusters": K,
        "n_channels": 32,
        "names": ["A", "B", "C", "D"],   # Standard naming
        "gev": 0.68,                      # Explained variance
        "method": "ModKMeans (pycrostates)"
    }

------------------------------------------------------------------------------
2.4 DISTANCE METRIC: CORRELATION-BASED (POLARITY-INVARIANT)
------------------------------------------------------------------------------

KEY INSIGHT: EEG microstates exhibit POLARITY INVARIANCE
    - A topography and its negative (-A) represent the SAME brain state
    - Euclidean distance incorrectly treats them as very different
    - Correlation-based distance naturally handles this

PYCROSTATES DISTANCE FORMULA (metrics_utils.py, lines 39-81):

    distance = |1/correlation| - 1

Example Values:
    correlation = 1.0  --> distance = 0.0   (identical)
    correlation = 0.5  --> distance = 1.0   (moderate)
    correlation = 0.2  --> distance = 4.0   (very different)
    correlation -> 0   --> distance -> inf  (orthogonal)

WHY |1/corr| - 1 INSTEAD OF 1 - |corr|?
    - More aggressive: low correlations produce much higher distances
    - Creates sharper cluster boundaries
    - Matches pycrostates library implementation exactly
    - Better distinguishes similar vs dissimilar topographies

POLARITY ALIGNMENT FOR SKLEARN METRICS (baseline.py, lines 1175-1186):
    1. Compute dot product between sample and assigned centroid
    2. If negative (opposite polarity), flip the sample
    3. Apply sklearn metrics on aligned data

    dot_products = sum(X * centroid_for_each_sample, axis=1)
    signs = sign(dot_products)
    X_aligned = X * signs[:, newaxis]

------------------------------------------------------------------------------
2.5 GLOBAL EXPLAINED VARIANCE (GEV)
------------------------------------------------------------------------------

GEV measures the percentage of global variance explained by cluster assignments.
It is the PRIMARY quality metric for microstate analysis.

FORMULA (baseline.py, lines 96-107):

    GEV = sum(GFP^2 * r^2) / sum(GFP^2)

Where:
    - GFP = spatial standard deviation of each peak (amplitude)
    - r = correlation with assigned centroid (polarity-invariant)
    - Weighting by GFP^2 gives more importance to high-amplitude peaks

INTERPRETATION:
    - GEV in [0, 1] - proportion of variance captured
    - Higher GEV = better microstate model
    - Typical values: 0.60 - 0.80 for K=4
    - Used for early stopping in VAE training

------------------------------------------------------------------------------
2.6 CLUSTERING METRICS
------------------------------------------------------------------------------

Three metric families are computed (baseline.py, lines 1155-1206):

FAMILY 1: SKLEARN RAW (No Polarity Alignment)
    sil_score_raw = sklearn_silhouette(X, labels)
    ch_score_raw = sklearn_ch(X, labels)
    db_score_raw = sklearn_db(X, labels)
    dunn_raw = compute_dunn_index(X, labels)

    Problem: Uses Euclidean distance, doesn't account for polarity.

FAMILY 2: SKLEARN ALIGNED (Polarity-Corrected)
    X_aligned = X * signs[:, newaxis]  # Flip to match centroids
    sil_score = sklearn_silhouette(X_aligned, labels)
    ch_score = sklearn_ch(X_aligned, labels)
    db_score = sklearn_db(X_aligned, labels)
    dunn = compute_dunn_index(X_aligned, labels)

    Better: Polarity alignment before Euclidean metrics.

FAMILY 3: PYCROSTATES-STYLE (Correlation-Based Distance)
    sil_score_corr = pycrostates_silhouette_score(X, labels)
    ch_score_corr = pycrostates_calinski_harabasz_score(X, labels, centroids)
    db_score_corr = pycrostates_davies_bouldin_score(X, labels, centroids)
    dunn_corr = pycrostates_dunn_score(X, labels)

    BEST: Uses |1/corr| - 1 distance, natively polarity-invariant.
    RECOMMENDED for microstate analysis.

METRIC DEFINITIONS:

Silhouette Score:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where: a(i) = within-cluster distance
           b(i) = nearest other cluster distance
    Range: [-1, 1], higher is better

Davies-Bouldin Index:
    DB = (1/K) * sum_i max_{j!=i}[(S_i + S_j) / d(C_i, C_j)]
    where: S_i = mean intra-cluster distance
           d(C_i, C_j) = centroid distance
    Range: [0, inf), LOWER is better

Calinski-Harabasz Index:
    CH = [B / (K-1)] / [W / (N-K)]
    where: B = between-cluster dispersion
           W = within-cluster dispersion
    Range: [0, inf), higher is better

Dunn Index:
    Dunn = min(inter-cluster distance) / max(intra-cluster diameter)
    Range: [0, inf), higher is better

------------------------------------------------------------------------------
2.7 POLARITY INVARIANCE HANDLING
------------------------------------------------------------------------------

PROBLEM:
    EEG microstate A and -A (inverted polarity) are the SAME brain state.
    Standard Euclidean distance treats them as maximally different.

SOLUTIONS IMPLEMENTED:

1. Label Assignment (correlation-based):
    correlations = abs(X @ centroids.T)
    labels = argmax(correlations, axis=1)

2. Polarity Alignment (for Euclidean metrics):
    signs = sign(dot(X, centroids[labels]))
    X_aligned = X * signs

3. Correlation Distance (metrics_utils.py):
    distance = |1/correlation| - 1
    Uses absolute correlation -> natively polarity-invariant

================================================================================
                3. LATENT SPACE ANALYSIS (VAE-GMM)
================================================================================

------------------------------------------------------------------------------
3.1 VAE ARCHITECTURE
------------------------------------------------------------------------------

ENCODER (model.py, lines 33-89):

Input: Grayscale EEG topomaps (1, 40, 40) = 1600D

Architecture:
    Conv2d layers with LeakyReLU + BatchNorm
    Progressive downsampling
    Adaptive average pooling to (1, 1)
    Fully connected: 1024 -> 512 -> latent_dim

Output:
    mu: (batch, latent_dim) - mean of posterior
    log_var: (batch, latent_dim) - log-variance of posterior

DECODER (model.py, lines 92-143):

Input: Sampled latent vector (batch, latent_dim)

Architecture:
    Linear expansion: latent_dim -> 512 -> 1024
    Reshape to (1024, 1, 1)
    ConvTranspose2d layers with progressive upsampling
    BatchNorm + LeakyReLU activations
    Sigmoid output (constrain to [0,1])

Output: Reconstructed topomap (1, 40, 40)

LATENT DIMENSION: 32 (configurable)
    - Much lower than input space (1600D)
    - Enables efficient clustering
    - Balances compression vs information preservation

------------------------------------------------------------------------------
3.2 EEG TOPOMAP ENCODING PROCESS
------------------------------------------------------------------------------

Data Flow:
    1. Input: 1600D topographic image (40x40 grayscale)
    2. Convolutional feature extraction
    3. Progressive spatial downsampling
    4. Global pooling -> 1024D features
    5. Bottleneck: 1024D -> 512D
    6. Final projection: 512D -> 32D (mu and log_var)

REPARAMETERIZATION TRICK (model.py, lines 695-705):
    z = mu + sigma * epsilon
    where epsilon ~ N(0, 1)

    Enables backpropagation through stochastic sampling.

KEY PROPERTY:
    Each point in 32D latent space encodes a complete EEG microstate
    as a compressed feature vector.

------------------------------------------------------------------------------
3.3 LATENT SPACE CLUSTERING (GMM)
------------------------------------------------------------------------------

CLUSTERING MECHANISM: Mixture of Gaussians (NOT traditional K-Means)

Parameters (model.py, lines 319-329):
    mu_c: Cluster means (K, 32) - initialized from KMeans on data
    log_var_c: Cluster variances (K, 32) - learned during training
    pi: Mixture weights (K,) - representing P(cluster k)

CLUSTER ASSIGNMENT (model.py, lines 816-819):
    1. Encode input to latent: z = mu
    2. Compute log-likelihood under each Gaussian:
       log p(z|c) = -0.5 * (log(var_c) + (z - mu_c)^2 / var_c)
    3. Compute posterior with mixture weight:
       log p(c|z) proportional to log(pi_c) + log(p(z|c))
    4. Assign to cluster with highest posterior: argmax_c p(c|z)

GMM ADVANTAGES OVER K-MEANS:
    - Learns cluster variance (uncertainty in cluster definition)
    - Soft posterior probabilities (not hard assignments)
    - More principled probabilistic framework
    - Better uncertainty quantification

INITIALIZATION FROM DATA (model.py, lines 534-614):
    1. Run KMeans on latent vectors from first epoch
    2. Initialize mu_c with KMeans centroids
    3. Initialize log_var_c from cluster-specific variances
    4. pi initialized uniformly
    -> Prevents poor initialization and dead clusters

------------------------------------------------------------------------------
3.4 DECODED OUTPUT FOR FAIR COMPARISON
------------------------------------------------------------------------------

DECODING PROCESS:
    Cluster centroid: z_c = mu_c (latent center of cluster c)
    Decode: x_c = decoder(z_c) -> (40, 40) topomap

TWO METRIC SPACES (clustering_trainer.py, lines 512-575):

PRIMARY METRICS (Decoded Space - 1600D):
    - Features: Full decoded reconstructions
    - Purpose: Fair comparison with baseline ModKMeans
    - Keys: primary_silhouette, primary_db, primary_ch
    - WHY: ModKMeans operates on 1600D electrode space

SECONDARY METRICS (Latent Space - 32D):
    - Features: Raw latent vectors
    - Purpose: Evaluate VAE's internal representation quality
    - Keys: secondary_silhouette, secondary_db, secondary_ch
    - WHY: Direct measure of clustering in learned representation

PAIRWISE DECODED METRICS (lines 551-562):
    - Correlation-based distance on decoded 1600D space
    - Most polarity-invariant fair comparison with ModKMeans
    - Keys: pairwise_decoded_silhouette, pairwise_decoded_db, etc.
    - RECOMMENDED for microstate analysis

------------------------------------------------------------------------------
3.5 LOSS FUNCTIONS
------------------------------------------------------------------------------

RECONSTRUCTION LOSS (model.py, lines 827-846):
    L_recon = MSE(x_reconstructed, x_input)
    Scaled by flat_size for proper gradient magnitude.

KL DIVERGENCE (Mixture-weighted) (model.py, lines 848-879):
    KL = E_q[log q(z|x)] - E_q[log p(z|c)] - H(q)
    Weighted mixture approach prevents posterior collapse.

CLUSTER ENTROPY REGULARIZATION (model.py, lines 365-378):
    L_entropy = -sum(pi_k * log(pi_k))
    Weight: 0.3
    Encourages uniform utilization of all K clusters.

CLUSTER SEPARATION LOSS (model.py, lines 648-693):
    L_sep = ReLU(threshold - min_distance) + 0.5 * ReLU(threshold - avg_distance)
    Weight: 0.2
    Prevents cluster centroids from collapsing together.

COMBINED LOSS (model.py, lines 907-975):
    L_total = L_recon + beta * L_KL + 0.3 * L_entropy + 0.2 * L_sep

BETA (KL Weight) ANNEALING:
    - Starts low (0.1) for reconstruction focus
    - Increases to max (2.0) over warmup epochs
    - Prevents posterior collapse
    - Adaptive weighting based on loss balance

------------------------------------------------------------------------------
3.6 METRICS: LATENT VS DECODED SPACE
------------------------------------------------------------------------------

DISTANCE FORMULA (metrics_utils.py, lines 39-81):
    distance = |1/correlation| - 1
    Same pycrostates formula used for both spaces.

SILHOUETTE SCORE VARIANTS:

1. PRIMARY (Decoded) - For Baseline Comparison:
   - Computed on 1600D decoded space
   - Directly comparable to ModKMeans
   - Euclidean distance metric

2. SECONDARY (Latent) - VAE Representation Quality:
   - Same formula on 32D latent space
   - Shows how well VAE's compressed representation clusters
   - Expected to be lower (compression reduces separation)

3. PAIRWISE-LATENT:
   - Correlation-based distance on latent space
   - May show poor results (correlation distance designed for high-dim)

4. PAIRWISE-DECODED:
   - Correlation-based distance on decoded 1600D space
   - Most polarity-invariant fair comparison with ModKMeans
   - RECOMMENDED for microstate analysis

GEV COMPUTATION (clustering_trainer.py, lines 920-987):
    GEV = sum[(GFP_i * |correlation_i|)^2] / sum[GFP_i^2]
    - GFP_i = spatial standard deviation of sample i
    - correlation_i = with assigned cluster centroid
    - Higher GEV = better fit
    - Standard microstate metric

------------------------------------------------------------------------------
3.7 GEV-BASED EARLY STOPPING
------------------------------------------------------------------------------

TRADITIONAL APPROACH: Stop when validation loss plateaus.

VAE APPROACH (clustering_trainer.py, lines 1272-1293):
    - Metric: Global Explained Variance (GEV) - higher is better
    - Default Patience: 40 epochs
    - Save Condition: if current_gev > best_gev
    - Stop Condition: No GEV improvement for 40 consecutive epochs

WHY GEV INSTEAD OF LOSS?
    - Loss = reconstruction + KL balancing (beta varies)
    - GEV = direct microstate quality metric
    - More interpretable for EEG analysis
    - Aligns with baseline ModKMeans evaluation

COMPOSITE SCORE (logged but not used for stopping):
    composite = sqrt(silhouette_norm * GEV)
    Balances cluster quality with variance explanation.

================================================================================
            4. VISUALIZATIONS FOR RESEARCH SUPPORT
================================================================================

------------------------------------------------------------------------------
4.1 MICROSTATE TOPOMAP VISUALIZATIONS
------------------------------------------------------------------------------

VAE MICROSTATE TOPOMAPS
    File: evaluate_polarity_merge.py - plot_vae_topomaps()
    Output: vae_microstate_topomaps.png
    Description: MNE-based electrode topomaps showing VAE-learned microstate
                 patterns, color-coded with RdBu_r colormap
    Research Value: Visualizes learned microstate templates comparable to
                    classical EEG microstate literature

VAE VS MODKMEANS TOPOMAPS (Critical Comparison Figure)
    File: evaluate_polarity_merge.py - plot_vae_vs_modkmeans_topomaps()
    Output: vae_vs_modkmeans_topomaps.png
    Description: Side-by-side comparison with automatic centroid matching
                 via correlation (handles polarity invariance)
    Research Value: Direct comparison demonstrates VAE clustering quality
                    against established baseline method

MERGED VAE TOPOMAPS
    Output: vae_merged_topomaps.png
    Description: Polarity-merged microstate topomaps after redundancy analysis
    Research Value: Shows microstate consolidation after handling polarity

ORIGINAL VS MERGED TOPOMAPS
    Output: vae_original_vs_merged_topomaps.png
    Description: Before/after comparison of polarity merging
    Research Value: Validates redundancy elimination; supports K selection

FINAL CLUSTER TOPOMAPS (Publication-Ready)
    Output: final_cluster_topomaps/ directory
    Description: MNE-rendered circular head outlines, nose, ear indicators,
                 microstate labels (A, B, C, D...)
    Research Value: Camera-ready figures for paper submission

MODKMEANS BASELINE TOPOMAPS
    Output: microstate_topomaps_named.png, baseline_merged_centroids.png
    Description: ModKMeans centroid visualization with GEV overlay
    Research Value: Baseline comparison point

------------------------------------------------------------------------------
4.2 LATENT SPACE VISUALIZATIONS
------------------------------------------------------------------------------

COMPREHENSIVE LATENT SPACE ANALYSIS (6 visualizations)
    Directory: latent_space_analysis/

    1. {prefix}_tsne_clusters_density.png
       - t-SNE scatter with cluster assignments + density contours (KDE-based)
       - Research Value: Shows cluster separability in latent space

    2. {prefix}_tsne_recon_error.png
       - t-SNE colored by reconstruction error + box plots per cluster
       - Research Value: Identifies hard-to-reconstruct samples

    3. {prefix}_cluster_distances.png
       - Inter-cluster distance heatmap + cluster size bar chart
       - Research Value: Shows cluster center separation and imbalance

    4. {prefix}_latent_distributions.png
       - Histograms of latent dimensions per cluster (2x4 grid)
       - Research Value: Shows how each latent dimension specializes

    5. {prefix}_comprehensive_dashboard.png
       - Multi-panel dashboard combining all above
       - Research Value: One-page summary of latent space organization

    6. {prefix}_cluster_quality_metrics.png
       - Global metrics + per-cluster quality
       - Research Value: Quantitative validation of clustering

t-SNE WITH STRATEGY COMPARISON
    Output: latent_space_{strategy_key}_k{K}.png
    Description: t-SNE scatter showing effect of polarity merging
    Research Value: Visualizes merge strategy impact

BEST MODEL VISUALIZATIONS (Per Epoch)
    Directory: best_model_visualizations/
    Output: best_{strategy_key}_epoch_{epoch}_k{K}.png
    Description: t-SNE visualizations whenever GEV improves
    Research Value: Tracks latent space evolution during training

LATENT SPACE EXPLAINABILITY
    Output: explainability/latent_explainability_topomaps.png
    Description: 8xK grid showing which topographic patterns each
                 latent dimension learns
    Research Value: Demonstrates interpretability of VAE representations

------------------------------------------------------------------------------
4.3 TRAINING CURVES AND LOSS PLOTS
------------------------------------------------------------------------------

COMPREHENSIVE TRAINING HISTORY (6-panel figure)
    File: model.py - plot_training_history()
    Output: comprehensive_training_history.png
    Panels:
        1. Total Loss (log scale, red)
        2. Reconstruction Loss (log scale, blue)
        3. KLD Loss (log scale, green)
        4. Beta Annealing Schedule (orange)
        5. Clustering Metrics: NMI, ARI, V-Measure
        6. Internal Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
    Research Value: Shows VAE loss convergence and KLD-reconstruction balance

TRAINING CURVES
    Output: training_curves.png
    Description: Loss curves over epochs with dual y-axes
    Research Value: Shows final training convergence

EPOCH METRICS JSON LOG
    Output: epoch_metrics_log.json
    Description: Cumulative log of all epoch metrics
    Research Value: Raw data for custom post-hoc analysis

------------------------------------------------------------------------------
4.4 CLUSTERING QUALITY PLOTS
------------------------------------------------------------------------------

CLUSTER QUALITY METRICS SUMMARY
    Output: {prefix}_cluster_quality_metrics.png
    Subpanels:
        - Bar chart: Silhouette, Davies-Bouldin, Calinski-Harabasz
        - Per-cluster reconstruction error with error bars
        - Per-cluster latent variance (compactness metric)
    Research Value: Quantifies clustering quality

CLUSTER CORRELATION MATRIX
    Output: centroid_correlation_matrix.png (full), absolute_correlation_matrix.png
    Description: Heatmaps showing spatial correlation between centroids
    Research Value: Identifies polarity-inverted and similar pairs

CENTROID SUMMARY VISUALIZATION
    Output: centroid_summary.png
    Layout: Channel activity bar plots + full correlation matrix
    Research Value: Combined view of microstate structure

------------------------------------------------------------------------------
4.5 VAE VS MODKMEANS COMPARISON PLOTS
------------------------------------------------------------------------------

TOPOMAPS COMPARISON (Most Critical)
    Output: vae_vs_modkmeans_topomaps.png
    Description: Side-by-side matched topomaps
    Research Value: Validates VAE against established method

STRATEGY COMPARISON
    Output: strategy_comparison.png
    Description: 4 merge strategies compared on clustering quality
    Research Value: Validates optimal polarity merging strategy

CROSS-K SUMMARY
    Output: {participant_id}/cross_k_summary.png
    Description: Compares different K values (4,5,6,7,8)
    Research Value: Shows optimal number of clusters

FAIR COMPARISON ANALYSIS
    Directory: fair_comparison/
    Description: Detailed metrics comparison VAE vs ModKMeans
    Research Value: Demonstrates VAE competitiveness or superiority

------------------------------------------------------------------------------
4.6 POLARITY AND REDUNDANCY ANALYSIS
------------------------------------------------------------------------------

POLARITY-MERGED TOPOMAPS
    Output: vae_merged_topomaps.png
    Research Value: Visualizes biological redundancy in microstate maps

POLARITY MERGE EVALUATION REPORT
    Output: Multiple strategy comparison images
    Description: Impact of each merge strategy on metrics
    Research Value: Scientific justification for microstate number

CENTROID INDIVIDUAL PLOTS
    Output: centroids_dir/centroid_{k+1}.png
    Description: Single high-resolution topomap per microstate
    Research Value: Inspection-quality visualizations for paper figures

------------------------------------------------------------------------------
4.7 GEV (GLOBAL EXPLAINED VARIANCE) PLOTS
------------------------------------------------------------------------------

GEV TRACKING
    Logged in: epoch_metrics_log.json
    Visualized in: Training curves and epoch summaries
    Research Value: Shows variance explained evolution

GEV IN COMPARISON REPORTS
    Description: All microstate topomaps show GEV in title (e.g., "GEV = 67.8%")
    Research Value: Standard EEG microstate reporting metric

------------------------------------------------------------------------------
4.8 TEMPORAL DYNAMICS VISUALIZATIONS
------------------------------------------------------------------------------

MICROSTATE SEGMENTATION PLOT
    File: model.py - plot_segmentation()
    Output: final_cluster_topomaps/vae_segmentation.png
    Layout:
        - Top: Global Field Potential with colored background by microstate
        - Bottom: Step plot of microstate labels over time
    Research Value: Shows temporal microstate sequences

MICROSTATE STATISTICS
    File: model.py - plot_microstate_statistics()
    Output: final_cluster_topomaps/vae_microstate_statistics.png
    3-panel bar chart:
        1. Mean Duration (ms) with std error bars
        2. Time Coverage (%)
        3. Occurrence Rate (per second)
    Also saves: vae_microstate_stats_peaks.json
    Research Value: Temporal properties comparable to ModKMeans

------------------------------------------------------------------------------
4.9 DEEP DIAGNOSTIC VISUALIZATIONS (Explainability)
------------------------------------------------------------------------------

*** CRITICAL FOR VAE-GMM PAPER ***

LATENT DIMENSION INDEPENDENCE HEATMAP
    Output: deep_diagnostics/Latent_Independence_Heatmap.png
    Description: Correlation matrix of all 32 latent dimensions
    Ideal: Zero correlations (white/grey heatmap)
    Research Value: PROVES latent dimensions are disentangled
                    Shows learned representations are independent

CLUSTER LATENT FINGERPRINTS (Radar Chart)
    Output: deep_diagnostics/Cluster_Latent_Fingerprints.png
    Description: Polar plot where each cluster is a line showing
                 mean values across all 32 latent dimensions
    Research Value: Shows how each microstate is "encoded" in latent space
                    Different fingerprints prove distinct representations

LATENT MANIFOLD TRAVERSALS GRID
    Output: deep_diagnostics/Latent_Traversals_Grid.png
    Layout: 32x7 grid (32 dimensions x 7 traversal steps)
    Description:
        - Each row: one latent dimension varied from -3sigma to +3sigma
        - Each column: decoded 40x40 topomap showing the effect
        - Uses RdBu_r colormap to show polarity shifts
    Research Value: *** MANIFOLD PROOF ***
                    - Demonstrates VAE learned smooth latent space
                    - Each dimension produces coherent EEG patterns
                    - Validates convolutional decoder learns topography

================================================================================
                    5. OUTPUT DIRECTORY STRUCTURE
================================================================================

outputs/
|-- best_model.pth
|-- checkpoint.pth
|-- training_curves.png
|-- comprehensive_training_history.png
|-- epoch_metrics_log.json
|-- best_model_metrics.json
|-- cluster_diagnosis.json
|-- unified_microstate_stats.json
|
|-- latent_space_analysis/
|   |-- final_tsne_clusters_density.png
|   |-- final_tsne_recon_error.png
|   |-- final_cluster_distances.png
|   |-- final_latent_distributions.png
|   |-- final_comprehensive_dashboard.png
|   |-- final_cluster_quality_metrics.png
|   |-- final_latent_analysis_stats.json
|
|-- best_model_visualizations/
|   |-- best_beta_epoch_*.png
|   |-- best_composite_epoch_*.png
|   |-- latest_best_*.png
|
|-- deep_diagnostics/
|   |-- Latent_Independence_Heatmap.png
|   |-- Cluster_Latent_Fingerprints.png
|   |-- Latent_Traversals_Grid.png
|
|-- explainability/
|   |-- latent_explainability_topomaps.png
|
|-- final_cluster_topomaps/
|   |-- vae_microstate_topomaps.png
|   |-- vae_segmentation.png
|   |-- vae_microstate_statistics.png
|   |-- vae_microstate_stats_peaks.json
|
|-- fair_comparison/
|   |-- (VAE vs ModKMeans detailed metrics)
|
|-- publication_ready/
|   |-- (Optimized publication-quality figures)
|
|-- baseline_modkmeans/
    |-- microstate_topomaps_named.png
    |-- baseline_merged_centroids.png
    |-- centroid_topomaps_mne.png
    |-- centroid_correlation_matrix.png
    |-- centroid_summary.png
    |-- cluster_validation_metrics.json

================================================================================
                        6. KEY FILES REFERENCE
================================================================================

ELECTRODE SPACE (ModKMeans Baseline):
    baseline.py              - Main ModKMeans implementation
    metrics_utils.py         - Correlation-based metrics (pycrostates formula)
    centroid_metrics.py      - Centroid-based metric definitions
    process_eeg_signals.py   - GFP peak extraction

LATENT SPACE (VAE-GMM):
    model.py                 - VAE architecture, GMM clustering, loss functions
    clustering_trainer.py    - Training loop, metric computation, visualizations
    centroid_analysis.py     - Post-hoc redundancy analysis

EVALUATION AND COMPARISON:
    evaluate_polarity_merge.py - Polarity analysis, three-column comparison
    test_pipeline.py         - Pipeline testing

CONFIGURATION:
    config/config.py         - Training hyperparameters
    config.toml              - Model configuration

================================================================================

SUMMARY TABLE: METRIC FAMILIES COMPARISON

| Aspect        | Sklearn Raw   | Sklearn Aligned | Pycrostates     |
|---------------|---------------|-----------------|-----------------|
| Distance      | Euclidean     | Euclidean       | Correlation     |
| Polarity      | None          | Aligned first   | Native          |
| Formula       | sklearn       | sklearn + flip  | |1/corr| - 1    |
| Best for      | Baseline      | Fairness check  | EEG data        |
| Recommended   | No            | Yes             | ***YES***       |

================================================================================

Document generated for research paper:
"Explainable EEG Microstate Discovery Using Convolutional VAE-GMM"

================================================================================
