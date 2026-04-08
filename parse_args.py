import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE Clustering Training")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["anomaly", "house", "eeg"],
        default="eeg",
        help="Dataset to use",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--latent_dim", type=int, help="Latent dimension (overrides config)"
    )
    parser.add_argument(
        "--n_clusters", type=int, help="Number of clusters (overrides config)"
    )
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=20, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="LEMON subject ID (e.g., 010004). Reads from config.toml if not specified.",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="A stable, unique ID for the training session to enable resumption. If not provided, a timestamped directory will be created.",
    )

    # Architecture parameters
    parser.add_argument(
        "--ndf", type=int, default=None,
        help="Channel width multiplier for encoder/decoder (e.g., 32, 64, 128). Overrides config.",
    )
    parser.add_argument(
        "--n_conv_layers", type=int, default=None,
        help="Number of conv layers in encoder/decoder (2-5). Overrides config.",
    )

    # Sweep mode
    parser.add_argument(
        "--fast-sweep",
        action="store_true",
        help="Tier 1 fast mode: reduced epochs, skip all visualization. Only saves metrics files.",
    )

    # Two-phase execution flags
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run EEG preprocessing (Phase 1). Saves artifacts to cache for parallel training.",
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Load preprocessed data from cache (Phase 2). Skips EEG processing and visualization.",
    )

    return parser.parse_args()
