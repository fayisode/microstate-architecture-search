import toml
from pathlib import Path


class Config:
    def __init__(self, target_file):
        self.file = target_file
        self.load_toml_config()

    def load_toml_config(self):
        with open(self.file, "r") as file:
            self.config = toml.load(file)

    def get_model_config(self):
        return self.config.get("vae", {})

    def get_eeg_config(self):
        """Get EEG config with LEMON-specific settings."""
        eeg_config = self.config.get("eeg", {})
        lemon_config = self.config.get("lemon", {})

        # Map LEMON config keys to EEGProcessor expected keys
        if "subject_id" in lemon_config:
            eeg_config["lemon_subject_id"] = lemon_config["subject_id"]
        if "condition" in lemon_config:
            eeg_config["lemon_condition"] = lemon_config["condition"]
        if "sample_freq" in lemon_config:
            eeg_config["sample_freq"] = lemon_config["sample_freq"]

        return eeg_config

    def get_lemon_config(self):
        """Get LEMON-specific configuration."""
        return self.config.get("lemon", {})

    def get_test_info(self):
        return self.config.get("data", {}).get("test_url", None), self.config.get(
            "data", {}
        ).get("test_dir", None)

    def get_train_info(self):
        return self.config.get("data", {}).get("train_url", None), self.config.get(
            "data", {}
        ).get("train_dir", None)

    def get_batch_size(self):
        return self.get_model_config().get("batch_size", 0)

    def get_clustering_config(self):
        """Get clustering configuration (thresholds for centroid merging)."""
        return self.config.get("clustering", {})

    def get_merge_thresholds(self):
        """Get SSIM and correlation thresholds for centroid merging."""
        clustering = self.get_clustering_config()
        return {
            "ssim_threshold": clustering.get("ssim_threshold", 0.95),
            "corr_threshold": clustering.get("corr_threshold", 0.95),
        }


config = Config(str(Path(__file__).parent / "config.toml"))
