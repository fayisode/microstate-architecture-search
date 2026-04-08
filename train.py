import logging
import os
import sys
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml

# Local imports
import helper_function as _g
import process_eeg_signals as _eeg
import seeding as _s
import parse_args as _pa
import train_cluster as _tc
from config.config import config as c

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_gpu_slot_assignments(reserve_gb=4.0, per_job_gb=1.5):
    """Build GPU assignment list based on available memory per GPU.

    Each GPU gets floor((free - reserve) / per_job) slots. Returns a list of
    physical GPU IDs with repeats, e.g. [0,0,0, 2,2, 3,3,3,3] means GPU 0
    gets 3 jobs, GPU 2 gets 2, GPU 3 gets 4. Workers beyond this list use CPU.
    """
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding='utf-8', timeout=10
        )
        if result.returncode != 0:
            return []

        reserve_mb = reserve_gb * 1024
        per_job_mb = per_job_gb * 1024
        assignments = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            idx, free_mb = line.split(',')
            idx = int(idx.strip())
            free = float(free_mb.strip())
            available = free - reserve_mb
            if available <= per_job_mb:
                continue
            slots = int(available / per_job_mb)
            assignments.extend([idx] * slots)
            print(f"  GPU {idx}: {slots} slots ({free:.0f}MB free, {available:.0f}MB usable)")
        print(f"  Total GPU slots: {len(assignments)} | Overflow -> CPU")
        return assignments

    except Exception as e:
        print(f"GPU slot detection failed: {e}")
        return []

# GPU selection: controlled by CUDA_VISIBLE_DEVICES environment variable
# Set externally (e.g., run_cpu_sweep.sh sets per-process)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("CUDA_VISIBLE_DEVICES not set — using all visible GPUs")
elif os.environ["CUDA_VISIBLE_DEVICES"].strip() == "":
    print("CUDA_VISIBLE_DEVICES is empty — CPU-only mode")
else:
    print("Using GPU(s): CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger("vae_clustering")


# NoDaemonProcess, NoDaemonContext, MyPool are defined lazily inside
# _train_parallel() to avoid allocating multiprocessing resources on the
# sequential code path (used by run_parallel.sh).


# Check CUDA availability (no allocation at import time)
if not torch.cuda.is_available():
    logger.warning("CUDA is not available - PyTorch will use CPU only")


# Worker function that creates DataLoaders locally
def _train_worker_function(
    task_id,
    n_clusters,
    batch_size,
    latent_dim,
    config,
    train_dataset,
    n_channels,
    available_gpu_ids,
    num_gpus,
    gfp_peaks=None,
    raw_mne=None,
    eval_set=None,
    train_set=None,
    norm_params=None,
    ndf=64,
    n_conv_layers=4,
    fast_sweep=False,
):
    """Worker function that creates DataLoaders locally to avoid serialization issues.

    Args:
        train_dataset: Augmented TensorDataset for training DataLoader (includes sign-flipped copies).
        train_set: Unaugmented Subset for GMM initialization (clean data without polarity duplicates).
        ndf: Channel width multiplier for encoder/decoder.
        n_conv_layers: Number of conv layers in encoder/decoder.
        fast_sweep: If True, skip visualization pipeline.
    """
    try:
        # Setup worker environment — pin to a physical GPU or fall back to CPU.
        # available_gpu_ids is a slot list (e.g. [0,0,0, 2,2, 3,3,3,3]);
        # tasks beyond the list length use CPU.
        if num_gpus > 0 and task_id < num_gpus:
            gpu_id = available_gpu_ids[task_id]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = torch.device("cuda:0")
            logger.info(f"Task {task_id} pinned to physical GPU {gpu_id} (cuda:0)")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = torch.device("cpu")
            logger.info(f"Task {task_id} using CPU (GPU slots exhausted)")

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create DataLoaders in worker process
        from torch.utils.data import DataLoader

        logger.info(
            f"Task {task_id}: Creating DataLoaders with batch_size={batch_size}"
        )
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": True if torch.cuda.is_available() else False,
            "drop_last": True,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)

        # Create eval DataLoader if eval_set was provided
        eval_loader = None
        if eval_set is not None:
            eval_loader = DataLoader(
                eval_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False,
            )
            logger.info(f"Task {task_id}: Eval DataLoader created ({len(eval_set)} samples)")

        logger.info(f"Task {task_id}: Train DataLoader created successfully")

        # Validate DataLoader works
        try:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            logger.info(
                f"Task {task_id}: train loader validated - batch shape {data.shape}"
            )
        except Exception as e:
            logger.error(f"Task {task_id}: DataLoader validation failed: {e}")
            raise

        # Run training with locally created DataLoaders
        # train_set (unaugmented) is used for GMM init; train_loader uses augmented data
        result = _tc.train_cluster(
            n_clusters=n_clusters,
            config_dict=config,
            train_loader=train_loader,
            train_set=train_set if train_set is not None else train_dataset,
            device=device,
            batch_size=batch_size,
            latent_dim=latent_dim,
            n_channels=n_channels,
            logger=logger,
            gfp_peaks=gfp_peaks,
            raw_mne=raw_mne,
            eval_loader=eval_loader,
            eval_set=eval_set,
            norm_params=norm_params,
            ndf=ndf,
            n_conv_layers=n_conv_layers,
            fast_sweep=fast_sweep,
        )

        logger.info(f"Task {task_id}: Training completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in worker {task_id} (n_clusters={n_clusters}): {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Cleanup on error
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

        # Return a minimal result structure to avoid breaking the main process
        return {
            "n_clusters": n_clusters,
            "best_train_loss": float("inf"),
            "loss_history": None,
            "error": str(e),
            "success": False,
        }


class ConfigManager:
    """Handles configuration loading and updates"""

    def __init__(self, args):
        self.args = args
        self.config = c.get_model_config()
        self._update_config_from_args()

    def _update_config_from_args(self):
        """Update config with command-line arguments"""
        updates = {
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "latent_dim": self.args.latent_dim,
            "n_clusters": self.args.n_clusters,
            "learning_rate": self.args.lr,
            "output_dir": self.args.output_dir,
            "ndf": getattr(self.args, 'ndf', None),
            "n_conv_layers": getattr(self.args, 'n_conv_layers', None),
        }
        for key, value in updates.items():
            if value is not None:
                self.config[key] = value

        # Fast sweep mode: flag only, all params from config.toml
        if getattr(self.args, 'fast_sweep', False):
            self.config["fast_sweep"] = True

    def get_config(self) -> Dict[str, Any]:
        return self.config


class DirectoryManager:
    """Manages output directories and paths"""

    def __init__(self, base_dir: str, participant_id: str = None, run_id: str = None):
        self.base_output_dir = Path(base_dir)
        self.participant_id = participant_id

        if run_id:
            run_name = run_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        if participant_id:
            self.run_dir = self.base_output_dir / participant_id / run_name
        else:
            self.run_dir = self.base_output_dir / run_name

    def setup_directories(self) -> Path:
        """Create necessary directories"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def get_comparison_dir(self) -> Path:
        comparison_dir = self.run_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        return comparison_dir


class DataLoaderFactory:
    """Factory for creating data loaders (90/10 train/eval split)"""

    @staticmethod
    def get_data_loaders(
        args, config: Dict[str, Any], participant_id: str = None
    ) -> Tuple:
        """Load data and return train/eval loaders with 90/10 split.

        Returns:
            Tuple of (n_channels, train_loader, train_set, gfp_peaks, raw_mne,
                       eval_loader, eval_set, norm_params)
        """
        if args.data == "eeg":
            processor = _eeg.EEGProcessor(c.get_eeg_config(), logger, participant_id)

            # Two-phase execution support
            use_cached = getattr(args, 'use_cached', False)
            preprocess_only = getattr(args, 'preprocess_only', False)

            # Phase 1: Preprocess and save cache
            if preprocess_only:
                datasets = processor.process(
                    use_cached=False,
                    save_cache=True,
                    generate_figures=True
                )
            # Phase 2: Load from cache
            elif use_cached:
                fast_sweep = getattr(args, 'fast_sweep', False)
                if fast_sweep:
                    # Fast sweep: numpy-only loader, no MNE/pycrostates needed
                    datasets = processor.load_from_cache_fast()
                else:
                    datasets = processor.process(
                        use_cached=True,
                        save_cache=False,
                        generate_figures=False
                    )
            # Standard: Full processing without caching
            else:
                datasets = processor.process()

            return (
                1,
                datasets.train,
                datasets.train_set,
                datasets.gfp_peaks,
                datasets.raw_mne,
                datasets.eval,
                datasets.eval_set,
                datasets.norm_params,
            )


class TrainingOrchestrator:
    """Manages the training process"""

    def __init__(
        self, config: Dict[str, Any], run_dir: Path, args, participant_id: str = None
    ):
        self.config = config
        self.run_dir = run_dir
        self.args = args
        self.participant_id = participant_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_gpu_utilization(self):
        """Get GPU utilization information using nvidia-smi"""
        try:
            import subprocess

            command = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            output = subprocess.check_output(command.split()).decode('utf-8')

            gpu_info = []
            for line in output.strip().split("\n"):
                values = [float(x) for x in line.split(", ")]
                gpu_id, gpu_util, mem_used, mem_total = values
                gpu_info.append(
                    {
                        "id": int(gpu_id),
                        "utilization": gpu_util,
                        "memory_used": mem_used,
                        "memory_total": mem_total,
                        "memory_used_percent": (mem_used / mem_total) * 100,
                    }
                )
            return gpu_info
        except Exception as e:
            logger.warning(f"Error getting GPU utilization: {str(e)}")
            return []

    def _get_param_ranges(self) -> List[Tuple[int, int, int, int, int]]:
        """Determine ranges for n_clusters, batch_size, latent_dim, n_conv_layers, ndf."""
        # Use specific n_clusters if provided via command line, otherwise default range
        if self.args.n_clusters is not None:
            cluster_range = [self.args.n_clusters]
        else:
            cluster_range = list(range(3, 21))  # 3 to 20 clusters
        batch_size_range = [128]
        latent_dim_range = [self.args.latent_dim] if self.args.latent_dim is not None else [32]
        n_conv_layers_range = [getattr(self.args, 'n_conv_layers', None) or self.config.get("n_conv_layers", 4)]
        ndf_range = [getattr(self.args, 'ndf', None) or self.config.get("ndf", 64)]
        param_combinations = list(
            product(cluster_range, batch_size_range, latent_dim_range, n_conv_layers_range, ndf_range)
        )

        participant_info = (
            f" for participant {self.participant_id}" if self.participant_id else ""
        )
        logger.info(
            f"Training over {len(param_combinations)} combinations{participant_info}: {param_combinations}"
        )
        return param_combinations

    def train(
        self, train_loader, train_set, n_channels: int,
        gfp_peaks=None, raw_mne=None,
        eval_loader=None, eval_set=None,
        norm_params=None,
    ) -> List[Dict]:
        """Execute training process with 90/10 train/eval split."""
        param_combinations = self._get_param_ranges()

        if len(param_combinations) > 1 and torch.cuda.device_count() > 1:
            results = self._train_parallel(
                param_combinations,
                train_loader,
                train_set,
                n_channels,
                gfp_peaks,
                raw_mne,
                eval_set,
                norm_params,
            )
            # Run baseline analysis after parallel VAE training (uses ALL data)
            if gfp_peaks is not None:
                self._run_baseline_for_all_clusters(
                    param_combinations, gfp_peaks, raw_mne
                )
            return results
        return self._train_sequential(
            param_combinations,
            train_loader,
            train_set,
            n_channels,
            gfp_peaks,
            raw_mne,
            eval_loader,
            eval_set,
            norm_params,
        )

    def _train_parallel(self, param_combinations, *args) -> List[Dict]:
        """Execute parallel training with capacity-proportional GPU assignment.

        Each GPU gets slots proportional to its free memory minus a safety reserve.
        Tasks beyond GPU capacity fall back to CPU.
        """
        if torch.cuda.is_available():
            # Capacity-proportional allocation: [gpu_id, gpu_id, ...] with repeats
            available_gpu_ids = get_gpu_slot_assignments(reserve_gb=4.0, per_job_gb=1.5)
            num_gpus = len(available_gpu_ids)  # Total slots, not unique GPUs
            if num_gpus > 0:
                unique_gpus = sorted(set(available_gpu_ids))
                logger.info(f"GPU slots: {num_gpus} across physical GPUs {unique_gpus}")
            else:
                logger.warning("No GPU slots available (insufficient free memory), using CPU")
        else:
            num_gpus = 0
            available_gpu_ids = []
            logger.warning("No GPUs available, using CPU")

        n_processes = max(len(param_combinations), 1)

        participant_info = (
            f" for participant {self.participant_id}" if self.participant_id else ""
        )
        logger.info(
            f"Starting parallel training with {n_processes} processes{participant_info}"
        )

        (train_loader, train_set, n_channels, gfp_peaks, raw_mne,
         eval_set, norm_params) = args

        train_dataset = train_loader.dataset

        logger.info("✅ Extracted dataset for worker processes (100% data mode)")

        if torch.cuda.is_available():
            # Parent process only needs minimal CUDA init; workers pin their own GPU
            torch.cuda.init()
            torch.cuda.set_device(0)

        mp.set_start_method("spawn", force=True)
        try:
            mp.set_sharing_strategy("file_system")
        except RuntimeError:
            pass

        # Define Pool infrastructure lazily — only when parallel path is actually used.
        # This avoids allocating POSIX semaphores on the sequential code path.
        from multiprocessing.pool import Pool as ProcessPool

        class NoDaemonProcess(mp.Process):
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, value):
                pass

        class NoDaemonContext(type(mp.get_context("spawn"))):
            Process = NoDaemonProcess

        class MyPool(ProcessPool):
            def __init__(self, *args, **kwargs):
                kwargs["context"] = NoDaemonContext()
                super().__init__(*args, **kwargs)

        pool = MyPool(processes=n_processes)
        results = []

        try:
            # This list will hold results from completed jobs found on disk
            final_results = []
            # This list will hold async jobs to be executed
            async_results = []

            fast_sweep = self.config.get("fast_sweep", False)

            for i, (n_clusters, batch_size, latent_dim, n_conv_layers, ndf) in enumerate(
                param_combinations
            ):

                label = f"{n_clusters}_{batch_size}_{latent_dim}_{n_conv_layers}_{ndf}"
                expected_output_dir = self.run_dir / f"cluster_{label}"
                success_marker = expected_output_dir / "summary_metrics.yaml"

                if success_marker.exists():
                    logger.info(
                        f"Skipping task {i} for {label} as it has already completed."
                    )
                    try:
                        with open(success_marker, "r") as f:
                            summary = yaml.safe_load(f)

                        final_results.append(
                            {
                                "n_clusters": summary.get("n_clusters"),
                                "best_train_loss": summary.get("best_train_loss", summary.get("best_loss")),
                                "loss_history": {
                                    "silhouette_scores": [summary.get("silhouette", -1)],
                                    "db_scores": [summary.get("davies_bouldin", -1)],
                                    "ch_scores": [summary.get("calinski_harabasz", -1)],
                                },
                                "output_dir": str(expected_output_dir),
                                "success": True,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not load summary for completed task {label}: {e}"
                        )
                    continue

                logger.info(
                    f"Scheduling task {i}{participant_info}: clusters={n_clusters}, batch={batch_size}, ld={latent_dim}, depth={n_conv_layers}, ndf={ndf}"
                )

                # Pass datasets to worker (eval_set/train_set are picklable, unlike DataLoader)
                # train_dataset = augmented data for DataLoader; train_set = unaugmented for GMM init
                result = pool.apply_async(
                    _train_worker_function,
                    args=(
                        i,
                        n_clusters,
                        batch_size,
                        latent_dim,
                        self.config,
                        train_dataset,
                        n_channels,
                        available_gpu_ids,
                        num_gpus,
                        gfp_peaks,
                        raw_mne,
                        eval_set,
                        train_set,
                        norm_params,
                        ndf,
                        n_conv_layers,
                        fast_sweep,
                    ),
                )
                async_results.append((i, result))

            task_timeout = 3600 * 24  # 24 hour timeout per task

            for i, result in async_results:
                try:
                    task_result = result.get(timeout=task_timeout)
                    final_results.append(task_result)
                    logger.info(f"Task {i}{participant_info} completed successfully")
                except mp.TimeoutError:
                    logger.error(
                        f"Task {i}{participant_info} timed out after {task_timeout} seconds"
                    )
                    final_results.append(
                        {
                            "n_clusters": param_combinations[i][0],
                            "best_train_loss": float("inf"),
                            "loss_history": None,
                            "error": f"Timeout after {task_timeout} seconds",
                            "success": False,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Task {i}{participant_info} failed with error: {str(e)}"
                    )
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    final_results.append(
                        {
                            "n_clusters": param_combinations[i][0],
                            "best_train_loss": float("inf"),
                            "loss_history": None,
                            "error": str(e),
                            "success": False,
                        }
                    )

            return final_results

        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt, terminating workers")
            pool.terminate()
            return []
        except Exception as e:
            logger.error(f"Error in parallel training orchestration: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            logger.info("Cleaning up process pool")
            try:
                pool.close()
                pool.join()
            except Exception as e:
                logger.error(f"Error during pool cleanup: {str(e)}")
                pool.terminate()
            # Explicitly delete pool to trigger SemLock.__del__ → sem_unlink()
            # for POSIX semaphores backing the pool's internal queues.
            import gc
            try:
                del pool
                gc.collect()
                logger.info("Pool object deleted and semaphores released")
            except Exception as e:
                logger.error(f"Error deleting pool: {str(e)}")
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("CUDA resources and memory cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning CUDA resources: {str(e)}")

    def _train_sequential(self, param_combinations, *args) -> List[Dict]:
        """Execute sequential training with 90/10 train/eval split."""
        participant_info = (
            f" for participant {self.participant_id}" if self.participant_id else ""
        )
        logger.info(
            f"Running sequential training for parameter combinations{participant_info}"
        )
        results = []

        (train_loader, train_set, n_channels, gfp_peaks, raw_mne,
         eval_loader, eval_set, norm_params) = args

        fast_sweep = self.config.get("fast_sweep", False)

        for n_clusters, batch_size, latent_dim, n_conv_layers, ndf in param_combinations:
            result = _tc.train_cluster(
                n_clusters=n_clusters,
                config_dict=self.config,
                train_loader=train_loader,
                train_set=train_set,
                device=self.device,
                batch_size=batch_size,
                latent_dim=latent_dim,
                n_channels=n_channels,
                logger=logger,
                gfp_peaks=gfp_peaks,
                raw_mne=raw_mne,
                eval_loader=eval_loader,
                eval_set=eval_set,
                norm_params=norm_params,
                ndf=ndf,
                n_conv_layers=n_conv_layers,
                fast_sweep=fast_sweep,
            )
            results.append(result)
        return results

    def _run_baseline_for_all_clusters(
        self,
        param_combinations: List[Tuple[int, int, int, int, int]],
        gfp_peaks,
        raw_mne,
    ) -> None:
        """Run baseline ModKMeans analysis for all cluster configurations (100% data mode).

        Uses 100% of GFP peaks for baseline fitting, same as VAE training.
        """
        import baseline as _b
        import json

        logger.info("\n" + "=" * 70)
        logger.info("RUNNING BASELINE (ModKMeans) FOR ALL CLUSTER CONFIGURATIONS")
        logger.info("Using 100% of data (same as VAE training)")
        logger.info("=" * 70)

        base_output_dir = Path(self.config.get("output_dir", "./outputs"))

        for n_clusters, batch_size, latent_dim, n_conv_layers, ndf in param_combinations:
            label = f"{n_clusters}_{batch_size}_{latent_dim}_{n_conv_layers}_{ndf}"
            cluster_output_dir = base_output_dir / f"cluster_{label}"

            if not cluster_output_dir.exists():
                logger.warning(f"Cluster directory not found: {cluster_output_dir}, skipping baseline")
                continue

            logger.info(f"\nRunning baseline for K={n_clusters} (100% data mode)...")

            try:
                baseline_handler = _b.BaselineHandler(
                    n_clusters=n_clusters,
                    device=self.device,
                    logger=logger,
                    output_dir=cluster_output_dir,
                )

                # 100% data mode: fit on ALL GFP peaks
                baseline_handler.fit(gfp_peaks)

                baseline_handler.plot()  # microstate_topomaps.png
                baseline_handler.plot_merged_centroids()  # baseline_merged_centroids.png
                try:
                    baseline_handler.plot_microstate_topomaps_named()
                except Exception as e:
                    logger.warning(f"Named topomap plot failed: {e}")

                # Additional visualizations
                try:
                    baseline_handler.plot_centroid_spatial_correlations()  # correlation matrix
                    baseline_handler.plot_centroid_summary()  # centroid summary plot
                except Exception as e:
                    logger.warning(f"Could not generate centroid analysis plots: {e}")

                if raw_mne is not None:
                    baseline_handler.plot_segmentation(raw_mne)
                    baseline_handler.plot_microstate_statistics(raw_mne)
                    try:
                        baseline_handler.plot_centroid_topomaps_mne(raw_mne.info)  # MNE topomaps
                    except Exception as e:
                        logger.warning(f"Could not generate MNE topomaps: {e}")

                # Electrode space analysis (t-SNE visualization on full data)
                try:
                    baseline_handler.generate_electrode_space_visualization(save_prefix="full_data")
                except Exception as e:
                    logger.warning(f"Could not generate electrode space visualizations: {e}")

                baseline_metrics = baseline_handler.evaluate()

                if raw_mne is not None:
                    baseline_metrics["predict_on_raw"] = baseline_handler.evaluate_on_raw(raw_mne)

                with open(cluster_output_dir / "baseline_metrics.json", "w") as f:
                    json.dump(baseline_metrics, f, indent=2, default=str)

                gev = baseline_metrics.get('gev', 'N/A')
                sil = baseline_metrics.get('silhouette', 'N/A')
                gev_str = f"{gev:.4f}" if isinstance(gev, (int, float)) else str(gev)
                sil_str = f"{sil:.4f}" if isinstance(sil, (int, float)) else str(sil)
                logger.info(f"  K={n_clusters}: GEV={gev_str}, Silhouette={sil_str}")

            except Exception as e:
                logger.error(f"Error running baseline for K={n_clusters}: {e}")

        logger.info("\n" + "=" * 70)
        logger.info("BASELINE ANALYSIS COMPLETE")
        logger.info("=" * 70)


class ResultsProcessor:
    """Processes and saves training results"""

    @staticmethod
    def process_results(
        results: List[Dict],
        comparison_dir: Path,
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Process training results and generate summaries"""
        if not results:
            logger.error("No results to process")
            return

        summary_data = []
        loss_history_mean = {
            "kld_losses": [],
            "reconstruct_losses": [],
            "epoch_losses": [],
            "nmi_scores": [],
            "ari_scores": [],
            "beta_scores": [],
            "silhouette_scores": [],
            "db_scores": [],
            "ch_scores": [],
        }

        for result in results:
            if result and result.get("loss_history") is not None:
                summary_data.append(ResultsProcessor._extract_summary(result))
                ResultsProcessor._update_loss_history(
                    loss_history_mean, result["loss_history"]
                )
            else:
                participant_info = (
                    f" for participant {participant_id}" if participant_id else ""
                )
                cluster_num = result.get("n_clusters", "N/A") if result else "N/A"
                logger.warning(
                    f"Skipping results for {cluster_num} clusters{participant_info} due to error or no history"
                )

        if summary_data:
            ResultsProcessor._save_results(
                loss_history_mean, comparison_dir, summary_data, run_dir, participant_id
            )
        else:
            logger.error("No valid results to save")

    @staticmethod
    def _extract_summary(result: Dict) -> Dict:
        """Extract summary data from a single result (100% data mode)."""
        metrics = [
            "nmi_scores",
            "ari_scores",
            "silhouette_scores",
            "db_scores",
            "ch_scores",
        ]
        summary = {"n_clusters": result["n_clusters"], "best_loss": result.get("best_train_loss", result.get("best_loss", float("inf")))}
        loss_history = result.get("loss_history") or {}
        for metric in metrics:
            scores = loss_history.get(metric, [])
            summary[metric.replace("_scores", "")] = scores[-1] if scores else None
        summary["output_dir"] = result.get("output_dir", "")
        return summary

    @staticmethod
    def _update_loss_history(loss_history_mean: Dict, loss_history: Dict) -> None:
        """Update mean loss history"""
        for key in loss_history_mean:
            if key in loss_history and loss_history[key]:
                value = (
                    np.mean(loss_history[key])
                    if isinstance(loss_history[key], (list, np.ndarray))
                    else loss_history[key]
                )
                loss_history_mean[key].append(value)

    @staticmethod
    def _save_results(
        loss_history_mean: Dict,
        comparison_dir: Path,
        summary_data: List[Dict],
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Save all results and generate plots"""
        try:
            filename_prefix = f"loss_history_mean_all_cluster"
            if participant_id:
                filename_prefix = f"loss_history_mean_all_cluster_{participant_id}"

            _g.save_loss_plots(loss_history_mean, str(comparison_dir / filename_prefix))
            logger.info(f"Saved combined loss plots to {comparison_dir}")

            if len(summary_data) > 1:
                ResultsAnalyzer.analyze_and_save(
                    summary_data, comparison_dir, run_dir, participant_id
                )
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


class ResultsAnalyzer:
    """Analyzes results and generates comparison metrics"""

    @staticmethod
    def analyze_and_save(
        summary_data: List[Dict],
        comparison_dir: Path,
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Analyze results and save comparisons"""
        df = pd.DataFrame(summary_data).sort_values("n_clusters")
        ResultsAnalyzer._save_summary_files(df, comparison_dir, participant_id)
        ResultsAnalyzer._generate_plots(df, comparison_dir, participant_id)
        ResultsAnalyzer._save_best_configurations(
            df, comparison_dir, run_dir, participant_id
        )

    @staticmethod
    def _save_summary_files(
        df: pd.DataFrame, comparison_dir: Path, participant_id: str = None
    ) -> None:
        """Save summary data in CSV and YAML formats"""
        csv_name = "cluster_comparison.csv"
        yaml_name = "cluster_comparison.yaml"
        if participant_id:
            csv_name = f"cluster_comparison_{participant_id}.csv"
            yaml_name = f"cluster_comparison_{participant_id}.yaml"

        df.to_csv(comparison_dir / csv_name, index=False)
        with open(comparison_dir / yaml_name, "w") as f:
            yaml.dump(df.to_dict("records"), f, default_flow_style=False)

    @staticmethod
    def _generate_plots(
        df: pd.DataFrame, comparison_dir: Path, participant_id: str = None
    ) -> None:
        """Generate comparison plots (100% data mode - training loss only)."""
        plt.figure(figsize=(12, 6))
        plt.plot(df["n_clusters"], df["best_loss"], "o-", linewidth=2)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Training Loss")

        title = "Training Loss vs Number of Clusters (100% Data)"
        if participant_id:
            title = f"Training Loss vs Number of Clusters (100% Data) - {participant_id}"
        plt.title(title)

        plt.grid(True)

        plot_name = "training_loss_by_clusters.png"
        if participant_id:
            plot_name = f"training_loss_by_clusters_{participant_id}.png"

        plt.savefig(comparison_dir / plot_name)
        plt.close()

    @staticmethod
    def _save_best_configurations(
        df: pd.DataFrame,
        comparison_dir: Path,
        run_dir: Path,
        participant_id: str = None,
    ) -> None:
        """Save best configurations and create model links (100% data mode)."""
        if df.empty or "best_loss" not in df.columns or df["best_loss"].isnull().all():
            logger.warning(
                "DataFrame is empty or has no valid losses; cannot determine best configuration."
            )
            return

        best_metrics = {
            "best_by_training_loss": df.loc[df["best_loss"].idxmin()].to_dict()
        }

        if participant_id:
            best_metrics["participant_id"] = participant_id

        config_name = "best_configurations.yaml"
        if participant_id:
            config_name = f"best_configurations_{participant_id}.yaml"

        with open(comparison_dir / config_name, "w") as f:
            yaml.dump(best_metrics, f, default_flow_style=False)


def main():
    """Main execution function"""

    # GPU selection is handled by run.sh (CUDA_VISIBLE_DEVICES) or auto-detection at top of file
    # NOTE: mp.set_start_method and mp.set_sharing_strategy are called lazily
    # inside _train_parallel() to avoid allocating multiprocessing resources
    # on the sequential code path (used by run_parallel.sh).

    try:
        try:
            args = _pa.parse_args()
        except Exception as e:
            logger.error(f"Error parsing arguments: {str(e)}")
            return

        try:
            _s.set_seed(args.seed)
        except Exception as e:
            logger.error(f"Error setting seed: {str(e)}")
            return

        try:
            config_manager = ConfigManager(args)
            config = config_manager.get_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return

        participant_id = args.participant
        if participant_id is None:
            participant_id = c.get_lemon_config().get("subject_id")
        if participant_id:
            logger.info(f"Training configured for participant: {participant_id}")

        try:
            dir_manager = DirectoryManager(
                config.get("output_dir", "./outputs"), participant_id, args.run_id
            )
            run_dir = dir_manager.setup_directories()
            config["output_dir"] = str(run_dir)

            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error setting up directories or saving config: {str(e)}")
            return

        logger.info(f"Starting new run in directory: {run_dir}")

        try:
            # 90/10 split: returns (n_channels, train_loader, train_set, gfp_peaks, raw_mne,
            #                        eval_loader, eval_set, norm_params)
            (n_channels, train_loader, train_set, gfp_peaks, raw_mne,
             eval_loader, eval_set, norm_params) = (
                DataLoaderFactory.get_data_loaders(args, config, participant_id)
            )
        except Exception as e:
            logger.error(f"Error creating data loaders: {str(e)}")
            return

        # Phase 1: Exit after preprocessing if --preprocess-only flag is set
        if getattr(args, 'preprocess_only', False):
            logger.info("=" * 60)
            logger.info("PREPROCESS-ONLY MODE: Preprocessing complete!")
            logger.info("Cache saved. Run parallel training jobs with --use-cached flag.")
            logger.info("=" * 60)
            return

        try:
            orchestrator = TrainingOrchestrator(config, run_dir, args, participant_id)
            results = orchestrator.train(
                train_loader, train_set, n_channels,
                gfp_peaks=gfp_peaks, raw_mne=raw_mne,
                eval_loader=eval_loader, eval_set=eval_set,
                norm_params=norm_params,
            )
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return

        try:
            ResultsProcessor.process_results(
                results, dir_manager.get_comparison_dir(), run_dir, participant_id
            )
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return

        participant_info = (
            f" for participant {participant_id}" if participant_id else ""
        )
        logger.info(f"Training pipeline completed successfully{participant_info}")

    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {str(e)}")
        logger.critical(traceback.format_exc())


if __name__ == "__main__":
    main()
