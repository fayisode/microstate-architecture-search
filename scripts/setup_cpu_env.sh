#!/bin/bash
# Setup CPU-only environment for IBM Power (ppc64le) server
# Follows MIT Satori docs: https://mit-satori.github.io/satori-ai-frameworks.html
# Usage: ./setup_cpu_env.sh
set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="sweep"

echo "============================================================"
echo "Setting up CPU-only environment (ppc64le)"
echo "============================================================"
echo "Arch: $(uname -m)"
echo "Python (system): $(python3 --version 2>&1)"

# ── Step 1: Install Miniconda if not present ──
if [ -f "$CONDA_DIR/bin/conda" ]; then
    echo "Miniconda already installed at $CONDA_DIR"
else
    echo ""
    echo "Installing Miniconda to $CONDA_DIR..."
    INSTALLER="/tmp/miniconda_installer.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"
fi

# Make conda available in this shell
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

echo "Conda: $(conda --version)"

# ── Step 2: Create conda environment ──
# Python 3.6 per Satori docs (WML-CE/Open-CE valid versions: 3.6, 3.7, 3.8)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Conda env '$ENV_NAME' already exists"
else
    echo ""
    echo "Creating conda env '$ENV_NAME' with Python 3.6..."
    conda create -y -n "$ENV_NAME" python=3.6
fi

conda activate "$ENV_NAME"

# ── Step 3: Configure channels (Satori approach) ──
# WML-CE + Open-CE channels for ppc64le packages
echo ""
echo "Configuring channels (Satori approach)..."
conda config --env --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

# Accept IBM license automatically
export IBM_POWERAI_LICENSE_ACCEPT=yes

# ── Step 4: Install all conda packages ──
echo ""
echo "Installing PyTorch + scientific stack..."
conda install -y \
    pytorch \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    scikit-image

# ── Step 5: Install pip dependencies ──
echo ""
echo "Installing pip dependencies..."
pip install --quiet \
    mne \
    tqdm \
    PyYAML \
    toml \
    python-wget \
    gdown

# ── Step 6: Verify ──
echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"
python -c "import torch; print('PyTorch {} (CUDA: {})'.format(torch.__version__, torch.cuda.is_available()))"
python -c "import numpy; print('NumPy {}'.format(numpy.__version__))"
python -c "import sklearn; print('scikit-learn {}'.format(sklearn.__version__))"
python -c "import scipy; print('SciPy {}'.format(scipy.__version__))"
python -c "import matplotlib; print('Matplotlib {}'.format(matplotlib.__version__))"
python -c "import mne; print('MNE {}'.format(mne.__version__))"
python -c "import yaml; print('PyYAML OK')"
python -c "import toml; print('TOML OK')"

# Verify project imports
echo ""
echo "Testing project import chain..."
python -c "
import sys
sys.path.insert(0, '.')
from config.config import config as c
print('  config: OK')
import process_eeg_signals
print('  process_eeg_signals: OK')
import model
print('  model: OK')
import clustering_trainer
print('  clustering_trainer: OK')
import train_cluster
print('  train_cluster: OK')
print('All imports OK - ready for training')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To activate, run BOTH lines:"
echo "  export PATH=\"$CONDA_DIR/bin:\$PATH\""
echo "  conda activate $ENV_NAME"
echo ""
echo "Or add to ~/.bashrc for persistence:"
echo "  echo 'export PATH=\"$CONDA_DIR/bin:\$PATH\"' >> ~/.bashrc"
echo "============================================================"
