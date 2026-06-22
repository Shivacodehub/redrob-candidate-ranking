#!/usr/bin/env bash
# setup_local_gpu.sh
# One-shot setup for running the pipeline on your own GPU machine.
# Run once: bash setup_local_gpu.sh
# Then activate the venv each session: source venv/bin/activate

set -e

echo "========================================"
echo " Redrob Candidate Ranking — GPU Setup"
echo "========================================"

# ── 1. Check Python ──────────────────────────────────────────────────────────
python_bin=""
for cmd in python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(sys.version_info >= (3,10))")
        if [ "$ver" = "True" ]; then
            python_bin="$cmd"
            break
        fi
    fi
done

if [ -z "$python_bin" ]; then
    echo "ERROR: Python 3.10+ not found. Install it first."
    exit 1
fi
echo "Using Python: $($python_bin --version)"

# ── 2. Check CUDA ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "── GPU info ──"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Make sure CUDA drivers are installed."
    echo "  Ubuntu: sudo apt install nvidia-driver-535 nvidia-cuda-toolkit"
    echo "Continuing anyway..."
fi

# ── 3. Create virtual environment ─────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $python_bin -m venv venv
fi
source venv/bin/activate

# ── 4. Install PyTorch with CUDA ──────────────────────────────────────────────
echo "Installing PyTorch (CUDA 12.1 build)..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible to PyTorch
$python_bin -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} | {props.total_memory/1e9:.1f} GB')
if not torch.cuda.is_available():
    print()
    print('WARNING: CUDA not available to PyTorch.')
    print('If you have a GPU, check that CUDA drivers match the PyTorch build.')
"

# ── 5. Install remaining dependencies ─────────────────────────────────────────
echo "Installing remaining dependencies..."
pip install --quiet \
    transformers>=4.40.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.43.0 \
    sentence-transformers>=2.7.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    jupyter ipykernel \
    pyyaml flask flask-cors

# Register the venv as a Jupyter kernel
$python_bin -m ipykernel install --user --name redrob-gpu --display-name "Redrob GPU (Python)"

# ── 6. Create data/ folder (gitignored) ───────────────────────────────────────
mkdir -p data output

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Copy your dataset:   cp /path/to/candidates.jsonl data/"
echo "  2. Activate the venv:   source venv/bin/activate"
echo "  3. Open the notebook:   jupyter notebook candidate_ranking_pipeline.ipynb"
echo "     Select kernel:       Redrob GPU (Python)"
echo ""
echo "  In the notebook, update these two lines in the CONFIG cell:"
echo "    CANDIDATES_PATH = \"data/candidates.jsonl\""
echo "    JOB_DESCRIPTION = \"\"\"<paste your JD>\"\"\""
echo ""
echo "  If you have only ONE GPU, also change:"
echo "    LLM_DEVICE_MAP = \"auto\"   # instead of {\"\": 1}"
echo ""
