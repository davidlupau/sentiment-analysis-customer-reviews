"""
utils.py
Utility functions for device detection, data loading and saving.
All data files are loaded from /data/ and outputs are saved to /analysis_output/.
"""

import platform
import pandas as pd
import torch
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    """Detect the best available compute device.

    Checks hardware availability in order of preference:
      1. CUDA  – NVIDIA GPU (fastest for most deep-learning workloads).
      2. MPS   – Apple Silicon GPU (Metal Performance Shaders).
      3. CPU   – universal fallback.

    Returns:
        torch.device: The most capable device available on this machine.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(file_name):
    """Load the dataset from the csv file
    Parameters:
         file_name (string): name of the csv file in /data folder
    Returns:
        dataframe containing the election results
    """
    print("\nLoading dataset...\n")
    try:
        # Get the project root (go up from src/)
        project_root = Path(__file__).parent.parent
        data_file = project_root / "data" / file_name

        # Check if file exists first
        if not data_file.exists():
            print(f"File not found: {data_file}")
            return None

        df = pd.read_csv(data_file)
        print(f"Successfully loaded {file_name} \n")
        return df
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None



