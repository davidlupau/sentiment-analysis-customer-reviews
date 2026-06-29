"""
utils.py
Utility functions for device detection, data loading and saving.
All data files are loaded from /data/ and outputs are saved to /analysis_output/.
"""

import platform
import re
import pandas as pd
import torch
from pathlib import Path

# Characters illegal in XML 1.0 (and therefore in .xlsx cell values)
_ILLEGAL_CHARS_RE = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]')


def _sanitise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with XML-illegal characters stripped from string columns."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(_ILLEGAL_CHARS_RE, '', regex=True)
    return df


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

def load_dataset(file_name: str) -> pd.DataFrame | None:
    """Load an Amazon Fashion reviews CSV from the project data directory.

    Parameters:
        file_name (str): Name of the CSV file inside the project's data/ folder.

    Returns:
        pd.DataFrame | None: Loaded DataFrame, or None if the file is not found
            or an error occurs.
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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_to_excel(
    data: pd.DataFrame | dict,
    file_name: str,
) -> str | None:
    """Save a DataFrame or dict of DataFrames to an Excel file in analysis_output/.

    Parameters:
        data (pd.DataFrame | dict): A single DataFrame written to one sheet,
            or a dict mapping sheet names to DataFrames for a multi-sheet workbook.
        file_name (str): Name of the output Excel file (e.g. 'results.xlsx').

    Returns:
        str | None: Absolute path to the saved file, or None if saving fails.
    """
    print(f"Saving analysis to {file_name}...\n")
    try:
        # Get the project root (go up from src/)
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "analysis_output"

        # Create directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / file_name

        if isinstance(data, dict):
            # Multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    _sanitise_df(df).to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Single DataFrame
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                _sanitise_df(data).to_excel(writer, sheet_name="Data", index=False)

        print(f"Successfully saved to {output_file}\n")
        return str(output_file)

    except Exception as e:
        print(f"Error saving {file_name}: {e}")
        return None

