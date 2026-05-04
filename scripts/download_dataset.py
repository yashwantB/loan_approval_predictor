from __future__ import annotations

import shutil
from pathlib import Path


DATASET_HANDLE = "mytalkwithyou/bank-loan-approval-dataset"
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_FILE = RAW_DATA_DIR / "loan_approval_dataset.csv"


def main() -> None:
    import kagglehub

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    matches = sorted(dataset_path.glob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV files found in Kaggle dataset: {dataset_path}")

    shutil.copy2(matches[0], RAW_DATA_FILE)
    print(f"Downloaded {DATASET_HANDLE}")
    print(f"Saved: {RAW_DATA_FILE}")


if __name__ == "__main__":
    main()
