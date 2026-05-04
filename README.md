# Loan Approval Predictor

A machine learning project for predicting loan approval outcomes from applicant, credit, and loan application attributes. The project compares Logistic Regression and Random Forest classifiers, generates evaluation plots, and saves the trained model bundle for reuse.

## Dataset

The project uses the Kaggle dataset [mytalkwithyou/bank-loan-approval-dataset](https://www.kaggle.com/datasets/mytalkwithyou/bank-loan-approval-dataset).

- Rows: 70,000
- Columns: 23
- Target: `loan_status`
- Local copy: `data/raw/loan_approval_dataset.csv`

The dataset can be downloaded again with `scripts/download_dataset.py` through `kagglehub`.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9475 | 0.9448 | 0.9506 | 0.9477 | 0.9909 |
| Logistic Regression | 0.9164 | 0.9033 | 0.9327 | 0.9178 | 0.9769 |

Random Forest achieved the best overall performance on the held-out test set.

## Project Structure

```text
.
├── data/raw/loan_approval_dataset.csv
├── models/loan_approval_models.joblib
├── notebooks/loan_approval_system.ipynb
├── reports/
│   ├── figures/
│   ├── metrics.csv
│   └── metrics_all.csv
├── scripts/
│   ├── download_dataset.py
│   └── run_notebook.py
├── src/loan_approval_system/
├── pyproject.toml
└── uv.lock
```

## Setup

This project uses `uv` for dependency management.

```bash
uv sync
```

To include the optional Apple Silicon MLX experiment:

```bash
uv sync --extra mlx
```

## Usage

Download or refresh the dataset:

```bash
uv run python scripts/download_dataset.py
```

Open the notebook:

```bash
uv run jupyter lab notebooks/loan_approval_system.ipynb
```

Execute the notebook from the terminal:

```bash
uv run python scripts/run_notebook.py
```

Run the lightweight package entry point:

```bash
uv run loan-approval-system
```

## Outputs

- `models/loan_approval_models.joblib`: saved preprocessing and model bundle
- `reports/metrics.csv`: final model metrics
- `reports/figures/confusion_matrices.png`: confusion matrix comparison
- `reports/figures/roc_curves.png`: ROC curve comparison
- `reports/figures/random_forest_feature_importance.png`: Random Forest feature importance

## Notes

- The notebook uses scikit-learn pipelines and column transformers to keep preprocessing fit only on training data.
- Logistic Regression uses `class_weight="balanced"`.
- Random Forest uses `class_weight="balanced_subsample"`.
- CUDA systems can use RAPIDS/cuML if installed separately in a compatible environment; the notebook detects `cuml` automatically.
- Many Kaggle loan approval datasets are anonymized or synthetic. For audited real-world finance use, replace the dataset with a verified institutional source and keep the same training pipeline.
