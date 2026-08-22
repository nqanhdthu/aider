# Aider Experiment Framework

A unified framework for image classification experiments on three datasets:
- **IP102**
- **Do**
- **Xie**

It supports:
- Training (`baseline` or `cbam`)
- Evaluation of trained models
- Backbone feature extraction from trained models
- Classification model training on extracted features (SVM, KNN, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost)
- Full end-to-end pipeline

---

## 1) Project Structure

```text
.
├── main.py
├── train_config.yaml
├── train_config_Do.yaml
├── train_config_xie.yaml
└── framework/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── modeling.py
    └── pipeline.py
```

- `main.py`: unified CLI entrypoint.
- `framework/config.py`: loads config automatically by dataset.
- `framework/data.py`: shared data loader for `.txt` (IP102) and `.csv` (Do/Xie).
- `framework/modeling.py`: model creation, checkpoint loading, backbone feature extraction.
- `framework/pipeline.py`: train / eval / feature-classifier / full pipeline logic.

---

## 2) Requirements

Minimum packages:
- `torch`, `torchvision`
- `timm`
- `pandas`, `numpy`
- `scikit-learn`
- `pyyaml`
- `Pillow`
- `tqdm`

Quick install:

```bash
pip install torch torchvision timm pandas numpy scikit-learn pyyaml pillow tqdm
```

---

## 3) Dataset Configuration

Default config mapping:
- `ip102` -> `train_config.yaml`
- `do` -> `train_config_Do.yaml`
- `xie` -> `train_config_xie.yaml`

Required config fields:
- `train_annotation`, `val_annotation`, `test_annotation`
- `image_dir`
- `batch_size`, `learning_rate`, `epochs`
- `model_name`, `input_key`, `input_size`
- `log_dir`, `early_stop_patience`
- `cbam_layers` (when using `--model cbam`)

---

## 4) Usage

### 4.1 Train

```bash
python3 main.py train --dataset ip102 --model baseline
python3 main.py train --dataset do --model cbam
```

### 4.2 Evaluate

```bash
python3 main.py eval --dataset ip102 --model baseline
python3 main.py eval --dataset xie --model cbam
```

Use a specific checkpoint:

```bash
python3 main.py eval --dataset ip102 --model baseline --checkpoint /path/to/checkpoint.pth
```

### 4.3 Feature Extraction + Classifier

```bash
python3 main.py svm --dataset ip102 --model baseline
python3 main.py svm --dataset do --model cbam --classifier random_forest
python3 main.py svm --dataset xie --model cbam --classifier xgboost
```

### 4.4 Full Pipeline (train + eval + feature-classifier)

```bash
python3 main.py full --dataset xie --model baseline
python3 main.py full --dataset xie --model baseline --classifier lightgbm
```

---

## 5) Output

Outputs are saved under:

```text
{log_dir}/framework/{dataset}/
```

Typical artifacts:
- `best_<model>.pth`
- `eval_metrics.json`
- `eval_per_class.csv`
- `features/`
  - `train_val_features.npy`, `train_val_labels.npy`
  - `test_features.npy`, `test_labels.npy`
- `<classifier>_model.pkl`
- `<classifier>_metrics.json`
- `<classifier>_scaler.pkl` (for scaled classifiers, e.g. SVM/KNN/Logistic Regression)
- `label_mapping.json`

---

## 6) CLI Options

```bash
python3 main.py --help
```

Supported modes:
- `train`
- `eval`
- `svm`
- `full`

Supported backbone models:
- `baseline`
- `cbam`

Supported feature classifiers (`--classifier`):
- `svm` (default)
- `knn`
- `logistic_regression`
- `decision_tree`
- `random_forest`
- `xgboost`
- `lightgbm`
- `catboost`

Supported datasets:
- `ip102`, `do`, `xie`
- aliases: `ip`, `do_dataset`
