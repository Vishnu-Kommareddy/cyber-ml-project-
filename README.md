# Cyber ML Project — Network Intrusion Detection

Machine learning model for network intrusion detection using the **NSL-KDD** dataset. Trains a Random Forest classifier to distinguish between normal and malicious network traffic.

## Dataset

**NSL-KDD** — an improved version of the KDD'99 dataset with no redundant or duplicate records, making for more realistic evaluation.

- `data/KDDTrain+.txt` — training set
- `data/KDDTest+.txt` — held-out test set

Both files are included in the `data/` directory.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python intrusion_detection.py
```

The script will:
1. Load and preprocess the training data (one-hot encode categorical features)
2. Train a Random Forest classifier (50 estimators)
3. Evaluate on the held-out test set (`KDDTest+.txt`)
4. Print confusion matrix and classification report
5. Save the trained model to `model.joblib`

## Results

| Metric | Value |
|--------|-------|
| Dataset | NSL-KDD (KDDTrain+ / KDDTest+) |
| Model | Random Forest (50 trees) |
| Task | Binary classification (normal vs attack) |

(Run the script to see exact accuracy, precision, recall, and F1-score.)

## Dependencies

- Python 3.8+
- pandas
- scikit-learn
- joblib

## Future Work

- Experiment with other classifiers (XGBoost, neural networks)
- Multi-class classification (identify specific attack types)
- Feature selection / dimensionality reduction
- Real-time inference pipeline
