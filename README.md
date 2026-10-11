# Manufacturing Quality Prediction Using Machine Learning

## Project status
Working educational prototype using **simulated data**. It is not trained or validated on real factory measurements. Do not present its metrics as real-world industrial performance.

## Objective
Demonstrate a reproducible machine-learning workflow for classifying a simulated manufacturing sample as `Defect` or `No defect` from five process measurements:
- temperature (°C)
- pressure (bar)
- machine speed (RPM)
- vibration (mm/s)
- material moisture (%)

## Tools
Python, NumPy, pandas, scikit-learn, Matplotlib.

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create a virtual environment (recommended):
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python main.py`

The script generates the dataset, evaluates a Random Forest classifier, saves metrics, and creates four plots.

## Reproducibility
- Fixed random seed: 42
- Train/test split: 75% / 25%, stratified
- Model: Random Forest classifier with 250 trees
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix and ROC curve

## Outputs
- `outputs/simulated_manufacturing_data.csv`
- `outputs/metrics.json`
- `outputs/classification_report.txt`
- `outputs/class_distribution.png`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`
- `outputs/roc_curve.png`
- `screenshots/run_results_summary.png`

## Limitations and next steps
- Data are synthetic and created from an illustrative rule; the model can learn patterns specific to that rule.
- These results do not demonstrate performance on actual factory data.
- A real research extension would require a public or approved industrial dataset, clear data provenance, a suitable validation strategy, domain review, and error analysis.
- Feature importance is descriptive, not proof of causality.

## Portfolio description
Built a reproducible Python ML pipeline for an educational manufacturing-quality classification task using simulated process data. Implemented data generation, stratified train/test splitting, preprocessing, Random Forest classification, evaluation metrics, confusion matrix, ROC curve and feature-importance visualization. Documented limitations and reproducibility steps.
