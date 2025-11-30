# Adult Census Income Prediction 📊

[![Repository](https://img.shields.io/badge/repo-Adult--Census--Income--Prediction-blue)](https://github.com/Hemanth7723/Adult-Census-Income-Prediction)
[![Languages](https://img.shields.io/github/languages/top/Hemanth7723/Adult-Census-Income-Prediction)](https://github.com/Hemanth7723/Adult-Census-Income-Prediction)
[![Last commit](https://img.shields.io/github/last-commit/Hemanth7723/Adult-Census-Income-Prediction)](https://github.com/Hemanth7723/Adult-Census-Income-Prediction/commits/main)

A reproducible data science project that analyzes the UCI "Adult" (Census Income) dataset and builds models to predict whether an individual's income exceeds $50K/yr. The project is notebook-centric (Jupyter Notebooks) and focuses on end-to-end steps: data ingestion, EDA, preprocessing, feature engineering, modeling, tuning, evaluation, and interpretability.

---

## **Table of contents**
- [**About the dataset**](#about-the-dataset)
- [**Repository structure**](#repository-structure)
- [**Quick start**](#quick-start)
  - [**Run in Google Colab**](#run-in-google-colab)
  - [**Run locally**](#run-locally)
- [**Environment & dependencies**](#environment--dependencies)
- [**Notebooks & workflow**](#notebooks--workflow)
- [**Modeling & evaluation**](#modeling--evaluation)
- [**Reproducibility & results**](#reproducibility--results)
- [**Tips for extension**](#tips-for-extension)
- [**Contributing**](#contributing)
- [**Acknowledgements**](#acknowledgements)

---

## **About the dataset**
- Source: UCI Machine Learning Repository — "Adult" / "Census Income" dataset.
- Goal: Predict whether an individual's annual income is >50K.
- Typical features: age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, etc.
- Target: income (<=50K, >50K).
- Notes: The dataset contains missing values encoded as '?', categorical attributes with varying cardinalities, and class imbalance that should be considered during modeling.

---

## **Repository structure**
- Notebooks/ or root .ipynb files — main analysis and modeling notebooks
- data/ (optional) — raw and processed datasets (if included)
- notebooks/ — exploratory and modeling notebooks (EDA, preprocessing, modeling, explainability)
- reports/ (optional) — model summaries, charts, or exported results
- requirements.txt — Python dependencies (if present)
- README.md — this file

(Exact file names may vary; open the repo to see the notebook filenames and adjust references if needed.)

---

## **Quick start**

### **Run in Google Colab**
1. Open the notebook in GitHub and click "Open in Colab" (or use the Colab link if provided).
2. If necessary, mount Google Drive to persist outputs:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Install required packages (if `requirements.txt` is present):
```bash
!pip install -r requirements.txt
```

### **Run locally**
1. Clone the repository:
```bash
git clone https://github.com/Hemanth7723/Adult-Census-Income-Prediction.git
cd Adult-Census-Income-Prediction
```
2. Create a Python environment (recommended):
```bash
# using venv
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

# or using conda
conda create -n acip python=3.9
conda activate acip
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Start Jupyter Lab / Notebook:
```bash
jupyter lab
# or
jupyter notebook
```
5. Open the notebooks and run cells in order (typically start with EDA notebook, then preprocessing, then modeling).

---

## **Environment & dependencies**
- Recommended Python: 3.8+ (3.9 or 3.10 are common choices)
- Core libraries commonly used: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost (optional), shap/eli5 (for explainability), joblib (for saving models)
- If a requirements.txt exists, install with `pip install -r requirements.txt`.
- For GPU-enabled training (if using large models), ensure appropriate CUDA and XGBoost/LightGBM builds; however, the Adult dataset is small and CPU is typically sufficient.
- After changes you can create your own requirements.txt file with `pip freeze > requirements.txt`.

---

## **Notebooks & workflow**
Typical notebook order (may vary by repo contents):
1. 00-data-exploration.ipynb — Data loading, missing values, distribution of features, class balance, initial visualizations.
2. 01-preprocessing-feature-engineering.ipynb — Encoding categorical features, scaling numeric features, handling missing values, feature creation.
3. 02-modeling.ipynb — Train test split, baseline models (Logistic Regression, Decision Trees), cross-validation, hyperparameter tuning.
4. 03-evaluation-and-interpretation.ipynb — Final evaluation metrics (confusion matrix, ROC AUC), feature importance and SHAP/interpretability.
5. 04-deployment-or-notes.ipynb — (Optional) model export, inference examples, or notes on deployment.

Run notebooks sequentially to reproduce results. Use kernel restart and run-all to ensure clean reproducibility.

---

## **Modeling & evaluation**
- Suggested models: Logistic Regression (baseline), Random Forest, Gradient Boosted Trees (XGBoost/LightGBM), and simple ensemble approaches.
- Important steps:
  - Use stratified train/test split to preserve class balance.
  - Apply cross-validation for robust performance estimates.
  - Address class imbalance (if present) via class weights, resampling (SMOTE), or threshold tuning.
- Common evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC AUC and precision-recall curves (especially useful when classes are imbalanced)
  - Confusion matrix to inspect error types
- Save best models with joblib or pickle for later inference.

---

## **Reproducibility & results**
- Set random seeds in notebooks (numpy, scikit-learn, any model-specific RNG) to help reproduce experiments.
- If results/figures are important, export them to the `reports/` or `outputs/` folder in a notebook cell so they are preserved.
- Document chosen hyperparameters, model versions, and experiment IDs in a short log or table inside the notebooks.

---

## **Tips for extension**
- Try advanced feature engineering: interaction terms, binning continuous features, or target encoding for high-cardinality categories.
- Experiment with pipeline automation: use scikit-learn Pipelines to chain preprocessing and modeling steps.
- Add CI for notebooks: use nbval or papermill to validate notebooks in continuous integration.
- Build a lightweight REST API for inference with FastAPI or Flask and containerize with Docker for deployment.

---

## **Contributing** 🤝
Contributions are welcome. Suggested workflow:
1. Fork the repo
2. Create a new branch: `git checkout -b feature/your-change`
3. Make changes and add or update notebooks/documentation
4. Commit and push, then open a pull request with a clear description and any reproducible steps

Please keep notebooks runnable (consider clearing outputs before committing large outputs) and add a short description of changes to the notebook's top cell.

---

## **Acknowledgements** 🙏
- The UCI Machine Learning Repository for the Adult Census Income dataset.
- Data science and ML community resources for modeling and interpretability best practices.

---
