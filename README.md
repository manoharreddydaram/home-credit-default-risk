# 🏦 Loan Default Risk Prediction
### Home Credit Default Risk | End-to-End Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.73-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📌 Project Overview

This project builds an end-to-end **loan default prediction pipeline** on 307,511 real-world home loan applications from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

The goal is to predict whether a loan applicant will **default (TARGET=1)** or **repay (TARGET=0)** — a core problem in credit risk modelling for any housing finance company. Only ~8% of applicants default, making this a **class imbalance** problem that requires careful metric selection (AUC-ROC over accuracy).

> **Domain relevance:** The techniques used here — debt-to-income ratio engineering, external credit score analysis, and borrower segmentation — are directly applicable to real-world credit underwriting at housing finance companies.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) |
| Primary file used | `application_train.csv` |
| Total records | 307,511 loan applications |
| Features | 122 columns (demographics, financials, credit scores, document flags) |
| Target | `TARGET` — 0: repaid, 1: defaulted |
| Class imbalance | 91.9% repaid · 8.1% defaulted |

**Why only `application_train.csv`?**
The dataset is a relational database across 7 CSV files. `application_train.csv` is the primary table containing the complete applicant profile and the TARGET label. The other files (`bureau.csv`, `installments_payments.csv` etc.) contain supplementary payment history that requires aggregation joins. For this baseline project, the primary table provides sufficient signal. In a production pipeline, joining `bureau.csv` on `SK_ID_CURR` would add external credit history features and is expected to improve AUC by 3–5%.

---

## 🔑 Key Columns Explained

| Column | Description |
|---|---|
| `TARGET` | 0 = repaid, 1 = defaulted — the prediction target |
| `AMT_CREDIT` | Total loan amount sanctioned |
| `AMT_INCOME_TOTAL` | Applicant's annual income |
| `AMT_ANNUITY` | Monthly EMI amount |
| `DAYS_BIRTH` | Age stored as negative days from application date (divide by -365 for years) |
| `DAYS_EMPLOYED` | Employment duration in negative days; 365243 = unemployed/pensioner |
| `EXT_SOURCE_1/2/3` | External credit bureau scores (0–1 scale) — most predictive features |
| `FLAG_OWN_CAR` | Whether applicant owns a car (asset ownership signal) |
| `FLAG_OWN_REALTY` | Whether applicant owns property |
| `FLAG_DOCUMENT_X` | ~18 binary columns indicating which documents were submitted |

---

## ⚙️ Project Pipeline

```
Raw Data (307,511 rows)
        │
        ▼
1. Exploratory Data Analysis
   ├── Class imbalance check (8% default rate)
   ├── Income vs default boxplot
   └── Feature correlation heatmap
        │
        ▼
2. Feature Engineering (5 domain features)
   ├── DEBT_TO_INCOME = AMT_CREDIT / AMT_INCOME_TOTAL
   ├── ANNUITY_TO_INCOME = AMT_ANNUITY / AMT_INCOME_TOTAL
   ├── AGE_YEARS = -DAYS_BIRTH / 365
   ├── EMPLOYMENT_YEARS = -DAYS_EMPLOYED.clip(upper=0) / 365
   └── EXT_SOURCE_MEAN = mean(EXT_SOURCE_2, EXT_SOURCE_3)
        │
        ▼
3. Preprocessing
   ├── Median imputation for missing values
   └── Stratified train/test split (80/20)
        │
        ▼
4. Modelling
   ├── Baseline: Logistic Regression
   └── Final: Random Forest (100 trees, max_depth=10)
        │
        ▼
5. Evaluation
   ├── AUC-ROC: 0.73
   └── Feature importance analysis
```

---

## 📈 Results

| Model | AUC-ROC |
|---|---|
| Logistic Regression (baseline) | ~0.72 |
| Random Forest | **0.73** |

**Why AUC-ROC and not accuracy?**
A naive model that predicts "repaid" for everyone achieves 92% accuracy — but catches zero defaulters. AUC-ROC measures the model's ability to rank defaulters above non-defaulters across all thresholds, making it the correct metric for imbalanced credit data.

---

## 🔍 Key Findings

1. **EXT_SOURCE_MEAN topped feature importance** — External credit bureau scores are the single strongest predictor of default. They encode years of repayment history from multiple bureaus, making them a distilled creditworthiness signal.

2. **DEBT_TO_INCOME ranked 10th** — Because `AMT_CREDIT` and `AMT_INCOME_TOTAL` are both present as raw features, Random Forest can implicitly learn their ratio. However, for **thin-bureau borrowers** (low-income segments where EXT_SOURCE is missing), engineered ratio features become relatively more important as fallback signals.

3. **Loan amount alone is a weak predictor** — The income and loan amount distributions for defaulters and non-defaulters overlap significantly. The ratio of the two captures risk far better than either individually.

4. **Class imbalance is a real challenge** — With only 8% defaults, standard accuracy metrics are misleading. Next steps include SMOTE oversampling and threshold tuning to improve recall on the minority class.

---

## 🚀 How to Run

### On Kaggle (recommended — no setup needed)
1. Go to the [Home Credit competition](https://www.kaggle.com/competitions/home-credit-default-risk)
2. Click **Code → New Notebook**
3. Copy the notebook cells from this repo
4. Click **Run All**

### Locally
```bash
# Clone the repo
git clone https://github.com/manoharreddydaram/home-credit-default-risk.git
cd home-credit-default-risk

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Download data from Kaggle
kaggle competitions download -c home-credit-default-risk

# Run the notebook
jupyter notebook loan_default_prediction.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas | Data loading, manipulation, feature engineering |
| NumPy | Numerical operations, array handling |
| Scikit-learn | ML models, train/test split, AUC-ROC |
| Matplotlib | Bar charts, histograms, feature importance plots |
| Seaborn | Correlation heatmap, distribution plots |
| Jupyter Notebook | Development environment |

---

## 📁 Repository Structure

```
home-credit-default-risk/
│
├── loan_default_prediction.ipynb   # Main notebook (full pipeline)
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## 🔮 Future Improvements

- [ ] Join `bureau.csv` to add external credit history features per applicant
- [ ] Apply **SMOTE** to address 8% class imbalance in training data
- [ ] Switch to **XGBoost** with hyperparameter tuning (expected AUC ~0.78+)
- [ ] Add **SHAP values** for model explainability — critical for regulatory compliance in lending
- [ ] Build a **Streamlit dashboard** for interactive default probability scoring

---

## 👤 Author

**Daram Manohar Reddy**
- 📧 manoharreddydaram@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/manoharreddydaram)
- 🐙 [GitHub](https://github.com/manoharreddydaram)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

> Dataset provided by Home Credit Group via Kaggle for educational purposes.
