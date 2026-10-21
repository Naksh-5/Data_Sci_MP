# Predictive Criminal Justice Analytics & Data Science Capstone

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Data Science curriculum and Capstone Mini-Project implementation analyzing official datasets from the **National Crime Records Bureau (NCRB)** and research from the **Commonwealth Human Rights Initiative (CHRI)** across 36 Indian States/UTs and over 900 police administrative districts (2014–2022).

---

## 📌 Repository Contents
* 📓 **[Data_Science_Complete_Curriculum_Project.ipynb](Data_Science_Complete_Curriculum_Project.ipynb)**: Complete, fully runnable Master Jupyter Notebook containing all 5 syllabus modules, complete code, statistical tests, visualizations, and the Capstone Machine Learning project.
* 📑 **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: Formal Academic & Executive Project Report with theoretical background, mathematical formulations, benchmarks, and policy findings.
* 📖 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**: Curriculum mapping, syllabus guide, and presentation talking points.
* 📊 **Data Files**: Verified longitudinal CSV datasets covering national, state, and district-level statistics, judicial outcomes, survey benchmarks (NFHS-5), and data dictionaries.
* ⚙️ **[build_notebook.py](build_notebook.py)**: Automated generator script for regenerating and validating the Master Notebook.

---

## 📚 Curriculum Structure & Modules

```
┌────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Tools, Environment & Python Basics                          │
│  - Jupyter Notebook shortcuts, lifecycle, variables, data structures  │
│  - Functions, lambdas, comprehensions, NumPy & Pandas foundations     │
├────────────────────────────────────────────────────────────────────────┤
│  MODULE 2: Data Acquisition & Cleaning                                 │
│  - Multi-source ingestion: CSV, JSON, SQLite relational database       │
│  - Web scraping & REST API integration patterns                       │
│  - Data auditing, missing value handling, Tukey's IQR & Z-score bounds │
├────────────────────────────────────────────────────────────────────────┤
│  MODULE 3: Visualization & Exploratory Data Analysis (EDA)            │
│  - Matplotlib & Seaborn aesthetics, KDE, boxplots, skewness/kurtosis   │
│  - Longitudinal trajectories (2014-2022), Top 10 states, heatmaps      │
│  - Pearson vs Spearman correlation, Zero-reporting audit (10.02%)      │
├────────────────────────────────────────────────────────────────────────┤
│  MODULE 4: Statistical & Algorithmic Data Modeling                     │
│  - Central Limit Theorem empirical simulation (n = 5, 30, 100)         │
│  - Hypothesis testing: Welch's t-test, Mann-Whitney U, Chi-Square      │
│  - Classification (Logistic Regression, Random Forest, ROC-AUC = 0.94) │
│  - Regression (Linear Regression, Random Forest Regressor, R², RMSE)   │
│  - Unsupervised Learning: K-Means District Clustering (k = 3)         │
│  - 5-Fold Stratified Cross-Validation & GridSearchCV hyperparameter    │
├────────────────────────────────────────────────────────────────────────┤
│  MODULE 5: Mini-Project Development & Executive Presentation           │
│  - Case Study: Predictive Judicial Clearance & High-Burden Risk        │
│  - Scikit-Learn Pipeline (Pipeline + ColumnTransformer + Imputer)      │
│  - 5-Model Benchmark (Dummy, Logistic, Tree, Random Forest, GBDT)      │
│  - Gini feature importance & National Judicial Funnel modeling         │
│  - Executive presentation, policy recommendations & future scope       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Naksh-5/Data_Sci_MP.git
cd Data_Sci_MP
```

### 2. Set Up Python Environment
Ensure Python 3.12+ is installed, then install the dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy ipykernel
```

### 3. Launch the Jupyter Notebook
```bash
python -m jupyter notebook Data_Science_Complete_Curriculum_Project.ipynb
```
*(Or open the repository in **VS Code**, select your Python 3.12 kernel in the top-right, and click **Run All**.)*

---

## 🔬 Key Empirical Findings

1. **High Caseload Concentration**:
   * A small minority (~15%) of metropolitan and urban commissionerates generate over 40% of all registered FIRs.
   * Prior year caseload (`lag_cases_1yr`) and 2-year rolling average are the strongest predictors (>70% Gini importance) of high-burden status.
2. **The "Zero-Reporting" Paradox**:
   * Approximately 10% of district-year observations report exactly 0 cases.
   * In criminological data science, a zero count frequently reflects structural reporting barriers, lack of police access, or refusal to register complaints rather than absence of crime.
3. **Judicial Funnel Attrition**:
   * While police chargesheeting rates are relatively high (~70–80%), court trial completion rates create the primary bottleneck with pendency exceeding 85%.

---

## 📜 Citation & Acknowledgments
* **NCRB (National Crime Records Bureau)**: Crime in India annual reports (2014–2022).
* **CHRI (Commonwealth Human Rights Initiative)**: Rape Statistics in India Analytical Reports (2024).
* **IIPS**: National Family Health Survey (NFHS-5) representative sample findings.
