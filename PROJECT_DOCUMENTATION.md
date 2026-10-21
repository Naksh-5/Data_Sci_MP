# Data Science Curriculum & Mini-Project Capstone Guide

## **Overview**
This repository contains a curriculum implementation and end-to-end Capstone Mini-Project constructed in a single Jupyter Notebook:
`Data_Science_Complete_Curriculum_Project.ipynb`

The project analyzes official datasets from the **National Crime Records Bureau (NCRB)** and research from the **Commonwealth Human Rights Initiative (CHRI)** across 36 States/UTs and over 900 police administrative districts from 2014 to 2022.

---

## **Syllabus Mapping to Notebook Modules**

| Syllabus Module | Sub-Topics Covered | Implementation in Notebook |
| :--- | :--- | :--- |
| **Module 1: Data Science Tools, Environment & Python Basics** | • Jupyter Notebook environment & keyboard shortcuts<br>• Introduction to Data Science lifecycle<br>• Python basics: types, control flow, functions, lambdas<br>• Core data structures: lists, tuples, dicts, sets<br>• NumPy arrays, vectorization, Pandas DataFrames | **Section 1 (Cells 1–5)**:<br>• System runtime diagnostics<br>• Python programming constructs & list/dict comprehensions<br>• Custom mathematical functions & lambda expressions<br>• NumPy array broadcasting & matrix arithmetic<br>• Pandas tabular indexing & slicing |
| **Module 2: Data Acquisition & Cleaning** | • Multi-source ingestion: CSV, JSON, Relational SQL<br>• Web scraping & REST API integration patterns<br>• Data auditing, type casting, missing value imputation<br>• Outlier detection: Tukey's IQR and Z-Score methods | **Section 2 (Cells 6–10)**:<br>• Ingesting 5 CSV datasets & round-trip JSON serialization<br>• In-memory SQLite database querying with aggregate SQL<br>• REST API call simulation & HTML table scraping via Regex<br>• Data cleaning, casing standardization, null handling<br>• IQR & Z-score outlier detection and 99th percentile Winsorization |
| **Module 3: Data Visualization & Exploratory Data Analysis (EDA)** | • Matplotlib & Seaborn visualization setups<br>• Univariate analysis: histograms, KDE, boxplots, skewness<br>• Longitudinal trajectories & state rankings<br>• Multivariate heatmaps, SC/ST demographic shares<br>• Pearson vs Spearman correlation matrices<br>• Domain insight: The Zero-Reporting Paradox | **Section 3 (Cells 11–15)**:<br>• Publication-grade plotting styles<br>• Skewness & kurtosis diagnostics + Log1p normalizations<br>• National time-series trajectory (2014–2022)<br>• Top 10 state rankings bar charts & annual state heatmaps<br>• Pearson ($r$) vs Spearman ($\rho$) correlation analysis<br>• Institutional audit of zero-reporting district years (10.02%) |
| **Module 4: Statistical & Algorithmic Data Modeling** | • Probability distributions & Central Limit Theorem (CLT)<br>• Hypothesis testing: Welch's t-test, Mann-Whitney U, $\chi^2$<br>• Scikit-Learn Supervised Classification & Regression<br>• Model metrics: Confusion Matrix, Precision, Recall, F1, ROC-AUC, MAE, RMSE, $R^2$<br>• Unsupervised Learning: K-Means District Clustering<br>• 5-Fold Stratified CV & `GridSearchCV` tuning | **Section 4 (Cells 16–21)**:<br>• Simulating CLT with $n \in \{5, 30, 100\}$ iterations<br>• Formal inferential hypothesis testing ($p < 0.05$ decision criteria)<br>• Lag feature engineering & moving window aggregations<br>• Logistic Regression vs Random Forest classifier (F1: 0.88, ROC-AUC: 0.94)<br>• Linear Regression vs Random Forest regressor ($R^2$, MAE, RMSE)<br>• K-Means clustering ($k=3$) for geographic risk tiering<br>• Hyperparameter optimization via `GridSearchCV` |
| **Module 5: Mini-Project Development & Presentation** | • Capstone formulation: Predictive Judicial Analytics<br>• Production ML pipeline (`Pipeline` + `ColumnTransformer`)<br>• 5-Model algorithmic benchmarking<br>• Gini feature importance extraction<br>• Justice system funnel modeling (FIR $\to$ Conviction)<br>• Executive presentation, policy recommendations & future scope | **Section 5 (Cells 22–26)**:<br>• End-to-end `sklearn.pipeline.Pipeline` with `SimpleImputer`, `StandardScaler`, `OneHotEncoder`<br>• Benchmarking Dummy Baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting<br>• Feature importance analysis revealing historical momentum drivers<br>• National Judicial Funnel analysis (quantifying trial pendency attrition)<br>• Executive decision framework & policy recommendations |

---

## **How to Open and Run the Notebook**

### **Option A: Using VS Code**
1. Open the project folder in VS Code (`File > Open Folder... > project`).
2. Click on `Data_Science_Complete_Curriculum_Project.ipynb`.
3. In the top-right corner of the notebook, click **Select Kernel** and choose Python 3.12 (`python.exe`).
4. Click **Run All** or execute cells individually using `Shift + Enter`.

### **Option B: Using Jupyter Notebook / JupyterLab via Terminal**
Open PowerShell in the project directory and run:
```bash
# Launch Jupyter Notebook
python -m jupyter notebook Data_Science_Complete_Curriculum_Project.ipynb
```
or
```bash
# Launch JupyterLab
python -m jupyter lab
```

---

## **Key Project Findings & Presentation Talking Points**

1. **High Caseload Concentration**:
   * A small subset of districts (~15%) generates over 40% of all registered FIRs.
   * Prior year caseload (`lag_cases_1yr`) and 2-year rolling average are the strongest predictors (>70% Gini importance) of high-burden status.

2. **The "Zero-Reporting" Paradox**:
   * Approximately 10% of district-year observations record exactly 0 cases.
   * In criminological data science, a zero count frequently signifies severe reporting barriers, lack of jurisdictional access, or refusal to register complaints rather than absence of crime.
   * *Policy Recommendation*: Autonomous audit triggers for zero-reporting police jurisdictions neighboring high-caseload districts.

3. **Trial Completion Bottleneck**:
   * National judicial funnel modeling reveals that while police chargesheeting rates are relatively high (~70-80%), the judicial trial completion rate is the primary bottleneck with pendency exceeding 85%.
   * *Policy Recommendation*: Prioritize Fast-Track Special Courts (FTSCs) in districts classified in K-Means Cluster 2 (high volume, high volatility).
