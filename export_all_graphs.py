import os
import io
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for clean image saving
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, r2_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Create output folder
output_dir = os.path.join(os.getcwd(), "graphs")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory for graphs: {output_dir}")

# Global Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300  # High-resolution publication quality

# ==============================================================================
# LOAD & PREPARE DATA
# ==============================================================================
df_district_raw = pd.read_csv("crime_statistics_district_annual.csv")
df_state_raw = pd.read_csv("crime_statistics_state_annual.csv")
df_outcomes_raw = pd.read_csv("investigation_judicial_outcomes.csv")
df_caste_raw = pd.read_csv("caste_profile_cases_scst.csv")

# Clean district data
df_district = df_district_raw.copy()
df_district.columns = [col.strip().lower() for col in df_district.columns]
df_district["state_ut"] = df_district["state_ut"].astype(str).str.strip().str.upper()
df_district["district"] = df_district["district"].astype(str).str.strip().str.title()
df_district = df_district[~df_district["state_ut"].isin(["NAN", "NONE", ""])]
df_district = df_district[~df_district["district"].isin(["Nan", "None", ""])]
df_district["cases_registered"] = pd.to_numeric(df_district["cases_registered"], errors="coerce").fillna(0).astype(int)
df_district["log_cases"] = np.log1p(df_district["cases_registered"])

# Clean state data
df_state = df_state_raw.copy()
df_state["cases_registered"] = pd.to_numeric(df_state["cases_registered"], errors="coerce").fillna(0)

# Clean caste data
df_caste = df_caste_raw.copy()
df_caste = df_caste[df_caste["year"] != "Total 2014-2022"]
df_caste["year"] = pd.to_numeric(df_caste["year"])
df_caste["total_rape_cases"] = pd.to_numeric(df_caste["total_rape_cases"])
df_caste["rape_cases_under_scst_poa_act"] = pd.to_numeric(df_caste["rape_cases_under_scst_poa_act"])
df_caste["pct_scst_of_total_cases"] = pd.to_numeric(df_caste["pct_scst_of_total_cases"])

# Feature engineering for ML
df_ml = df_district.sort_values(by=["district", "year"]).copy()
df_ml["lag_cases_1yr"] = df_ml.groupby("district")["cases_registered"].shift(1)
df_ml["rolling_2yr_mean"] = df_ml.groupby("district")["cases_registered"].shift(1).rolling(2, min_periods=1).mean()
state_historical_mean = df_ml.groupby("state_ut")["cases_registered"].transform("mean")
df_ml["state_avg_load"] = state_historical_mean
feature_cols = ["lag_cases_1yr", "rolling_2yr_mean", "state_avg_load", "year"]
df_ml.dropna(subset=feature_cols + ["cases_registered"], inplace=True)
threshold_75 = df_ml["cases_registered"].quantile(0.75)
df_ml["target_high_burden"] = (df_ml["cases_registered"] >= threshold_75).astype(int)

X = df_ml[feature_cols].copy()
y_clf = df_ml["target_high_burden"].copy()
y_reg = df_ml["cases_registered"].copy()

print("Data loaded and preprocessed successfully.")

# ==============================================================================
# GRAPH 1: UNIVARIATE CASELOAD DISTRIBUTIONS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_district["cases_registered"], kde=True, ax=axes[0], color="#2b5c8f", bins=50)
axes[0].set_title("District Caseload Distribution (Raw)")
axes[0].set_xlabel("Cases Registered")
axes[0].set_ylabel("Frequency")

sns.boxplot(y=df_district["cases_registered"], ax=axes[1], color="#709775", showmeans=True)
axes[1].set_title("Caseload Dispersion & Outliers (Boxplot)")
axes[1].set_ylabel("Cases Registered")

sns.histplot(df_district["log_cases"], kde=True, ax=axes[2], color="#c85a17", bins=30)
axes[2].set_title("Log1p-Transformed Distribution (Normalized)")
axes[2].set_xlabel("log(1 + Cases Registered)")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
p1 = os.path.join(output_dir, "01_univariate_caseload_distributions.png")
plt.savefig(p1, bbox_inches="tight")
plt.close()
print(f"Saved: {p1}")

# ==============================================================================
# GRAPH 2: NATIONAL TREND & TOP 10 STATES
# ==============================================================================
annual_national = df_state.groupby("year")["cases_registered"].sum().reset_index()
top10_states = (
    df_state.groupby("state_ut")["cases_registered"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
axes[0].plot(annual_national["year"], annual_national["cases_registered"], marker="o", linewidth=2.8, color="#1b4965")
axes[0].set_title("National Registered Cases Trajectory (2014-2022)", fontweight="bold")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Total Cases Registered")
axes[0].set_xticks(annual_national["year"])
for _, row in annual_national.iterrows():
    axes[0].annotate(f"{int(row['cases_registered']):,}", 
                     (row['year'], row['cases_registered']), 
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

sns.barplot(data=top10_states, x="cases_registered", y="state_ut", ax=axes[1], hue="state_ut", palette="Blues_r", legend=False)
axes[1].set_title("Top 10 States by Cumulative Registered Volume (2014-2022)", fontweight="bold")
axes[1].set_xlabel("Cumulative Cases Registered")
axes[1].set_ylabel("State / UT")
for i, v in enumerate(top10_states["cases_registered"]):
    axes[1].text(v + 500, i, f" {int(v):,}", va='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
p2 = os.path.join(output_dir, "02_national_trend_and_top_states.png")
plt.savefig(p2, bbox_inches="tight")
plt.close()
print(f"Saved: {p2}")

# ==============================================================================
# GRAPH 3: MULTIVARIATE HEATMAP & SC/ST PROPORTION TREND
# ==============================================================================
top7_state_names = top10_states["state_ut"].head(7).tolist()
df_top_states = df_state[df_state["state_ut"].isin(top7_state_names)]
state_pivot = df_top_states.pivot(index="state_ut", columns="year", values="cases_registered")

fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
sns.heatmap(state_pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[0], cbar_kws={'label': 'Cases Registered'})
axes[0].set_title("Annual Caseload Matrix for Top States (2014-2022)", fontweight="bold")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("State / UT")

axes[1].plot(df_caste["year"], df_caste["pct_scst_of_total_cases"], marker="s", color="#b7094c", linewidth=2.8, label="SC/ST Share (%)")
axes[1].set_title("SC/ST Share of Total Registered Cases (2014-2022)", fontweight="bold")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Share of Registered Cases (%)")
axes[1].set_xticks(df_caste["year"])
axes[1].legend(loc="lower right")

for _, row in df_caste.iterrows():
    axes[1].annotate(f"{row['pct_scst_of_total_cases']:.1f}%", 
                     (row['year'], row['pct_scst_of_total_cases']), 
                     textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
p3 = os.path.join(output_dir, "03_state_caseload_heatmap_and_scst_share.png")
plt.savefig(p3, bbox_inches="tight")
plt.close()
print(f"Saved: {p3}")

# ==============================================================================
# GRAPH 4: CENTRAL LIMIT THEOREM (CLT) SIMULATION
# ==============================================================================
pop_cases = df_district["cases_registered"].values
pop_mean = np.mean(pop_cases)
pop_std = np.std(pop_cases)
sample_sizes = [5, 30, 100]
n_experiments = 1000

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for idx, n in enumerate(sample_sizes):
    sample_means = [np.mean(np.random.choice(pop_cases, size=n, replace=True)) for _ in range(n_experiments)]
    sns.histplot(sample_means, kde=True, ax=axes[idx], color="#0077b6", stat="density", bins=30)
    x_axis = np.linspace(min(sample_means), max(sample_means), 100)
    theory_se = pop_std / np.sqrt(n)
    axes[idx].plot(x_axis, stats.norm.pdf(x_axis, pop_mean, theory_se), 'r--', linewidth=2, label=f"Normal N({pop_mean:.1f}, {theory_se:.1f})")
    axes[idx].set_title(f"Sample Size n = {n} ({n_experiments} iterations)", fontweight="bold")
    axes[idx].set_xlabel("Sample Mean")
    axes[idx].legend()

plt.suptitle(f"Central Limit Theorem Convergence (Population Mean = {pop_mean:.2f})", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
p4 = os.path.join(output_dir, "04_central_limit_theorem_simulation.png")
plt.savefig(p4, bbox_inches="tight")
plt.close()
print(f"Saved: {p4}")

# ==============================================================================
# GRAPH 5: CLASSIFICATION CONFUSION MATRICES & ROC-AUC CURVES
# ==============================================================================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(X_train_c_scaled, y_train_c)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf_rf.fit(X_train_c, y_train_c)

y_pred_lr = clf_lr.predict(X_test_c_scaled)
y_prob_lr = clf_lr.predict_proba(X_test_c_scaled)[:, 1]
y_pred_rf = clf_rf.predict(X_test_c)
y_prob_rf = clf_rf.predict_proba(X_test_c)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cm_lr = confusion_matrix(y_test_c, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title(f"Logistic Regression (F1: {f1_score(y_test_c, y_pred_lr):.3f})", fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

cm_rf = confusion_matrix(y_test_c, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
axes[1].set_title(f"Random Forest (F1: {f1_score(y_test_c, y_pred_rf):.3f})", fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

fpr_lr, tpr_lr, _ = roc_curve(y_test_c, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test_c, y_prob_rf)
auc_lr = roc_auc_score(y_test_c, y_prob_lr)
auc_rf = roc_auc_score(y_test_c, y_prob_rf)

axes[2].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="blue", linewidth=2)
axes[2].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", color="green", linewidth=2.5)
axes[2].plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")
axes[2].set_title("ROC Curves Comparison", fontweight="bold")
axes[2].set_xlabel("False Positive Rate (1 - Specificity)")
axes[2].set_ylabel("True Positive Rate (Recall)")
axes[2].legend(loc="lower right")

plt.tight_layout()
p5 = os.path.join(output_dir, "05_classification_confusion_matrices_and_roc.png")
plt.savefig(p5, bbox_inches="tight")
plt.close()
print(f"Saved: {p5}")

# ==============================================================================
# GRAPH 6: REGRESSION ACTUAL VS PREDICTED
# ==============================================================================
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)
reg_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
reg_rf.fit(X_train_r, y_train_r)
y_pred_reg_rf = reg_rf.predict(X_test_r)

plt.figure(figsize=(8.5, 6))
plt.scatter(y_test_r, y_pred_reg_rf, alpha=0.35, color="#1d3557", edgecolors="none", label="District Caseloads")
max_val = max(y_test_r.max(), y_pred_reg_rf.max())
plt.plot([0, max_val], [0, max_val], "r--", linewidth=2.2, label="Ideal Fit (y = x)")
plt.title(f"Random Forest Regressor: Actual vs Predicted Caseloads (R² = {r2_score(y_test_r, y_pred_reg_rf):.3f})", fontweight="bold")
plt.xlabel("Actual Registered Cases")
plt.ylabel("Predicted Registered Cases")
plt.legend(loc="upper left")
plt.tight_layout()
p6 = os.path.join(output_dir, "06_regression_actual_vs_predicted.png")
plt.savefig(p6, bbox_inches="tight")
plt.close()
print(f"Saved: {p6}")

# ==============================================================================
# GRAPH 7: K-MEANS DISTRICT CLUSTERING
# ==============================================================================
district_profiles = df_district.groupby("district").agg(
    mean_cases=("cases_registered", "mean"),
    std_cases=("cases_registered", "std"),
    max_cases=("cases_registered", "max")
).dropna()

scaler_k = StandardScaler()
X_cluster = scaler_k.fit_transform(district_profiles)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
district_profiles["cluster"] = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(9.5, 6))
scatter = plt.scatter(
    district_profiles["mean_cases"], 
    district_profiles["std_cases"], 
    c=district_profiles["cluster"], 
    cmap="viridis", 
    alpha=0.65,
    edgecolors="k", 
    s=55
)
plt.title("District Caseload Risk Stratification (K-Means, k=3)", fontweight="bold")
plt.xlabel("Mean Annual Registered Cases")
plt.ylabel("Caseload Standard Deviation (Volatility)")
cbar = plt.colorbar(scatter)
cbar.set_label("Cluster Priority Tier", rotation=270, labelpad=15)
plt.tight_layout()
p7 = os.path.join(output_dir, "07_kmeans_district_risk_clusters.png")
plt.savefig(p7, bbox_inches="tight")
plt.close()
print(f"Saved: {p7}")

# ==============================================================================
# GRAPH 8: GINI FEATURE IMPORTANCE (RANDOM FOREST)
# ==============================================================================
features_numeric = ["lag_cases_1yr", "rolling_2yr_mean", "state_avg_load"]
features_categorical = ["year"]
target = "target_high_burden"
df_project = df_ml.copy()
df_project.dropna(subset=features_numeric + features_categorical + [target], inplace=True)
X_proj = df_project[features_numeric + features_categorical]
y_proj = df_project[target]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_proj, y_proj, test_size=0.20, random_state=42, stratify=y_proj
)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, features_numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), features_categorical)
    ]
)
best_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
pipe_best = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", best_rf)
])
pipe_best.fit(X_train_p, y_train_p)

raw_feature_names = pipe_best.named_steps["preprocessor"].get_feature_names_out()
clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in raw_feature_names]
importances = pipe_best.named_steps["classifier"].feature_importances_

df_feat_imp = pd.DataFrame({
    "Feature": clean_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(9.5, 5))
bars = plt.barh(df_feat_imp["Feature"], df_feat_imp["Importance"], color="#2a9d8f")
plt.title("Random Forest Feature Importance Analysis (Gini Impurity)", fontweight="bold")
plt.xlabel("Gini Importance Score")
plt.ylabel("Predictor Variable")
for bar, v in zip(bars, df_feat_imp["Importance"]):
    plt.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%", va='center', fontsize=9.5, fontweight='bold')
plt.xlim(0, max(df_feat_imp["Importance"]) * 1.15)
plt.tight_layout()
p8 = os.path.join(output_dir, "08_feature_importance_random_forest.png")
plt.savefig(p8, bbox_inches="tight")
plt.close()
print(f"Saved: {p8}")

# ==============================================================================
# GRAPH 9: NATIONAL JUDICIAL FUNNEL (2022 NCRB DATA)
# ==============================================================================
df_outcomes = df_outcomes_raw[df_outcomes_raw["crime_category"] == "Rape (IPC 376)"].copy()
outcomes_2022 = df_outcomes[df_outcomes["year"] == 2022].set_index("indicator")["value"].to_dict()

funnel_stages = [
    "Cases requiring investigation (new)",
    "Total cases for investigation during year",
    "Cases where chargesheet was submitted",
    "Cases where trial was completed",
    "Cases ending in conviction"
]

stage_labels = [
    "New Cases Registered (FIRs)",
    "Total Cases for Investigation (w/ Backlog)",
    "Chargesheets Submitted by Police",
    "Trials Completed by Courts",
    "Cases Ending in Conviction"
]

funnel_data = []
for stage, label in zip(funnel_stages, stage_labels):
    if stage in outcomes_2022:
        funnel_data.append({"Stage": label, "Cases": outcomes_2022[stage]})

df_funnel = pd.DataFrame(funnel_data)

plt.figure(figsize=(11, 5.5))
bars = plt.barh(df_funnel["Stage"][::-1], df_funnel["Cases"][::-1], color="#457b9d")
plt.title("National Judicial Funnel & Case Attrition (2022 NCRB Returns)", fontweight="bold")
plt.xlabel("Number of Cases")
for bar in bars:
    w = bar.get_width()
    plt.text(w + 500, bar.get_y() + bar.get_height()/2, f" {int(w):,} cases", va='center', fontsize=9.5, fontweight='bold')
plt.xlim(0, max(df_funnel["Cases"]) * 1.18)
plt.tight_layout()
p9 = os.path.join(output_dir, "09_national_judicial_funnel_2022.png")
plt.savefig(p9, bbox_inches="tight")
plt.close()
print(f"Saved: {p9}")

# ==============================================================================
# GRAPH 10: ZERO-REPORTING TEMPORAL SHARE
# ==============================================================================
zero_by_year = df_district[df_district["cases_registered"] == 0].groupby("year")["district"].count()
total_by_year = df_district.groupby("year")["district"].count()
zero_share_year = (zero_by_year / total_by_year * 100).round(2).reset_index()
zero_share_year.columns = ["year", "zero_pct"]

plt.figure(figsize=(9, 4.5))
plt.plot(zero_share_year["year"], zero_share_year["zero_pct"], marker="o", color="#d90429", linewidth=2.5)
plt.title("Zero-Reporting Districts Share Over Time (2015-2022)", fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Districts Reporting 0 Cases (%)")
plt.xticks(zero_share_year["year"])
for _, row in zero_share_year.iterrows():
    plt.annotate(f"{row['zero_pct']:.1f}%", 
                 (row['year'], row['zero_pct']), 
                 textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9.5, fontweight='bold')
plt.tight_layout()
p10 = os.path.join(output_dir, "10_zero_reporting_districts_trend.png")
plt.savefig(p10, bbox_inches="tight")
plt.close()
print(f"Saved: {p10}")

print("=" * 60)
print(f"[SUCCESS] All 10 high-resolution graph photos generated and saved to:")
print(f"          {output_dir}")
print("=" * 60)
