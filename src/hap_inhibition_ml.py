"""
MM3330 Biomaterials ML Project
Regression-based ML of tumour-cell inhibition rate of hydroxyapatite nanoparticles
Models: Random Forest, K-Nearest Neighbours, XGBoost
Cross-validation: Repeated K-Fold (5 folds x 10 repeats) + Leave-One-Out for comparison

Run from the repository root:  python src/hap_inhibition_ml.py
Figures go to outputs/figures/, tables to outputs/tables/
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RepeatedKFold, KFold, LeaveOneOut,
                                     cross_validate, cross_val_predict, GridSearchCV)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="HAp nanoparticle inhibition-rate regression")
parser.add_argument("--data", default=ROOT / "data" / "hap_inhibition_dataset.csv", help="input CSV")
parser.add_argument("--out", default=ROOT / "outputs", help="output folder")
args = parser.parse_args()

SEED = 42
FIG = Path(args.out) / "figures"
TAB = Path(args.out) / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)

# ------------------------------------------------------------------
# (i) Data understanding
# ------------------------------------------------------------------
df = pd.read_csv(args.data)
df.columns = ["Diameter_nm", "Time_days", "Concentration", "Inhibition_pct"]
FEATURES = ["Diameter_nm", "Time_days", "Concentration"]
TARGET = "Inhibition_pct"
LABELS = {"Diameter_nm": "Sphere diameter (nm)",
          "Time_days": "Inhibition time (days)",
          "Concentration": "Particle concentration",
          "Inhibition_pct": "Inhibition rate (%)"}

print("Shape:", df.shape)
print("Missing values per column:\n", df.isna().sum())
print("Unique levels per input:")
for f in FEATURES:
    print(f"  {f}: {sorted(df[f].unique().tolist())}")

summary = df.describe().T
summary["median"] = df.median()
summary["skewness"] = df.skew()
summary = summary[["count", "mean", "std", "min", "25%", "median", "75%", "max", "skewness"]]
print("\nSummary statistics:\n", summary.round(2))
summary.round(3).to_csv(TAB / "table1_summary_statistics.csv")

corr = df.corr(method="pearson")
print("\nPearson correlation:\n", corr.round(3))
corr.round(3).to_csv(TAB / "table2_correlation.csv")

plt.figure(figsize=(6.5, 5))
sns.heatmap(corr.rename(index=LABELS, columns=LABELS), annot=True, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
plt.title("Pearson correlation matrix")
plt.tight_layout(); plt.savefig(FIG / "fig0_correlation_heatmap.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# (ii) Pairwise scatterplots
# ------------------------------------------------------------------
# (a) full pair plot, coloured by diameter
g = sns.pairplot(df.rename(columns=LABELS), hue=LABELS["Diameter_nm"],
                 palette="viridis", diag_kind="hist", plot_kws={"s": 45, "alpha": .8})
g.figure.suptitle("Pairwise scatterplot matrix (coloured by sphere diameter)", y=1.02)
g.savefig(FIG / "fig1_pairplot.png", dpi=300); plt.close()

# (b) each input vs target, with jitter (inputs are discrete levels)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
rng = np.random.default_rng(SEED)
for ax, f in zip(axes, FEATURES):
    span = df[f].max() - df[f].min()
    x = df[f] + rng.uniform(-0.015, 0.015, len(df)) * span
    sc = ax.scatter(x, df[TARGET], c=df["Concentration"] if f != "Concentration" else df["Diameter_nm"],
                    cmap="plasma", s=50, edgecolor="k", linewidth=.4)
    means = df.groupby(f)[TARGET].mean()
    ax.plot(means.index, means.values, "k--o", lw=1.5, ms=5, label="group mean")
    ax.set_xlabel(LABELS[f]); ax.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Concentration" if f != "Concentration" else "Diameter (nm)", fontsize=9)
axes[0].set_ylabel(LABELS[TARGET])
fig.suptitle("Inputs vs inhibition rate")
plt.tight_layout(); plt.savefig(FIG / "fig2_inputs_vs_target.png", dpi=300); plt.close()

# (c) interaction view: time-course per diameter & concentration
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, sorted(df["Diameter_nm"].unique())):
    sub = df[df["Diameter_nm"] == d]
    for c, col in zip(sorted(sub["Concentration"].unique()), ["#4C72B0", "#DD8452", "#C44E52"]):
        s = sub[sub["Concentration"] == c].sort_values("Time_days")
        ax.plot(s["Time_days"], s[TARGET], "-o", color=col, label=f"conc. {c}")
    ax.set_title(f"Diameter = {d} nm"); ax.set_xlabel("Inhibition time (days)")
axes[0].set_ylabel(LABELS[TARGET]); axes[0].legend(fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "fig3_time_course_interactions.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# (iii) Regression models + cross-validation
# ------------------------------------------------------------------
X = df[FEATURES].values
y = df[TARGET].values

# Light hyper-parameter tuning with an inner 5-fold CV (kept small: n = 63)
inner = KFold(5, shuffle=True, random_state=SEED)
grids = {
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [300], "max_depth": [None, 6],
                       "min_samples_leaf": [1, 2, 3], "max_features": [1.0, 0.67]}),
    "KNN": (Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]),
            {"knn__n_neighbors": [2, 3, 4, 5, 7, 9], "knn__weights": ["uniform", "distance"],
             "knn__p": [1, 2]}),
    "XGBoost": (XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=1),
                {"n_estimators": [200, 500], "learning_rate": [0.03, 0.1],
                 "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]}),
}
best_models = {}
for name, (est, grid) in grids.items():
    gs = GridSearchCV(est, grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X, y)
    best_models[name] = gs.best_estimator_
    print(f"{name} best params: {gs.best_params_}")
pd.Series({k: str(v.get_params()) for k, v in best_models.items()}).to_csv(TAB / "model_params.csv")

# Outer CV 1: Repeated 5-fold (10 repeats = 50 train/test splits)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
scoring = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}
rows, fold_scores = [], {}
for name, model in best_models.items():
    cv = cross_validate(model, X, y, cv=rkf, scoring=scoring, return_train_score=True, n_jobs=-1)
    r2, rmse, mae = cv["test_R2"], -cv["test_RMSE"], -cv["test_MAE"]
    fold_scores[name] = r2
    rows.append({"Model": name,
                 "R2 (mean ± sd)": f"{r2.mean():.3f} ± {r2.std():.3f}",
                 "RMSE % (mean ± sd)": f"{rmse.mean():.2f} ± {rmse.std():.2f}",
                 "MAE % (mean ± sd)": f"{mae.mean():.2f} ± {mae.std():.2f}",
                 "Train R2": f"{cv['train_R2'].mean():.3f}",
                 "R2_mean": r2.mean(), "RMSE_mean": rmse.mean(), "MAE_mean": mae.mean()})
table_rkf = pd.DataFrame(rows)

# Outer CV 2: Leave-One-Out (pooled metrics over 63 held-out predictions)
loo = LeaveOneOut()
loo_rows, oof = [], {}
for name, model in best_models.items():
    pred = cross_val_predict(model, X, y, cv=loo, n_jobs=-1)
    oof[name] = pred
    loo_rows.append({"Model": name,
                     "LOOCV R2": round(r2_score(y, pred), 3),
                     "LOOCV RMSE (%)": round(np.sqrt(mean_squared_error(y, pred)), 2),
                     "LOOCV MAE (%)": round(mean_absolute_error(y, pred), 2)})
table_loo = pd.DataFrame(loo_rows)

comparison = table_rkf.drop(columns=["R2_mean", "RMSE_mean", "MAE_mean"]).merge(table_loo, on="Model")
print("\n=== Model comparison table ===\n", comparison.to_string(index=False))
comparison.to_csv(TAB / "table3_model_comparison.csv", index=False)

# Figure: comparison table rendered as an image (handy for the report)
fig, ax = plt.subplots(figsize=(14, 1.6)); ax.axis("off")
t = ax.table(cellText=comparison.values, colLabels=comparison.columns, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
plt.tight_layout(); plt.savefig(FIG / "fig4_comparison_table.png", dpi=300, bbox_inches="tight"); plt.close()

# Figure: bar chart of metrics + distribution of fold R2
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
pal = {"Random Forest": "#2A9D8F", "KNN": "#E9C46A", "XGBoost": "#E76F51"}
for ax, (m, lab) in zip(axes[:3], [("R2_mean", "R²"), ("RMSE_mean", "RMSE (%)"), ("MAE_mean", "MAE (%)")]):
    ax.bar(table_rkf["Model"], table_rkf[m], color=[pal[k] for k in table_rkf["Model"]], edgecolor="k")
    for i, v in enumerate(table_rkf[m]): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Repeated 5-fold CV: {lab}")
sns.boxplot(data=pd.DataFrame(fold_scores), ax=axes[3], palette=pal)
axes[3].set_title("R² across 50 CV folds"); axes[3].set_ylabel("R²")
plt.tight_layout(); plt.savefig(FIG / "fig5_metric_comparison.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# (iv-a) Predicted vs actual (out-of-fold LOOCV predictions)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
lim = [-5, 85]
for ax, name in zip(axes, best_models):
    p = oof[name]
    sc = ax.scatter(y, p, c=df["Diameter_nm"], cmap="viridis", s=55, edgecolor="k", linewidth=.4)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual inhibition rate (%)"); ax.set_ylabel("Predicted inhibition rate (%)")
    ax.set_title(name)
    ax.text(0.04, 0.96, f"R² = {r2_score(y, p):.3f}\nRMSE = {np.sqrt(mean_squared_error(y, p)):.2f}\n"
            f"MAE = {mean_absolute_error(y, p):.2f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=.85), fontsize=10)
cb = fig.colorbar(sc, ax=axes, pad=0.01); cb.set_label("Sphere diameter (nm)")
fig.suptitle("Predicted vs actual (out-of-fold, Leave-One-Out CV)")
plt.savefig(FIG / "fig6_predicted_vs_actual.png", dpi=300, bbox_inches="tight"); plt.close()

# Residuals
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), sharey=True)
for ax, name in zip(axes, best_models):
    ax.scatter(oof[name], y - oof[name], color=pal[name], edgecolor="k", s=45)
    ax.axhline(0, color="k", ls="--"); ax.set_title(name)
    ax.set_xlabel("Predicted (%)")
axes[0].set_ylabel("Residual = actual - predicted (%)")
plt.tight_layout(); plt.savefig(FIG / "fig7_residuals.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# (iv-b) Feature importance: RF & XGBoost (built-in) + permutation importance (all 3)
# ------------------------------------------------------------------
fitted = {k: v.fit(X, y) for k, v in best_models.items()}
nice = [LABELS[f] for f in FEATURES]
imp = pd.DataFrame({
    "Random Forest (impurity)": fitted["Random Forest"].feature_importances_,
    "XGBoost (gain)": pd.Series(fitted["XGBoost"].get_booster().get_score(importance_type="gain"))
                        .reindex([f"f{i}" for i in range(3)]).fillna(0).values,
}, index=nice)
imp["XGBoost (gain)"] /= imp["XGBoost (gain)"].sum()

perm = {}
for name, m in fitted.items():
    r = permutation_importance(m, X, y, n_repeats=50, random_state=SEED, scoring="r2")
    perm[name] = pd.Series(r.importances_mean, index=nice)
    perm[name + "_sd"] = pd.Series(r.importances_std, index=nice)
perm_df = pd.DataFrame(perm)
imp_all = pd.concat([imp, perm_df], axis=1)
print("\nFeature importances:\n", imp_all.round(3))
imp_all.round(4).to_csv(TAB / "table4_feature_importance.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, col, c in zip(axes, imp.columns, ["#2A9D8F", "#E76F51"]):
    s = imp[col].sort_values()
    ax.barh(s.index, s.values, color=c, edgecolor="k")
    for i, v in enumerate(s.values): ax.text(v, i, f" {v:.3f}", va="center", fontsize=10)
    ax.set_title(col); ax.set_xlabel("Normalised importance"); ax.set_xlim(0, s.max() * 1.25)
fig.suptitle("Model-based feature importance")
plt.tight_layout(); plt.savefig(FIG / "fig8_feature_importance_RF_XGB.png", dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
w, xs = 0.26, np.arange(len(nice))
for i, name in enumerate(fitted):
    ax.bar(xs + (i - 1) * w, perm_df[name], w, yerr=perm_df[name + "_sd"], capsize=3,
           color=pal[name], edgecolor="k", label=name)
ax.set_xticks(xs); ax.set_xticklabels(nice)
ax.set_ylabel("Mean drop in R² when shuffled"); ax.legend()
ax.set_title("Permutation feature importance (model-agnostic)")
plt.tight_layout(); plt.savefig(FIG / "fig9_permutation_importance.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# (iv-c) Response surfaces from the best model: what drives inhibition?
# ------------------------------------------------------------------
best_name = table_rkf.sort_values("RMSE_mean").iloc[0]["Model"]
best = fitted[best_name]
print(f"\nBest model by repeated-CV RMSE: {best_name}")

days = np.linspace(1, 7, 61); concs = np.linspace(140, 560, 61)
D, C = np.meshgrid(days, concs)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
for ax, d in zip(axes, [20, 125, 250]):
    grid = np.c_[np.full(D.size, d), D.ravel(), C.ravel()]
    Z = best.predict(grid).reshape(D.shape)
    cs = ax.contourf(D, C, Z, levels=np.linspace(0, 80, 17), cmap="magma", extend="both")
    ax.set_title(f"{best_name}: diameter = {d} nm"); ax.set_xlabel("Inhibition time (days)")
axes[0].set_ylabel("Particle concentration")
cb = fig.colorbar(cs, ax=axes, pad=0.01); cb.set_label("Predicted inhibition (%)")
plt.savefig(FIG / "fig10_response_surface.png", dpi=300, bbox_inches="tight"); plt.close()

# Partial-dependence style curves (average prediction while varying one input)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
ranges = {"Diameter_nm": np.linspace(20, 250, 47), "Time_days": np.linspace(1, 7, 61),
          "Concentration": np.linspace(140, 560, 61)}
for ax, (j, f) in zip(axes, enumerate(FEATURES)):
    for name, m in fitted.items():
        pdp = []
        for v in ranges[f]:
            Xt = X.copy().astype(float); Xt[:, j] = v; pdp.append(m.predict(Xt).mean())
        ax.plot(ranges[f], pdp, color=pal[name], lw=2, label=name)
    ax.set_xlabel(LABELS[f])
axes[0].set_ylabel("Avg. predicted inhibition (%)"); axes[0].legend(fontsize=9)
fig.suptitle("Partial dependence of inhibition rate on each input")
plt.tight_layout(); plt.savefig(FIG / "fig11_partial_dependence.png", dpi=300); plt.close()

print(f"\nFigures saved to {FIG}\nTables saved to {TAB}")
