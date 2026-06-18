"""
02_model_training.py
---------------------
Feature engineering + model training + evaluation for fraud detection.

Key ML concepts demonstrated here (good interview talking points):
1. Train/test split with stratification (preserves class ratio)
2. Handling severe class imbalance via class_weight (no SMOTE library
   available, so this uses scikit-learn's built-in weighting instead --
   a perfectly standard and often preferred approach in practice)
3. Comparing a simple baseline (Logistic Regression) against a more
   powerful model (Random Forest)
4. Why accuracy is a misleading metric for imbalanced problems, and
   what to use instead (precision, recall, F1, ROC-AUC, PR-AUC)
5. Feature importance / interpreting the model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load + feature engineering
# ---------------------------------------------------------------
df = pd.read_csv("transactions.csv")

# One-hot encode merchant_category (a categorical variable)
df_model = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)

feature_cols = [c for c in df_model.columns
                 if c not in ("transaction_id", "is_fraud")]
X = df_model[feature_cols]
y = df_model["is_fraud"]

print(f"Features used ({len(feature_cols)}):")
print(feature_cols)

# ---------------------------------------------------------------
# 2. Train/test split (stratified so both sets keep ~1.2% fraud rate)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain size: {len(X_train)}  Fraud in train: {y_train.sum()}")
print(f"Test size:  {len(X_test)}  Fraud in test:  {y_test.sum()}")

# Scale numeric features (helps Logistic Regression converge properly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Baseline model: Logistic Regression with class_weight='balanced'
#    This upweights the rare fraud class during training instead of
#    needing to artificially resample the data (a SMOTE alternative).
# ---------------------------------------------------------------
log_reg = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)
print(classification_report(y_test, y_pred_lr, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_lr), 4))
print("PR-AUC (average precision):", round(average_precision_score(y_test, y_proba_lr), 4))

# ---------------------------------------------------------------
# 4. Stronger model: Random Forest with class_weight='balanced'
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)  # tree models don't need scaling
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)
print(classification_report(y_test, y_pred_rf, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_rf), 4))
print("PR-AUC (average precision):", round(average_precision_score(y_test, y_proba_rf), 4))

# ---------------------------------------------------------------
# 5. Confusion matrices
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CONFUSION MATRICES (rows=actual, cols=predicted)")
print("=" * 60)
print("Logistic Regression:\n", confusion_matrix(y_test, y_pred_lr))
print("Random Forest:\n", confusion_matrix(y_test, y_pred_rf))

# ---------------------------------------------------------------
# 6. Visualization: ROC curves + Precision-Recall curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, proba in [("Logistic Regression", y_proba_lr), ("Random Forest", y_proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

for name, proba in [("Logistic Regression", y_proba_lr), ("Random Forest", y_proba_rf)]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=120)
print("\nSaved chart -> model_evaluation.png")

# ---------------------------------------------------------------
# 7. Feature importance (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCES (Random Forest)")
print("=" * 60)
print(importances.head(10))

plt.figure(figsize=(8, 5))
importances.head(10).sort_values().plot(kind="barh", color="#1A5276")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
print("Saved chart -> feature_importance.png")
