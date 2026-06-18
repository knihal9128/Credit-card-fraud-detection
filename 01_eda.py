"""
01_eda.py
---------
Exploratory Data Analysis on the transactions dataset.
Run this after generate_data.py.

Goal: understand the data, spot patterns that separate fraud from
legit transactions, and decide what features matter before modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("transactions.csv")

print("=" * 60)
print("BASIC INFO")
print("=" * 60)
print(df.info())
print()
print(df.describe())

print()
print("=" * 60)
print("CLASS BALANCE")
print("=" * 60)
print(df["is_fraud"].value_counts())
print(df["is_fraud"].value_counts(normalize=True) * 100)

print()
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print()
print("=" * 60)
print("MEAN FEATURE VALUES: FRAUD VS LEGIT")
print("=" * 60)
numeric_cols = ["transaction_amount", "transaction_hour", "customer_age",
                 "account_age_days", "distance_from_home_km",
                 "num_transactions_last_24h"]
print(df.groupby("is_fraud")[numeric_cols].mean().round(2))

# ---- Visualizations ----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.boxplot(data=df, x="is_fraud", y="transaction_amount", ax=axes[0, 0])
axes[0, 0].set_title("Transaction Amount by Fraud Label")

sns.histplot(data=df, x="transaction_hour", hue="is_fraud", multiple="dodge",
             bins=24, ax=axes[0, 1], stat="density", common_norm=False)
axes[0, 1].set_title("Hour of Day Distribution (normalised)")

sns.boxplot(data=df, x="is_fraud", y="distance_from_home_km", ax=axes[0, 2])
axes[0, 2].set_title("Distance From Home by Fraud Label")
axes[0, 2].set_ylim(0, 500)

sns.barplot(data=df, x="is_fraud", y="is_online", ax=axes[1, 0], errorbar=None)
axes[1, 0].set_title("Share of Online Transactions")

sns.barplot(data=df, x="is_fraud", y="card_present", ax=axes[1, 1], errorbar=None)
axes[1, 1].set_title("Share of Card-Present Transactions")

sns.boxplot(data=df, x="is_fraud", y="num_transactions_last_24h", ax=axes[1, 2])
axes[1, 2].set_title("Transactions in Last 24h by Fraud Label")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
print("\nSaved chart -> eda_overview.png")

# Merchant category breakdown
plt.figure(figsize=(8, 5))
fraud_rate_by_merchant = df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False)
fraud_rate_by_merchant.plot(kind="bar", color="#1A5276")
plt.ylabel("Fraud rate")
plt.title("Fraud Rate by Merchant Category")
plt.tight_layout()
plt.savefig("fraud_by_merchant.png", dpi=120)
print("Saved chart -> fraud_by_merchant.png")

print()
print("=" * 60)
print("FRAUD RATE BY MERCHANT CATEGORY")
print("=" * 60)
print((fraud_rate_by_merchant * 100).round(2).astype(str) + "%")
