
# Credit Card Fraud Detection — Project Walkthrough

**Read this fully before listing the project on your resume or talking about
it in an interview.** This explains *what* the project does and *why* each
decision was made, so you can speak about it confidently and honestly. Run
the scripts yourself, look at the output and the charts, and try changing a
few parameters — that's what will make this genuinely yours.

---

## 1. Why this project, and what problem it solves

Card networks process huge numbers of transactions per second, and only a
tiny fraction (often under 2%) are fraudulent. The challenge is twofold:

1. **Severe class imbalance** — if you just predicted "not fraud" for every
   transaction, you'd be 98%+ "accurate" while catching zero fraud. Accuracy
   is a useless metric here.
2. **Cost asymmetry** — missing a fraud case (a false negative) usually
   costs a lot more than a false alarm (a false positive). The right model
   depends on which mistake the business is more willing to accept.

This project builds a small end-to-end pipeline: generate/explore data,
engineer features, train two different models, and evaluate them with
metrics that are actually meaningful for imbalanced problems.

## 2. The dataset

`generate_data.py` creates a **synthetic** dataset of 20,000 transactions
with a realistic ~1.2% fraud rate. I built it synthetically (rather than
using the well-known Kaggle "Credit Card Fraud" dataset) on purpose: that
dataset's features are anonymised via PCA (`V1`...`V28`), which makes it
impossible to reason about what's actually happening. Here, every column is
a plain, interpretable feature:

- `transaction_amount`, `transaction_hour`, `customer_age`
- `account_age_days`, `distance_from_home_km`
- `is_online`, `card_present`, `num_transactions_last_24h`
- `merchant_category`

Fraudulent transactions are generated with deliberately different
distributions — larger amounts, odd hours, far from the customer's home,
mostly online/card-not-present, and more rapid repeat transactions. This
mirrors real-world fraud patterns and is why the EDA charts show such clean
separation between the two classes.

**Be upfront in an interview that this is a synthetic/simulated dataset
built to mimic real fraud patterns, not real financial data** — that's a
completely normal and honest thing to say, and it shows you understand why
real fraud data is sensitive and hard to access as a student.

## 3. Exploratory Data Analysis (`01_eda.py`)

This step checks data quality (no missing values here, by construction)
and compares fraud vs. legitimate transactions across each feature. Key
findings (see `eda_overview.png` and `fraud_by_merchant.png`):

- Fraudulent transactions are **on average ~4x larger** in amount.
- Fraud is concentrated in **late-night/early-morning hours** (00:00–05:00).
- Fraud transactions happen **much farther from the customer's home**.
- Fraud is **overwhelmingly online and card-not-present**.
- Customers committing/experiencing fraud show **more transactions in the
  last 24 hours** (velocity is a classic fraud signal).
- Fraud rate varies a lot by merchant category — highest in **travel** and
  **electronics**, lowest in **grocery** and **restaurant**.

## 4. Feature engineering & train/test split (`02_model_training.py`)

- `merchant_category` (categorical text) is **one-hot encoded** into
  separate 0/1 columns since models need numeric input.
- Data is split into train (75%) and test (25%) sets using **stratified
  sampling**, so both sets keep the same ~1.2% fraud ratio. Without
  stratification, a random split could leave the test set with very few
  (or zero) fraud cases, making evaluation unreliable.
- Numeric features are **standardised** (mean 0, std 1) before Logistic
  Regression, since it's sensitive to feature scale. Random Forest doesn't
  need this since tree splits don't care about scale.

## 5. Handling class imbalance

No SMOTE library was available in this environment, so I used
**`class_weight="balanced"`**, a built-in scikit-learn option that
automatically up-weights the rare (fraud) class during training, instead of
making the model treat every transaction as equally important. This is a
completely standard, often-preferred alternative to oversampling — it's
worth knowing that SMOTE (Synthetic Minority Oversampling) is another common
approach, and you can mention you'd consider it as a comparison if asked.

## 6. Models trained, and why two

- **Logistic Regression** — a simple, interpretable baseline. Good first
  step in any classification problem: if a simple model already does well,
  that tells you the problem may not need something more complex.
- **Random Forest** — an ensemble of decision trees, generally stronger at
  capturing non-linear patterns and feature interactions.

Comparing both is good practice: it shows whether the extra complexity of
Random Forest is actually buying you anything.

## 7. Results — and how to talk about the precision/recall tradeoff

From the run on this dataset:

| Model | Precision (fraud) | Recall (fraud) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.68 | **1.00** | 0.9999 |
| Random Forest | **1.00** | 0.88 | 0.9998 |

This is the most important thing to understand and be ready to explain:

- **Logistic Regression caught every single fraud case in the test set**
  (recall = 1.00) but flagged 28 legitimate transactions as fraud too
  (lower precision).
- **Random Forest** had zero false alarms (precision = 1.00) but missed 7
  out of 60 fraud cases.

Neither is "better" in isolation — it depends on business priorities. A
bank that wants to minimize missed fraud (and is okay with more manual
review of flagged transactions) would prefer the Logistic Regression
behavior here. A bank that wants to minimize customer friction from false
declines might lean toward the Random Forest behavior. **This tradeoff is
exactly the kind of thing interviewers like to probe** — being able to
explain precision vs. recall in plain language, using your own confusion
matrix numbers, is more valuable than memorizing the formulas.

Also worth knowing: **ROC-AUC and PR-AUC** are both used because accuracy is
misleading on imbalanced data. ROC-AUC measures how well the model ranks
fraud above non-fraud across all thresholds; PR-AUC (average precision) is
often considered more informative for highly imbalanced problems
specifically because it focuses on performance on the rare positive class.

## 8. Feature importance

`feature_importance.png` shows Random Forest's top predictive features:
`distance_from_home_km`, `transaction_amount`, and
`num_transactions_last_24h` dominate — which lines up with the EDA findings
and with how fraud was simulated. Being able to connect "what the model
learned" back to "what we saw in EDA" is a good sign of genuine
understanding, not just running code.

## 9. Honest limitations (good to mention proactively)

- The dataset is synthetic, so results are likely **better than you'd see
  on real, noisier data** — real fraud patterns are subtler and evolve over
  time (fraudsters adapt).
- No **temporal/sequential modeling** here (e.g. tracking a single card's
  behavior change over time) — a production system would likely add that.
- No **threshold tuning / cost-based decisioning** — in practice you'd pick
  a probability threshold based on the actual cost of false positives vs.
  false negatives, not just use the default 0.5 cutoff.
- No **SMOTE comparison** since the library wasn't available in this
  environment — `class_weight balanced` was used instead, which is a valid
  but different approach worth contrasting if asked.

## 10. How to practice explaining this

Before an interview, try explaining out loud (or to a friend):
1. Why accuracy is a bad metric here and what you'd use instead.
2. What "stratified split" means and why it matters.
3. The precision/recall tradeoff using the actual numbers above.
4. One thing you'd improve if you had real production data.

If you can do that without looking at notes, you genuinely understand the
project — and that's what matters far more than the resume bullet itself.
