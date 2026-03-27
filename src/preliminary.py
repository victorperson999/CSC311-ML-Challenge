"""
CSC311 ML Challenge — Exploration & Model Comparison
=====================================================
This script does:
  1. Quick EDA (class balance, feature distributions, missing values)
  2. Full preprocessing pipeline (Likert ordinal, numeric cleaning,
     multi-hot categoricals, bag-of-words text features)
  3. Train/val split (80/20, stratified)
  4. Compare 4 model families: Logistic Regression, Random Forest,
     Naive Bayes, MLP Neural Network
  5. Hyperparameter sweeps with results tables

Usage:  python3 explore.py
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("ml_challenge_dataset.csv")

# Short column aliases for convenience
COL_NAMES = {
    "unique_id": "id",
    "Painting": "target",
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
    "Describe how this painting makes you feel.": "text_feelings",
    "This art piece makes me feel sombre.": "likert_sombre",
    "This art piece makes me feel content.": "likert_content",
    "This art piece makes me feel calm.": "likert_calm",
    "This art piece makes me feel uneasy.": "likert_uneasy",
    "How many prominent colours do you notice in this painting?": "num_colors",
    "How many objects caught your eye in the painting?": "num_objects",
    "How much (in Canadian dollars) would you be willing to pay for this painting?": "price",
    "If you could purchase this painting, which room would you put that painting in?": "room",
    "If you could view this art in person, who would you want to view it with?": "who",
    "What season does this art piece remind you of?": "season",
    "If this painting was a food, what would be?": "text_food",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "text_soundtrack",
}
df.rename(columns=COL_NAMES, inplace=True)

print("=" * 60)
print("1. QUICK EDA")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nMissing % per column:")
for col in df.columns:
    pct = df[col].isnull().mean() * 100
    if pct > 0:
        print(f"  {col}: {pct:.1f}%")

# ─────────────────────────────────────────────
# 2. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# --- 2a. Encode target ---
label_enc = LabelEncoder()
df["target_enc"] = label_enc.fit_transform(df["target"])
print(f"Target classes: {list(label_enc.classes_)}")
# 0 = The Persistence of Memory, 1 = The Starry Night, 2 = The Water Lily Pond

# --- 2b. Likert columns → ordinal integers (1-5) ---
def parse_likert(val):
    """Extract leading integer from '4 - Agree' → 4"""
    if pd.isna(val):
        return np.nan
    m = re.match(r"(\d)", str(val))
    return int(m.group(1)) if m else np.nan

likert_cols = ["likert_sombre", "likert_content", "likert_calm", "likert_uneasy"]
for col in likert_cols:
    df[col] = df[col].apply(parse_likert)

print(f"Likert columns parsed. Sample:\n{df[likert_cols].head()}")

# --- 2c. Price column → numeric ---
def parse_price(val):
    """Clean price: strip $, commas, spaces, words; extract first number → float or NaN"""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip()
    # Try direct conversion first
    try:
        v = float(s)
        return v if v >= 0 else np.nan
    except ValueError:
        pass
    # Try to extract the first number from messy text like "300 dollars" or "5 000 000"
    m = re.findall(r"[\d]+(?:\.\d+)?", s.replace(" ", ""))
    if m:
        try:
            v = float(m[0])
            return v if v >= 0 else np.nan
        except ValueError:
            return np.nan
    return np.nan

df["price"] = df["price"].apply(parse_price)

# Log-transform price to tame huge range (values span $0 to $1e8+)
# Add 1 to handle $0 entries, then log
df["price"] = np.log1p(df["price"])
print(f"Price (log1p): median={df['price'].median():.2f}, max={df['price'].max():.2f}, NaN={df['price'].isna().sum()}")

# --- 2d. Numeric columns — fill NaN with median ---
numeric_cols = ["emotion_intensity", "num_colors", "num_objects", "price"]
for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  {col}: filled NaN with median={median_val:.1f}")

# Fill likert NaN with median (3 = neutral)
for col in likert_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# --- 2e. Multi-select categorical → multi-hot ---
def multi_hot_encode(series, col_prefix):
    """
    Expand comma-separated column into multi-hot binary columns.
    e.g. "Bedroom,Bathroom" → bedroom=1, bathroom=1, others=0
    """
    # Collect all unique categories
    all_cats = set()
    for val in series.dropna():
        for cat in str(val).split(","):
            all_cats.add(cat.strip())
    all_cats = sorted(all_cats)

    # Build multi-hot DataFrame
    result = pd.DataFrame(0, index=series.index, columns=[f"{col_prefix}_{c}" for c in all_cats])
    for idx, val in series.items():
        if pd.notna(val):
            for cat in str(val).split(","):
                cname = f"{col_prefix}_{cat.strip()}"
                if cname in result.columns:
                    result.at[idx, cname] = 1
    return result, all_cats

room_df, room_cats = multi_hot_encode(df["room"], "room")
who_df, who_cats = multi_hot_encode(df["who"], "who")
season_df, season_cats = multi_hot_encode(df["season"], "season")

print(f"\nMulti-hot categories:")
print(f"  Room:   {room_cats}")
print(f"  Who:    {who_cats}")
print(f"  Season: {season_cats}")

# --- 2f. Bag-of-Words for text columns ---
# We'll build BoW for: text_feelings, text_food, text_soundtrack
# Using a limited vocabulary to keep feature count manageable

text_cols_to_bow = ["text_feelings", "text_food", "text_soundtrack"]
bow_dfs = {}
vectorizers = {}

for tcol in text_cols_to_bow:
    df[tcol] = df[tcol].fillna("")
    vec = CountVectorizer(
        max_features=50,       # top 50 words per text column
        stop_words="english",  # remove common words
        min_df=5,              # word must appear in at least 5 docs
        binary=True            # presence/absence, not counts
    )
    bow_matrix = vec.fit_transform(df[tcol])
    bow_col_names = [f"{tcol}_{w}" for w in vec.get_feature_names_out()]
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_col_names, index=df.index)
    bow_dfs[tcol] = bow_df
    vectorizers[tcol] = vec
    print(f"  BoW {tcol}: {len(bow_col_names)} features, vocab sample: {list(vec.get_feature_names_out())[:10]}")

# ─────────────────────────────────────────────
# 3. ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. FEATURE MATRIX ASSEMBLY")
print("=" * 60)

# Combine all feature groups
feature_parts = {
    "numeric": df[numeric_cols + likert_cols],
    "room": room_df,
    "who": who_df,
    "season": season_df,
}
# Add BoW features
for tcol, bdf in bow_dfs.items():
    feature_parts[tcol] = bdf

X = pd.concat(feature_parts.values(), axis=1)
# Safety: fill any remaining NaN with 0 (multi-hot missing → 0, numeric already filled)
nan_count = X.isna().sum().sum()
if nan_count > 0:
    print(f"  WARNING: {nan_count} NaN values remaining, filling with 0")
    X = X.fillna(0)
y = df["target_enc"].values

print(f"Total features: {X.shape[1]}")
for name, part in feature_parts.items():
    print(f"  {name}: {part.shape[1]} features")

# ─────────────────────────────────────────────
# 4. TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# Scale numeric features (important for LogReg and MLP)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

cols_to_scale = numeric_cols + likert_cols
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

# ─────────────────────────────────────────────
# 5. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. MODEL COMPARISON")
print("=" * 60)

results = []

def evaluate_model(name, model, Xtr, Xv, ytr, yv):
    """Train, predict, and record accuracy."""
    model.fit(Xtr, ytr)
    train_acc = accuracy_score(ytr, model.predict(Xtr))
    val_acc = accuracy_score(yv, model.predict(Xv))
    results.append({"Model": name, "Train Acc": train_acc, "Val Acc": val_acc,
                     "Gap": train_acc - val_acc})
    print(f"  {name:45s} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")
    return model

# ── 4a. Logistic Regression (various regularization strengths) ──
print("\n--- Logistic Regression ---")
for C in [0.01, 0.1, 1.0, 10.0]:
    evaluate_model(
        f"LogReg (C={C})",
        LogisticRegression(C=C, max_iter=1000, solver="lbfgs"),
        X_train_scaled, X_val_scaled, y_train, y_val
    )

# ── 4b. Random Forest (various n_estimators and max_depth) ──
print("\n--- Random Forest ---")
for n_est in [50, 100, 200]:
    for depth in [5, 10, 20, None]:
        evaluate_model(
            f"RF (n={n_est}, depth={depth})",
            RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1),
            X_train, X_val, y_train, y_val  # RF doesn't need scaling
        )

# ── 4c. Naive Bayes ──
print("\n--- Naive Bayes ---")
# GaussianNB on scaled features
evaluate_model("GaussianNB", GaussianNB(), X_train_scaled, X_val_scaled, y_train, y_val)

# ComplementNB works well for text/count features (needs non-negative input)
# Use unscaled data since multi-hot + BoW are already non-negative
# But numeric cols can be negative after scaling, so use unscaled
X_train_nonneg = X_train.copy()
X_val_nonneg = X_val.copy()
# Clip any negatives to 0 for ComplementNB
X_train_nonneg = X_train_nonneg.clip(lower=0)
X_val_nonneg = X_val_nonneg.clip(lower=0)

for alpha in [0.1, 0.5, 1.0, 2.0]:
    evaluate_model(
        f"ComplementNB (alpha={alpha})",
        ComplementNB(alpha=alpha),
        X_train_nonneg, X_val_nonneg, y_train, y_val
    )

# ── 4d. MLP Neural Network ──
print("\n--- MLP Neural Network ---")
for hidden in [(64,), (128,), (64, 32), (128, 64)]:
    for alpha in [0.001, 0.01]:
        evaluate_model(
            f"MLP (hidden={hidden}, alpha={alpha})",
            MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=500,
                          random_state=42, early_stopping=True, validation_fraction=0.15),
            X_train_scaled, X_val_scaled, y_train, y_val
        )

# ─────────────────────────────────────────────
# 6. RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. RESULTS SUMMARY (sorted by Val Acc)")
print("=" * 60)
results_df = pd.DataFrame(results).sort_values("Val Acc", ascending=False)
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. BEST MODEL — DETAILED REPORT
# ─────────────────────────────────────────────
best_row = results_df.iloc[0]
print(f"\nBest model: {best_row['Model']} with Val Acc = {best_row['Val Acc']:.4f}")

# ─────────────────────────────────────────────
# 8. CROSS-VALIDATION ON TOP 3 MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 5-FOLD CROSS-VALIDATION ON TOP MODELS")
print("=" * 60)

# Re-scale all of X for cross-val
X_all_scaled = X.copy()
X_all_scaled[cols_to_scale] = StandardScaler().fit_transform(X[cols_to_scale])
X_all_nonneg = X.clip(lower=0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_models = [
    ("LogReg (C=1.0)", LogisticRegression(C=1.0, max_iter=1000), X_all_scaled),
    ("LogReg (C=10.0)", LogisticRegression(C=10.0, max_iter=1000), X_all_scaled),
    ("RF (n=200, depth=20)", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42), X),
    ("RF (n=200, depth=None)", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42), X),
    ("MLP (128,64) alpha=0.01", MLPClassifier(hidden_layer_sizes=(128,64), alpha=0.01, max_iter=500, random_state=42, early_stopping=True), X_all_scaled),
    ("ComplementNB (alpha=1.0)", ComplementNB(alpha=1.0), X_all_nonneg),
]

for name, model, X_cv in cv_models:
    scores = cross_val_score(model, X_cv, y, cv=cv, scoring="accuracy")
    print(f"  {name:40s} | Mean: {scores.mean():.4f} ± {scores.std():.4f} | Folds: {scores}")

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE (from best RF)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)
rf_best = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_best.fit(X_train, y_train)
importances = pd.Series(rf_best.feature_importances_, index=X.columns)
top_20 = importances.sort_values(ascending=False).head(20)
print("Top 20 features:")
for feat, imp in top_20.items():
    print(f"  {feat:45s} {imp:.4f}")

# ─────────────────────────────────────────────
# 10. CONFUSION MATRIX FOR BEST MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. CONFUSION MATRIX (best single-split model)")
print("=" * 60)
# Use LogReg C=1.0 as a representative
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_val_scaled)
print(f"Classes: {list(label_enc.classes_)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_val, y_pred, target_names=label_enc.classes_)}")

print("\n" + "=" * 60)
print("DONE. Use these results to pick your final model.")
print("=" * 60)