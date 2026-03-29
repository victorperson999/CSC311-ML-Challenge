"""
CSC311 ML Challenge — train.py
================================
Trains the final model on ALL training data and exports everything
pred.py needs for inference into model_params.npz.

Final model: Logistic Regression (C=0.1, softmax / multinomial)
Features:
  - 4 numeric: emotion_intensity, num_colors, num_objects, price (log1p)
  - 4 likert:  sombre, content, calm, uneasy (ordinal 1-5)
  - 5 room multi-hot
  - 5 who multi-hot
  - 4 season multi-hot
  - 150 BoW from text_feelings (binary unigram, top 150, min_df=5)
  - 150 BoW from text_food     (binary unigram, top 150, min_df=5)
  - text_soundtrack DROPPED (ablation showed it hurts)
Total: ~322 features (exact count depends on vectorizer)

Exports to model_params.npz:
  - W:              (3, D) weight matrix
  - b:              (3,)   intercept/bias vector
  - scaler_mean:    (8,)   mean for numeric+likert cols
  - scaler_std:     (8,)   std  for numeric+likert cols
  - median_vals:    (8,)   median for NaN imputation (numeric+likert)
  - vocab_feelings: (V1,)  vocabulary for text_feelings BoW
  - vocab_food:     (V2,)  vocabulary for text_food BoW
  - room_cats:      list of room categories (sorted)
  - who_cats:       list of who categories (sorted)
  - season_cats:    list of season categories (sorted)
  - class_names:    (3,)   label encoder classes

Usage: python3 train.py
"""

import pandas as pd
import numpy as np
import re
import json

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("ml_challenge_dataset.csv")

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
print(f"Loaded {len(df)} rows")

# ─────────────────────────────────────────────
# 2. ENCODE TARGET
# ─────────────────────────────────────────────
label_enc = LabelEncoder()
df["target_enc"] = label_enc.fit_transform(df["target"])
class_names = list(label_enc.classes_)
print(f"Classes: {class_names}")
# 0 = The Persistence of Memory
# 1 = The Starry Night
# 2 = The Water Lily Pond

# ─────────────────────────────────────────────
# 3. PREPROCESS FEATURES
# ─────────────────────────────────────────────

# --- 3a. Likert → ordinal integer ---
def parse_likert(val):
    if pd.isna(val): return np.nan
    m = re.match(r"(\d)", str(val))
    return int(m.group(1)) if m else np.nan

likert_cols = ["likert_sombre", "likert_content", "likert_calm", "likert_uneasy"]
for col in likert_cols:
    df[col] = df[col].apply(parse_likert)

# --- 3b. Price → numeric (log-transformed) ---
def parse_price(val):
    if pd.isna(val): return np.nan
    s = str(val).replace("$", "").replace(",", "").strip()
    try:
        return max(float(s), 0)
    except ValueError:
        pass
    m = re.findall(r"[\d]+(?:\.\d+)?", s.replace(" ", ""))
    if m:
        try: return max(float(m[0]), 0)
        except ValueError: pass
    return np.nan

df["price"] = df["price"].apply(parse_price)
df["price"] = np.log1p(df["price"])

# --- 3c. Fill NaN with median (record medians for export) ---
numeric_cols = ["emotion_intensity", "num_colors", "num_objects", "price"]
impute_cols = numeric_cols + likert_cols  # 8 columns total

median_vals = {}
for col in impute_cols:
    med = df[col].median()
    median_vals[col] = med
    df[col] = df[col].fillna(med)
    print(f"  {col}: median={med:.2f}, NaN filled → remaining NaN: {df[col].isna().sum()}")

# Store median values as array in consistent order
median_array = np.array([median_vals[c] for c in impute_cols])

# --- 3d. Multi-hot encode categorical columns ---
def multi_hot_encode(series, col_prefix):
    """Returns DataFrame + sorted category list."""
    all_cats = set()
    for val in series.dropna():
        for cat in str(val).split(","):
            all_cats.add(cat.strip())
    all_cats = sorted(all_cats)

    result = pd.DataFrame(0, index=series.index,
                          columns=[f"{col_prefix}_{c}" for c in all_cats])
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

print(f"  Room categories:   {room_cats}")
print(f"  Who categories:    {who_cats}")
print(f"  Season categories: {season_cats}")

# --- 3e. Bag-of-Words for text_feelings and text_food ---
#     (text_soundtrack DROPPED — ablation showed it hurts accuracy)
text_cols_to_bow = ["text_feelings", "text_food"]
bow_dfs = {}
vectorizers = {}

for tcol in text_cols_to_bow:
    df[tcol] = df[tcol].fillna("")
    vec = CountVectorizer(
        max_features=150,
        stop_words="english",
        min_df=5,
        binary=True,
        ngram_range=(1, 1)
    )
    bow_matrix = vec.fit_transform(df[tcol])
    bow_col_names = [f"{tcol}_{w}" for w in vec.get_feature_names_out()]
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_col_names, index=df.index)
    bow_dfs[tcol] = bow_df
    vectorizers[tcol] = vec
    vocab = vec.get_feature_names_out()
    print(f"  BoW {tcol}: {len(vocab)} words, sample: {list(vocab[:5])}...{list(vocab[-5:])}")

# ─────────────────────────────────────────────
# 4. ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────
X = pd.concat([
    df[impute_cols],        # 8 numeric/likert
    room_df,                # 5 room multi-hot
    who_df,                 # 5 who multi-hot
    season_df,              # 4 season multi-hot
    bow_dfs["text_feelings"],  # BoW feelings
    bow_dfs["text_food"],      # BoW food
], axis=1).fillna(0)

y = df["target_enc"].values

print(f"\nFeature matrix: {X.shape} ({X.shape[1]} features)")
print(f"Numeric+Likert: {len(impute_cols)}")
print(f"Room: {len(room_cats)}")
print(f"Who: {len(who_cats)}")
print(f"Season: {len(season_cats)}")
print(f"BoW feelings: {bow_dfs['text_feelings'].shape[1]}")
print(f"BoW food: {bow_dfs['text_food'].shape[1]}")

# Record the exact feature column order (critical for pred.py alignment)
feature_names = list(X.columns)

# ─────────────────────────────────────────────
# 5. SCALE NUMERIC/LIKERT FEATURES
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[impute_cols] = scaler.fit_transform(X[impute_cols])

scaler_mean = scaler.mean_     # shape (8,)
scaler_std = scaler.scale_     # shape (8,)

print(f"\nScaler mean: {scaler_mean}")
print(f"Scaler std:  {scaler_std}")

# ─────────────────────────────────────────────
# 6. TRAIN LOGISTIC REGRESSION ON ALL DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING: LogisticRegression(C=0.1) on ALL data")
print("=" * 60)

model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
model.fit(X_scaled, y)

train_acc = accuracy_score(y, model.predict(X_scaled))
print(f"Training accuracy (full dataset): {train_acc:.4f}")

W = model.coef_        # shape (3, D)
b = model.intercept_   # shape (3,)
print(f"Weight matrix shape: {W.shape}")
print(f"Intercept shape:     {b.shape}")

# ─────────────────────────────────────────────
# 7. SANITY CHECK: verify our manual softmax matches sklearn
# ─────────────────────────────────────────────
def softmax(z):
    """Numerically stable softmax (from lab06)."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Manual prediction: softmax(X @ W.T + b) → argmax
logits = X_scaled.values @ W.T + b
probs = softmax(logits)
manual_preds = np.argmax(probs, axis=1)
manual_acc = accuracy_score(y, manual_preds)
print(f"Manual softmax accuracy: {manual_acc:.4f}")
assert manual_acc == train_acc, f"MISMATCH! sklearn={train_acc:.4f} vs manual={manual_acc:.4f}"
print("PASS: manual prediction matches sklearn exactly")

# ─────────────────────────────────────────────
# 8. EXPORT MODEL PARAMETERS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPORTING MODEL PARAMETERS")
print("=" * 60)

# Vocabularies as arrays
vocab_feelings = np.array(vectorizers["text_feelings"].get_feature_names_out())
vocab_food = np.array(vectorizers["text_food"].get_feature_names_out())

# Category lists as arrays
room_cats_arr = np.array(room_cats)
who_cats_arr = np.array(who_cats)
season_cats_arr = np.array(season_cats)
class_names_arr = np.array(class_names)

# Feature column order (as JSON string stored in array for npz compatibility)
feature_names_json = np.array([json.dumps(feature_names)])

# save to an npz file used by Numpy library to store our multidimentional arrays in a single file to be used by train and pred.py
np.savez("model_params.npz",
    # Model weights
    W=W,
    b=b,
    # Scaler parameters
    scaler_mean=scaler_mean,
    scaler_std=scaler_std,
    # Imputation medians (order: emotion_intensity, num_colors, num_objects, price, likert_sombre, likert_content, likert_calm, likert_uneasy)
    median_vals=median_array,
    # BoW vocabularies
    vocab_feelings=vocab_feelings,
    vocab_food=vocab_food,
    # Multi-hot category lists
    room_cats=room_cats_arr,
    who_cats=who_cats_arr,
    season_cats=season_cats_arr,
    # Class names
    class_names=class_names_arr,
    # Feature column order
    feature_names_json=feature_names_json,
)

import os
fsize = os.path.getsize("model_params.npz")
print(f"Saved model_params.npz ({fsize / 1024:.1f} KB)")
print(f"W: {W.shape}")
print(f"b: {b.shape}")
print(f"scaler_mean: {scaler_mean.shape}")
print(f"scaler_std: {scaler_std.shape}")
print(f"median_vals:{median_array.shape}")
print(f"vocab_feelings: {vocab_feelings.shape}")
print(f"vocab_food: {vocab_food.shape}")
print(f"room_cats: {room_cats_arr.shape}")
print(f"who_cats: {who_cats_arr.shape}")
print(f"season_cats: {season_cats_arr.shape}")
print(f"class_names: {class_names_arr.shape}")

# ─────────────────────────────────────────────
# 9. VERIFY LOADING WORKS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION: reload and predict")
print("=" * 60)

params = np.load("model_params.npz", allow_pickle=True)
W_loaded = params["W"]
b_loaded = params["b"]

logits2 = X_scaled.values @ W_loaded.T + b_loaded
preds2 = np.argmax(softmax(logits2), axis=1)
reload_acc = accuracy_score(y, preds2)
print(f"Reloaded model accuracy: {reload_acc:.4f}")
assert reload_acc == train_acc, "MISMATCH after reload!"
print("PASS: reloaded params produce identical predictions")

# Check total file size (must be < 10MB)
total_size = fsize
print(f"\nTotal model file size: {total_size / 1024:.1f} KB (limit: 10 MB)")
print(f"Well under the 10 MB limit.")

print("\n" + "=" * 60)
print("DONE. model_params.npz is ready for pred.py")
print("=" * 60)