"""
CSC311 ML Challenge — Phase 2: Feature Tuning & Ensembles
==========================================================
Builds on explore.py results. Experiments with:
  1. BoW hyperparams (vocab size, ngrams, binary vs counts)
  2. Feature ablation (which groups matter?)
  3. Ensemble methods (soft voting, stacking)
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# PREPROCESSING (same as explore.py)
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

# Encode target
label_enc = LabelEncoder()
df["target_enc"] = label_enc.fit_transform(df["target"])

# Likert → ordinal
def parse_likert(val):
    if pd.isna(val): return np.nan
    m = re.match(r"(\d)", str(val))
    return int(m.group(1)) if m else np.nan

likert_cols = ["likert_sombre", "likert_content", "likert_calm", "likert_uneasy"]
for col in likert_cols:
    df[col] = df[col].apply(parse_likert)

# Price → numeric (log-transformed)
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

# Fill numeric NaN
numeric_cols = ["emotion_intensity", "num_colors", "num_objects", "price"]
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in likert_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Multi-hot encode
def multi_hot_encode(series, col_prefix):
    all_cats = set()
    for val in series.dropna():
        for cat in str(val).split(","):
            all_cats.add(cat.strip())
    all_cats = sorted(all_cats)
    result = pd.DataFrame(0, index=series.index, columns=[f"{col_prefix}_{c}" for c in all_cats])
    for idx, val in series.items():
        if pd.notna(val):
            for cat in str(val).split(","):
                cname = f"{col_prefix}_{cat.strip()}"
                if cname in result.columns:
                    result.at[idx, cname] = 1
    return result

room_df = multi_hot_encode(df["room"], "room")
who_df = multi_hot_encode(df["who"], "who")
season_df = multi_hot_encode(df["season"], "season")

# Fill text NaN
for tcol in ["text_feelings", "text_food", "text_soundtrack"]:
    df[tcol] = df[tcol].fillna("")

y = df["target_enc"].values

# ─────────────────────────────────────────────
# CONSISTENT SPLIT FOR ALL EXPERIMENTS
# ─────────────────────────────────────────────
# Use the SAME indices across all experiments
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)
y_train, y_val = y[train_idx], y[val_idx]

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
print(f"Classes: {list(label_enc.classes_)}")

# ─────────────────────────────────────────────
# HELPER: build feature matrix with given BoW config
# ─────────────────────────────────────────────
def build_features(bow_type="count", max_features=50, ngram_range=(1,1),
                   min_df=5, binary=True, include_text=True,
                   text_cols=["text_feelings", "text_food", "text_soundtrack"]):
    """
    Build the full feature matrix with configurable BoW settings.
    Returns X_train, X_val (DataFrames), and the vectorizer objects for export.
    """
    # Structured features (always included)
    struct_df = pd.concat([
        df[numeric_cols + likert_cols],
        room_df, who_df, season_df
    ], axis=1).fillna(0)

    if not include_text:
        X = struct_df
        return X.iloc[train_idx], X.iloc[val_idx], {}

    # Build BoW features
    bow_parts = []
    vectorizers = {}
    for tcol in text_cols:
        if bow_type == "count":
            vec = CountVectorizer(
                max_features=max_features, stop_words="english",
                min_df=min_df, binary=binary, ngram_range=ngram_range
            )
        else:  # tfidf
            vec = TfidfVectorizer(
                max_features=max_features, stop_words="english",
                min_df=min_df, ngram_range=ngram_range
            )
        # Fit on TRAIN only, transform both
        train_text = df[tcol].iloc[train_idx]
        val_text = df[tcol].iloc[val_idx]
        bow_train = vec.fit_transform(train_text)
        bow_val = vec.transform(val_text)

        col_names = [f"{tcol}_{w}" for w in vec.get_feature_names_out()]
        bow_train_df = pd.DataFrame(
            bow_train.toarray(), columns=col_names, index=train_idx
        )
        bow_val_df = pd.DataFrame(
            bow_val.toarray(), columns=col_names, index=val_idx
        )
        bow_parts.append((bow_train_df, bow_val_df))
        vectorizers[tcol] = vec

    # Combine
    struct_train = struct_df.iloc[train_idx].reset_index(drop=True)
    struct_val = struct_df.iloc[val_idx].reset_index(drop=True)

    bow_train_all = pd.concat([b[0].reset_index(drop=True) for b in bow_parts], axis=1)
    bow_val_all = pd.concat([b[1].reset_index(drop=True) for b in bow_parts], axis=1)

    X_train = pd.concat([struct_train, bow_train_all], axis=1)
    X_val = pd.concat([struct_val, bow_val_all], axis=1)

    return X_train, X_val, vectorizers


def scale_and_eval(X_train, X_val, model_name="LogReg", C=0.1):
    """Scale numeric cols and evaluate a model."""
    scaler = StandardScaler()
    X_tr = X_train.copy()
    X_v = X_val.copy()

    # Only scale the structured numeric/likert cols (not binary BoW features)
    cols_to_scale = [c for c in numeric_cols + likert_cols if c in X_tr.columns]
    if cols_to_scale:
        X_tr[cols_to_scale] = scaler.fit_transform(X_tr[cols_to_scale])
        X_v[cols_to_scale] = scaler.transform(X_v[cols_to_scale])

    model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    model.fit(X_tr, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_tr))
    val_acc = accuracy_score(y_val, model.predict(X_v))
    return train_acc, val_acc, model, scaler


# ═════════════════════════════════════════════
# EXPERIMENT 1: BoW HYPERPARAMETER SWEEP
# ═════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 1: BoW HYPERPARAMETER SWEEP (LogReg C=0.1)")
print("=" * 70)

bow_configs = [
    # (label, bow_type, max_features, ngram_range, min_df, binary)
    ("No text",           "count", 50,  (1,1), 5, True),   # text excluded
    ("Count 30w unigram", "count", 30,  (1,1), 5, True),
    ("Count 50w unigram", "count", 50,  (1,1), 5, True),
    ("Count 75w unigram", "count", 75,  (1,1), 5, True),
    ("Count 100w unigram","count", 100, (1,1), 5, True),
    ("Count 150w unigram","count", 150, (1,1), 5, True),
    ("Count 50w bigram",  "count", 50,  (1,2), 5, True),
    ("Count 100w bigram", "count", 100, (1,2), 5, True),
    ("Count 150w bigram", "count", 150, (1,2), 5, True),
    ("TF-IDF 50w unigram","tfidf", 50,  (1,1), 5, False),
    ("TF-IDF 100w unigram","tfidf",100, (1,1), 5, False),
    ("TF-IDF 100w bigram","tfidf", 100, (1,2), 5, False),
    ("TF-IDF 150w bigram","tfidf", 150, (1,2), 5, False),
    ("Count 50w min_df=3","count", 50,  (1,1), 3, True),
    ("Count 100w min_df=3","count",100, (1,1), 3, True),
    ("Count 100w bigram min_df=3","count",100,(1,2),3,True),
]

bow_results = []
for label, btype, mf, ngram, mdf, binary in bow_configs:
    include = label != "No text"
    X_tr, X_v, _ = build_features(
        bow_type=btype, max_features=mf, ngram_range=ngram,
        min_df=mdf, binary=binary, include_text=include
    )
    train_acc, val_acc, _, _ = scale_and_eval(X_tr, X_v)
    gap = train_acc - val_acc
    bow_results.append((label, X_tr.shape[1], train_acc, val_acc, gap))
    print(f"  {label:30s} | feats={X_tr.shape[1]:4d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Gap: {gap:+.4f}")

print("\nTop 5 by Val Acc:")
for row in sorted(bow_results, key=lambda x: -x[3])[:5]:
    print(f"  {row[0]:30s} | Val: {row[3]:.4f} | Gap: {row[4]:+.4f}")


# ═════════════════════════════════════════════
# EXPERIMENT 2: FEATURE ABLATION
# ═════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: FEATURE ABLATION (best BoW config from above)")
print("=" * 70)

# Use the best BoW config — we'll figure out which from the sweep
# For now, build with a good default (100w bigram)
best_bow = sorted(bow_results, key=lambda x: -x[3])[0]
print(f"Using BoW config: {best_bow[0]}")

# Full feature set as baseline
X_tr_full, X_v_full, _ = build_features(
    bow_type="count", max_features=100, ngram_range=(1,2), min_df=5, binary=True
)
_, full_val, _, _ = scale_and_eval(X_tr_full, X_v_full)
print(f"\n  {'FULL (all features)':40s} | Val: {full_val:.4f}")

# Ablate each group
struct_df_base = pd.concat([
    df[numeric_cols + likert_cols], room_df, who_df, season_df
], axis=1).fillna(0)

feature_groups = {
    "numeric (intensity, colors, objects, price)": numeric_cols,
    "likert (sombre, content, calm, uneasy)": likert_cols,
    "room (multi-hot)": [c for c in room_df.columns],
    "who (multi-hot)": [c for c in who_df.columns],
    "season (multi-hot)": [c for c in season_df.columns],
}

# Drop each group and measure impact
print("\n  Ablation: remove one group at a time")
for group_name, group_cols in feature_groups.items():
    X_tr_abl = X_tr_full.drop(columns=[c for c in group_cols if c in X_tr_full.columns], errors="ignore")
    X_v_abl = X_v_full.drop(columns=[c for c in group_cols if c in X_v_full.columns], errors="ignore")
    _, val_abl, _, _ = scale_and_eval(X_tr_abl, X_v_abl)
    delta = val_abl - full_val
    print(f"  Remove {group_name:45s} | Val: {val_abl:.4f} | Delta: {delta:+.4f}")

# Drop each text column
for tcol in ["text_feelings", "text_food", "text_soundtrack"]:
    drop_cols = [c for c in X_tr_full.columns if c.startswith(tcol + "_")]
    X_tr_abl = X_tr_full.drop(columns=drop_cols, errors="ignore")
    X_v_abl = X_v_full.drop(columns=drop_cols, errors="ignore")
    _, val_abl, _, _ = scale_and_eval(X_tr_abl, X_v_abl)
    delta = val_abl - full_val
    print(f"  Remove {('BoW: ' + tcol):45s} | Val: {val_abl:.4f} | Delta: {delta:+.4f}")

# Only structured features (no text)
X_tr_struct = struct_df_base.iloc[train_idx].reset_index(drop=True)
X_v_struct = struct_df_base.iloc[val_idx].reset_index(drop=True)
_, struct_val, _, _ = scale_and_eval(X_tr_struct, X_v_struct)
print(f"\n  {'STRUCTURED ONLY (no text)':40s} | Val: {struct_val:.4f}")

# Only text features (no structured)
text_only_cols = [c for c in X_tr_full.columns if c not in struct_df_base.columns]
X_tr_text = X_tr_full[text_only_cols]
X_v_text = X_v_full[text_only_cols]
_, text_val, _, _ = scale_and_eval(X_tr_text, X_v_text)
print(f"  {'TEXT ONLY (no structured)':40s} | Val: {text_val:.4f}")


# ═════════════════════════════════════════════
# EXPERIMENT 3: ENSEMBLE METHODS
# ═════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 3: ENSEMBLE METHODS")
print("=" * 70)

# Build feature matrices for different model needs
X_tr_best, X_v_best, vecs_best = build_features(
    bow_type="count", max_features=100, ngram_range=(1,2), min_df=5, binary=True
)

# Scaled version for LogReg/MLP
scaler = StandardScaler()
X_tr_s = X_tr_best.copy()
X_v_s = X_v_best.copy()
cols_to_scale = [c for c in numeric_cols + likert_cols if c in X_tr_s.columns]
X_tr_s[cols_to_scale] = scaler.fit_transform(X_tr_s[cols_to_scale])
X_v_s[cols_to_scale] = scaler.transform(X_v_s[cols_to_scale])

# Non-negative version for ComplementNB
X_tr_nn = X_tr_best.clip(lower=0)
X_v_nn = X_v_best.clip(lower=0)

# Individual models
print("\n--- Individual models (best configs from explore.py) ---")
models = {
    "LogReg C=0.1": (LogisticRegression(C=0.1, max_iter=1000), X_tr_s, X_v_s),
    "LogReg C=0.05": (LogisticRegression(C=0.05, max_iter=1000), X_tr_s, X_v_s),
    "LogReg C=0.2": (LogisticRegression(C=0.2, max_iter=1000), X_tr_s, X_v_s),
    "RF n=200 d=10": (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), X_tr_best, X_v_best),
    "RF n=200 d=15": (RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42), X_tr_best, X_v_best),
    "RF n=300 d=10": (RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42), X_tr_best, X_v_best),
    "CompNB a=0.1": (ComplementNB(alpha=0.1), X_tr_nn, X_v_nn),
    "MLP (128,64) a=0.01": (MLPClassifier(hidden_layer_sizes=(128,64), alpha=0.01, max_iter=500, random_state=42, early_stopping=True), X_tr_s, X_v_s),
    "MLP (256,128) a=0.01": (MLPClassifier(hidden_layer_sizes=(256,128), alpha=0.01, max_iter=500, random_state=42, early_stopping=True), X_tr_s, X_v_s),
}

fitted_models = {}
for name, (model, X_tr, X_v) in models.items():
    model.fit(X_tr, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_tr))
    v_acc = accuracy_score(y_val, model.predict(X_v))
    fitted_models[name] = (model, X_tr, X_v, v_acc)
    print(f"  {name:30s} | Train: {tr_acc:.4f} | Val: {v_acc:.4f}")

# --- Manual soft voting ensemble ---
print("\n--- Soft Voting Ensembles (manual) ---")

def soft_vote_ensemble(model_list, X_val_list):
    """Average predicted probabilities across models, pick argmax."""
    probs = np.zeros((len(X_val_list[0]), 3))
    for model, X_v in zip(model_list, X_val_list):
        probs += model.predict_proba(X_v)
    probs /= len(model_list)
    return np.argmax(probs, axis=1)

# Ensemble combos to try
combos = [
    ("LogReg + RF", ["LogReg C=0.1", "RF n=200 d=10"]),
    ("LogReg + RF + MLP", ["LogReg C=0.1", "RF n=200 d=10", "MLP (128,64) a=0.01"]),
    ("LogReg + RF + NB", ["LogReg C=0.1", "RF n=200 d=10", "CompNB a=0.1"]),
    ("LogReg + RF + MLP + NB", ["LogReg C=0.1", "RF n=200 d=10", "MLP (128,64) a=0.01", "CompNB a=0.1"]),
    ("2xLogReg + RF", ["LogReg C=0.05", "LogReg C=0.2", "RF n=200 d=10"]),
    ("2xLogReg + RF + MLP", ["LogReg C=0.05", "LogReg C=0.2", "RF n=200 d=10", "MLP (128,64) a=0.01"]),
    ("LogReg + 2xRF", ["LogReg C=0.1", "RF n=200 d=10", "RF n=300 d=10"]),
    ("LogReg + RF(d=15) + MLP", ["LogReg C=0.1", "RF n=200 d=15", "MLP (128,64) a=0.01"]),
    ("LogReg + RF + bigMLP", ["LogReg C=0.1", "RF n=200 d=10", "MLP (256,128) a=0.01"]),
]

ensemble_results = []
for combo_name, model_names in combos:
    m_list = [fitted_models[n][0] for n in model_names]
    xv_list = [fitted_models[n][2] for n in model_names]
    preds = soft_vote_ensemble(m_list, xv_list)
    val_acc = accuracy_score(y_val, preds)
    ensemble_results.append((combo_name, val_acc, model_names))
    print(f"  {combo_name:40s} | Val: {val_acc:.4f}")

# ═════════════════════════════════════════════
# EXPERIMENT 4: CROSS-VALIDATE BEST CONFIGS
# ═════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 4: 5-FOLD CV ON BEST INDIVIDUAL + ENSEMBLE")
print("=" * 70)

# For CV we need to rebuild features on each fold properly
# Simpler approach: fit vectorizers on full data (slight optimism, but
# consistent with how we'll deploy — train on ALL data for final model)
X_full, _, vecs_full = build_features(
    bow_type="count", max_features=100, ngram_range=(1,2), min_df=5, binary=True
)
# Combine train+val since build_features split them
X_all = pd.concat([X_full, _], axis=0).sort_index()
X_all = X_all.fillna(0)

# Scale all
X_all_s = X_all.copy()
scaler_all = StandardScaler()
X_all_s[cols_to_scale] = scaler_all.fit_transform(X_all[cols_to_scale])
X_all_nn = X_all.clip(lower=0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_configs = [
    ("LogReg C=0.1", LogisticRegression(C=0.1, max_iter=1000), X_all_s),
    ("LogReg C=0.05", LogisticRegression(C=0.05, max_iter=1000), X_all_s),
    ("RF n=200 d=10", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), X_all),
    ("RF n=200 d=15", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42), X_all),
    ("MLP (128,64)", MLPClassifier(hidden_layer_sizes=(128,64), alpha=0.01, max_iter=500, random_state=42, early_stopping=True), X_all_s),
    ("CompNB a=0.1", ComplementNB(alpha=0.1), X_all_nn),
]

print("\n--- Individual 5-fold CV ---")
for name, model, X_cv in cv_configs:
    scores = cross_val_score(model, X_cv, y, cv=cv, scoring="accuracy")
    print(f"  {name:25s} | Mean: {scores.mean():.4f} ± {scores.std():.4f} | {scores.round(4)}")

# Manual CV for ensemble
print("\n--- Ensemble 5-fold CV ---")
best_ensemble_combos = sorted(ensemble_results, key=lambda x: -x[1])[:3]

for combo_name, _, model_names in best_ensemble_combos:
    fold_accs = []
    for train_i, val_i in cv.split(X_all, y):
        y_tr_f, y_v_f = y[train_i], y[val_i]
        probs = np.zeros((len(val_i), 3))

        for mname in model_names:
            if "LogReg" in mname:
                C_val = float(mname.split("C=")[1])
                m = LogisticRegression(C=C_val, max_iter=1000)
                X_use = X_all_s
            elif "RF" in mname:
                parts = mname.split()
                n_est = int(parts[1].split("=")[1])
                depth_str = parts[2].split("=")[1]
                depth = None if depth_str == "None" else int(depth_str)
                m = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
                X_use = X_all
            elif "CompNB" in mname:
                alpha_val = float(mname.split("a=")[1])
                m = ComplementNB(alpha=alpha_val)
                X_use = X_all_nn
            elif "MLP" in mname:
                # Parse hidden layers
                hidden_str = mname.split("(")[1].split(")")[0]
                hidden = tuple(int(x) for x in hidden_str.split(","))
                alpha_val = float(mname.split("a=")[1])
                m = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha_val,
                                  max_iter=500, random_state=42, early_stopping=True)
                X_use = X_all_s

            m.fit(X_use.iloc[train_i], y_tr_f)
            probs += m.predict_proba(X_use.iloc[val_i])

        probs /= len(model_names)
        preds = np.argmax(probs, axis=1)
        fold_accs.append(accuracy_score(y_v_f, preds))

    fold_accs = np.array(fold_accs)
    print(f"  {combo_name:40s} | Mean: {fold_accs.mean():.4f} ± {fold_accs.std():.4f} | {fold_accs.round(4)}")


# ═════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: BEST CONFIGURATIONS")
print("=" * 70)
print("\nBest BoW configs (single-split):")
for row in sorted(bow_results, key=lambda x: -x[3])[:3]:
    print(f"  {row[0]:30s} | feats={row[1]:4d} | Val: {row[3]:.4f}")

print("\nBest individual models (single-split):")
for name, (_, _, _, v) in sorted(fitted_models.items(), key=lambda x: -x[1][3])[:3]:
    print(f"  {name:30s} | Val: {v:.4f}")

print("\nBest ensembles (single-split):")
for row in sorted(ensemble_results, key=lambda x: -x[1])[:3]:
    print(f"  {row[0]:40s} | Val: {row[1]:.4f} | Models: {row[2]}")

print("\nDone! Use these results to decide final model for pred.py")