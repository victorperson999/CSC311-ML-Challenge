import pandas as pd
import numpy as np
import sys
import csv
import random
import re
import os

# use numpy, pandas, sys, csv, random..

def load_model():
    """
    load  the trained model params from model_params.npz!
    """
    model_dir = os.path.dirname(os.path.abspath(__file__))
    params = np.load(os.path.join(model_dir, "model_params.npz"), allow_pickle=True)

    return {
        "W": params["W"], # the (3, 322) weight matrix
        "b": params["b"], # (3,) bias vector
        "scaler_mean": params["scaler_mean"], # (8,) standardScaler means
        "scaler_std": params["scaler_std"], # (8,) standardScaler stds
        "median_vals": params["median_vals"], # (8,) median for Nan implementation
        "vocab_feelings": list(params["vocab_feelings"]), # 150 words
        "vocab_food": list(params["vocab_food"]), # 150 words
        "room_cats": list(params["room_cats"]), # ['Bathroom','Bedroom','Dining room','Living room','Office'], uhh more later perhaps tweak
        "who_cats": list(params["who_cats"]), # ['By yourself','Coworkers/Classmates','Family members','Friends','Strangers']....
        "season_cats": list(params["season_cats"]), # ['spring','summer','fall','winter']
        "class_names": list(params["class_names"]), #['The Persistence of Memory','The Starry Night','The Water Lily Pond']
    }


def softmax(z):
    """
    Softmax over rows of z
    """
    z_temp = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_temp)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def parse_likert(val):
    """
    Get the integer from the Likert string, ex) "4 - Agree" becomes 4
    """
    if pd.isna(val):
        return np.nan
    m = re.match(r"(\d)", str(val))
    if (m):
        return int(m.group(1))
    else:
        return np.nan
    

def parse_price(val):
    """
    parse price strings into numeric values, then log transform them
    """
    if pd.isna(val):
        return np.nan
    
    s = str(val).replace("$", "").replace(",", "").strip()
    try:
        return max(float(s), 0)
    except ValueError:
        pass

    m = re.findall(r"[\d]+(?:\.\d+)?", s.replace(" ", ""))
    if m:
        try:
            return max(float(m[0]), 0)
        except ValueError:
            pass
    
    return np.nan


def multi_hot_encode(series, categories, prefix):
    """
    Encode a comma-seperated column into the needed multi-hot columns
    """
    res = np.zeros((len(series), len(categories)))
    for i, val in enumerate(series):
        if pd.notna(val):
            for cat in str(val).split(","):
                cat = cat.strip()
                if cat in categories:
                    res[i, categories.index(cat)] = 1
    col_names = [f"{prefix}_{c}" for c in categories]
    
    return res, col_names

def bow_encode(series, vocab, prefix):
    """
    Gets a binary Bag-of-words in binary using sklearns tokenizer regex pattern.
    1 if the word is in the text, 0 otherwise.
    """

    vocab_set = {w: j for j, w in enumerate(vocab)}
    result = np.zeros((len(series), len(vocab)))
    # sklearn's default: r'(?u)\b\w\w+\b'
    token_pattern = re.compile(r"(?u)\b\w\w+\b")

    for i, text in enumerate(series):
        if pd.isna(text) or text == "":
            continue
        tokens = token_pattern.findall(str(text).lower())
        for w in tokens:
            if w in vocab_set:
                result[i, vocab_set[w]] = 1
    col_names = [f"{prefix}_{w}" for w in vocab]

    return result, col_names

def preprocess(df, model):
    """
    transform raw test dataframe into feature matrix X, also mirrors the exact
    preprocessing pipeline from train.pu
    """
    col_map = { # matches train.py's columns map also
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
    }
    df = df.rename(columns=col_map)
    N = len(df)
    medians = model["median_vals"]

    #numeric features
    emotion = pd.to_numeric(df["emotion_intensity"], errors="coerce").values
    num_colors = pd.to_numeric(df["num_colors"], errors="coerce").values
    num_objects = pd.to_numeric(df["num_objects"], errors="coerce").values
 
    price_raw = df["price"].apply(parse_price).values
    price_log = np.log1p(np.where(np.isnan(price_raw), np.nan, price_raw))

    #likert features
    likert_sombre = df["likert_sombre"].apply(parse_likert).values
    likert_content = df["likert_content"].apply(parse_likert).values
    likert_calm = df["likert_calm"].apply(parse_likert).values
    likert_uneasy = df["likert_uneasy"].apply(parse_likert).values

    # stack the numeric + likert
    numeric_block = np.column_stack([
        emotion, num_colors, num_objects, price_log,
        likert_sombre, likert_content, likert_calm, likert_uneasy
    ])
    for j in range(8):
        mask = np.isnan(numeric_block[:, j])
        numeric_block[mask, j] = medians[j]

    #stdize
    numeric_scaled = (numeric_block - model["scaler_mean"]) / model["scaler_std"]

    # multi-hot categories
    room_block, _ = multi_hot_encode(df["room"].values, model["room_cats"], "room")
    who_block, _ = multi_hot_encode(df["who"].values, model["who_cats"], "who")
    season_block, _ = multi_hot_encode(df["season"].values, model["season_cats"], "season")

    #bag of words text features
    feelings_block, _ = bow_encode(df["text_feelings"].values, model["vocab_feelings"], "text_feelings")
    food_block, _ = bow_encode(df["text_food"].values, model["vocab_food"], "text_food")

    #build bull feature matrix
    X = np.hstack([
        numeric_scaled, # 8 cols
        room_block, # 5 cols
        who_block, # 5 cols
        season_block, # 4 cols
        feelings_block, # 150 cols
        food_block, # 150 cols
    ])

    return X


def predict_all(file_path: str):
    """
    Load test CSV, preprocess, and return predictions.
 
    Parameters:
        file_path: path to test CSV (same format as training data, minus target)
 
    Returns:
        list of predicted painting names
    """
    model = load_model()
    test_data = pd.read_csv(file_path)

    X = preprocess(test_data, model)

    # logistic regression
    logits = X @ model["W"].T + model["b"]
    probs = softmax(logits)
    pred_indicies = np.argmax(probs, axis=1)

    # map indicies back to class names
    predictions = [model["class_names"][i] for i in pred_indicies]

    return predictions

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "ml_challenge_dataset.csv"
 
    preds = predict_all(file_path)
    print(f"Made {len(preds)} predictions")
    print(f"Distribution: { {p: preds.count(p) for p in set(preds)} }")
    print(f"First 10: {preds[:10]}")

