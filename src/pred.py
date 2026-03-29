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
        "medial_vals": params["medial_vals"], # (8,) median for Nan implementation
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
    """

def parse_price(val):
    """
    """

def multi_hot_encode(series, categories, prefix):
    """
    """

def bow_encode(series, vocab, prefix):
    """
    """

def preprocess(df, model):
    """
    transform raw test dataframe into feature matrix X, also mirrors the exact
    preprocessing pipeline from train.pu
    """
    col_map = {

    }
    df = df.rename(columns=col_map)
    N = len(df)
    medians = model["median_vals"]

    #numeric features

    #likert features

    #numeric + likert

    #stdize

    # multi-hot categories

    #bag of words text features

    #build bull feature matrix
    X = np.hstack([...])

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
    ...

