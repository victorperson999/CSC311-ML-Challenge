import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys, csv, random

def predict_all(file_path: str):
    # to be done
    test_data = pd.read_csv(file_path)
    #handle features.......