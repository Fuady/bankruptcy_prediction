"""
train.py
--------
Train final best model and save it to disk
"""

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# =========================
# 1. Load data
# =========================
DATA_PATH = "data/insurance_claims.csv"   # adjust path if needed
MODEL_PATH = "models/fraud_model.pkl"

df = pd.read_csv(DATA_PATH)