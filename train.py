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
MODEL_PATH = "models/fraud_model.pkl"

year_1 = arff.loadarff("data/1year.arff")
df_year_1 = pd.DataFrame(year_1[0])


# =========================
# 2. Preprocessing
# =========================
col= []
for a in range(1,65):
    col.append('x'+str(a))
col.append('y')

df_year_1.columns = col
df_year_1['y'] = df_year_1['y'].str.decode('utf-8')

df_year_1.replace('?',np.nan,inplace=True)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df_year_1)
df_imputed_array_1 = imp.transform(df_year_1)
df_imputed_1 = pd.DataFrame(df_imputed_array_1, columns=df_year_1.columns)

df = pd.read_csv(DATA_PATH)