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

# separating the feature and target columns

# outcome
y = df_imputed_1['y'].astype('int')

# features
X = df_imputed_1.drop('y',axis = 1)

# =========================
# 3. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# =========================
# 4. Training Model
# =========================

classifiers = [
    ['CatBoostClassifier',CatBoostClassifier(silent=True)],
    ['LogisticRegression',LogisticRegression(class_weight='balanced',random_state=1)],
    ['DecisionTree',DecisionTreeClassifier(class_weight='balanced',random_state=1)],
    ['LightGBM',LGBMClassifier(class_weight='balanced',metric='binary_logloss')],
    ['SVC', SVC()],
    ['KNN', KNeighborsClassifier(n_neighbors = 30)],
    ['GradientBoosting', GradientBoostingClassifier()],
    ['RandomForest', RandomForestClassifier()],
    ['XGBoost', XGBClassifier()]]


def modelling(classifiers, X_train, y_train, X_test, y_test):
    """
    Train multiple classifiers and collect predictions, probabilities, and runtime
    """
    
    model_results = []

    for name, model in classifiers:
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        runtime = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities (handle models without predict_proba)
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_train_proba = model.decision_function(X_train)
            y_test_proba = model.decision_function(X_test)
        else:
            y_train_proba = None
            y_test_proba = None
        
        model_results.append({
            "model_name": name,
            "runtime": runtime,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
            "y_train_proba": y_train_proba,
            "y_test_proba": y_test_proba
        })
        
    return model_results

def process_eval(model_results, overfit_threshold=10):
    """
    Evaluate models and return comparison table
    """
    
    rows = []

    for res in model_results:
        y_train = res["y_train"]
        y_test = res["y_test"]
        
        y_train_pred = res["y_train_pred"]
        y_test_pred = res["y_test_pred"]
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # AUC (safe handling)
        if res["y_train_proba"] is not None:
            train_auc = roc_auc_score(y_train, res["y_train_proba"])
            test_auc = roc_auc_score(y_test, res["y_test_proba"])
        else:
            train_auc = np.nan
            test_auc = np.nan
        
        # Overfitting detection
        diff = train_accuracy - test_accuracy
        diff_percentage = diff * 100
        is_overfitting = diff_percentage > overfit_threshold
        
        rows.append({
            "model_name": res["model_name"],
            "runtime": round(res["runtime"], 3),
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "train_recall": round(train_recall, 4),
            "test_recall": round(test_recall, 4),
            "train_precision": round(train_precision, 4),
            "test_precision": round(test_precision, 4),
            "train_f1_score": round(train_f1, 4),
            "test_f1_score": round(test_f1, 4),
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "diff_percentage": round(diff_percentage, 2),
            "diff": round(diff, 4),
            "is_overfitting": is_overfitting
        })

    df = pd.DataFrame(rows)
    
    # Define best model (highest test F1, fallback to AUC)
    best_idx = (
        df["test_f1_score"]
        .fillna(0)
        .idxmax()
    )
    
    df["is_best_model"] = False
    df.loc[best_idx, "is_best_model"] = True
    
    return df


model_result=modelling(classifiers,X_train,y_train,X_test,y_test)
result=process_eval(model_result)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def get_param_grid(model_name):
    grids = {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear"]
        },
        "RandomForest": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        },
        "XGBoost": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.01, 0.1]
        },
        "LightGBM": {
            "model__n_estimators": [100, 300],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.01, 0.1]
        }
    }
    return grids.get(model_name, {})

def tune_best_model(
    best_model_name,
    classifiers,
    X_train,
    y_train,
    scoring="f1"
):
    # Get model
    model = dict(classifiers)[best_model_name]
    
    pipe = Pipeline([
        ("model", model)
    ])
    
    param_grid = get_param_grid(best_model_name)
    
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=10,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_

