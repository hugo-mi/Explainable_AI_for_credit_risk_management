import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier

def train_model(df, X_train, y_train, imbalance=True):

    params = {'n_estimators': 86, 'max_depth': 410, 'learning_rate': 0.13438707410414805,
              'subsample': 0.46670221632867176, 'num_leaves': 42,
              'feature_fraction': 0.7569445635207415, 'sub_bin': 84053
        }

    model = LGBMClassifier(**params, class_weight='balanced', n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    return model

def predict_model(model, X_test):
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    
    return y_preds, y_probs