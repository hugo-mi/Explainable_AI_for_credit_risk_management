import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from train_model import train_model, predict_model
from eval_model import evaluate_model
from model_calibration import calibrate_model_isotonic, find_best_threshold_recall
from portfolio_optimization_strategy import (port_optimization_strategy, 
                                             plot_portfolio_credit_risk_strategy, 
                                             plot_strategy_portfolio_optimization,)

current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Load preprocessed data
data = pd.read_parquet(current_script_directory+'/data/final_categorical.pq')

def preprocess_data(data):
    # Convert data to categorical
    data.set_index('SK_ID_CURR', drop=True, inplace=True)
     
    floats = data.select_dtypes(float).columns.to_list()
    categorical_columns = (data.nunique() == 2).index.to_list()
    
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le


    X = data.drop(columns=['TARGET'])
    y = data['TARGET']
    return X, y

X, y = preprocess_data(data)

# Split data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = train_model(data, X_train, y_train)

# Calibrate model
calibrated_model = calibrate_model_isotonic(model, X_test, y_test)

# Get predictions with the calibrated model
y_preds, y_probs = predict_model(calibrated_model, X_test)
y_probs_default = y_probs[:,1]

# Adjust threshold
threshold = find_best_threshold_recall(X_test, y_test, y_probs_default)

# Eval model with the threshold
evaluate_model(calibrated_model, X_test, y_test, threshold)

# Apply portfolio credit risk strategy
strategy_table = port_optimization_strategy(calibrated_model, X_test, y_test)

# Vizualize the portfolio credit risk strategy
plot_strategy_portfolio_optimization(strategy_table)

# Visualize the portfolio credit risk strategy
plot_portfolio_credit_risk_strategy(strategy_table, elev=10, azim=-50)
