# -*- coding: utf-8 -*-
 

import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectKBest, SelectPercentile, SequentialFeatureSelector, RFE, RFECV
from sklearn.ensemble import IsolationForest
import json
from sklearn.calibration import calibration_curve
import shap
import os
import matplotlib.pyplot as plt
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
with open('best_params.json', 'r') as f:
    loaded_params = json.load(f)

current_dir = os.getcwd()

# Define the relative path to the Parquet file
relative_path = os.path.join(current_dir, 'data', 'final_categorical.pq')
df = pd.read_parquet(relative_path).set_index('SK_ID_CURR')
cat = df.select_dtypes('O').columns.to_list()

causal_graph = {
    'YEAR_EMPLOYED': ['AGE', 'NAME_EDUCATION_TYPE'],
    'AMT_CREDIT': ['CAR_ESTIMATED_VALUE', 'AMT_INCOME_TOTAL', 'REALTY_ESTIMATED_VALUE', 'N_CREDIT_ACTIVE', 'AMT_GOODS_PRICE', 'MEAN_RATE_DOWN_PAYMENT'],
    'AMT_GOODS_PRICE': ['AMT_CREDIT'],
    'AMT_ANNUITY': ['AMT_CREDIT', 'CREDIT_DOWNPAYMENT'],
    'N_CREDIT_ACTIVE': ['REALTY_ESTIMATED_VALUE', 'CAR_ESTIMATED_VALUE', 'ORGANIZATION_TYPE'],
    'AMT_INCOME_TOTAL': ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'AGE', 'NAME_EDUCATION_TYPE'],
    'CREDIT_DOWNPAYMENT': ['AMT_INCOME_TOTAL', 'AMT_CREDIT'],
    'CRED_INC_RATIO': ['AMT_CREDIT', 'AMT_INCOME_TOTAL'],
    'NAME_INCOME_TYPE': ['NAME_EDUCATION_TYPE', 'AGE', 'OCCUPATION_TYPE']
}

def get_residuals(causal_graph, df):
    ord_encode = OrdinalEncoder(cols=cat, return_df=True).fit(df)
    new_df = df.copy()
    new_df[new_df.columns.to_list()] = ord_encode.transform(df)

    for k in causal_graph.keys():
        if df[k].dtype == object:
            classif = lgb.LGBMClassifier(n_estimators=50, subsample=0.8, min_split_gain=0.01)
            classif.fit(new_df[causal_graph[k]], new_df[k])
            df[k + '_RESID'] = classif.predict(new_df[causal_graph[k]])
        else:
            classif = LGBMRegressor(n_estimators=50, subsample=0.8, min_split_gain=0.01, objective='mse')
            classif.fit(new_df[causal_graph[k]], new_df[k])
            df[k + '_RESID'] = (new_df[k] - classif.predict(new_df[causal_graph[k]]))

    return df

def select_important_features(df):
    ord_encode = OrdinalEncoder(cols=cat, return_df=True).fit(df)
    new_df = df.copy()
    new_df[new_df.columns.to_list()] = ord_encode.transform(df)
    new_df = new_df.sample(frac=0.8)
    X_train, y_train = new_df.drop('TARGET', axis=1), new_df['TARGET']
    classif = lgb.LGBMClassifier(class_weight='balanced', min_split_gain=0.01, reg_lambda=0.01, num_leaves=20)
    chosen_features = RFE(classif).fit(X_train, y_train)
    to_keep = chosen_features.get_feature_names_out()
    return to_keep

# Calculate Residuals and Select Important Features
df = get_residuals(causal_graph, df)
to_keep = select_important_features(df)
cat = df[to_keep].select_dtypes('O').columns.to_list()
ord_encode = OrdinalEncoder(cols=cat).fit(df[to_keep])
processed_df = df.copy()[to_keep]
processed_df = ord_encode.transform(df[to_keep])

# Anomaly Detection
df_encode = df.copy().drop('TARGET', axis=1)
df_encode[df_encode.columns.to_list()] = OrdinalEncoder().fit_transform(df_encode).fillna(0)
iso_forest = IsolationForest(bootstrap=True, max_features=50).fit(df_encode)
processed_df['ANOMALY'] = iso_forest.decision_function(df_encode)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(processed_df, df['TARGET'], test_size=0.25)

# Initialize Classifier with Loaded Parameters
classif_loaded = lgb.LGBMClassifier(**loaded_params)
classif_loaded.fit(X_train, y_train)

# Predictions and Evaluation
preds = classif_loaded.predict(X_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
print(roc_auc_score(y_test, preds))

explainer = shap.TreeExplainer(classif)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", plot_size=(10,15))
shap.summary_plot(shap_values, X_test)

shap_values

output_dir = 'Decision plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

y_preds = preds[:10]

if isinstance(X_test, pd.DataFrame):

    for i in range(min(10, X_test.shape[0])):  # Ensure we don't go out of bounds
        true_label = y_test.iloc[i]
        predicted_outcome = y_preds[i]

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.decision_plot(explainer.expected_value, shap_values[i], feature_names=X_test.columns.values)
        ax.set_title(f'SHAP Decision Plot for Instance {i}\nPredicted: {predicted_outcome}, True: {true_label}')
        plt.savefig(f'{output_dir}/shap_decision_plot_instance_{i}.jpeg')
        plt.close()

    print(f'Plots saved in {output_dir}')
else:
    print("X_test is not a DataFrame. Please check your data.")

pdp_dir = 'pdp_plots'
from sklearn.inspection import PartialDependenceDisplay

for dir_name in [pdp_dir]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

 if isinstance(X_test, pd.DataFrame):
    #
    features_to_plot = X_test.columns[:30]  # Change this list as needed

    # Generate and save PDPs for the selected features
    for feature in features_to_plot:
        # PDP
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(classif_loaded, X_test, [feature], ax=ax)
        plt.title(f'Partial Dependence Plot for Feature: {feature}')
        plt.savefig(f'{pdp_dir}/pdp_{feature}.png')
        plt.close()



    print(f'Partial Dependence Plots saved in {pdp_dir}')
else:
    print("X_test is not a DataFrame. Please check your data.")

def permutation_feature_importance(model, X_test, y_test, metric=log_loss, n_repeats=5):
    baseline_score = metric(y_test, model.predict_proba(X_test)[:, 1])   
    importances = pd.DataFrame(index=X_test.columns, columns=range(n_repeats))

    for col in X_test.columns:
        for n in range(n_repeats):
            X_permuted = X_test.copy()
            X_permuted[col] = shuffle(X_test[col].values)
            permuted_score = metric(y_test, model.predict_proba(X_permuted)[:, 1])
            importances.loc[col, n] = baseline_score - permuted_score

    mean_importances = importances.mean(axis=1)
    return mean_importances.sort_values(ascending=False)

#
feature_importances = permutation_feature_importance(classif_loaded, X_test, y_test)

#
top_5_features = feature_importances.head(5)
print("Top 5 feature importances:\n", top_5_features)

#
plt.figure(figsize=(10, 6))
top_5_features.plot(kind='barh')
plt.title('Top 5 Permutation Feature Importances')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()  #
plt.show()

 
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.50, random_state=42)  # 60-20-20 split

 
y_pred_proba = classif_loaded.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)
mapie_score = MapieClassifier(estimator=classif_loaded, cv="prefit", method="lac")
mapie_score.fit(X_calib, y_calib)
alpha = [0.2, 0.1, 0.05]
y_pred_score, y_ps_score = mapie_score.predict(X_test, alpha=alpha)


 

def plot_scores(n, alphas, scores, quantiles):
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}  # Updated colors
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins="auto")
    for i, quantile in enumerate(quantiles):
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax=400,
            color=colors[i],
            ls="dashed",
            label=f"alpha = {alphas[i]}"
        )
    plt.title("Distribution of scores")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.show()

plot_scores(len(y_pred_proba_max), alpha, y_pred_proba_max, mapie_score.quantiles_)

def plot_scores_and_stats(alphas, scores, quantiles):
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}  # Updated colors
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins="auto", alpha=0.7, label='Scores')

    stats = {}
    for i, quantile in enumerate(quantiles):
        below = np.sum(scores <= quantile)
        above = np.sum(scores > quantile)
        stats[alphas[i]] = {'below': below, 'above': above}
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax=max(np.histogram(scores, bins='auto')[0]),
            color=colors[i],
            ls="dashed",
            label=f"alpha = {alphas[i]}, below = {below}, above = {above}"
        )

    plt.title("Distribution of scores with quantiles")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.show()

    return stats

quantiles = [np.quantile(y_pred_proba_max, 1 - a) for a in alpha]


stats = plot_scores_and_stats(alpha, y_pred_proba_max, quantiles)
print("Stats for each alpha:")
for a, stat in stats.items():
    print(f"Alpha: {a}, Below: {stat['below']}, Above: {stat['above']}")

# Predict probabilities and scores
y_pred_proba = classif_loaded.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)
mapie_score = MapieClassifier(estimator=classif_loaded, cv="prefit", method="lac")
mapie_score.fit(X_calib, y_calib)
alpha = [0.2, 0.1, 0.05]
y_pred_score, y_ps_score = mapie_score.predict(X_test, alpha=alpha)

def plot_class_probability_distribution(y_pred_proba):
    plt.figure(figsize=(7, 5))
    plt.hist(y_pred_proba, bins='auto', alpha=0.7, label=['Class 0', 'Class 1'])
    plt.title("Class Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_calibration_curve(y_true, y_pred_proba):
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.title("Calibration Curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.show()

def plot_reliability_diagram(y_true, y_pred_proba):
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.bar(prob_pred, prob_true - prob_pred, width=0.1, edgecolor='k', label='Reliability diagram')
    plt.plot([0, 1], [0, 0], linestyle='--', color='gray')
    plt.title("Reliability Diagram")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives - Predicted probability")
    plt.legend()
    plt.show()

def plot_coverage_vs_significance_level(alphas, y_ps_score, y_test):
    coverages = [np.mean(np.any(y_ps_score[:, :, i] == y_test[:, np.newaxis], axis=1)) for i in range(len(alphas))]
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, coverages, marker='o', label='Coverage vs Significance Level')
    plt.title("Coverage vs. Significance Level")
    plt.xlabel("Significance Level")
    plt.ylabel("Coverage")
    plt.legend()
    plt.show()
    print(coverages)



plot_class_probability_distribution(y_pred_proba_max)
plot_calibration_curve(y_test, y_pred_proba_max)
plot_reliability_diagram(y_test, y_pred_proba_max)
plot_coverage_vs_significance_level(alpha, y_ps_score, y_test.values)
