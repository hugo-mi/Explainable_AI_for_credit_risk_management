
#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import optuna
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import dice_ml

df = pd.read_parquet('/Users/Utilisateur/Desktop/S2/Data for Etienne/final_categorical.pq')#.clip(-1e9, 1e9)
df.set_index('SK_ID_CURR', drop=True, inplace=True)
 
floats = df.select_dtypes(float).columns.to_list()
categorical_columns = (df.nunique() == 2).index.to_list()


#%%
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=['TARGET'])
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# LightGBM optimization 
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 52),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256, log=True),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
    }
    model = LGBMClassifier(**params, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred)

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=2)
print("Best hyperparameters for LightGBM:", study_lgbm.best_params)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

# Train with hyperparameters 
model = LGBMClassifier(**study_lgbm.best_params, class_weight='balanced')
model.fit(X_train, y_train)
evaluate_model(model, X_test, y_test)

# LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=["non_defaults", "defaults"],
    categorical_features=[i for i, col in enumerate(X.columns) if col in categorical_columns],
    verbose=True,
    discretize_continuous=True,
    random_state=42,
    mode='classification'
)

def LIME_explainer(df_index, explainer, clf):
    pos_index = X_test.index.get_loc(df_index)
    print("\n\n\n**************************************************************************************************************************")
    print(f"************************************************************* EXAMPLE INDEX {df_index} *********************************************************")
    print("**************************************************************************************************************************")
    print("True Label: ", y_test.iloc[pos_index])
    exp = explainer.explain_instance(X_test.iloc[pos_index].values, clf.predict_proba, num_features=10, num_samples=5000)
    exp.show_in_notebook(show_table=True, show_all=True)
    exp.as_pyplot_figure()
    plt.show()
    print(exp.as_list())

# Example  
LIME_explainer(225738, explainer, model)


#%%
# DICE implementation
# Conterfactual here 

def generate_dice_explanation(pos_index, model, df, categorical_features):
    pos_index = X_test.index.get_loc(index)
    dice_data = dice_ml.Data(dataframe=df, continuous_features=[col for col in df.columns if col not in categorical_features + ['TARGET']], outcome_name='TARGET')
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    dice_exp = dice_ml.Dice(dice_data, dice_model)
    query_instance = df.iloc[[pos_index]].fillna(0).drop('TARGET', axis=1)
    counterfactuals = dice_exp.generate_counterfactuals(query_instance, stopping_threshold=0.5,total_CFs=5, desired_class="opposite")
    return counterfactuals.visualize_as_dataframe(show_only_changes=True)

# Example same as before 
cf_example_100 = generate_dice_explanation(800, model, df, [categorical_columns])
cf_example_100

#%%

# Import for analysis
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

#%%

# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Analysis 
analysis_df = X_test.copy()
analysis_df['Actual'] = y_test.values
analysis_df['Predicted'] = y_pred
random_state = 42

correct_non_delinquent = analysis_df[(analysis_df['Actual'] == 0) & (analysis_df['Predicted'] == 0)].sample(9, random_state=random_state)
correct_delinquent = analysis_df[(analysis_df['Actual'] == 1) & (analysis_df['Predicted'] == 1)].sample(9, random_state=random_state)
misclassified_as_delinquent = analysis_df[(analysis_df['Actual'] == 0) & (analysis_df['Predicted'] == 1)].sample(9, random_state=random_state)
misclassified_as_non_delinquent = analysis_df[(analysis_df['Actual'] == 1) & (analysis_df['Predicted'] == 0)].sample(9, random_state=random_state)

results = {
    'correct_non_default': correct_non_delinquent,
    'correct_default': correct_delinquent,
    'misclassified_as_default': misclassified_as_delinquent,
    'misclassified_as_non_default': misclassified_as_non_delinquent,
}


# Several tests

# Non-default and good classification
index = results['correct_non_default'].index[0]
LIME_explainer(index, explainer, model)
cf_correct_non_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_correct_non_delinquent)

index = results['correct_non_default'].index[1]
LIME_explainer(index, explainer, model)
cf_correct_non_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_correct_non_delinquent)

index = results['correct_non_default'].index[2]
LIME_explainer(index, explainer, model)
cf_correct_non_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_correct_non_delinquent)

index = results['correct_non_default'].index[3]
LIME_explainer(index, explainer, model)
cf_correct_non_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_correct_non_delinquent)

index = results['correct_non_default'].index[4]
LIME_explainer(index, explainer, model)
cf_correct_non_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_correct_non_delinquent)

# default and good classification
index = results['correct_default'].index[0]
print(index)
LIME_explainer(index, explainer, model)
cf_correct_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_correct_delinquent)

# Non-default and bad classification
index = results['misclassified_as_default'].index[0]
LIME_explainer(index, explainer, model)
cf_misclassified_as_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_misclassified_as_delinquent)

# default and  bad classification
index = results['misclassified_as_non_default'].index[0]
LIME_explainer(index, explainer, model)
cf_misclassified_as_non_delinquent = generate_dice_explanation(index, model, df, [categorical_columns])
print(cf_misclassified_as_non_delinquent)





