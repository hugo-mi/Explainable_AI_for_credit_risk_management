#%%
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
df.loc[:,np.invert(df.columns.str.contains(r'ISNA|RESID'))].columns
#%%
df = pd.read_parquet('data/final_categorical.pq').set_index('SK_ID_CURR')
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

df = get_residuals(causal_graph, df)
to_keep = select_important_features(df)

cat = df[to_keep].select_dtypes('O').columns.to_list()

ord_encode = OrdinalEncoder(cols=cat).fit(df[to_keep])
processed_df = df.copy()[to_keep]
processed_df = ord_encode.transform(df[to_keep])
#%%
df_encode = df.copy().drop('TARGET', axis=1)
df_encode[df_encode.columns.to_list()] = OrdinalEncoder().fit_transform(df_encode).fillna(0) 

iso_forest = IsolationForest(bootstrap=True, max_features=50).fit(df_encode)
processed_df['ANOMALY'] = iso_forest.decision_function(df_encode)

X_train, X_test, y_train, y_test = train_test_split(processed_df, df['TARGET'], test_size=0.25)

classif = lgb.LGBMClassifier(class_weight='balanced', 
                             min_split_gain=0.001, 
                             reg_lambda=0.0001, 
                             num_leaves=30, 
                             min_child_samples=100, 
                             learning_rate=0.1)
classif.fit(X_train, y_train)


preds = classif.predict(X_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
print(roc_auc_score(y_test, preds))
#%%
import shapash 

xpl = shapash.SmartExplainer(classif, preprocessing=ord_encode)
xpl.compile(x=X_test.sort_index(),
            y_target=y_test.sort_index(),
            additional_data=df.loc[df.index.isin(X_test.index), np.invert(df.columns.isin(X_test))].sort_index()
    )
xpl.run_app(title_story='Credit Default Risk', port=8020)

#%%
import dice_ml

# DICE implementation
# Conterfactual here 

def generate_dice_explanation(pos_index, model, df, categorical_features):
    pos_index = df.loc[df.index == pos_index].index
    print(df['TARGET'])
    dice_data = dice_ml.Data(dataframe=df, continuous_features=[col for col in df.columns if col not in categorical_features + ['TARGET']], outcome_name='TARGET')
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    dice_exp = dice_ml.Dice(dice_data, dice_model)
    
    print(pos_index)
    query_instance = df.loc[pos_index].fillna(0).drop('TARGET', axis=1)

    columns_to_vary_to_deal = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'RATIO_CREDIT_ANNUITY', 'N_CREDIT_ACTIVE_RESID', 'CAR_ESTIMATED_VALUE']

    counterfactuals_deal = dice_exp.generate_counterfactuals(query_instance, proximity_weight=0,stopping_threshold=0.5,total_CFs=10, desired_class="opposite", features_to_vary = columns_to_vary_to_deal)

    return counterfactuals_deal

# Example same as before 
cf_deal = generate_dice_explanation(105455, classif, pd.merge(X_test, y_test, on='SK_ID_CURR'), [cat])
cf_deal.visualize_as_dataframe(show_only_changes=True)
#%%
pd.merge(X_test, y_test, on='SK_ID_CURR').loc[400348]
#%%
cf_deal.cf_examples_list[0].final_cfs_df[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'RATIO_CREDIT_ANNUITY', 'N_CREDIT_ACTIVE_RESID', 'CAR_ESTIMATED_VALUE']]
