#%%
import pandas as pd 
import numpy as np 
import glob 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import QuantileTransformer
import os

current_script_directory = os.path.dirname(os.path.abspath(__file__))

df = pd.read_parquet('data/application_train.pq')

do_quick_check = False

#%%
demographics = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

count = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS']

duration = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE']

    
social = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
          'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

contact = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
           'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']

address = ['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
           'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
           'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

region = ['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT',
          'REGION_RATING_CLIENT_W_CITY']

process = ['HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START']

external = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

amount = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

inquiry = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
           'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

document = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
            'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

building = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
            'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG',
            'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
            'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
            'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
            'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
            'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
            'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
            'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
            'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
            'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
            'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
            'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
            'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
            'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
            'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

#%%

dict_cols = {
                      'demographic': demographics,
                      'count': count, 
                      'duration': duration, 
                      'social': social, 
                      'contact_info': contact,
                      'adress': address,
                      'region': region,
                      'process': process,
                      'external': external,
                      'amount': amount,
                      'inquiry': inquiry,
                      'document': document,
                      'building': building
                      }
# %%

def quick_check(cols, df):

    objects = df[cols].select_dtypes('O').columns.to_list()
    nums = df[cols].select_dtypes(np.number).columns.to_list()

    for col in objects:
        print(col.upper())
        print(df[col].value_counts(dropna=False)) 

    for col in nums:
        print(col.upper())
        df[col].hist(bins=100)
        plt.show()

if do_quick_check:
    for cols in dict_cols.values():
        quick_check(cols, df)
#%%
#============================================================================================ DEMOGRAPHIC

df = df.drop(['CODE_GENDER', 'NAME_TYPE_SUITE'], axis=1)                                       # ! Drop GENDER because it is discriminant
df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].replace('unknown', 'Single / not married') # ! Replace unknown with Single
df[['FLAG_OWN_REALTY', 'FLAG_OWN_CAR']] = df[['FLAG_OWN_REALTY', 'FLAG_OWN_CAR']].replace(['N', 'Y'], [0, 1])

df['CAR_ESTIMATED_VALUE'] = df['FLAG_OWN_CAR'] / (df['OWN_CAR_AGE'] + 4)                       # ! Deprecation
df['REALTY_ESTIMATED_VALUE'] = df['APARTMENTS_MEDI'] * df['FLAG_OWN_REALTY']


df = df.drop(['OWN_CAR_AGE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'], axis=1)
#============================================================================================ COUNT

df = df.drop('CNT_FAM_MEMBERS', axis=1)                           # ! Both are highly correlated, we drop the one with more NaN

#============================================================================================ DURATION

df = df.drop(['DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'], axis=1)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, df['DAYS_EMPLOYED'].median()) # ! Replace outliers with the median

df['AGE'] = (-df['DAYS_BIRTH'] / 365).astype(int)                                       # ! Transform DAYS_BIRTH with AGE
df['YEAR_EMPLOYED'] = (-df['DAYS_EMPLOYED'] / 365).astype(int)
df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1)

df['EMPLOYMENT_PERCENTAGE'] = df['YEAR_EMPLOYED'] / df['AGE']
#============================================================================================ SOCIAL

df = df.drop(dict_cols['social'], axis=1)                         # ! To confirm but this seems very not fair to me

#============================================================================================ CONTACT INFORMATION

df = df.drop(dict_cols['contact_info'], axis=1)                   # ! Mostly irrelevant and discriminative 

#============================================================================================ ADRESS

df = df.drop(dict_cols['adress'], axis=1)

#============================================================================================ REGION

df = df.drop(['REGION_RATING_CLIENT'], axis=1)                    # ! Highly correlated, we drop the one with a weird value

region_id = df['REGION_POPULATION_RELATIVE'].unique()             # ! Create variable Region ID
df['REGION_ID'] = df['REGION_POPULATION_RELATIVE'].apply(lambda x : (x == region_id).argmax()).astype(str)
df = df.drop('REGION_POPULATION_RELATIVE', axis=1)

#============================================================================================ PROCESS

df = df.drop(dict_cols['process'], axis=1)                          # ! Cannot be explained to a customer or regulator

#============================================================================================ EXTERNAL SOURCES

df = df.drop(dict_cols['external'], axis=1)                         # ! Black-Box

#============================================================================================ AMOUNT

df['CREDIT_DOWNPAYMENT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT'] # ! Create new variable downpayment
df['RATIO_CREDIT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']   # ! Create ratio because correlated, Discuss if we remove one of them
df['ANN_INC_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CRED_INC_RATIO']= df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

#============================================================================================ INQUIRY

dict_cols['inquiry'].remove('AMT_REQ_CREDIT_BUREAU_MON')             # !Keep only the number of inquiries during prev month 
df = df.drop(dict_cols['inquiry'], axis=1)

#============================================================================================ DOCUMENT

df = df.drop(dict_cols['document'], axis=1) # ! Drop All, irrelevant

#============================================================================================ BUILDING

df = df.drop(dict_cols['building'], axis=1) # ! Drop rest because they are either irrelevant and redundant 
#%%
sns.heatmap(df.select_dtypes(np.number).corr())


#%%
df.to_parquet('data/base_df.pq')

#================================================ CHEKPOINT ===============================================
#==========================================================================================================
#==================================== MOVE TO FEATURE ENGINEERING =========================================
# %%

df = pd.read_parquet('data/base_df.pq').dropna(subset='TARGET').set_index('SK_ID_CURR', drop=True)

bureau = pd.read_parquet('data/bureau.pq')
bureau_bal = pd.read_parquet('data/bureau_balance.pq')
pos_cash = pd.read_parquet('data/POS_CASH_balance.pq')
installments = pd.read_parquet('data/installments_payments.pq')
prev_applications = pd.read_parquet('data/previous_application.pq')

# ! Create variable current debt

approval = prev_applications['NAME_CONTRACT_STATUS'].replace(['Approved', 'Refused', 'Canceled', 'Unused offer'], [1, 0, 0, 0])
current_debt = (
        approval * 
        ((prev_applications['AMT_CREDIT'] + (1 + prev_applications['RATE_INTEREST_PRIMARY']) * (-prev_applications['DAYS_FIRST_DUE']) / 365) 
        - (prev_applications['AMT_ANNUITY'] * prev_applications['CNT_PAYMENT'])))

current_debt.index = prev_applications['SK_ID_CURR']
current_debt = current_debt.groupby('SK_ID_CURR').sum()
current_debt.name = 'CURRENT_DEBT'
df = pd.merge(df, current_debt, left_index=True, right_index=True, how='left')

#%%
# ! Create variable ANNUITY_TO_MAX_INSTALLMENT_RATIO
max_installment = installments.groupby('SK_ID_CURR')['AMT_INSTALMENT'].max() 
max_installment.rename('MAX_AMT_INSTALMENT', inplace=True)

df = pd.merge(df, max_installment, right_index=True, left_index=True, how='left')
df['ANNUITY_TO_MAX_INSTALLMENT_RATIO'] = df['AMT_ANNUITY'] / (df['MAX_AMT_INSTALMENT'] + 0.01)


# ! Create variable N_CREDIT_ACTIVE and N_CREDIT_PROLONGATION
bureau['N_CREDIT_ACTIVE'] = bureau['CREDIT_ACTIVE'].replace(['Closed', 'Active', 'Sold', 'Bad debt'], [0, 1, 0, np.nan])

n_credit_active = bureau.groupby('SK_ID_CURR')['N_CREDIT_ACTIVE'].sum() 

n_credit_prolongation = bureau.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].sum()
n_credit_prolongation.rename('N_CREDIT_PROLONG', inplace=True)

bureau['IS_BAD_DEBT'] = bureau['CREDIT_ACTIVE'].replace(['Closed', 'Active', 'Sold', 'Bad debt'], [0, 0, 0, 1])
flag_bad_debt = bureau.groupby('SK_ID_CURR')['IS_BAD_DEBT'].sum()
flag_bad_debt.rename('N_BAD_DEBT', inplace=True)

pos_cash = pos_cash.groupby('SK_ID_CURR')[['SK_DPD', 'SK_DPD_DEF', 'CNT_INSTALMENT_FUTURE']].sum()
pos_cash.columns = ['HISTORICAL_DPD', 'HISTORICAL_DPD_DEF', 'N_FUTURE_INSTALLMENT']

df = pd.merge(df, n_credit_active, right_index=True, left_index=True, how='left')
df = pd.merge(df, n_credit_prolongation, right_index=True, left_index=True, how='left')
df = pd.merge(df, flag_bad_debt, right_index=True, left_index=True, how='left')
df = pd.merge(df, pos_cash, left_index=True, right_index=True, how='left')

# ! Create variable MEAN_RATE_DOWN_PAYMENT
mean_rate_down_payment = prev_applications.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].mean()
mean_rate_down_payment.rename('MEAN_RATE_DOWN_PAYMENT', inplace=True)
df = pd.merge(df, mean_rate_down_payment, right_index=True, left_index=True, how='left')

#================================================ CHEKPOINT ===============================================
#==========================================================================================================
#==================================== MOVE TO SCALING / ENCODING ==========================================
#%%

def flag_nans(df, min_nan=20000, fill_nan=True):
    """
    This function creates a new dummy column if there 
    are more than min_nan NaN that indicates whether the 
    value was missing 
    """

    for col in df.columns:


        if (df[col].isna().sum() > min_nan):

            if df[col].dtype in [float, int]:

                df[col + '_ISNA'] = df[col].isna() * 1
                df[col].fillna(0)

            else:
                df[col] = df[col].fillna('NaN')

    return df

df = flag_nans(df)
df.reset_index().to_parquet('data/final_categorical.pq')


#%%
categorical = df.select_dtypes('O').columns.to_list()

df = pd.get_dummies(df, columns=categorical) * 1

to_quantile_scale = df.columns.to_list()
df[to_quantile_scale] = QuantileTransformer().fit_transform(df[to_quantile_scale])

df.columns = [re.sub('[^a-zA-Z0-9_]', '', col) for col in df.columns]

df.reset_index().to_parquet('data/final_df.pq')

# %%

