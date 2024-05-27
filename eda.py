#%% Import libs
import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import squarify 

from scipy.stats import chi2_contingency
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")


# Define custom colors/cmaps/palettes for visualization purposes.
denim='#6F8FAF'
salmon='#FA8072'
slate_gray = '#404040'
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("",[denim,salmon])
palette = 'colorblind'
sns.set_style('darkgrid')

current_script_directory = os.path.dirname(os.path.abspath(__file__))

#%% Load Dataset
data = pd.read_parquet(current_script_directory+'/data/application_train.pq')
print(data)
print(data.describe())
#%% Selection of the feature of interest

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

feature_groups = {'demographic': demographics,
                  'family_count': count,
                  'age_duration': duration,
                  'social': social,
                  'contact': contact,
                  'address': address,
                  "region": region,
                  'process': process,
                  'external': external,
                  'amounts': amount,
                  'inquiry': inquiry,
                  'document': document,
                  "building": building
                 }

#=== DEMOGRAPHIC ===

data = data.drop(['CODE_GENDER', 'NAME_TYPE_SUITE'], axis=1)     # ! Drop GENDER because it is discriminant

#=== COUNT ===

data = data.drop('CNT_FAM_MEMBERS', axis=1) # ! Drop CNT_FAM_MEMBERS because redundant with CNT_CHILDREN and contain lot of missing values

#=== DURATION ===

data = data.drop(['DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'], axis=1)

#=== SOCIAL ===

data = data.drop(feature_groups['social'], axis=1)       # ! To confirm but this seems very not fair to me

#=== CONTACT INFORMATION ===

data = data.drop(feature_groups['contact'], axis=1) # ! Mostly irrelevant and discriminative 

#=== ADRESS ===

data = data.drop(feature_groups['address'], axis=1) # not relevant

#=== PROCESS ===

data = data.drop(feature_groups['process'], axis=1)  # ! Cannot be explained to a customer or regulator

#=== EXTERNAL SOURCES ===

data = data.drop(feature_groups['external'], axis=1) # ! Cannot make sense of the column description

#=== INQUIRY ===

feature_groups['inquiry'].remove('AMT_REQ_CREDIT_BUREAU_MON') # !Keep number of inquiries during prev month 
data = data.drop(feature_groups['inquiry'], axis=1)

#=== DOCUMENT ===

data = data.drop(feature_groups['document'], axis=1) # ! Drop All, irrelevant

#=== BUILDING ===

data = data.drop(feature_groups['building'], axis=1) # ! Drop rest because they are either irrelevant and redundant 

#=== BUILDING ===

data = data.drop("SK_ID_CURR", axis=1) # ! Drop the ID of the applicants because irrelevant for credit scoring

print(data)
#%% Redefine our dictionnay of features
demographics = ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

family_count = ['CNT_CHILDREN']

age_duration = ['DAYS_BIRTH', 'DAYS_EMPLOYED']

region = ['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT',
          'REGION_RATING_CLIENT_W_CITY']

amounts = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

inquiry = ['AMT_REQ_CREDIT_BUREAU_MON']

feature_groups = {'demographic': demographics,
                  'family_count': family_count,
                  'age_duration': age_duration,
                  'region': region,
                  'amounts': amounts,
                  'inquiry': inquiry
                 }
#%% Visualize the proportion of missing values

def missing_values_by_feature(date):
    # Calculate the percentage of missing values
    missing_percentage = data.isna().sum()/len(data)*100
    
    # Convert the Series to a DataFrame for easier plotting
    missing_percentage_data = missing_percentage.reset_index()
    missing_percentage_data.columns = ['feature', 'missing_percentage']
    
    # Create the histogram
    plt.figure(figsize=(20, 8))
    bars = plt.bar(missing_percentage_data['feature'], missing_percentage_data['missing_percentage'], color='skyblue', edgecolor='k', alpha=0.7)
    
    # Annotate each bin with the percentage
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom')
    
    # Customize the plot
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.title('Percentage of Missing Values for Each Feature')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.show()

missing_values_by_feature(data)
#%% Analyze the distribution of the labels

def target_distribution(data):
    # Create the histogram with custom bin edges and colors
    plt.figure(figsize=(8, 6))
    bins = plt.hist(data["TARGET"], bins=[-0.5, 0.5, 1.5], edgecolor='k', alpha=0.7)
    
    # Calculate and annotate percentages
    bin_counts = bins[0]
    bin_edges = bins[1]
    total = len(data["TARGET"])
    
    for count, edge in zip(bin_counts, bin_edges):
        if edge != bin_edges[-1]:  # Skip the last edge
            percentage = (count / total) * 100
            plt.text(edge + 0.5, count, f'{percentage:.2f}%', ha='center', va='bottom')
    
    # Customize the plot
    plt.xlabel('Target')
    plt.ylabel('Frequency')
    plt.title('TARGET Feature Distribution')
    plt.xticks([0, 1], labels=['0', '1'])
    plt.grid(axis='y')
    plt.show()

target_distribution(data) # We can observe that our dataset is unbalanced, because the loans with non late payments are over represented compared to loans with late payments.
#%% Outliers dectection
def summarize_data(df):
    # Initialize empty dictionaries to store top 10 max and min values for each feature
    top_10_max = {}
    top_10_min = {}
    
    # Initialize an empty DataFrame to store unique values
    unique_values_df = pd.DataFrame(index=df.columns, columns=['Unique_Values'])
    
    # Iterate through each column in the dataframe
    for col in df.columns:
        # Exclude non-numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Get top 10 max and min values for the current feature
            top_10_max[col] = df[col].nlargest(10).tolist()
            top_10_min[col] = df[col].nsmallest(10).tolist()

    for col in df.columns:
        unique_values_df.loc[col, 'Unique_Values'] = df[col].unique()
    
    # Plot boxplot for each numeric feature
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()
    
    return top_10_max, top_10_min, unique_values_df

### Outliers for demographics features ###
print("----------DEMOGRAPHICS------------")
top_10_max, top_10_min, unique_values_df = summarize_data(data[demographics])
print("\nUnique Values:")
print(unique_values_df)

### Outliers for family_count features ###
print("\n\n----------FAMILY_COUNT------------")
top_10_max, top_10_min, unique_values_df = summarize_data(data[family_count])
print("Top 10 Maximum Values:")
print(top_10_max)
print("\nTop 10 Minimum Values:")
print(top_10_min)
print("\nUnique Values:")
print(unique_values_df)

### Outliers for age_duration features ###
print("\n\n----------AGE_DURATION------------")
top_10_max, top_10_min, unique_values_df = summarize_data(data[age_duration])
print("Top 10 Maximum Values:")
print(top_10_max)
print("\nTop 10 Minimum Values:")
print(top_10_min)
print("\nUnique Values:")
print(unique_values_df)

### Outliers for region features ###
print("\n\n----------REGION------------")
top_10_max, top_10_min, unique_values_df = summarize_data(data[region])
print("Top 10 Maximum Values:")
print(top_10_max)
print("\nTop 10 Minimum Values:")
print(top_10_min)
print("\nUnique Values:")
print(unique_values_df)

### Outliers for amounts features ###
top_10_max, top_10_min, unique_values_df = summarize_data(data[amounts])
print("\n\n----------AMOUNTS------------")
print("Top 10 Maximum Values:")
print(top_10_max)
print("\nTop 10 Minimum Values:")
print(top_10_min)
print("\nUnique Values:")
print(unique_values_df)


### Outliers for inquiry features ###
top_10_max, top_10_min, unique_values_df = summarize_data(data[inquiry])
print("\n\n----------INQUIRY------------")
print("Top 10 Maximum Values:")
print(top_10_max)
print("\nTop 10 Minimum Values:")
print(top_10_min)
print("\nUnique Values:")
print(unique_values_df)
#%% Handling outliers

# For the feature NAME_FAMILY_STATUS: Replace Widow and unknown with Single
# For the feature DAYS_EMPLOYED: Replace outliers with the median
# Transform DAYS_BIRTH with AGE in years

data['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS'].replace('unknown', 'Single / not married')
data['AGE'] = (-data['DAYS_BIRTH'] / 365).astype(int) # ! Transform DAYS_BIRTH with AGE
data['YEAR_EMPLOYED'] = (-data['DAYS_EMPLOYED'] / 365).astype(int)
data = data.drop('DAYS_BIRTH', axis=1)
data = data.drop('DAYS_EMPLOYED', axis=1)

# Re-update the "age_duration" group because we drop the column "DAYS_BIRTH" (resp. "DAYS_EMPLOYED") and replace by "AGE" (resp. "YEAR_EMPLOYED")
age_duration = ['AGE', 'YEAR_EMPLOYED']

feature_groups = {'demographic': demographics,
                  'family_count': family_count,
                  'age_duration': age_duration,
                  'region': region,
                  'amounts': amounts,
                  'inquiry': inquiry
                 }
# Check if it exists one client in our database who work more year than his actual age
filtered_data = data[data['YEAR_EMPLOYED'] >= data['AGE']] 
if len(filtered_data) == 0:
    print("NO OUTILERS")
#%% Feature engineering

# Percent Applicant has been employed in thier life:
#   ['EMPLOYMENT_PERCENTAGE'] = ['DAYS_EMPLOYED'] / ['DAYS_BIRTH']
# Anunity Income Ratio:
#   ['ANN_INC_RATIO'] = ['AMT_ANNUITY'] / ['AMT_INCOME_TOTAL']
# Credit Income Ratio:
#   ['CRED/INC_RATIO'] = ['AMT_CREDIT'] / ['AMT_INCOME_TOTAL']
# GOODS_PRICE/CREDIT:
#   ['GOODS_PRICE/CREDIT'] = ['AMT_GOODS_PRICE'] / ['AMT_CREDIT']

# Percent Applicant has been employed in thier life
data['EMPLOYMENT_PERCENTAGE'] = data['YEAR_EMPLOYED'] / data['AGE']

# Anunity Income Ratio
data['ANN_INC_RATIO']=data['AMT_ANNUITY']/data['AMT_INCOME_TOTAL']

#Credit Income Ratio 
data['CRED_INC_RATIO']=data['AMT_CREDIT']/data['AMT_INCOME_TOTAL']

# GOODS_PRICE/CREDIT
data['GOODS_PRICE_CREDIT']=data['AMT_GOODS_PRICE']/data['AMT_CREDIT']

# Re-update the "age_duration" group because we drop the column "DAYS_BIRTH" (resp. "DAYS_EMPLOYED") and replace by "AGE" (resp. "YEAR_EMPLOYED")
amounts = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', "EMPLOYMENT_PERCENTAGE", "ANN_INC_RATIO", "CRED_INC_RATIO", "GOODS_PRICE_CREDIT"]

feature_groups = {'demographic': demographics,
                  'family_count': family_count,
                  'age_duration': age_duration,
                  'region': region,
                  'amounts': amounts,
                  'inquiry': inquiry
                 }
#%% Select qualitative variables and quantitative variables
target = 'TARGET'
X_train = data.drop(target,axis=1).copy()
y_train = data[target].copy()

# Numerical (continuous/discrete) and categorical features

num_feats = X_train.select_dtypes(include='number').columns.tolist()

thresh = 25

cont_feats = [feat for feat in num_feats if X_train[feat].nunique() > thresh]
disc_feats = [feat for feat in num_feats if X_train[feat].nunique() <= thresh]

cat_feats = X_train.select_dtypes(exclude='number').columns.tolist()

print(f'Features: {X_train.shape[1]}\n\n\
Continuous: {len(cont_feats)}\n\
{cont_feats}\n\n\
Discrete: {len(disc_feats)}\n\
{disc_feats}\n\n\
Categorical: {len(cat_feats)}\n\
{cat_feats}')
#%% Define the functions to analyze continuous, discrete and continuous variable

def summary(feat):
    
    if feat in cont_feats:
        cont_summary(feat)
        cont_plots(feat)
    elif feat in disc_feats:
        disc_summary(feat)
        disc_plots(feat)
    else:
        cat_summary(feat)
        cat_plots(feat)
    
    missing_flag_plot(feat)
    
    return

# --------------------------------
# Customized correlation heatmap for each feature group

def corr_heatmap(key):

    sns.set_style('white')

    group = feature_groups[key].copy()
    scale = 1 if len(pd.get_dummies(X_train[group])) < 5 else 2
    
    corr = pd.concat([pd.get_dummies(X_train[group]),y_train],axis=1)\
        .corr(numeric_only=True)
    fig = plt.figure(figsize=(6.4*scale,5.6*scale))
    ax = sns.heatmap(corr,annot=True,fmt='.2f',cmap='viridis')
    ax.set_title(f'Correlation heatmap: {key}')

    fig.tight_layout()
    plt.show()
    sns.set_style('darkgrid')
    
    return

# --------------------------------
# Customized description for continuous features

def cont_summary(feat):

    # Create an empty summary
    columns = ['dtype', 'count', 'unique', 'top_value_counts', 'missing_count',
               'missing_percentage','mean', 'std', 'min', 'median', 'max',
               'corr_with_target']
    summary = pd.DataFrame(index=[feat],columns=columns,dtype=float)
    
    # Pull the feature column in question
    col = X_train[feat].copy()
    
    # Basic statistics using the original describe method
    summary.loc[feat,['count','mean', 'std', 'min', 'median', 'max']]\
        = col.describe(percentiles=[.5]).values.transpose()
    
    # Number of unique values
    summary.loc[feat,'unique'] = col.nunique()

    # Missing values count
    summary.loc[feat,'missing_count'] = col.isnull().sum()

    # Missing values percentage
    summary.loc[feat,'missing_percentage'] = col.isnull().sum()/len(col)*100

    # Correlation with target
    summary.loc[feat,'corr_with_target'] = col.corr(y_train)
    
    int_cols = ['count', 'unique', 'missing_count']
    summary[int_cols] = summary[int_cols].astype(int)
    summary = summary.round(2).astype(str)

    # Top 3 value_counts
    value_counts = X_train[feat].value_counts().head(3)
    value_counts.index = value_counts.index.astype(float).to_numpy().round(2)
    summary.loc[feat,'top_value_counts'] = str(value_counts.to_dict())

    # Data type
    summary.loc[feat,'dtype'] = col.dtypes
    
    # Top 10 maximum values
    top10_max = col.nlargest(10).to_numpy()
    summary.loc[feat, 'top10_max_values'] = str(top10_max.round(2).tolist())

    # Top 10 minimum values
    top10_min = col.nsmallest(10).to_numpy()
    summary.loc[feat, 'top10_min_values'] = str(top10_min.round(2).tolist())
    
    return print(summary)

# --------------------------------
# Customized plots for continuous features

def cont_plots(feat,bins='auto'):
    
    n_cols = 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6.4*n_cols, 4.8))
    
    # Histogram
    sns.histplot(data=X_train,
                 x=feat,
                 bins=bins,
                 ax=axes[0],
                 color=slate_gray)
    
    # Box plots with the target as hue
    sns.boxplot(data=X_train,
                x=feat,
                y=y_train,
                ax=axes[1],
                palette=palette,
                orient='h')
    
#     KDE plots with the target as hue
    sns.kdeplot(data=X_train,
                x=feat,
                hue=y_train,
                palette=palette,
                fill=True,
                common_norm=False,
                ax=axes[2])
    
    axes[0].title.set_text('Histogram')
    axes[1].title.set_text('Box Plots')
    axes[2].title.set_text('KDE Plots')
    
    fig.tight_layout()
    plt.show()
    return

# --------------------------------
# Customized description for discrete features

def disc_summary(feat):
    
    # Create an empty summary
    columns = ['dtype', 'count', 'unique', 'missing_count',
               'missing_percentage', 'mean', 'std', 'min', 'median',
               'max', 'cv', 'corr_with_target']
    summary = pd.DataFrame(index=[feat],columns=columns,dtype=float)
    
    # Pull the feature column in question
    col = X_train[feat].copy()
    
    # Basic statistics using the original describe method
    summary.loc[feat,['count','mean', 'std', 'min', 'median', 'max']]\
    = col.describe(percentiles=[.5]).values.transpose()

    # Number of unique values
    summary.loc[feat,'unique'] = col.nunique()

    # Coefficient of Variation (CV)    
    summary.loc[feat,'cv'] = np.NaN if not col.mean() else col.std()/col.mean()

    # Missing values count
    summary.loc[feat,'missing_count'] = col.isnull().sum()

    # Missing values percentage
    summary.loc[feat,'missing_percentage'] = col.isnull().sum()/len(col)*100
    
    # Correlation with target
    summary.loc[feat,'corr_with_target'] = col.corr(y_train)
    
    int_cols = ['count','unique','missing_count']
    summary[int_cols] = summary[int_cols].astype(int)
    summary = summary.round(2).astype(str)
    
    # Data type
    summary.loc[feat,'dtype'] = col.dtypes
        
    return print(summary)

# --------------------------------
# Customized plots for discrete features

def disc_plots(feat):

    col = X_train[feat].copy()    

    n_rows = 3
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(20 * n_cols, 8.2 * n_rows))

    # Sort unique values
    unique_values = col.dropna().unique()
    unique_values.sort()

    # Value counts
    val_counts = col.dropna().value_counts()
    val_counts = val_counts.reindex(unique_values)
    val_counts_pct = val_counts/len(col)*100
    
    # Countplot
    sns.countplot(x=col, order=unique_values, palette=palette, ax=axes[0])
    axes[0].xaxis.grid(False)
    axes[0].set_title("Countplot")
    
    # Show count value if rare (less than 1%)
    lp_thresh = 1
    for i, p in enumerate(axes[0].patches):
        pct = val_counts_pct.iloc[i]
        axes[0].annotate(f'{pct:.2f}%',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0,0),
                         textcoords='offset points')
        if pct < lp_thresh:
            axes[0].annotate(val_counts.iloc[i],
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', xytext=(0,10),
                             textcoords='offset points',color='red')
    
    # Barplot
    df = pd.concat([X_train,y_train],axis=1).groupby(feat)[target].mean()*100
    df = df.reindex(unique_values)  # Reindex to match the order
    sns.barplot(x=df.index, y=df.values, palette=palette, ax=axes[1])
    axes[1].set_ylabel('Default %')
    axes[1].set_title('Bar plot of Default % by '+feat)
    axes[1].xaxis.grid(False)
    
    # Violin plot
    sns.violinplot(x=feat, y='TARGET', data=data, scale='width', inner='quartile', palette=palette, ax=axes[2])
    axes[2].set_title('Violin plot of TARGET by '+feat)

    fig.tight_layout()
    plt.show()
    
    
    # Filter out zero sizes
    df = df[df > 0]

    sizes = df.values
    labels = df.index

    # Plot the treemap 
    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, alpha=.8, color=sns.color_palette(palette, len(sizes)), ax=ax)

    # Add the class label and percentage annotations
    for i, rect in enumerate(ax.patches):
        # Print the percentage below the class label
        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2 - rect.get_height() / 6,
                 f'{sizes[i]:.2f}%', ha='center', va='top', fontsize=10, color='black')

    plt.title('Treemap of Default % by ' + feat)
    plt.axis('off')
    plt.show()
    
    return

# --------------------------------
# Customized description for categorical features

def cat_summary(feat):
    
    # Create an empty summary
    columns = ['dtype', 'count', 'unique', 'missing_count',
               'missing_percentage']
    summary = pd.DataFrame(index=[feat],columns=columns,dtype=float)
    
    # Pull the feature column in question
    col = X_train[feat].copy()
    
    # Count
    summary.loc[feat,'count'] = col.count()

    # Number of unique values
    summary.loc[feat,'unique'] = col.nunique()

    # Missing values count
    summary.loc[feat,'missing_count'] = col.isnull().sum()

    # Missing values percentage
    summary.loc[feat,'missing_percentage'] = col.isnull().sum()/len(col)*100
    
    int_cols = ['count', 'unique', 'missing_count']
    summary[int_cols] = summary[int_cols].astype(int)
    summary = summary.round(2).astype(str)

    # Data type
    summary.loc[feat,'dtype'] = col.dtypes
    
    return print(summary)

# --------------------------------
# Customized plots for categorical features

def cat_plots(feat):
    
    col = X_train[feat].copy()
    
    n_rows = 3
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(20 * n_cols, 8.2 * n_rows))
    
    # Value counts
    val_counts = col.dropna().value_counts()
    
    # Unique values
    unique_values = val_counts.index

    # Countplot with sorted order
    sns.countplot(x=col, order=unique_values, palette=palette, ax=axes[0])
    axes[0].xaxis.grid(False)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_title("Countplot")

    val_counts_pct = val_counts/len(col)*100
    
    # Show count value if rare (less than 1%)
    lp_thresh = 1
    for i, p in enumerate(axes[0].patches):
        pct = val_counts_pct.iloc[i]
        axes[0].annotate(f'{pct:.2f}%',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0,0),
                         textcoords='offset points')
        if pct < lp_thresh:
            axes[0].annotate(val_counts.iloc[i],
                             (p.get_x() + p.get_width()/2., p.get_height()),
                             ha='center', va='bottom', xytext=(0,10),
                             textcoords='offset points',color='red')
            
    # Barplot with the same order
    df = pd.concat([X_train,y_train],axis=1).groupby(feat)[target]\
        .mean()*100
    sns.barplot(x=df.index, y=df.values, order=unique_values, palette=palette,
                ax=axes[1])
    axes[1].set_ylabel('Default %')
    axes[1].xaxis.grid(False)
    axes[1].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[1].set_title('Bar plot of Default % by '+feat)
    
    # Violin plot
    sns.violinplot(x=feat, y='TARGET', data=data, scale='width', inner='quartile', ax=axes[2])
    axes[2].set_title('Violin Plot of TARGET by '+feat)
    axes[2].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    plt.show()

    


    # Filter out zero sizes
    df = df[df > 0]

    sizes = df.values
    labels = df.index

    # Plot the treemap 
    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, alpha=.8, color=sns.color_palette(palette, len(sizes)), ax=ax)

    # Add the class label and percentage annotations
    for i, rect in enumerate(ax.patches):
        # Print the percentage below the class label
        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2 - rect.get_height() / 6,
                 f'{sizes[i]:.2f}%', ha='center', va='top', fontsize=10, color='black')

    plt.title('Treemap of Default % by ' + feat)
    plt.axis('off')
    plt.show()
    
    return

# --------------------------------
# Plot for the missing flag associated with a feature

def missing_flag_plot(feat):
    col = X_train[feat].isnull().astype(int)

    if not col.sum():
        return

    df = (pd.concat([col,y_train],axis=1).groupby(feat).mean()*100)\
        .reset_index()
    cols = [f'MISSING_FLAG_{feat}', 'Default %']
    df.columns = cols
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = sns.barplot(data=df,x=cols[0], y=cols[1], palette=palette)
    
    fig.tight_layout()
    plt.show()
    
    return

def feature_analysis(key):
    print("\n\n---------- "+ key +" ------------")
    group = feature_groups[key].copy()
    print(pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1))
    print("")

    for feat in group:
        print("\n\n"+"#"*80)
        print('#'*25 + " " +  feat + " " +  '#'*25)
        print("#"*80)
        summary(feat)
    
    corr_heatmap(key)

### Analysis for demographic features ###
feature_analysis("demographic")

### Analysis for family_count features ###
feature_analysis("family_count")

### Analysis for age_duration features ###
feature_analysis("age_duration")

### Analysis for region features ###
feature_analysis("region")


### Analysis for amounts features ###
feature_analysis("amounts")

### Analysis for inquiry features ###
feature_analysis("inquiry")
#%% Bivariate relationships between each pair of numerical variables

# df = pd.concat([X_train,y_train],axis=1)
# g = sns.pairplot(df, hue="TARGET")
# g.add_legend()

#%% Information Value
# The Information Value (IV) is a statistical measure used primarily in the context of credit scoring and binary classification problems in machine learning. It quantifies the predictive power of a particular feature (or variable) in distinguishing between two classes, typically the good (non-default) and bad (default) cases in credit scoring. 
# The Information Value helps in feature selection by indicating which features contribute most to the model's ability to make accurate predictions.

# Interpretation of Information Value The IV provides a quantitative measure of the feature's ability to predict the target variable. The interpretation of IV values is as follows:

# IV < 0.02: Predictive power is considered to be insignificant.
# 0.02 ≤ IV < 0.1: Predictive power is considered to be weak.
# 0.1 ≤ IV < 0.3: Predictive power is considered to be medium.
# 0.3 ≤ IV < 0.5: Predictive power is considered to be strong.
# IV ≥ 0.5: Predictive power is considered to be suspiciously high (could indicate overfitting or data leakage).

df = pd.concat([X_train,y_train],axis=1)
numeric_feats = disc_feats + cont_feats

def create_features(data, numeric_feats):
    # Generate all pairwise combinations of numeric features for interaction and ratio
    data = data.dropna()
    for comb in combinations(numeric_feats, 2):
        # Interaction term
        data[f'{comb[0]}*{comb[1]}'] = data[comb[0]] * data[comb[1]]
        # Ratio term, avoid division by zero
        data[f'{comb[0]}/{comb[1]}'] = data[comb[0]] / (data[comb[1]] + 0.0001)
    return data

def calculate_iv(df, feature, target):
    """
    Calculate the information value (IV) of a feature in a dataset,
    with added diagnostics to troubleshoot issues with the data or calculations.

    Parameters:
    df (pandas.DataFrame): the dataset
    feature (str): the name of the feature to calculate IV for
    target (str): the name of the target variable

    Returns:
    float: the information value (IV) of the feature
    """
    df = df[[feature, target]].dropna()
    n = df.shape[0]
    good = df[target].sum()
    bad = n - good

    # Early exit if there are no good or bad cases, which could corrupt further calculations.
    if good == 0 or bad == 0:
        print("Warning: No positive or negative cases available.")
        return 0

    unique_values = df[feature].unique()
    iv = 0
    for value in unique_values:
        n1 = df[df[feature] == value].shape[0]
        good1 = df[(df[feature] == value) & (df[target] == 1)].shape[0]
        bad1 = n1 - good1

        if good1 == 0 or bad1 == 0:
            #print(f"Skipping value {value} with good1 {good1} or bad1 {bad1}")
            continue

        good_rate = (good1 + 1e-10) / (good + 1e-10)
        bad_rate = (bad1 + 1e-10) / (bad + 1e-10)

        # Check before taking the log
        if good_rate <= 0 or bad_rate <= 0:
            #print(f"Invalid rates for log: good_rate={good_rate}, bad_rate={bad_rate}")
            continue

        woe = np.log(good_rate / bad_rate)
        iv_contribution = (good_rate - bad_rate) * woe
        iv += iv_contribution
        #print(f"Value: {value}, WoE: {woe}, IV contribution: {iv_contribution}")

    return iv

def iv_feature(data_feats):
    
    iv_values = {}
    for feature in data_feats.columns:
         if '*' in feature or '/' in feature:  # filter to only ratio/interaction features
             iv_values[feature] = calculate_iv(data_feats, feature, target)
             print("IV " + feature + ": " + str(iv_values[feature]))

    # Sort features by IV in descending order
    print("IV sorted (descending order)")
    sorted_iv = sorted(iv_values.items(), key=lambda x: x[1], reverse=True)
    for feature, iv in sorted_iv:
        print(feature, iv)

data_feats = create_features(df, numeric_feats)

# Example usage
iv = calculate_iv(data_feats, 'CNT_CHILDREN/AMT_INCOME_TOTAL', target)
print("IV:", iv)

#iv_feature(data_feats)
#%% Corralation for categorical variables with Cramer's Matrix
# Cramer's V heatmap is a visualization tool used to examine the association between categorical variables in a dataset. 
# It uses Cramer's V statistic, which is a measure of the strength of association between two nominal variables, to generate a heatmap that highlights these relationships. This is particularly useful in exploratory data analysis to identify potential interactions or dependencies between categorical features.

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

def create_cramers_v_matrix(df):
    df[target] =  df[target].astype('object')
    cols = df.select_dtypes(include=['category', 'object']).columns
    cramers = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for col1 in cols:
        for col2 in cols:
            cramers.loc[col1, col2] = cramers_v(df[col1], df[col2])
    np.fill_diagonal(cramers.values, 1)  # Fill diagonal with 1s for self-comparison
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cramers, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Cramer\'s V Heatmap of Categorical Variables')
    plt.show()
    
    return cramers

# Assuming df is your DataFrame
cramers_v_matrix = create_cramers_v_matrix(df)
#%% Corralation matrix for numerical variables
def numeric_corr_matrix(numearic_feats):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_feats].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()
