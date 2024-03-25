# %% [markdown]
# # Loading Libraries

# %%
# !pip install category_encoders

# %%
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from google.colab import drive
# drive.mount("/content/drive")
# os.chdir("/content/drive/MyDrive/Hackathon")

# %% [markdown]
# # Loading the data

# %%
train = pd.read_csv("train_set.csv", sep=";", decimal=',')
test = pd.read_csv("test_set.csv", sep=";", decimal=',')

# %%
train.columns

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ### Datetime

# %% [markdown]
# We decide to take the column "decision_date" and split the information it contains in year, month and day. We do it for the training and the test set. 

# %%
train["decision_date"] = pd.to_datetime(train["decision_date"])
train["year"] = train["decision_date"].dt.year
train["month"] = train["decision_date"].dt.month
train["day"] = train["decision_date"].dt.day

# %%
test["decision_date"] = pd.to_datetime(test["decision_date"])
test["year"] = test["decision_date"].dt.year
test["month"] = test["decision_date"].dt.month
test["day"] = test["decision_date"].dt.day

# %% [markdown]
# ### Cyclical Encoding

# %% [markdown]
# Cyclical encoding is used to deal with periodical patterns, such as month and day. By encoding cyclical data in a way that preserves their cyclic nature, such as using sine and cosine functions, it helps the algorithm to improve its classification performance.

# %%
train["month_sin"] = np.sin(2 * np.pi * train["month"]/31.0)
train["month_cos"] = np.cos(2 * np.pi * train["month"]/31.0)

# %%
test["month_sin"] = np.sin(2 * np.pi * test["month"]/31.0)
test["month_cos"] = np.cos(2 * np.pi * test["month"]/31.0)

# %%
train["day_sin"] = np.sin(2 * np.pi * train["day"]/31.0)
train["day_cos"] = np.cos(2 * np.pi * train["day"]/31.0)

# %%
test["day_sin"] = np.sin(2 * np.pi * test["day"]/31.0)
test["day_cos"] = np.cos(2 * np.pi * test["day"]/31.0)

# %% [markdown]
# ### Company Occurences

# %% [markdown]
# Here the column company_ID is not very informative, thus some form of transformation is needed. We apply target encoding, meaning that we exploit the conditional distribution of the transformed column, conditioned on the target column.

# %%
rep_id = train.groupby("company_ID")["application_ID"].nunique()
train["company"] = train["company_ID"].map(rep_id)

# %%
rep_id_test = test.groupby("company_ID")["application_ID"].nunique()
test["company"] = test["company_ID"].map(rep_id_test)

# %% [markdown]
# ### Late Payment Score

# %% [markdown]
# Since this column presents several null values, we impute them with 0, as this column has a range [1.0 - 20.0]. Our assumption was that - since the values is missing - then it means that the company is not late with the payments. Thus, by assigning 0, this means that the company has a good reputation of paying on time. We want our model to exploit this information.

# %%
train["late_payment_score"] = train["late_payment_score"].fillna(0)

# %%
test["late_payment_score"] = test["late_payment_score"].fillna(0)

# %% [markdown]
# ### juridical_form

# %% [markdown]
# The test set has some categories that are missing in the training set. Hence, we must replace with NaN the category that is not in the training set, but appears in the test set, because in this way the sklearn pipeline will effectively manage the missing value as one additional category.

# %%
train["juridical_form"].loc[train["juridical_form"] == "PS"] = np.NaN

# %% [markdown]
# ## Feature creation

# %% [markdown]
# Feature creation aims at constructing new variables as combination of the already existing ones, with the goal to enhance the model performance.

# %% [markdown]
# 1. variable age is positively skewed (long tailed on the right). For this reason, add a new variable age_sqrt

# %%
train['age_sqrt'] = np.sqrt(train['age'])
test['age_sqrt'] = np.sqrt(test['age'])

# %% [markdown]
# 2. moreover, age may have some relationship with the revenues. Let's include one interaction term

# %%
train['age_x_revenues'] = train['age']*train['revenues']
test['age_x_revenues'] = test['age']*test['revenues']

# %% [markdown]
# 3) Revenues themselves suffer the same skewness age suffers from

# %%
train['revenues_sqrt'] = np.sqrt(train['revenues'])
test['revenues_sqrt'] = np.sqrt(test['revenues'])

# %% [markdown]
# 4. cash asset ratio is strongly skewed to the right as well.

# %%
train['cash_asset_ratio_sqrt'] = np.sqrt(train['cash_asset_ratio'])
test['cash_asset_ratio_sqrt'] = np.sqrt(test['cash_asset_ratio'])

# %% [markdown]
# 5. tangible asset ratio, instead, is negatively skewed, but it's also a bit u-shaped. Thus try to add a squared term.

# %%
train['tangible_assets_ratio_squared'] = train['tangible_assets_ratio']*train['tangible_assets_ratio']
test['tangible_assets_ratio_squared'] = test['tangible_assets_ratio']*test['tangible_assets_ratio']

# %% [markdown]
# # Preprocessing

# %% [markdown]
# We divide each column in a specified list, based on the technique we want to apply to it. Then we use indexing in the ColumnTransformer pipeline in order to apply to each column the desired technique.

# %%
cols_target = ["province", "juridical_form", "industry_sector"]

cols_ordinal = ["year", "last_statement_age"]

cols_rest = ["month_sin", "month_cos", "day_sin", "day_cos", "cr_available"]

cols_standard = ["company", "external_score_ver01", "age", "external_score_ver02", "late_payment_score",
                 "external_score_adverse", "gross_margin_ratio", "core_income_ratio", "cash_asset_ratio",
                 "consolidated_liabilities_ratio", "tangible_assets_ratio", "revenues",
                 'age_sqrt', 'age_x_revenues', 'revenues_sqrt', 'cash_asset_ratio_sqrt', 'tangible_assets_ratio_squared']

cols_pca = ['overrun_freq_a_revoca_autoliquidanti',
       'avg_tension_a_revoca_autoliquidanti',
       'std_tension_a_revoca_autoliquidanti',
       'max_tension_a_revoca_autoliquidanti',
       'last_tension_a_revoca_autoliquidanti',
       'avg_rel_used_a_revoca_autoliquidanti',
       'std_rel_used_a_revoca_autoliquidanti',
       'max_rel_used_a_revoca_autoliquidanti',
       'last_rel_used_a_revoca_autoliquidanti', 'overrun_freq_a_scadenza',
       'avg_rel_used_a_scadenza', 'std_rel_used_a_scadenza',
       'max_rel_used_a_scadenza', 'last_rel_used_a_scadenza',
       'avg_count_enti_affidanti', 'std_count_enti_affidanti',
       'max_count_enti_affidanti', 'last_count_enti_affidanti',
       'avg_count_numero_prima_info', 'std_count_numero_prima_info',
       'max_count_numero_prima_info', 'last_count_numero_prima_info']
col_list = cols_target + cols_ordinal + cols_rest + cols_standard + cols_pca

X = np.array(train[col_list])

y = np.array(train["days_to_default"])

# %% [markdown]
# # Training and HPT

# %% [markdown]
# In this phase, we perform the data pre-processing exploiting the pipeline experimented in day 1, which yielded positive results. We then define a set of parameters to tune, hoping that these refinements will improve the model performance. We limit our experimentations to 3 parameters, as we have computational limits.

# %% [markdown]
# Moreover, we decided to use the column "target" which consists of the predictions generated in the notebook "Hackathon Day1". 

# %%
# Selecting rows where target != 0 for the model training
train_default = train.loc[train["target"] != 0]
X = np.array(train_default[col_list])
y = np.array(train_default["days_to_default"])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
ct = ColumnTransformer(transformers=[("target_encoder", TargetEncoder(handle_unknown=-1), [0, 1, 2]),
                                     ("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [3, 4]),
                                     ("standard_scaler", StandardScaler(), [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
                                     ('pca', PCA(), [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])],
                       remainder="passthrough")

model = xgb.XGBRegressor(scale_pos_weight = 3.64)

pipeline = Pipeline(steps= [("column_transformer", ct),
                            ("xgb", model)])

# %% [markdown]
# We then define a set of parameters and their associated values to be tuned, in order to perform hyperparameter tuning via gridsearch.

# %%
param_grid = {
    'xgb__max_depth': range(2, 10, 2),
    'xgb__n_estimators': range(60, 220, 40),
    'xgb__learning_rate': [0.1, 0.05, 0.001, 0.0001],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

# %%
grid.fit(X_train, y_train)

# %%
train["predictions"] = 1498
indeces = np.array(train["predictions"].loc[train["target"] != 0].index)
X = np.array(train[col_list])
train["predictions"].loc[train["target"] != 0] = grid.predict(X[indeces, :])

# %%
mean_absolute_error(np.array(train["predictions"]), np.array(train["days_to_default"]))

# %% [markdown]
# # Final predictions

# %% [markdown]
# ### Train Model on entire Train Data

# %%
# Get rows where target != 0
train_default = train.loc[train["target"] != 0]
X = np.array(train_default[col_list])
y = np.array(train_default["days_to_default"])

# %%
grid.fit(X, y)

# %% [markdown]
# ## Load target predictions from yesterday

# %%
target_preds = pd.read_csv("predictions-day1.csv")
test["target"] = target_preds

# %%
# Default prediction is 1498
test["predictions"] = 1498

# Filter the rows where the target prediction is != 0
indeces = np.array(test["predictions"].loc[test["target"] != 0].index)
X_test = np.array(test[col_list])

# %%
# Generate predictions where predicted default != 0
test["predictions"].loc[test["target"] != 0] = grid.predict(X_test[indeces, :])
final_preds = test["predictions"]

# %%
final_preds

# %%
final_preds.to_csv("predictions-def-day2.csv", index=False, header=False)


