
# # Loading Libraries

import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import string


# # Loading the data

train = pd.read_csv("dataset/train_set.csv", sep=";", decimal=',')
test = pd.read_csv("dataset/test_set.csv", sep=";", decimal=',')

# # Feature Engineering

# ### Datetime

train["decision_date"] = pd.to_datetime(train["decision_date"])
train["year"] = train["decision_date"].dt.year
train["month"] = train["decision_date"].dt.month
train["day"] = train["decision_date"].dt.day

test["decision_date"] = pd.to_datetime(test["decision_date"])
test["year"] = test["decision_date"].dt.year
test["month"] = test["decision_date"].dt.month
test["day"] = test["decision_date"].dt.day

# ### Cyclical Encoding


train["month_sin"] = np.sin(2 * np.pi * train["month"]/31.0)
train["month_cos"] = np.cos(2 * np.pi * train["month"]/31.0)
test["month_sin"] = np.sin(2 * np.pi * test["month"]/31.0)
test["month_cos"] = np.cos(2 * np.pi * test["month"]/31.0)
train["day_sin"] = np.sin(2 * np.pi * train["day"]/31.0)
train["day_cos"] = np.cos(2 * np.pi * train["day"]/31.0)
test["day_sin"] = np.sin(2 * np.pi * test["day"]/31.0)
test["day_cos"] = np.cos(2 * np.pi * test["day"]/31.0)

# ### Company Occurences

rep_id = train.groupby("company_ID")["application_ID"].nunique()
train["company"] = train["company_ID"].map(rep_id)
rep_id_test = test.groupby("company_ID")["application_ID"].nunique()
test["company"] = test["company_ID"].map(rep_id_test)

# ### Late Payment Score

train["late_payment_score"] = train["late_payment_score"].fillna(0)
test["late_payment_score"] = test["late_payment_score"].fillna(0)

# ### juridical_form

train["juridical_form"].loc[train["juridical_form"] == "PS"] = np.NaN

# ## Score_ver03

dic = {letter : (26 -inx) for inx, letter in enumerate(list(string.ascii_uppercase))}
dic['MISSING'] = None
train['external_score_ver03'] = train['external_score_ver03'].map(dic)
test['external_score_ver03'] = test['external_score_ver03'].map(dic)


# # Preprocessing


train['ver_01_squared'] = train['external_score_ver01']*train['external_score_ver01']
train['ver_02_squared'] = train['external_score_ver02']*train['external_score_ver02']
test['ver_01_squared'] = test['external_score_ver01']*test['external_score_ver01']
test['ver_02_squared'] = test['external_score_ver02']*test['external_score_ver02']
train['ver_avg'] = train['external_score_ver01']*train['external_score_ver02']
test['ver_avg'] = test['external_score_ver01']*test['external_score_ver02']


cols_target = ["province", "juridical_form", "industry_sector", "region", "geo_area", ]
cols_ordinal = ["year", "last_statement_age", "external_score_ver03"]
cols_rest = ["month_sin", "month_cos", "day_sin", "day_cos", "cr_available"]
cols_standard = ["company", "external_score_ver01", "age", "external_score_ver02", "late_payment_score", "external_score_adverse",
                 "external_score_moderate", 
                 "gross_margin_ratio", "core_income_ratio", "cash_asset_ratio",
                 "consolidated_liabilities_ratio", "tangible_assets_ratio", "revenues", 'ver_01_squared', 'ver_02_squared']
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
y = np.array(train["target"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


ct = ColumnTransformer(transformers=[("target_encoder", OneHotEncoder(handle_unknown='ignore'), [0, 1, 2, 3, 4]),
                                     ("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [5,6,7]),
                                     ("standard_scaler", StandardScaler(), [13,14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,]),
                                     ('pca', PCA(), [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])],
                       remainder="passthrough")

model = xgb.XGBClassifier(objective="binary:logistic",
                          scale_pos_weight = 10,
                          max_depth = 24,
                          n_estimators = 180,
                          learning_rate = 0.1,
                          reg_lambda = 6,
                          base_score = 0.2,
                          alpha = 1,
)

pipeline = Pipeline(steps= [("column_transformer", ct),
                            ("xgb", model)])

pipeline.fit(X_train, y_train)
f1_score(pipeline.predict(X_test), y_test)


# Reference for how we tune hyperparamters

#param_grid = {
    #'xgb__scale_pos_weight' : [8,10,12],
    #'xgb__max_depth' : [8, 12, 14],
    #'xgb__n_estimators':[140, 180, 220],
    #'xgb__lambda' : [2,4,6],
    #'xgb__learning_rate' : [0.1, 0.05, 0.01],
    #'xgb__min_child_weight' : [1, 5]
#}

#grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=10)
#grid.fit(X_train, y_train)



# # Final predictions


pipeline.fit(X, y)
final_preds = pipeline.predict(np.array(test[col_list]))
final_preds = pd.DataFrame(final_preds)
final_preds.to_csv("predictions.csv", index=False, header=["label"])




