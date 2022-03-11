##### register
#
#


import lime
import lime.lime_tabular

import pandas as pd
import numpy as np
import lightgbm as lgb

# For converting textual categories to integer labels
from sklearn.preprocessing import LabelEncoder

# for creating train test split
from sklearn.model_selection import train_test_split

# specify your configurations as a dict
lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': None,
    'num_iteration': 100,
    'num_threads': 7,
    'max_depth': 12,
    'min_data_in_leaf': 100,
    'alpha': 0.5}

# reading the titanic data
df_titanic = pd.read_csv(r'/Users/300011432/Downloads/all/train.csv')

# data preparation
df_titanic.fillna(0, inplace=True)

le = LabelEncoder()

feat = ['PassengerId', 'Pclass_le', 'Sex_le', 'SibSp_le', 'Parch', 'Fare']

# label encoding textual data
df_titanic['Pclass_le'] = le.fit_transform(df_titanic['Pclass'])
df_titanic['SibSp_le'] = le.fit_transform(df_titanic['SibSp'])
df_titanic['Sex_le'] = le.fit_transform(df_titanic['Sex'])

# using train test split to create validation set
X_train, X_test, y_train, y_test = train_test_split(df_titanic[feat], df_titanic[['Survived']], test_size=0.3)

# def lgb_model(X_train,y_train,X_test,y_test,lgb_params):
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

# training the lightgbm model
model = lgb.train(lgb_params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)


# this is required as LIME requires class probabilities in case of classification example
# LightGBM directly returns probability for class 1 by default

def prob(data):
    return np.array(list(zip(1 - model.predict(data), model.predict(data))))


explainer = lime.lime_tabular.LimeTabularExplainer(df_titanic[model.feature_name()].astype(int).values,
                                                   mode='classification', training_labels=df_titanic['Survived'], feature_names=model.feature_name())

# asking for explanation for LIME model
i = 1
exp = explainer.explain_instance(df_titanic.loc[i, feat].astype(int).values, prob, num_features=5)

# Code for SP-LIME
import warnings
from lime import submodular_pick

# Remember to convert the dataframe to matrix values
# SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model
sp_obj = submodular_pick.SubmodularPick(explainer, df_titanic[model.feature_name()].values, prob, num_features=5,num_exps_desired=10)

[exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]