import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


def rank(df, target, sensitive, protected):
    categorical_features = []
    numerical_features = []
    for i in list(df):
        if i != target and df[i].dtype == object:
            categorical_features.extend([i])
        elif i != target and df[i].dtype != object:
            numerical_features.extend([i])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    preprocessor_2 = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)], remainder='passthrough')

    pipeline = Pipeline(steps=[('preprocessor', preprocessor_2),
                                       ('clf',
                                        LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                           max_iter=100000, multi_class='auto'))])

    X = df.loc[:, [i for i in list(df) if i != target]]
    y = np.ravel(df.loc[:, target].to_numpy())

    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)

    df['proba'] = proba[:, 1]

    pos = df.loc[(df[sensitive] == protected) & (df[target] == 0), 'proba']
    neg = df.loc[(df[sensitive] != protected) & (df[target] == 1), 'proba']

    promotion = pos.reset_index(drop=False)
    demotion = neg.reset_index(drop=False)
    promotion.sort_values(by='proba', inplace=True, ascending=False)
    demotion.sort_values(by='proba', inplace=True, ascending=True)

    return promotion, demotion


def massaging(df, target, sensitive, protected):

    promotion, demotion = rank(credit_df, target, sensitive, protected)

    non_protected_pos = df.loc[(df[sensitive] != protected) & (df[target] == 1)].shape[0]
    non_protected = df.loc[(df[sensitive] != protected)].shape[0]
    protected_pos = df.loc[(df[sensitive] == protected) & (df[target] == 1)].shape[0]
    protected = df.loc[(df[sensitive] == protected)].shape[0]

    M = ((protected * non_protected_pos) - (non_protected * protected_pos)) / (protected + non_protected)

    promotion_idx = promotion.iloc[:int(M), 0]
    demotion_idx = demotion.iloc[:int(M), 0]

    df_ = df.copy()

    df_.iloc[promotion_idx, list(df).index(target)] = 1
    df_.iloc[demotion_idx, list(df).index(target)] = 0

    return df_





