import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from fmeasures.ROD import ROD
from numpy.linalg import norm
from sklearn.model_selection import KFold
from evolutionary import evolution
import time
from causality.causal_filter import causal_filter

# home = str(Path.home())
#
# adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
# results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/Final_results'
# COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'
# COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')
# germanc_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
# credit_df = pd.read_csv(germanc_path + '/german_credit.csv', sep=',', header=0)
# adult_df = pd.read_csv(adult_path + '/adult.csv', sep=',', header=None, names=["age", "workclass", "fnlwgt",
#                         "education", "education_num",
#                         "marital_status", "occupation",
#                         "relationship", "race", "sex",
#                         "capital_gain", "capital_loss",
#                         "hours_per_week", "native_country", "income"])
#
# def label(row):
#     if row['income'] == ' <=50K':
#         return 0
#     else:
#         return 1
#
# adult_df['income'] = adult_df.apply(lambda row: label(row), axis=1)
# adult_df.drop(columns=['relationship', 'race', 'native_country', 'fnlwgt', 'education_num'], inplace=True)
#
# ####################### COMPAS
#
# COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
#                     (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
#                     & (COMPAS['is_recid'] != -1)
#                     & (COMPAS['race'].isin(['African-American','Caucasian']))
#                     & (COMPAS['c_charge_degree'].isin(['F', 'M']))
#                     , ['race', 'age', 'age_cat', 'priors_count', 'is_recid', 'c_charge_degree']]
#
# ###################### German
#
# credit_df.dropna(inplace=True)
# credit_df.drop(columns='foreign_worker', inplace=True)
#
#
# def discretize_age(row, mean):
#    if row['age'] > mean:
#       return 'old'
#    else:
#        return 'young'
#
#
# def label_2(row):
#     if row['class'] == 'bad':
#         return 0
#     else:
#         return 1
#
# age_mean = credit_df['age'].mean()
#
# credit_df['age'] = credit_df.apply(lambda row: discretize_age(row, age_mean), axis=1)
# credit_df['class'] = credit_df.apply(lambda row: label_2(row), axis=1)

###################################


def evaluate_NSGAII(array=None, df=None, target=None, sensitive=None, inadmissible=None, protected=None, sampling=None):

    if inadmissible != None:
        admissible_features = [i for i in list(df) if
                               i not in inadmissible and i != sensitive and i != target]
    else:
        admissible_features = [i for i in list(df) if i != sensitive and i != target]

    if inadmissible != None:
        sensitive_features = [sensitive, inadmissible]
    else:
        sensitive_features = [sensitive]

    all_features = list(df)
    all_features.remove(target)
    train_df = df.copy()

    train_df.reset_index(inplace=True, drop=True)
    y_train = np.ravel(train_df.loc[:, target].to_numpy())

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    selected_array = evolution(array, y_train, scorers=[f1], cv_splitter=2, df_train=train_df,
                               sensitive_feature=sensitive, protected=protected, sensitive_features=sensitive_features,
                               admissible_features=admissible_features, sampling=sampling)

    feature_sets = selected_array.X
    scores = selected_array.F

    scores = scores[:, [0, 1]]

    scores[:, 0] = np.ravel(MinMaxScaler().fit_transform(scores[:, 0].reshape((scores.shape[0], 1))))

    ideal_point = np.asarray([1, 0])
    dist = np.empty((scores.shape[0], 1))

    for idx, i in enumerate(scores):
        dist[idx] = norm(i - ideal_point)

    min_dist = np.argmin(dist)

    selected_mask = feature_sets[min_dist]

    return selected_mask

# if __name__ == '__main__':
#     mp.set_start_method('fork')
#     #COMPAS_NSGAII = evaluate_NSGAII(COMPAS, target='is_recid', sensitive= 'race', inadmissible='', protected='African-American', name= 'COMPAS')
#     adult_NSGAII = evaluate_NSGAII(adult_df, target='income', sensitive= 'sex', inadmissible='marital_status', protected=' Female', name= 'adult',
#                                    sampling=0.05)
#     #german_credit = evaluate_NSGAII(credit_df, target='class', sensitive= 'age', inadmissible='', protected='young', name= 'german')
#
#     print(adult_NSGAII[['ROD', 'F1', 'Fold']])
