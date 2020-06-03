import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from d_separation import d_separation
import multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy.linalg import norm
import time
from capuchin import repair_dataset

home = str(Path.home())

germanc_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

credit_df = pd.read_csv(germanc_path + '/german_credit.csv', sep=',', header=0)
credit_df.dropna(inplace=True)
credit_df.drop(columns='foreign_worker', inplace=True)


def discretize_age(row, mean):
   if row['age'] > mean:
      return 'old'
   else:
       return 'young'


def label(row):
    if row['class'] == 'bad':
        return 0
    else:
        return 1

age_mean = credit_df['age'].mean()

sensitive_feature = 'age'
inadmissible_features = []
target = 'class'
admissible_features = [i for i in list(credit_df) if
                       i not in inadmissible_features and i != sensitive_feature and i != target]

credit_df['age'] = credit_df.apply(lambda row: discretize_age(row, age_mean), axis=1)
credit_df['class'] = credit_df.apply(lambda row: label(row), axis=1)

all_features = list(credit_df)
all_features.remove(target)
all_2_combinations = list(itertools.combinations(all_features, 2))

def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in [target, 'outcome'] and (df_[i].dtype != object and len(df_[i].unique()) > 4):
            out = pd.cut(df_[i], bins=2)
            df_.loc[:, i] = out.astype(str)

    return df_

complexity = 2
CF = False
count = 0
method_list = []
kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf1.split(credit_df):

    runtimes = [len(all_2_combinations), complexity]
    count_transformations = 0
    filtered_transformations = 0
    time_2_create_transformations = 0
    time_2_CF = 0
    time_2_FR = 0
    time_2_SR = 0

    train_df = credit_df.iloc[train_index]
    test_df = credit_df.iloc[test_index]

    X_train = train_df.loc[:, all_features]

    y_train = train_df.loc[:, target]

    X_test = test_df.loc[:, all_features]

    y_test = test_df.loc[:, target]

    capuchin_df = credit_df.copy()
    capuchin_columns = list(capuchin_df)
    capuchin_columns.remove(target)

    categorical = []
    for i in capuchin_columns:
        if i != target:
            categorical.extend([i])

    categorical_transformer_3 = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor_3 = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer_3, categorical)],
        remainder='passthrough')

    capuchin_repair_pipeline = Pipeline(steps=[('generate_binned_df', FunctionTransformer(generate_binned_df)),
                                               ('repair', FunctionTransformer(repair_dataset, kw_args={
                                                   'admissible_attributes': admissible_features,
                                                   'sensitive_attribute': sensitive_feature,
                                                   'target': target}))])

    capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
                                        ('clf',
                                         LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                            max_iter=100000, multi_class='auto'))])

    start_time_capuchin = time.time()

    print('Start repairing training set with capuchin')
    to_repair = pd.concat([X_train.loc[:, capuchin_columns], y_train], axis=1)
    train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)
    end_time_capuchin = time.time() - start_time_capuchin

    runtimes.extend([end_time_capuchin])

    print(runtimes)

    print('Finished repairing training set with capuchin')
    y_train_repaired = train_repaired.loc[:, [target]].to_numpy()
    X_train_repaired = train_repaired.loc[:, capuchin_columns]

    X_test_capuchin = generate_binned_df(X_test.loc[:, capuchin_columns])

    capuchin_pipeline.fit(X_train_repaired, np.ravel(y_train_repaired))
    predicted_capuchin = capuchin_pipeline.predict(X_test_capuchin)
    predicted_capuchin_proba = capuchin_pipeline.predict_proba(X_test_capuchin)[:, 1]
    rod_capuchin = ROD.ROD(y_pred=predicted_capuchin_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                           admissible=X_test.loc[:, admissible_features],
                           protected='young', name='capuchin_credit_df')

    f1_capuchin = f1_score(np.ravel(y_test.to_numpy()), predicted_capuchin)

    method_list.append(['capuchin', rod_capuchin, f1_capuchin, capuchin_columns, len(capuchin_columns), count + 1])

    print('ROD capuchin ' + ': ' + str(rod_capuchin))
    print('F1 capuchin ' + ': ' + str(f1_capuchin))