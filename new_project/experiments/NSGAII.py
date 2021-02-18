import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from causality.d_separation import d_separation
import multiprocessing as mp
from ROD import ROD
from numpy.linalg import norm
from sklearn.model_selection import KFold
from new_project.tests.test_evolutionary import evolution
import time
from causal_filter import causal_filter

home = str(Path.home())

adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/Final_results'
COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'
COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')
germanc_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
credit_df = pd.read_csv(germanc_path + '/german_credit.csv', sep=',', header=0)
adult_df = pd.read_csv(adult_path + '/adult.csv', sep=',', header=None, names=["age", "workclass", "fnlwgt",
                        "education", "education_num",
                        "marital_status", "occupation",
                        "relationship", "race", "sex",
                        "capital_gain", "capital_loss",
                        "hours_per_week", "native_country", "income"])

def label(row):
    if row['income'] == ' <=50K':
        return 0
    else:
        return 1

adult_df['income'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['relationship', 'race', 'native_country', 'fnlwgt', 'education_num'], inplace=True)

####################### COMPAS

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
                    (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
                    & (COMPAS['is_recid'] != -1)
                    & (COMPAS['race'].isin(['African-American','Caucasian']))
                    & (COMPAS['c_charge_degree'].isin(['F', 'M']))
                    , ['race', 'age', 'age_cat', 'priors_count', 'is_recid', 'c_charge_degree']]

###################### German

credit_df.dropna(inplace=True)
credit_df.drop(columns='foreign_worker', inplace=True)


def discretize_age(row, mean):
   if row['age'] > mean:
      return 'old'
   else:
       return 'young'


def label_2(row):
    if row['class'] == 'bad':
        return 0
    else:
        return 1

age_mean = credit_df['age'].mean()

credit_df['age'] = credit_df.apply(lambda row: discretize_age(row, age_mean), axis=1)
credit_df['class'] = credit_df.apply(lambda row: label_2(row), axis=1)

###################################


def evaluate_NSGAII(df=None, target=None, sensitive=None, inadmissible=None, protected=None, name=None, sampling=None):

    admissible_features = [i for i in list(df) if
                           i not in inadmissible and i != sensitive and i != target]

    sensitive_features = [sensitive, inadmissible]

    all_features = list(df)
    all_features.remove(target)

    count = 0
    results = []
    kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf1.split(df):
        start_time = time.time()
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        X_train = train_df.loc[:, all_features]

        y_train = train_df.loc[:, target]

        X_test = test_df.loc[:, all_features]

        y_test = test_df.loc[:, target]

        f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

        numerical_features = []
        categorical_features = []

        for f in all_features:
            if df[f].dtype != np.dtype('O'):
                numerical_features.append(f)
            else:
                categorical_features.append(f)

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)], remainder='passthrough')

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)


        encoded_features = preprocessor.transformers_[0][1].named_steps[
            'encoder'].get_feature_names(categorical_features)
        all_encoded_features = encoded_features.tolist() + numerical_features

        selected_array = evolution(X_train, np.ravel(y_train.to_numpy()), scorers=[f1], cv_splitter=3, df_train=train_df,
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
        selected_features = [f for idf, f in enumerate(all_encoded_features) if selected_mask[idf] == True]
        size = len(selected_features)

        test_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                      max_iter=100000, multi_class='auto')

        test_clf.fit(X_train[:, selected_mask], np.ravel(y_train.to_numpy()))

        predicted_genetic = test_clf.predict(X_test[:, selected_mask])
        predicted_genetic_proba = test_clf.predict_proba(X_test[:, selected_mask])[:, 1]
        sensitive_df = test_df.loc[:, sensitive_features]
        sensitive_df.reset_index(inplace=True, drop=True)
        outcomes_df = pd.DataFrame(predicted_genetic_proba, columns=['outcome'])
        features_df = test_df.loc[:, admissible_features]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([sensitive_df, features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_test = ROD(y_pred=predicted_genetic_proba, df=test_df, sensitive=sensitive,
                        admissible=admissible_features, protected=protected, mb=mb)
        f1_genetic = f1_score(np.ravel(y_test.to_numpy()), predicted_genetic)

        end_time = time.time() - start_time

        count += 1

        print('ROD: ' + str(rod_test))
        print('F1: ' + str(f1_genetic))
        print('size: ' + str(size))

        results.append(['NSGAII', rod_test, f1_genetic, selected_features, size, end_time, count])


    results_df = pd.DataFrame(results, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Runtime', 'Fold'])

    results_df.to_csv(path_or_buf=results_path + '/NSGAII_raw/' + name + '.csv',
                      index=False)

    return results_df

if __name__ == '__main__':
    mp.set_start_method('fork')
    #COMPAS_NSGAII = evaluate_NSGAII(COMPAS, target='is_recid', sensitive= 'race', inadmissible='', protected='African-American', name= 'COMPAS')
    adult_NSGAII = evaluate_NSGAII(adult_df, target='income', sensitive= 'sex', inadmissible='marital_status', protected=' Female', name= 'adult',
                                   sampling=0.05)
    #german_credit = evaluate_NSGAII(credit_df, target='class', sensitive= 'age', inadmissible='', protected='young', name= 'german')

    print(adult_NSGAII[['ROD', 'F1', 'Fold']])
