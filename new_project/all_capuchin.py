import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
import time
from capuchin import repair_dataset

home = str(Path.home())

adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'


############## Read datasets

### Adult
adult_df = pd.read_csv(adult_path + '/adult.csv', sep=',', header=0)


def label(row):
    if row['class'] == ' <=50K':
        return 0
    else:
        return 1

adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class', 'relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)

sensitive_feature_adult = 'sex'
target_adult = 'target'
inadmissible_features_adult = ['marital-status']
protected_adult = ' Female'

## COMPAS

COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'

COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
                    (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
                    & (COMPAS['is_recid'] != -1)
                    & (COMPAS['race'].isin(['African-American','Caucasian']))
                    & (COMPAS['c_charge_degree'].isin(['F','M']))
                    , ['race', 'age_cat', 'priors_count', 'is_recid', 'c_charge_degree']]

sensitive_feature_COMPAS = 'race'
inadmissible_features_COMPAS = []
target_COMPAS = 'is_recid'
protected_COMPAS = 'African-American'

## German credit

germanc_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'

credit_df = pd.read_csv(germanc_path + '/german_credit.csv', sep=',', header=0)
credit_df.dropna(inplace=True)
credit_df.drop(columns='foreign_worker', inplace=True)


def discretize_age(row, mean):
   if row['age'] > mean:
      return 'old'
   else:
       return 'young'


def label_german(row):
    if row['class'] == 'bad':
        return 0
    else:
        return 1

age_mean = credit_df['age'].mean()

credit_df['age'] = credit_df.apply(lambda row: discretize_age(row, age_mean), axis=1)
credit_df['class'] = credit_df.apply(lambda row: label_german(row), axis=1)

sensitive_feature_german = 'age'
inadmissible_features_german = []
target_german = 'class'
protected_german = 'young'

########



def repair_with_capuchin(df, sensitive_feature, protected, inadmissible_features, target, name):


    def generate_binned_df(df_1):
        columns2_drop = []
        df_ = df_1.copy()
        for i in list(df_):
            if i not in [target, 'outcome'] and (df_[i].dtype != object and len(df_[i].unique()) > 4):
                out = pd.cut(df_[i], bins=2)
                df_.loc[:, i] = out.astype(str)

        return df_


    all_features = list(df)
    all_features.remove(target)
    admissible_features = [i for i in list(df) if
                           i not in inadmissible_features and i != sensitive_feature and i != target]

    count = 0
    method_list = []
    runtime_list = []
    kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf1.split(df):

        runtimes = [name, df.shape[0], df.shape[1]]

        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        X_train = train_df.loc[:, all_features]

        y_train = train_df.loc[:, target]

        X_test = test_df.loc[:, all_features]

        y_test = test_df.loc[:, target]

        capuchin_df = df.copy()
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

        capuchin_repair_pipeline = Pipeline(steps=[('generate_binned_df', FunctionTransformer(generate_binned_df, validate=False)),
                                                   ('repair', FunctionTransformer(repair_dataset, kw_args={
                                                       'admissible_attributes': admissible_features,
                                                       'sensitive_attribute': sensitive_feature,
                                                       'target': target}, validate=False))])

        capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
                                            ('clf',
                                             LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                                max_iter=100000, multi_class='auto'))])

        start_time_capuchin = time.time()

        print('Start repairing training set with capuchin')
        to_repair = pd.concat([X_train.loc[:, capuchin_columns], y_train], axis=1)

        train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)
        end_time_capuchin = time.time() - start_time_capuchin

        runtimes.extend([end_time_capuchin, count + 1])

        runtime_list.append(runtimes)

        print('Finished repairing training set with capuchin')
        y_train_repaired = train_repaired.loc[:, [target]].to_numpy()
        X_train_repaired = train_repaired.loc[:, capuchin_columns]

        X_test_capuchin = generate_binned_df(X_test.loc[:, capuchin_columns])

        capuchin_pipeline.fit(X_train_repaired, np.ravel(y_train_repaired))
        predicted_capuchin = capuchin_pipeline.predict(X_test_capuchin)
        predicted_capuchin_proba = capuchin_pipeline.predict_proba(X_test_capuchin)[:, 1]
        rod_capuchin = ROD.ROD(y_pred=predicted_capuchin_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                               admissible=X_test.loc[:, admissible_features],
                               protected=protected, name=name)

        f1_capuchin = f1_score(np.ravel(y_test.to_numpy()), predicted_capuchin)

        predicted_capuchin_train = capuchin_pipeline.predict(X_train_repaired)
        predicted_capuchin_proba_train = capuchin_pipeline.predict_proba(X_train_repaired)[:, 1]
        rod_capuchin_train = ROD.ROD(y_pred=predicted_capuchin_proba_train, sensitive=X_train_repaired.loc[:, [sensitive_feature]],
                               admissible=X_train_repaired.loc[:, admissible_features],
                               protected=protected, name=name)

        f1_capuchin_train = f1_score(np.ravel(y_train_repaired), predicted_capuchin_train)




        method_list.append([name, 'test', rod_capuchin, f1_capuchin, capuchin_columns, len(capuchin_columns), count + 1])
        method_list.append(
            [name, 'train', rod_capuchin_train, f1_capuchin_train, capuchin_columns, len(capuchin_columns), count + 1])


        count += 1

    method_df = pd.DataFrame(method_list, columns=['Problem', 'Set', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])

    method_df.to_csv(
            path_or_buf=results_path + '/capuchin_sets_' + name + '.csv', index=False)

    runtimes_df = pd.DataFrame(runtime_list, columns=['Problem', 'Rows', 'Columns', 'Runtime', 'Fold'])

    runtimes_df.to_csv(
            path_or_buf=results_path + '/capuchin_sets_runtimes_' + name + '.csv', index=False)


#repair_with_capuchin(credit_df, sensitive_feature_german, protected_german, inadmissible_features_german, target_german, 'german_credit')
#repair_with_capuchin(COMPAS, sensitive_feature_COMPAS, protected_COMPAS, inadmissible_features_COMPAS, target_COMPAS, 'COMPAS')
repair_with_capuchin(adult_df, sensitive_feature_adult, protected_adult, inadmissible_features_adult, target_adult, 'adult')

