from pathlib import Path
import pandas as pd
import subprocess
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
from causal_filter import causal_filter
import ROD
from sklearn.metrics import f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame, selection_rate, true_positive_rate, true_negative_rate
from fairexp import evaluate
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from capuchin import repair_dataset
import time
from CDP import CDP

home = str(Path.home())

capuchin_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/capuchin'
feldman_folder = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/feldman'

capuchin_credit_df = pd.read_csv(capuchin_path + '/bin_german_credit.csv', sep=',', header=0)

capuchin_credit_df.drop(columns=['Unnamed: 0', 'purpose', 'personal_status_sex', 'other_debtors',
                                 'property', 'other_installment_plans', 'housing', 'job', 'telephone',
                                 'foreign_worker', 'credit_history'], inplace=True)

target = 'default'
sensitive = 'age'
protected = 0


def encode_account_check_status(row):

    if row['account_check_status'] == '< 0 DM':
        return 0
    elif row['account_check_status'] == 'no checking account':
        return 1
    elif row['account_check_status'] == '0 <= ... < 200 DM':
        return 2
    else:
        return 3


def encode_savings(row):
    if row['savings'] == 'unknown/ no savings account':
        return 0
    elif row['savings'] == '... < 100 DM':
        return 1
    elif row['savings'] == '100 <= ... < 500 DM':
        return 2
    elif row['savings'] == '500 <= ... < 1000 DM ':
        return 3
    else:
        return 4

def encode_present_emp_since(row):
    if row['present_emp_since'] == 'unemployed':
        return 0
    elif row['present_emp_since'] == '... < 1 year ':
        return 1
    elif row['present_emp_since'] == '1 <= ... < 4 years':
        return 2
    elif row['present_emp_since'] == '4 <= ... < 7 years':
        return 3
    else:
        return 4

def prepare_df_calmon(df):
    df_ = df.copy()
    df_['account_check_status'] = df_.apply(lambda row: encode_account_check_status(row), axis=1)
    df_['savings'] = df_.apply(lambda row: encode_savings(row), axis=1)
    df_['present_emp_since'] = df_.apply(lambda row: encode_present_emp_since(row), axis=1)

    return df_


fold = 0
kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf1.split(capuchin_credit_df):
    train_df = capuchin_credit_df.iloc[train_index]
    test_df = capuchin_credit_df.iloc[test_index]
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    train_df_enc = prepare_df_calmon(train_df)
    test_df_enc = prepare_df_calmon(test_df)

    train_df_enc.to_csv(path_or_buf=feldman_folder + '/train_german_' + str(fold) + '.csv', index=False)

    subprocess.run("BlackBoxAuditing-repair " + feldman_folder + '/train_german_' + str(fold) + '.csv '
                   + feldman_folder + '/repaired_train_german_' + str(fold) + '.csv ' + '1.0 ' + "'True' " + '-p '
                   + 'age ' + '-i ' + 'default', shell=True)

    repaired_train_df = pd.read_csv(feldman_folder + '/repaired_train_german_' + str(fold) + '.csv',
                                    sep=',', header=0)

    numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, [i for i in list(repaired_train_df) if i not in (target, sensitive)])],
    remainder='passthrough')

    feldman_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('clf',
                                         LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                            max_iter=100000, multi_class='auto'))])

    model = GridSearchCV(feldman_pipeline, param_grid={
        'clf__penalty': ['l2'], 'clf__C': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5], 'clf__solver': ['lbfgs'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
    },
        n_jobs=-1,
        scoring='f1', cv=5)

    ##### Feldman

    X_train = repaired_train_df.loc[:, [i for i in list(repaired_train_df) if i != target]]
    y_train = np.ravel(repaired_train_df.loc[:, target].to_numpy())

    X_test = test_df_enc.loc[:, [i for i in list(test_df) if i != target]]
    y_test = np.ravel(test_df_enc.loc[:, target].to_numpy())

    admissible_features = [i for i in list(train_df) if i not in (target, sensitive)]

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    predicted_proba = model.predict_proba(X_test)[:, 1]

    outcomes_df = pd.DataFrame(predicted_proba, columns=['outcome'])
    features_df = test_df.reset_index(drop=True)

    candidate_df = pd.concat([features_df, outcomes_df], axis=1)

    JCIT, mb = causal_filter(candidate_df, [sensitive])

    rod = ROD.ROD(y_pred=predicted_proba, df=test_df, sensitive=sensitive,
                          admissible=admissible_features, protected=protected, mb=mb)

    dp = demographic_parity_difference(y_test, predicted,
                                               sensitive_features=test_df.loc[:, sensitive])
    tpr = MetricFrame(true_positive_rate, y_test, predicted,
                              sensitive_features=test_df.loc[:, sensitive])
    tpb = tpr.difference()
    tnr = MetricFrame(true_negative_rate, y_test, predicted,
                              sensitive_features=test_df.loc[:, sensitive])
    tnb = tnr.difference()

    f1 = f1_score(y_test, predicted)

    print('ROD feldman ' + ': ' + str(rod))
    print('F1 feldman ' + ': ' + str(f1))
    print('DP feldman ' + ': ' + str(dp))
    print('TPB feldman ' + ': ' + str(tpb))
    print('TNB feldman ' + ': ' + str(tnb))

    fold += 1


    ##### Capuchin

    X_test_capuchin = test_df.loc[:, [i for i in list(test_df) if i != target]]
    y_test_capuchin = np.ravel(test_df.loc[:, target].to_numpy())

    admissible_features = [i for i in list(train_df) if i not in (target, sensitive)]

    all_features_capuchin = [i for i in list(train_df) if i != target]

    start_time_capuchin = time.time()

    capuchin_repair_pipeline = Pipeline(steps=[
                                               ('repair', FunctionTransformer(repair_dataset, kw_args={
                                                   'admissible_attributes': admissible_features,
                                                   'sensitive_attribute': sensitive,
                                                   'target': target}))])
    
    categorical = []
    for i in list(train_df):
        if i != target and train_df[i].dtype == object:
            categorical.extend([i])

    numerical = []
    for i in list(train_df):
        if i != target and train_df[i].dtype != object:
            numerical.extend([i])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('minmax', MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical), ('num', numerical_transformer, numerical)],
        remainder='passthrough')

    capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('clf',
                                         LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                            max_iter=100000, multi_class='auto', n_jobs=-1))])

    capuchin_model = GridSearchCV(capuchin_pipeline, param_grid={
        'clf__penalty': ['l2'], 'clf__C': [0.2, 0.5, 1.0, 1.2, 1.5, 2.0], 'clf__solver': ['lbfgs'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
    },
                                  n_jobs=-1,
                                  scoring='f1', cv=5)

    train_repaired = capuchin_repair_pipeline.fit_transform(train_df)
    print(train_repaired.shape, train_df.shape)
    X_train_repaired = train_repaired.loc[:, all_features_capuchin]
    y_train_capuchin = np.ravel(train_repaired.loc[:, target].to_numpy())

    capuchin_model.fit(X_train_repaired, y_train_capuchin)

    predicted_capuchin = capuchin_model.predict(X_test_capuchin)
    predicted_proba_capuchin = capuchin_model.predict_proba(X_test_capuchin)[:, 1]

    outcomes_df = pd.DataFrame(predicted_proba_capuchin, columns=['outcome'])
    features_df = test_df.reset_index(drop=True)

    candidate_df = pd.concat([features_df, outcomes_df], axis=1)

    JCIT, mb = causal_filter(candidate_df, [sensitive])

    rod_capuchin = ROD.ROD(y_pred=predicted_proba_capuchin, df=test_df, sensitive=sensitive,
                  admissible=admissible_features, protected=protected, mb=mb)

    dp_capuchin = demographic_parity_difference(y_test, predicted_capuchin,
                                       sensitive_features=test_df.loc[:, sensitive])
    tpr_capuchin = MetricFrame(true_positive_rate, y_test, predicted_capuchin,
                      sensitive_features=test_df.loc[:, sensitive])
    tpb_capuchin = tpr_capuchin.difference()
    tnr_capuchin = MetricFrame(true_negative_rate, y_test, predicted_capuchin,
                      sensitive_features=test_df.loc[:, sensitive])
    tnb_capuchin = tnr_capuchin.difference()
    cdp_capuchin = CDP(y_test, predicted_capuchin, test_df, sensitive, admissible_features)
    f1_capuchin = f1_score(y_test, predicted_capuchin)

    print('ROD capuchin ' + ': ' + str(rod_capuchin))
    print('F1 capuchin ' + ': ' + str(f1_capuchin))
    print('DP capuchin ' + ': ' + str(dp_capuchin))
    print('TPB capuchin ' + ': ' + str(tpb_capuchin))
    print('TNB capuchin ' + ': ' + str(tnb_capuchin))
    print('CDP capuchin ' + ': ' + str(cdp_capuchin))



