import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
import time
from capuchin import repair_dataset
from causal_filter import causal_filter
from sklearn.feature_selection import mutual_info_classif
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame, selection_rate, true_positive_rate, true_negative_rate


home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'
capuchin_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/capuchin'
germanc_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/'

credit_df = pd.read_csv(germanc_path + '/german_credit.csv', sep=',', header=0)
credit_df.dropna(inplace=True)
credit_df.drop(columns='foreign_worker', inplace=True)

capuchin_credit_df = pd.read_csv(capuchin_path + '/bin_german_credit.csv', sep=',', header=0)
capuchin_credit_df.dropna(inplace=True)
capuchin_credit_df.drop(columns=['foreign_worker', 'Unnamed: 0'], inplace=True)


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

def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in [target, 'outcome'] and (df_[i].dtype != object and len(df_[i].unique()) > 3):
            out = pd.cut(df_[i], bins=2)
            df_.loc[:, i] = out.astype(str)

    return df_

age_mean = credit_df['age'].mean()

sensitive_feature = 'age'
inadmissible_features = []
target = 'class'

credit_df['age'] = credit_df.apply(lambda row: discretize_age(row, age_mean), axis=1)
credit_df['class'] = credit_df.apply(lambda row: label(row), axis=1)
admissible_features = [i for i in list(credit_df) if
                       i not in inadmissible_features and i != sensitive_feature and i != target]

protected = 'young'
dataset = 'german_credit'

all_features = list(credit_df)
all_features.remove(target)


CF = False
count = 0
method_list = []
runtimes_list = []
kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf1.split(credit_df):

    runtimes = []
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

    rod_score = make_scorer(ROD.ROD, greater_is_better=True, needs_proba=True,
                            sensitive=X_train.loc[:, sensitive_feature],
                            admissible=X_train.loc[:, admissible_features],
                            protected=protected, name=dataset)

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    ########### Dropped

    categorical_features_2 = []
    numerical_features_2 = []
    start_time_dropped = time.time()
    for i in list(credit_df):
        if i != target and i not in inadmissible_features and i != sensitive_feature and credit_df[i].dtype == np.dtype(
                'O'):
            categorical_features_2.extend([i])
        elif i != target and i not in inadmissible_features and i != sensitive_feature and credit_df[
            i].dtype != np.dtype('O'):
            numerical_features_2.extend([i])

    categorical_transformer_2 = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer_2 = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    preprocessor_2 = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer_2, categorical_features_2),
            ('num', numerical_transformer_2, numerical_features_2)], remainder='passthrough')

    dropped_pipeline = Pipeline(steps=[('preprocessor', preprocessor_2),
                                       ('clf',
                                        LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                           max_iter=100000, multi_class='auto'))])

    dropped_model = GridSearchCV(dropped_pipeline, param_grid={
        'clf__penalty': ['l2'], 'clf__C': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5], 'clf__solver': ['lbfgs'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
    },
                                  n_jobs=-1,
                                  scoring='f1', cv=5)

    X_train_dropped = X_train.drop(columns=inadmissible_features + [sensitive_feature])
    X_test_dropped = X_test.drop(columns=inadmissible_features + [sensitive_feature])

    dropped_model.fit(X_train_dropped, np.ravel(y_train.to_numpy()))
    predicted_dropped = dropped_model.predict(X_test_dropped)
    predicted_dropped_proba = dropped_model.predict_proba(X_test_dropped)[:, 1]

    outcomes_df = pd.DataFrame(predicted_dropped_proba, columns=['outcome'])
    features_df = test_df.reset_index(drop=True)

    candidate_df = pd.concat([features_df, outcomes_df], axis=1)

    JCIT, mb = causal_filter(candidate_df, [sensitive_feature])

    rod_dropped = ROD.ROD(y_pred=predicted_dropped_proba, df=test_df, sensitive=sensitive_feature,
                           admissible=admissible_features, protected=protected, mb=mb)

    dp_dropped = demographic_parity_difference(np.ravel(y_test.to_numpy()), predicted_dropped,
                                                sensitive_features=test_df.loc[:, sensitive_feature])
    tpr_dropped = MetricFrame(true_positive_rate, np.ravel(y_test.to_numpy()), predicted_dropped,
                               sensitive_features=test_df.loc[:, sensitive_feature])
    tpb_dropped = tpr_dropped.difference()
    tnr_dropped = MetricFrame(true_negative_rate, np.ravel(y_test.to_numpy()), predicted_dropped,
                               sensitive_features=test_df.loc[:, sensitive_feature])
    tnb_dropped = tnr_dropped.difference()

    f1_dropped = f1_score(np.ravel(y_test.to_numpy()), predicted_dropped)

    end_time_dropped = time.time() - start_time_dropped

    method_list.append(['Dropped', rod_dropped, dp_dropped, tpb_dropped, tnb_dropped, f1_dropped,
                        admissible_features, len(admissible_features), count + 1])

    #runtimes.extend(
    #    ['Dropped', X_train.shape[0], X_train.shape[0], X_train.shape[1], end_time_dropped, count + 1])
    runtimes_list.append(['Dropped', X_train.shape[0], X_train.shape[0], X_train.shape[1], end_time_dropped, count + 1])

    print('ROD dropped ' + ': ' + str(rod_dropped))
    print('F1 dropped ' + ': ' + str(f1_dropped))

    ##################### Original

    categorical_features = []
    numerical_features = []
    start_time_original = time.time()
    for i in list(credit_df):
        if i != target and credit_df[i].dtype == np.dtype('O'):
            categorical_features.extend([i])
        elif i != target and credit_df[i].dtype != np.dtype('O'):
            numerical_features.extend([i])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)], remainder='passthrough')

    print('original shape: ' + str(preprocessor.fit_transform(X_train).shape))

    original_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('clf',
                                         LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                            max_iter=100000, multi_class='auto'))])

    original_model = GridSearchCV(original_pipeline, param_grid={
        'clf__penalty': ['l2'], 'clf__C': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5], 'clf__solver': ['lbfgs'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
    },
                                  n_jobs=-1,
                                  scoring='f1', cv=5)

    original_model.fit(X_train, np.ravel(y_train.to_numpy()))


    predicted_original = original_model.predict(X_test)
    predicted_original_proba = original_model.predict_proba(X_test)[:, 1]
    outcomes_df = pd.DataFrame(predicted_original_proba, columns=['outcome'])
    features_df = test_df.reset_index(drop=True)

    candidate_df = pd.concat([features_df, outcomes_df], axis=1)

    JCIT, mb = causal_filter(candidate_df, inadmissible_features + [sensitive_feature])

    rod_original = ROD.ROD(y_pred=predicted_original_proba, df=test_df, sensitive=sensitive_feature,
                           admissible=admissible_features, protected=protected, mb=mb)

    dp_original = demographic_parity_difference(y_test, predicted_original,
                                                sensitive_features=test_df.loc[:, sensitive_feature])
    tpr_original = MetricFrame(true_positive_rate, y_test, predicted_original,
                               sensitive_features=test_df.loc[:, sensitive_feature])
    tpb_original = tpr_original.difference()
    tnr_original = MetricFrame(true_negative_rate, y_test, predicted_original,
                               sensitive_features=test_df.loc[:, sensitive_feature])
    tnb_original = tnr_original.difference()

    f1_original = f1_score(np.ravel(y_test.to_numpy()), predicted_original)

    end_time_original = time.time() - start_time_original

    #runtimes.extend(
    #    ['Original', X_train.shape[0], X_train.shape[0], X_train.shape[1], end_time_original, count + 1])
    runtimes_list.append(['Original', X_train.shape[0], X_train.shape[0], X_train.shape[1], end_time_original, count + 1])

    method_list.append(['Original', rod_original, dp_original, tpb_original, tnb_original,
                        f1_original, all_features, len(all_features), count + 1])

    print('ROD original ' + ': ' + str(rod_original))
    print('F1 original ' + ': ' + str(f1_original))

    count += 1

############################## Capuchin ####################################
# Remove the sensitive when training and check results --> does ROD decrease variance? : No, bad results, go back

fold_capuchin = 0
for train_index_capuchin, test_index_capuchin in kf1.split(credit_df):

    capuchin_train_df = credit_df.iloc[train_index_capuchin]
    capuchin_test_df = credit_df.iloc[test_index_capuchin]

    capuchin_train_df.reset_index(inplace=True, drop=True)
    capuchin_test_df.reset_index(inplace=True, drop=True)

    capuchin_train_df.drop(columns=['personal_status', 'purpose', 'installment_commitment',
                                    'other_parties', 'property_magnitude', 'other_payment_plans',
                                    'own_telephone', 'housing', 'existing_credits', 'job', 'residence_since'], inplace=True)
    capuchin_test_df.drop(columns=['personal_status', 'purpose', 'installment_commitment',
                                    'other_parties', 'property_magnitude', 'other_payment_plans',
                                    'own_telephone', 'housing', 'existing_credits', 'job', 'residence_since'], inplace=True)

    admissible_features_capuchin = [i for i in list(capuchin_train_df) if
                                    i not in inadmissible_features and i != sensitive_feature and i != target]

    all_features_capuchin = [i for i in list(capuchin_train_df) if i != target]

    start_time_capuchin = time.time()
    categorical = []
    for i in list(capuchin_train_df):
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
                                                   'admissible_attributes': admissible_features_capuchin,
                                                   'sensitive_attribute': sensitive_feature,
                                                   'target': target}))])

    capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
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

    print('Start repairing training set with capuchin')
    #to_repair = pd.concat([X_train, y_train], axis=1)

    train_repaired = capuchin_repair_pipeline.fit_transform(capuchin_train_df)

    print('Finished repairing training set with capuchin')

    print(train_repaired.groupby(sensitive_feature)[target].mean())
    print(capuchin_train_df.groupby(sensitive_feature)[target].mean())

    y_train_repaired = train_repaired.loc[:, [target]].to_numpy()
    X_train_repaired = train_repaired.loc[:, all_features_capuchin]

    print('capuchin shape: ' + str(preprocessor_3.fit_transform(X_train_repaired).shape))

    X_test_capuchin = credit_df.iloc[test_index_capuchin, [idx for idx, i in enumerate(list(credit_df)) if i != target]]
    X_test_capuchin.reset_index(inplace=True, drop=True)
    y_test_capuchin = credit_df.iloc[test_index_capuchin, [idx for idx, i in enumerate(list(credit_df))]]
    y_test_capuchin.reset_index(inplace=True, drop=True)
    y_test_capuchin.dropna(inplace=True)
    y_test_capuchin = y_test_capuchin.loc[:, target]
    X_test_capuchin.dropna(inplace=True)


    capuchin_model.fit(X_train_repaired, np.ravel(y_train_repaired))

    X_test_capuchin = generate_binned_df(X_test_capuchin)

    for i in list(X_test_capuchin):
        if X_test_capuchin[i].dtype != object:
            X_test_capuchin[i] = X_test_capuchin[i].astype('object')
    #X_test_capuchin = preprocessor_3.fit_transform(X_test_capuchin)

    #runtimes.extend(['Capuchin', X_train.shape[0], X_train_repaired.shape[0], X_train.shape[1], end_time_capuchin, count +1])

    predicted_capuchin = capuchin_model.predict(X_test_capuchin)
    predicted_capuchin_proba = capuchin_model.predict_proba(X_test_capuchin)[:, 1]
    outcomes_df = pd.DataFrame(predicted_capuchin_proba, columns=['outcome'])
    features_df = X_test_capuchin.reset_index(drop=True)

    candidate_df = pd.concat([features_df, outcomes_df], axis=1)

    JCIT, mb = causal_filter(candidate_df, inadmissible_features + [sensitive_feature])

    rod_capuchin = ROD.ROD(y_pred=predicted_capuchin_proba, df=X_test_capuchin, sensitive=sensitive_feature,
                           admissible=admissible_features, protected=protected, mb=mb)

    dp_capuchin = demographic_parity_difference(y_test_capuchin, predicted_capuchin, sensitive_features=X_test_capuchin.loc[:, sensitive_feature])
    tpr_capuchin = MetricFrame(true_positive_rate, y_test_capuchin, predicted_capuchin,
                      sensitive_features=X_test_capuchin.loc[:, sensitive_feature])
    tpb_capuchin = tpr_capuchin.difference()
    tnr_capuchin = MetricFrame(true_negative_rate, y_test_capuchin, predicted_capuchin,
                      sensitive_features=X_test_capuchin.loc[:, sensitive_feature])
    tnb_capuchin = tnr_capuchin.difference()

    f1_capuchin = f1_score(np.ravel(y_test_capuchin.to_numpy()), predicted_capuchin)
    end_time_capuchin = time.time() - start_time_capuchin

    method_list.append(['Capuchin', rod_capuchin, dp_capuchin, tpb_capuchin, tnb_capuchin,
                        f1_capuchin, all_features_capuchin, len(all_features_capuchin), fold_capuchin + 1])

    runtimes_list.append(
        ['Capuchin', capuchin_train_df.shape[0], X_train_repaired.shape[0], X_train_repaired.shape[1], end_time_capuchin, fold_capuchin + 1])

    print('ROD capuchin ' + ': ' + str(rod_capuchin))
    print('F1 capuchin ' + ': ' + str(f1_capuchin))

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'DP', 'TPB', 'TNB', 'F1', 'Representation', 'Size', 'Fold'])
runtimes_df = pd.DataFrame(runtimes_list, columns=['Method', 'Rows', 'Modified_rows', 'Columns', 'Repair_time_capuchin', 'Fold'])


print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

if CF:
    summary_df.to_csv(path_or_buf=results_path + '/baselines_' +dataset + '_results' + '_CF.csv', index=False)
    runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_' + dataset + '_capuchin_' + '.csv',
                       index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/baselines_' +dataset + '_results' + '.csv', index=False)
    runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_' + dataset + '_capuchin_' + '.csv',
                       index=False)
