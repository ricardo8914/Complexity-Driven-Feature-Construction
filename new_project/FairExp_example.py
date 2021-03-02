import openml
from sklearn.model_selection import KFold
import multiprocessing as mp
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fairexp_optimistic import extend_dataframe_complete, repair_algorithm
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from causality.fair_feature_selection import test_d_separation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from causality.causal_filter import causal_filter
from fmeasures import ROD, CDP, CTPB, CTNB
import time
from fairlearn.metrics import demographic_parity_difference, MetricFrame, true_positive_rate, true_negative_rate

from pathlib import Path

from fastsklearnfeature.configuration.Config import Config
home = str(Config.get('path_to_project'))
#home = str(Path.home())

results_path = Path(home + '/Complexity-Driven-Feature-Construction/results')
results_path.mkdir(parents=True, exist_ok=True)

#### Adult dataset
dataset = openml.datasets.get_dataset(1590)

adult, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)

f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

kf1 = KFold(n_splits=5, random_state=42, shuffle=True)

## Drop other sensitive information
adult.drop(columns=['fnlwgt', 'relationship', 'race', 'native-country'], inplace=True)

adult['income'] = y.apply(lambda x: 1 if x == '>50K' else 0)

adult.dropna(inplace=True)

target = 'income'
sensitive_feature = 'sex'
inadmissible_feature = 'marital-status'
protected = 'Female'
sensitive_features = [sensitive_feature] + [inadmissible_feature]
admissible_features = [i for i in list(adult) if i not in sensitive_features and i != target]

y = np.ravel(adult['income'].to_numpy())

if __name__ == '__main__':

    mp.set_start_method('fork')

    fold = 1
    results = []
    for train_index, test_index in kf1.split(adult):

        start_time = time.time()

        X, names, retained_indices, valid_indices = extend_dataframe_complete(df=adult, complexity=2, scoring=f1,
                                                                              target=target, sampling=0.05,
                                                                              train_indices=train_index)

        test_indices = np.intersect1d(valid_indices, test_index)

        X_train = X[retained_indices]
        y_train = y[retained_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        train_df_e = adult.iloc[retained_indices]
        test_df_e = adult.iloc[test_indices]

        selected_features_ = repair_algorithm(X_train, names, train_df_e, y_train, sensitive_feature,
                                                         sensitive_features, protected,
                                                         admissible_features, target,
                                                         LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                                            class_weight='balanced',
                                                                            max_iter=100000, multi_class='auto'),
                                                         sampling=0.05)

        selected_train = X_train[:, selected_features_]
        selected_test = X_test[:, selected_features_]
        selected_names = [names[i] for i in selected_features_]

        intersection = [i for i in selected_names if i in list(adult)]

        cit_df = pd.DataFrame(selected_train, columns=selected_names)
        cit_df.drop(columns=intersection, inplace=True)
        cit_df_test = pd.DataFrame(selected_test, columns=selected_names)
        cit_df_test.drop(columns=intersection, inplace=True)
        train_df_e.reset_index(inplace=True, drop=True)
        test_df_e.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([train_df_e, cit_df], axis=1)
        candidate_df_test = pd.concat([test_df_e, cit_df_test], axis=1)

        fair_features = test_d_separation(candidate_df, sensitive_features=sensitive_features,
                                          admissible=admissible_features, target=target)

        features2_scale = []
        features2_encode = []

        for i in fair_features + admissible_features:
            if candidate_df.loc[:, i].dtype in (int, float) and i != target:
                features2_scale.append(i)
            elif i != target:
                features2_encode.append(i)

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='if_binary'))])
        #
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', numerical_transformer, features2_scale),
                ('encode', categorical_transformer, features2_encode)],
            remainder='passthrough')

        FairExp_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('clf',
                                             LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                                class_weight='balanced',
                                                                max_iter=100000, multi_class='auto'))])

        FairExp_model = GridSearchCV(FairExp_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [0.5, 1.0, 5], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                      n_jobs=-1,
                                      scoring='f1', cv=5)

        j = fair_features + admissible_features

        fair_df_train = candidate_df.loc[:, j]
        fair_y_train = np.ravel(candidate_df.loc[:, target].to_numpy())

        fair_df_test = candidate_df_test.loc[:, j]

        FairExp_model.fit(fair_df_train, fair_y_train)
        y_pred= FairExp_model.predict(fair_df_test)
        y_pred_proba = FairExp_model.predict_proba(fair_df_test)[:, 1]

        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = test_df_e.loc[:, [i for i in test_df_e.columns if i != target]]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_FairExp_ = ROD.ROD(y_pred=y_pred_proba, df=test_df_e,
                                           sensitive=sensitive_feature,
                                           admissible=admissible_features, protected=protected, mb=mb)

        dp_FairExp = demographic_parity_difference(y_test, y_pred,
                                                                sensitive_features=test_df_e.loc[:, sensitive_feature])
        tpr_FairExp_ = MetricFrame(true_positive_rate, y_test, y_pred,
                                               sensitive_features=test_df_e.loc[:, sensitive_feature])
        tpb_FairExp_ = tpr_FairExp_.difference()
        tnr_FairExp_ = MetricFrame(true_negative_rate, y_test, y_pred,
                                               sensitive_features=test_df_e.loc[:, sensitive_feature])
        tnb_FairExp_ = tnr_FairExp_.difference()

        cdp_FairExp_ = CDP.CDP(y_test, y_pred, test_df_e, sensitive_feature,
                                       admissible_features)
        ctpb_FairExp_ = CTPB.CTPB(y_test, y_pred, test_df_e, sensitive_feature,
                                         admissible_features)
        ctnb_FairExp_ = CTNB.CTNB(y_test, y_pred, test_df_e, sensitive_feature,
                                         admissible_features)

        f1_FairExp_ = f1_score(y_test, y_pred)

        end_time = time.time() - start_time

        print('ROD FairExp: ' + str(rod_FairExp_))
        print('DP FairExp: ' + str(dp_FairExp))
        print('TPB FairExp: ' + str(tpb_FairExp_))
        print('TNB FairExp: ' + str(tnb_FairExp_))
        print('CTPB FairExp: ' + str(ctpb_FairExp_))
        print('CTNB FairExp: ' + str(ctnb_FairExp_))
        print('F1 FairExp: ' + str(f1_FairExp_))

        results.append(
            ['Adult', 'FairExp', selected_names, fold, rod_FairExp_,
             dp_FairExp, tpb_FairExp_, tnb_FairExp_,
             cdp_FairExp_,
             ctpb_FairExp_, ctnb_FairExp_, f1_FairExp_, end_time])

        fold += 1

    results_df = pd.DataFrame(results, columns=['Dataset', 'Method', 'Representation', 'Fold', 'ROD', 'DP', 'TPB', 'TNB',
                              'CDP', 'CTPB', 'CTNB', 'F1', 'Runtime'])

    results_df.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/FairExp_experiment_adult.csv',
                      index=False)


