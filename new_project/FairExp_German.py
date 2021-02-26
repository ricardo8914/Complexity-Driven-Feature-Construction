from sklearn.model_selection import KFold
import multiprocessing as mp
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fairexp import extend_dataframe_complete, repair_algorithm
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from causality.fair_feature_selection import test_d_separation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from causality.causal_filter import causal_filter
from benchmark.capuchin import repair_dataset
from fmeasures import ROD, CDP, CTPB, CTNB
import time
from fairlearn.metrics import demographic_parity_difference, MetricFrame, true_positive_rate, true_negative_rate
from experiments.NSGAII import evaluate_NSGAII
from benchmark.kamiran.massaging import massaging
from benchmark.kamiran.reweighting import reweighting

from pathlib import Path

home = str(Path.home())

capuchin_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/capuchin'
results_path = Path(home + '/Complexity-Driven-Feature-Construction/results')
results_path.mkdir(parents=True, exist_ok=True)

COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'

df = pd.read_csv(COMPAS_path + '/compas-scores-two-years.csv')

df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
                    'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
ix = df['days_b_screening_arrest'] <= 30
ix = (df['days_b_screening_arrest'] >= -30) & ix
ix = (df['is_recid'] != -1) & ix
ix = (df['c_charge_degree'] != "O") & ix
ix = (df['score_text'] != 'N/A') & ix
df = df.loc[ix, :]
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

dfcut = df.loc[~df['race'].isin(['Native American', 'Hispanic', 'Asian', 'Other']), :]

dfcutQ = dfcut[['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text', 'priors_count', 'is_recid',
                'two_year_recid', 'length_of_stay']].copy()


# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x):
    if x <= 0:
        return '0'
    elif 1 <= x <= 3:
        return '1 to 3'
    else:
        return 'More than 3'


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return '<week'
    if 8 < x <= 93:
        return '<3months'
    else:
        return '>3 months'


# Quantize length of stay
def adjustAge(x):
    if x == '25 - 45':
        return '25 to 45'
    else:
        return x


# Quantize score_text to MediumHigh
def quantizeScore(x):
    if (x == 'High') | (x == 'Medium'):
        return 'MediumHigh'
    else:
        return x


dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

features = ['race', 'age_cat', 'c_charge_degree', 'priors_count', 'is_recid']

# Pass vallue to df
COMPAS_binned = dfcutQ[features]
COMPAS = dfcut[features]

f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

kf1 = KFold(n_splits=5, random_state=42, shuffle=True)

target = 'is_recid'
sensitive_feature = 'race'
inadmissible_feature = ''
protected = 'African-American'
sensitive_features = [sensitive_feature]
admissible_features = [i for i in list(COMPAS) if i not in sensitive_features and i != target]

y = np.ravel(COMPAS['is_recid'].to_numpy())

if __name__ == '__main__':

    mp.set_start_method('fork')

    fold = 1
    results = []
    for train_index, test_index in kf1.split(COMPAS):

        start_time = time.time()

        X, names, retained_indices, valid_indices = extend_dataframe_complete(df=COMPAS, complexity=4, scoring=f1,
                                                                              target=target, sampling=0.05,
                                                                              train_indices=train_index)

        end_time_fc = time.time() - start_time

        test_indices = np.intersect1d(valid_indices, test_index)

        X_train_fairexp = X[retained_indices]
        y_train_fairexp = y[retained_indices]
        X_test_fairexp = X[test_indices]
        y_test_fairexp = y[test_indices]

        train_df_e = COMPAS.iloc[retained_indices]
        test_df_e = COMPAS.iloc[test_indices]
        train_df_e.reset_index(inplace=True, drop=True)
        test_df_e.reset_index(inplace=True, drop=True)

        selected_features_ = repair_algorithm(X_train_fairexp, names, train_df_e, y_train_fairexp, sensitive_feature,
                                              sensitive_features, protected,
                                              admissible_features, target,
                                              LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                                 class_weight='balanced',
                                                                 max_iter=100000, multi_class='auto'),
                                              sampling=0.05)

        selected_train = X_train_fairexp[:, selected_features_]
        selected_test = X_test_fairexp[:, selected_features_]
        selected_names = [names[i] for i in selected_features_]

        intersection = [i for i in selected_names if i in list(COMPAS)]

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
            'clf__penalty': ['l2'], 'clf__C': [1.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                     n_jobs=-1,
                                     scoring='f1', cv=5)

        j = fair_features + admissible_features

        fair_df_train = candidate_df.loc[:, j]
        fair_y_train = np.ravel(candidate_df.loc[:, target].to_numpy())

        fair_df_test = candidate_df_test.loc[:, j]

        FairExp_pipeline.fit(fair_df_train, fair_y_train)
        y_pred = FairExp_pipeline.predict(fair_df_test)
        y_pred_proba = FairExp_pipeline.predict_proba(fair_df_test)[:, 1]

        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = test_df_e.loc[:, [i for i in test_df_e.columns if i != target]]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_FairExp_ = ROD.ROD(y_pred=y_pred_proba, df=test_df_e,
                               sensitive=sensitive_feature,
                               admissible=admissible_features, protected=protected, mb=mb)

        dp_FairExp = demographic_parity_difference(y_test_fairexp, y_pred,
                                                   sensitive_features=test_df_e.loc[:, sensitive_feature])
        tpr_FairExp_ = MetricFrame(true_positive_rate, y_test_fairexp, y_pred,
                                   sensitive_features=test_df_e.loc[:, sensitive_feature])
        tpb_FairExp_ = tpr_FairExp_.difference()
        tnr_FairExp_ = MetricFrame(true_negative_rate, y_test_fairexp, y_pred,
                                   sensitive_features=test_df_e.loc[:, sensitive_feature])
        tnb_FairExp_ = tnr_FairExp_.difference()

        cdp_FairExp_ = CDP.CDP(y_test_fairexp, y_pred, test_df_e, sensitive_feature,
                               admissible_features)
        ctpb_FairExp_ = CTPB.CTPB(y_test_fairexp, y_pred, test_df_e, sensitive_feature,
                                  admissible_features)
        ctnb_FairExp_ = CTNB.CTNB(y_test_fairexp, y_pred, test_df_e, sensitive_feature,
                                  admissible_features)

        f1_FairExp_ = f1_score(y_test_fairexp, y_pred)

        end_time = time.time() - start_time

        print('ROD FairExp: ' + str(rod_FairExp_))
        print('DP FairExp: ' + str(dp_FairExp))
        print('TPB FairExp: ' + str(tpb_FairExp_))
        print('TNB FairExp: ' + str(tnb_FairExp_))
        print('CTPB FairExp: ' + str(ctpb_FairExp_))
        print('CTNB FairExp: ' + str(ctnb_FairExp_))
        print('F1 FairExp: ' + str(f1_FairExp_))

        results.append(
            ['COMPAS', 'FairExp', j, fold, rod_FairExp_,
             dp_FairExp, tpb_FairExp_, tnb_FairExp_,
             cdp_FairExp_,
             ctpb_FairExp_, ctnb_FairExp_, f1_FairExp_, end_time])

        ###### Original

        start_time = time.time()

        train_df = COMPAS.iloc[train_index]
        test_df = COMPAS.iloc[test_index]
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        X_train = COMPAS.iloc[train_index, [list(COMPAS).index(i) for i in list(COMPAS) if i != target]]
        X_test = COMPAS.iloc[test_index, [list(COMPAS).index(i) for i in list(COMPAS) if i != target]]

        y_train = y[train_index]
        y_test = y[test_index]

        features2_scale = []
        features2_encode = []

        for i in list(COMPAS):
            if COMPAS.loc[:, i].dtype in (int, float) and i != target:
                features2_scale.append(i)
            elif i != target:
                features2_encode.append(i)

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='if_binary'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', numerical_transformer, features2_scale),
                ('encode', categorical_transformer, features2_encode)],
            remainder='passthrough')

        original_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('clf',
                                             LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                                class_weight='balanced',
                                                                max_iter=100000, multi_class='auto'))])

        original_model = GridSearchCV(original_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [1.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                      n_jobs=-1,
                                      scoring='f1', cv=5)

        original_pipeline.fit(X_train, y_train)

        y_pred = original_pipeline.predict(X_test)
        y_pred_proba = original_pipeline.predict_proba(X_test)[:, 1]

        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = test_df.loc[:, [i for i in test_df.columns if i != target]]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_original = ROD.ROD(y_pred=y_pred_proba, df=test_df,
                               sensitive=sensitive_feature,
                               admissible=admissible_features, protected=protected, mb=mb)

        dp_original = demographic_parity_difference(y_test, y_pred,
                                                    sensitive_features=test_df.loc[:, sensitive_feature])
        tpr_original = MetricFrame(true_positive_rate, y_test, y_pred,
                                   sensitive_features=test_df.loc[:, sensitive_feature])
        tpb_original = tpr_original.difference()
        tnr_original = MetricFrame(true_negative_rate, y_test, y_pred,
                                   sensitive_features=test_df.loc[:, sensitive_feature])
        tnb_original = tnr_original.difference()

        cdp_original = CDP.CDP(y_test, y_pred, test_df, sensitive_feature,
                               admissible_features)
        ctpb_original = CTPB.CTPB(y_test, y_pred, test_df, sensitive_feature,
                                  admissible_features)
        ctnb_original = CTNB.CTNB(y_test, y_pred, test_df, sensitive_feature,
                                  admissible_features)

        f1_original = f1_score(y_test, y_pred)

        end_time = time.time() - start_time

        print('ROD original: ' + str(rod_original))
        print('DP original: ' + str(dp_original))
        print('TPB original: ' + str(tpb_original))
        print('TNB original: ' + str(tnb_original))
        print('CTPB original: ' + str(ctpb_original))
        print('CTNB original: ' + str(ctnb_original))
        print('F1 original: ' + str(f1_original))

        results.append(
            ['COMPAS', 'Original', admissible_features + sensitive_features, fold, rod_original,
             dp_original, tpb_original, tnb_original,
             cdp_original,
             ctpb_original, ctnb_original, f1_original, end_time])

        ####### Dropped

        start_time = time.time()

        dropped_train_df = COMPAS.iloc[
            train_index, [list(COMPAS).index(i) for i in list(COMPAS) if i != target and i not in sensitive_features]]
        dropped_test_df = COMPAS.iloc[
            test_index, [list(COMPAS).index(i) for i in list(COMPAS) if i != target and i not in sensitive_features]]

        features2_scale = []
        features2_encode = []
        for i in list(dropped_train_df):
            if COMPAS.loc[:, i].dtype in (int, float) and i != target:
                features2_scale.append(i)
            elif i != target:
                features2_encode.append(i)

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='if_binary'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', numerical_transformer, features2_scale),
                ('encode', categorical_transformer, features2_encode)],
            remainder='passthrough')

        dropped_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('clf',
                                            LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                               class_weight='balanced',
                                                               max_iter=100000, multi_class='auto'))])

        dropped_model = GridSearchCV(dropped_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [1.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                     n_jobs=-1,
                                     scoring='f1', cv=5)

        dropped_pipeline.fit(dropped_train_df, y_train)

        y_pred = dropped_pipeline.predict(dropped_test_df)
        y_pred_proba = dropped_pipeline.predict_proba(dropped_test_df)[:, 1]

        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = test_df.loc[:, [i for i in test_df.columns if i != target]]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_dropped = ROD.ROD(y_pred=y_pred_proba, df=test_df,
                              sensitive=sensitive_feature,
                              admissible=admissible_features, protected=protected, mb=mb)

        dp_dropped = demographic_parity_difference(y_test, y_pred,
                                                   sensitive_features=test_df.loc[:, sensitive_feature])
        tpr_dropped = MetricFrame(true_positive_rate, y_test, y_pred,
                                  sensitive_features=test_df.loc[:, sensitive_feature])
        tpb_dropped = tpr_dropped.difference()
        tnr_dropped = MetricFrame(true_negative_rate, y_test, y_pred,
                                  sensitive_features=test_df.loc[:, sensitive_feature])
        tnb_dropped = tnr_dropped.difference()

        cdp_dropped = CDP.CDP(y_test, y_pred, test_df, sensitive_feature,
                              admissible_features)
        ctpb_dropped = CTPB.CTPB(y_test, y_pred, test_df, sensitive_feature,
                                 admissible_features)
        ctnb_dropped = CTNB.CTNB(y_test, y_pred, test_df, sensitive_feature,
                                 admissible_features)

        f1_dropped = f1_score(y_test, y_pred)

        end_time = time.time() - start_time

        print('ROD dropped: ' + str(rod_dropped))
        print('DP dropped: ' + str(dp_dropped))
        print('TPB dropped: ' + str(tpb_dropped))
        print('TNB dropped: ' + str(tnb_dropped))
        print('CTPB dropped: ' + str(ctpb_dropped))
        print('CTNB dropped: ' + str(ctnb_dropped))
        print('F1 dropped: ' + str(f1_dropped))

        results.append(
            ['COMPAS', 'Dropped', admissible_features, fold, rod_dropped,
             dp_dropped, tpb_dropped, tnb_dropped,
             cdp_dropped,
             ctpb_dropped, ctnb_dropped, f1_dropped, end_time])

        ###### NSGAII

        start_time = time.time()

        selected_NSGAII = evaluate_NSGAII(X_train_fairexp, df=train_df_e, target=target, sensitive=sensitive_feature,
                                          protected=protected, sampling=0.05)

        selected_names_NSGAII = [names[idf] for idf, f in enumerate(selected_NSGAII) if f == True]

        NSGAII_pipeline = Pipeline(steps=[
            ('clf',
             LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                class_weight='balanced',
                                max_iter=100000, multi_class='auto', n_jobs=-1))])

        NSGAII_model = GridSearchCV(NSGAII_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [1.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                    n_jobs=-1,
                                    scoring='f1', cv=5)

        X_train_NSGAII = X_train_fairexp[:, selected_NSGAII]
        X_test_NSGAII = X_test_fairexp[:, selected_NSGAII]

        NSGAII_pipeline.fit(X_train_NSGAII, y_train_fairexp)

        predicted_NSGAII = NSGAII_pipeline.predict(X_test_NSGAII)
        predicted_NSGAII_proba = NSGAII_pipeline.predict_proba(X_test_NSGAII)[:, 1]
        outcomes_df = pd.DataFrame(predicted_NSGAII_proba, columns=['outcome'])
        features_df = test_df_e.reset_index(drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_NSGAII = ROD.ROD(y_pred=predicted_NSGAII_proba, df=test_df_e,
                             sensitive=sensitive_feature,
                             admissible=admissible_features, protected=protected, mb=mb)

        dp_NSGAII = demographic_parity_difference(y_test_fairexp, predicted_NSGAII,
                                                  sensitive_features=test_df_e.loc[:, sensitive_feature])
        tpr_NSGAII = MetricFrame(true_positive_rate, y_test_fairexp, predicted_NSGAII,
                                 sensitive_features=test_df_e.loc[:, sensitive_feature])
        tpb_NSGAII = tpr_NSGAII.difference()
        tnr_NSGAII = MetricFrame(true_negative_rate, y_test_fairexp, predicted_NSGAII,
                                 sensitive_features=test_df_e.loc[:, sensitive_feature])
        tnb_NSGAII = tnr_NSGAII.difference()

        cdp_NSGAII = CDP.CDP(y_test_fairexp, predicted_NSGAII, test_df_e, sensitive_feature,
                             admissible_features)
        ctpb_NSGAII = CTPB.CTPB(y_test_fairexp, predicted_NSGAII, test_df_e, sensitive_feature,
                                admissible_features)
        ctnb_NSGAII = CTNB.CTNB(y_test_fairexp, predicted_NSGAII, test_df_e, sensitive_feature,
                                admissible_features)

        f1_NSGAII = f1_score(y_test_fairexp, predicted_NSGAII)

        end_time = time.time() - start_time + end_time_fc

        print('ROD NSGAII: ' + str(rod_NSGAII))
        print('DP NSGAII: ' + str(dp_NSGAII))
        print('TPB NSGAII: ' + str(tpb_NSGAII))
        print('TNB NSGAII: ' + str(tnb_NSGAII))
        print('CTPB NSGAII: ' + str(ctpb_NSGAII))
        print('CTNB NSGAII: ' + str(ctnb_NSGAII))
        print('F1 NSGAII: ' + str(f1_NSGAII))

        results.append(
            ['COMPAS', 'NSGAII', selected_names_NSGAII, fold, rod_NSGAII,
             dp_NSGAII, tpb_NSGAII, tnb_NSGAII,
             cdp_NSGAII,
             ctpb_NSGAII, ctnb_NSGAII, f1_NSGAII, end_time])

        ##### Kamiran massaging

        start_time = time.time()

        kamiran_massaging_df = massaging(train_df, target, sensitive_feature, protected)

        print(kamiran_massaging_df.groupby(sensitive_feature)[target].mean())
        print(train_df.groupby(sensitive_feature)[target].mean())

        X_train_kamiran = kamiran_massaging_df.loc[:, [i for i in kamiran_massaging_df if i != target]]
        X_test_kamiran = test_df.loc[:, [i for i in kamiran_massaging_df if i != target]]

        y_train_kamiran = np.ravel(train_df.loc[:, target].to_numpy())
        y_test_kamiran = np.ravel(test_df.loc[:, target].to_numpy())

        features2_scale = []
        features2_encode = []
        for i in list(train_df):
            if COMPAS.loc[:, i].dtype in (int, float) and i != target:
                features2_scale.append(i)
            elif i != target:
                features2_encode.append(i)

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='if_binary'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', numerical_transformer, features2_scale),
                ('encode', categorical_transformer, features2_encode)],
            remainder='passthrough')

        kamiran_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('clf',
                                            LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                               max_iter=100000, multi_class='auto'))])

        kamiran_model = GridSearchCV(kamiran_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [0.5, 1.0, 5.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                     n_jobs=-1,
                                     scoring='f1', cv=5)

        kamiran_pipeline.fit(X_train_kamiran, y_train_kamiran)

        y_pred = kamiran_pipeline.predict(X_test_kamiran)
        y_pred_proba = kamiran_pipeline.predict_proba(X_test_kamiran)[:, 1]

        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = test_df.loc[:, [i for i in test_df.columns if i != target]]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_kamiran = ROD.ROD(y_pred=y_pred_proba, df=test_df,
                              sensitive=sensitive_feature,
                              admissible=admissible_features, protected=protected, mb=mb)

        dp_kamiran = demographic_parity_difference(y_test_kamiran, y_pred,
                                                   sensitive_features=test_df.loc[:, sensitive_feature])
        tpr_kamiran = MetricFrame(true_positive_rate, y_test_kamiran, y_pred,
                                  sensitive_features=test_df.loc[:, sensitive_feature])
        tpb_kamiran = tpr_kamiran.difference()
        tnr_kamiran = MetricFrame(true_negative_rate, y_test_kamiran, y_pred,
                                  sensitive_features=test_df.loc[:, sensitive_feature])
        tnb_kamiran = tnr_kamiran.difference()

        cdp_kamiran = CDP.CDP(y_test_kamiran, y_pred, test_df, sensitive_feature,
                              admissible_features)
        ctpb_kamiran = CTPB.CTPB(y_test_kamiran, y_pred, test_df, sensitive_feature,
                                 admissible_features)
        ctnb_kamiran = CTNB.CTNB(y_test_kamiran, y_pred, test_df, sensitive_feature,
                                 admissible_features)

        f1_kamiran = f1_score(y_test_kamiran, y_pred)

        end_time = time.time() - start_time

        print('ROD kamiran massaging: ' + str(rod_kamiran))
        print('DP kamiran massaging: ' + str(dp_kamiran))
        print('TPB kamiran massaging: ' + str(tpb_kamiran))
        print('TNB kamiran massaging: ' + str(tnb_kamiran))
        print('CTPB kamiran massaging: ' + str(ctpb_kamiran))
        print('CTNB kamiran massaging: ' + str(ctnb_kamiran))
        print('F1 kamiran massaging: ' + str(f1_kamiran))

        results.append(
            ['COMPAS', 'Kamiran-massaging', list(COMPAS), fold, rod_kamiran,
             dp_kamiran, tpb_kamiran, tnb_kamiran,
             cdp_kamiran,
             ctpb_kamiran, ctnb_kamiran, f1_kamiran, end_time])

        ##### Kamiran reweighting

        start_time = time.time()

        kamiran_weights = reweighting(train_df, target, sensitive_feature, protected)

        X_train = train_df.loc[:, [i for i in list(COMPAS) if i != target]]
        y_train = np.ravel(train_df.loc[:, target].to_numpy())

        kamiran_model = GridSearchCV(kamiran_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [0.5, 1.0, 5.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                     n_jobs=-1,
                                     scoring='f1', cv=5)

        kamiran_pipeline.fit(X_train, y_train, clf__sample_weight=kamiran_weights)

        y_pred = kamiran_pipeline.predict(X_test_kamiran)
        y_pred_proba = kamiran_pipeline.predict_proba(X_test_kamiran)[:, 1]

        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = test_df.loc[:, [i for i in test_df.columns if i != target]]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_kamiran = ROD.ROD(y_pred=y_pred_proba, df=test_df,
                              sensitive=sensitive_feature,
                              admissible=admissible_features, protected=protected, mb=mb)

        dp_kamiran = demographic_parity_difference(y_test_kamiran, y_pred,
                                                   sensitive_features=test_df.loc[:, sensitive_feature])
        tpr_kamiran = MetricFrame(true_positive_rate, y_test_kamiran, y_pred,
                                  sensitive_features=test_df.loc[:, sensitive_feature])
        tpb_kamiran = tpr_kamiran.difference()
        tnr_kamiran = MetricFrame(true_negative_rate, y_test_kamiran, y_pred,
                                  sensitive_features=test_df.loc[:, sensitive_feature])
        tnb_kamiran = tnr_kamiran.difference()

        cdp_kamiran = CDP.CDP(y_test_kamiran, y_pred, test_df, sensitive_feature,
                              admissible_features)
        ctpb_kamiran = CTPB.CTPB(y_test_kamiran, y_pred, test_df, sensitive_feature,
                                 admissible_features)
        ctnb_kamiran = CTNB.CTNB(y_test_kamiran, y_pred, test_df, sensitive_feature,
                                 admissible_features)

        f1_kamiran = f1_score(y_test_kamiran, y_pred)

        end_time = time.time() - start_time

        print('ROD kamiran: ' + str(rod_kamiran))
        print('DP kamiran: ' + str(dp_kamiran))
        print('TPB kamiran: ' + str(tpb_kamiran))
        print('TNB kamiran: ' + str(tnb_kamiran))
        print('CTPB kamiran: ' + str(ctpb_kamiran))
        print('CTNB kamiran: ' + str(ctnb_kamiran))
        print('F1 kamiran: ' + str(f1_kamiran))

        results.append(
            ['COMPAS', 'Kamiran-reweighting ', list(COMPAS), fold, rod_kamiran,
             dp_kamiran, tpb_kamiran, tnb_kamiran,
             cdp_kamiran,
             ctpb_kamiran, ctnb_kamiran, f1_kamiran, end_time])

        fold += 1

        ###### Capuchin

    capuchin_df = COMPAS_binned
    #capuchin_df.drop(columns=['educationnum', 'nativecountry', 'race', 'relationship'], inplace=True)

    y = np.ravel(capuchin_df.loc[:, target].to_numpy())

    fold = 1
    for train_index, test_index in kf1.split(capuchin_df):

        start_time = time.time()

        capuchin_train_df = capuchin_df.iloc[train_index]
        capuchin_test_df = capuchin_df.iloc[test_index]

        y_test = y[test_index]

        start_time_capuchin = time.time()
        categorical = []
        for i in list(capuchin_df):
            if i != target:
                categorical.extend([i])

        categorical_transformer_3 = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor_3 = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer_3, categorical)],
            remainder='passthrough')

        admissible_features_capuchin = [f for f in list(capuchin_df) if f not in ('race', 'is_recid')]
        all_features_capuchin = [f for f in list(capuchin_df) if f != target]

        capuchin_repair_pipeline = Pipeline(steps=[  # ('generate_binned_df', FunctionTransformer(generate_binned_df)),
            ('repair', FunctionTransformer(repair_dataset, kw_args={
                'admissible_attributes': admissible_features_capuchin,
                'sensitive_attribute': sensitive_feature,
                'target': target}))])

        capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
                                            ('clf',
                                             LogisticRegression(penalty='l2', C=1, solver='lbfgs',
                                                                max_iter=100000, multi_class='auto', n_jobs=-1))])

        capuchin_model = GridSearchCV(capuchin_pipeline, param_grid={
            'clf__penalty': ['l2'], 'clf__C': [1.0], 'clf__solver': ['lbfgs'],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [100000], 'clf__multi_class': ['auto'], 'clf__n_jobs': [-1]
        },
                                      n_jobs=-1,
                                      scoring='f1', cv=5)

        print('Start repairing training set with capuchin')
        # to_repair = pd.concat([X_train, y_train], axis=1)

        train_repaired = capuchin_repair_pipeline.fit_transform(capuchin_train_df)

        print('Finished repairing training set with capuchin')

        y_train_repaired = train_repaired.loc[:, [target]].to_numpy()
        X_train_repaired = train_repaired.loc[:, all_features_capuchin]

        X_test_capuchin = capuchin_df.iloc[
            test_index, [idx for idx, i in enumerate(list(capuchin_df)) if i != target]]
        X_test_capuchin.reset_index(inplace=True, drop=True)
        y_test_capuchin = capuchin_df.iloc[test_index, [idx for idx, i in enumerate(list(capuchin_df))]]
        y_test_capuchin.reset_index(inplace=True, drop=True)
        y_test_capuchin.dropna(inplace=True)
        y_test_capuchin = np.ravel(y_test_capuchin.loc[:, target].to_numpy())
        X_test_capuchin.dropna(inplace=True)

        capuchin_pipeline.fit(X_train_repaired, np.ravel(y_train_repaired))

        predicted_capuchin = capuchin_pipeline.predict(X_test_capuchin)
        predicted_capuchin_proba = capuchin_pipeline.predict_proba(X_test_capuchin)[:, 1]
        outcomes_df = pd.DataFrame(predicted_capuchin_proba, columns=['outcome'])
        features_df = X_test_capuchin.reset_index(drop=True)

        candidate_df = pd.concat([features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        rod_capuchin = ROD.ROD(y_pred=predicted_capuchin_proba, df=X_test_capuchin,
                               sensitive=sensitive_feature,
                               admissible=admissible_features_capuchin, protected=protected, mb=mb)

        dp_capuchin = demographic_parity_difference(y_test_capuchin, predicted_capuchin,
                                                    sensitive_features=X_test_capuchin.loc[:, sensitive_feature])
        tpr_capuchin = MetricFrame(true_positive_rate, y_test_capuchin, predicted_capuchin,
                                   sensitive_features=X_test_capuchin.loc[:, sensitive_feature])
        tpb_capuchin = tpr_capuchin.difference()
        tnr_capuchin = MetricFrame(true_negative_rate, y_test_capuchin, predicted_capuchin,
                                   sensitive_features=X_test_capuchin.loc[:, sensitive_feature])
        tnb_capuchin = tnr_capuchin.difference()

        cdp_capuchin = CDP.CDP(y_test_capuchin, predicted_capuchin, X_test_capuchin, sensitive_feature,
                               admissible_features_capuchin)
        ctpb_capuchin = CTPB.CTPB(y_test_capuchin, predicted_capuchin, X_test_capuchin, sensitive_feature,
                                  admissible_features_capuchin)
        ctnb_capuchin = CTNB.CTNB(y_test_capuchin, predicted_capuchin, X_test_capuchin, sensitive_feature,
                                  admissible_features_capuchin)

        f1_capuchin = f1_score(y_test_capuchin, predicted_capuchin)

        end_time = time.time() - start_time

        print('ROD capuchin: ' + str(rod_capuchin))
        print('DP capuchin: ' + str(dp_capuchin))
        print('TPB capuchin: ' + str(tpb_capuchin))
        print('TNB capuchin: ' + str(tnb_capuchin))
        print('CTPB capuchin: ' + str(ctpb_capuchin))
        print('CTNB capuchin: ' + str(ctnb_capuchin))
        print('F1 capuchin: ' + str(f1_capuchin))

        results.append(
            ['COMPAS', 'Capuchin', all_features_capuchin, fold, rod_capuchin,
             dp_capuchin, tpb_capuchin, tnb_capuchin,
             cdp_capuchin,
             ctpb_capuchin, ctnb_capuchin, f1_capuchin, end_time])

    results_df = pd.DataFrame(results,
                              columns=['Dataset', 'Method', 'Representation', 'Fold', 'ROD', 'DP', 'TPB', 'TNB',
                                       'CDP', 'CTPB', 'CTNB', 'F1', 'Runtime'])

    results_df.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/FairExp_experiment_COMPAS.csv',
                      index=False)




