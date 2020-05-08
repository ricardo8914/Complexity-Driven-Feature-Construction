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
from sklearn.model_selection import train_test_split
from d_separation import d_separation
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
import time
from test_evolutionary import evolution
import sys

sys.path.insert(0, '/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from methods.capuchin import repair_dataset

home = str(Path.home())

adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

adult_df = pd.read_csv(adult_path + '/adult.csv', sep=';', header=0)


def label(row):
    if row['class'] == ' <=50K':
        return 0
    else:
        return 1


def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in ['target', 'outcome'] and (df_[i].dtype != np.dtype('O') and len(df_[i].unique()) > 4):
            out, bins = pd.cut(df_[i], bins=2, retbins=True, duplicates='drop')
            df_.loc[:, i] = out.astype(str)

    return df_


sensitive_feature = 'sex'
target = 'target'
inadmissible_features = ['marital-status']
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class', 'relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)
admissible_features = [i for i in list(adult_df) if
                       i not in inadmissible_features and i != sensitive_feature and i != target]

all_features = list(adult_df)
all_features.remove(target)
all_2_combinations = list(itertools.combinations(all_features, 2))

complexity = 4
CF = False
count = 0
method_list = []
kf1 = KFold(n_splits=3, random_state=42)
for train_index, test_index in kf1.split(adult_df):

    runtimes = [len(all_2_combinations), complexity]
    count_transformations = 0
    filtered_transformations = 0
    time_2_create_transformations = 0
    time_2_CF = 0
    time_2_FR = 0

    train_df = adult_df.iloc[train_index]
    test_df = adult_df.iloc[test_index]

    X_train = train_df.loc[:, all_features]

    y_train = train_df.loc[:, 'target']

    X_test = test_df.loc[:, all_features]

    y_test = test_df.loc[:, 'target']

    rod_score = make_scorer(ROD.ROD, greater_is_better=True, needs_proba=True,
                            sensitive=X_train.loc[:, sensitive_feature],
                            admissible=X_train.loc[:, admissible_features],
                            protected=' Female', name='train_adult')

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    ########### Dropped

    categorical_features_2 = []
    numerical_features_2 = []

    for i in list(adult_df):
        if i != target and i not in inadmissible_features and i != sensitive_feature and adult_df[i].dtype == np.dtype(
                'O'):
            categorical_features_2.extend([i])
        elif i != target and i not in inadmissible_features and i != sensitive_feature and adult_df[
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

    X_train_dropped = X_train.drop(columns=['sex', 'marital-status'])
    X_test_dropped = X_test.drop(columns=['sex', 'marital-status'])

    rod_dropped_train = cross_val_score(dropped_pipeline, X_train_dropped,
                                        np.ravel(y_train.to_numpy()), cv=5,
                                        scoring=rod_score)

    f1_dropped_train = cross_val_score(dropped_pipeline, X_train_dropped,
                                        np.ravel(y_train.to_numpy()), cv=5,
                                        scoring=f1)

    max_rod = rod_dropped_train.mean()
    min_f1 = f1_dropped_train.mean()

    dropped_pipeline.fit(X_train_dropped, np.ravel(y_train.to_numpy()))

    predicted_dropped = dropped_pipeline.predict(X_test_dropped)
    predicted_dropped_proba = dropped_pipeline.predict_proba(X_test)[:, 1]
    rod_dropped = ROD.ROD(y_pred=predicted_dropped_proba, sensitive=X_test.loc[:, ['sex']],
                          admissible=X_test.loc[:, admissible_features],
                          protected=' Female', name='dropped_adult')

    f1_dropped = f1_score(np.ravel(y_test.to_numpy()), predicted_dropped)

    method_list.append(['dropped', rod_dropped, f1_dropped, admissible_features, count + 1])

    print('ROD dropped ' + ': ' + str(rod_dropped))
    print('F1 dropped ' + ': ' + str(f1_dropped))

    ############################## Capuchin ####################################
    # Remove the sensitive when training and check results --> does ROD decrease variance? : No, bad results, go back

    capuchin_df = adult_df.copy()

    categorical = []
    for i in list(capuchin_df):
        if i != 'target':
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
    to_repair = pd.concat([X_train, y_train], axis=1)
    train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)
    end_time_capuchin = time.time() - start_time_capuchin

    print(runtimes)

    print('Finished repairing training set with capuchin')
    y_train_repaired = train_repaired.loc[:, ['target']].to_numpy()
    X_train_repaired = train_repaired.loc[:,
                       ['workclass', 'education', 'occupation', 'age', 'sex', 'marital-status', 'capital-gain',
                        'capital-loss', 'hours-per-week']]

    X_test_capuchin = (generate_binned_df(X_test)).loc[:,
                      ['workclass', 'education', 'occupation', 'age', 'sex', 'marital-status',
                       'capital-gain', 'capital-loss', 'hours-per-week']]

    capuchin_pipeline.fit(X_train_repaired, np.ravel(y_train_repaired))
    predicted_capuchin = capuchin_pipeline.predict(X_test_capuchin)
    predicted_capuchin_proba = capuchin_pipeline.predict_proba(X_test_capuchin)[:, 1]
    rod_capuchin = ROD.ROD(y_pred=predicted_capuchin_proba, sensitive=X_test.loc[:, ['sex']],
                           admissible=X_test.loc[:, admissible_features],
                           protected=' Female', name='capuchin_adult')

    f1_capuchin = f1_score(np.ravel(y_test.to_numpy()), predicted_capuchin)

    method_list.append(['capuchin', rod_capuchin, f1_capuchin, all_features, count + 1])

    print('ROD capuchin ' + ': ' + str(rod_capuchin))
    print('F1 capuchin ' + ': ' + str(f1_capuchin))

    ##################### Original

    categorical_features = []
    numerical_features = []
    for i in list(adult_df):
        if i != target and adult_df[i].dtype == np.dtype('O'):
            categorical_features.extend([i])
        elif i != target and adult_df[i].dtype != np.dtype('O'):
            numerical_features.extend([i])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)], remainder='passthrough')

    original_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('clf',
                                         LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                                            max_iter=100000, multi_class='auto'))])

    rod_original_train = cross_val_score(original_pipeline, X_train,
                                        np.ravel(y_train.to_numpy()), cv=5,
                                        scoring=rod_score)

    f1_original_train = cross_val_score(original_pipeline, X_train,
                                       np.ravel(y_train.to_numpy()), cv=5,
                                       scoring=f1)

    min_rod = rod_original_train.mean()
    max_f1 = f1_original_train.mean()

    original_pipeline.fit(X_train, np.ravel(y_train.to_numpy()))
    predicted_original = original_pipeline.predict(X_test)
    predicted_original_proba = original_pipeline.predict_proba(X_test)[:, 1]
    rod_original = ROD.ROD(y_pred=predicted_original_proba, sensitive=X_test.loc[:, ['sex']],
                           admissible=X_test.loc[:, admissible_features],
                           protected=' Female', name='original_adult')

    f1_original = f1_score(np.ravel(y_test.to_numpy()), predicted_original)

    method_list.append(['original', rod_original, f1_original, all_features, count + 1])

    method_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Fold'])
    if CF:
        method_df.to_csv(
            path_or_buf=results_path + '/complete_adult_results_joinscore_complexity_' + str(complexity) + '_CF_' + str(
                count) + '.csv', index=False)
    else:
        method_df.to_csv(path_or_buf=results_path + '/complete_adult_results_joinscore_complexity_' + str(complexity) + '_' + str(
            count) + '.csv',
                         index=False)

    print('ROD original ' + ': ' + str(rod_original))
    print('F1 original ' + ': ' + str(f1_original))


    ################ My algorithm


    accepted_features = []
    unique_features = []
    unique_representations = []
    transformed_train = np.empty((X_train.shape[0], 1))
    transformed_test = np.empty((X_test.shape[0], 1))
    all_allowed_train = np.empty((X_train.shape[0], 1))
    all_allowed_test = np.empty((X_test.shape[0], 1))
    allowed_names = []
    runtimes.extend([X_train.shape[0]])
    print(runtimes)
    registered_representations = []
    dropped_features = []
    global_join_score = 0
    for idx, i in enumerate(all_2_combinations):

        features2_build = []
        features2_build.extend(i)

        features2_build_cat = []
        features2_build_num = []

        for x in features2_build:
            if adult_df[x].dtype != np.dtype('O'):
                features2_build_num.extend([x])
            else:
                features2_build_cat.extend([x])

        features2_scale = []
        for x in features2_build:
            if adult_df[x].dtype != np.dtype('O'):
                features2_scale.extend([features2_build.index(x)])
            else:
                pass

        numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, features2_scale)], remainder='passthrough')

        new_order = features2_build_num + features2_build_cat
        features2_build_mask = ([False] * len(features2_build_num)) + ([True] * len(features2_build_cat))

        column_transformation = Pipeline([('new_construction',
                                           ConstructionTransformer(c_max=complexity, max_time_secs=1000000, scoring=f1, n_jobs=7,
                                                                   model=LogisticRegression(),
                                                                   parameter_grid={'penalty': ['l2'], 'C': [1],
                                                                                   'solver': ['lbfgs'],
                                                                                   'class_weight': ['balanced'],
                                                                                   'max_iter': [100000],
                                                                                   'multi_class': ['auto']}, cv=5,
                                                                   epsilon=-np.inf,
                                                                   feature_names=new_order,
                                                                   feature_is_categorical=features2_build_mask))])

        transformed_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                               ('feature_construction', column_transformation)])

        X_train_t = X_train.loc[:, features2_build].to_numpy()
        X_test_t = X_test.loc[:, features2_build].to_numpy()

        start_time_transform = time.time()
        transformed_train_i = transformed_pipeline.fit_transform(X_train_t, np.ravel(y_train.to_numpy()))
        time_2_create = (time.time() - start_time_transform)
        all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
            'new_construction'].all_features_set
        transformed_test_i = transformed_pipeline.transform(X_test_t)
        count_transformations += len(all_transformations)
        time_2_create_transformations += time_2_create

        #########Paralelize!!!!

        def causal_filter(candidate):

            j = (candidate.get_name()).strip()

            feature_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                             max_iter=100000, multi_class='auto')

            result = False

            if j != sensitive_feature:

                transformed_train_c = candidate.pipeline.transform(preprocessor.fit_transform(X_train_t))
                transformed_test_c = candidate.pipeline.transform(preprocessor.transform(X_test_t))

                if (np.isnan(transformed_train_c).sum() == 0 and np.isinf(transformed_train_c).sum() == 0) \
                        and (np.isnan(transformed_test_c).sum() == 0 and np.isinf(transformed_test_c).sum() == 0):

                    feature_clf.fit(transformed_train_c, np.ravel(y_train.to_numpy()))
                    outcome_candidate = feature_clf.predict(transformed_test_c)

                    outcome_df = pd.DataFrame(data=outcome_candidate, columns=['outcome'])
                    sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                                columns=[sensitive_feature])
                    selected_df_causal = pd.DataFrame(data=transformed_test_c, columns=[j])
                    test_df_causal = pd.concat([sensitive_df, selected_df_causal, outcome_df], axis=1)

                    if np.unique(transformed_test_c).shape[0] == 1 or np.unique(outcome_candidate).shape[0] == 1:
                        pass
                    elif d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):
                        result = True
                    else:
                        pass
                else:
                    pass
            else:
                pass

            return result


        transformations2_generate = [t for t in all_transformations if (t.get_name()).strip() not in unique_features]
        transformations2_generate_idx = [idx for idx, t in enumerate(all_transformations) if
                                         (t.get_name()).strip() not in unique_features]
        all_names = [(t.get_name()).strip() for t in all_transformations]

        unique_features.extend([(t.get_name()).strip() for t in transformations2_generate])

        if CF:
            start_time_CF = time.time()

            pool = mp.Pool(7)
            results = pool.map(causal_filter, transformations2_generate)
            pool.close()

            end_time_CF = time.time()-start_time_CF

            accepted_list = list(itertools.chain(*[results]))

            accepted_idx = np.argwhere(np.array(accepted_list))

            time_2_CF += end_time_CF

            mask = [x for idx, x in enumerate(transformations2_generate_idx) if accepted_list[idx]]
            filtered_transformations += len(mask)
        else:
            time_2_CF += 0
            mask = [x for x in transformations2_generate_idx]

        test_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                      max_iter=100000, multi_class='auto')

        print('round 1: Try to improve objective in 1 direction : ')

        start_time_FR = time.time()
        for idj, j in enumerate(mask):

            ##### step 1: Try to add a feature :

            transformed_train = np.concatenate((transformed_train, transformed_train_i[:, [j]]), axis=1)
            transformed_test = np.concatenate((transformed_test, transformed_test_i[:, [j]]), axis=1)
            all_allowed_train = np.concatenate((all_allowed_train, transformed_train_i[:, [j]]), axis=1)
            all_allowed_test = np.concatenate((all_allowed_test, transformed_test_i[:, [j]]), axis=1)
            allowed_names.extend([all_names[j]])
            accepted_features.extend([all_names[j]])
            accepted_features.sort()

            if idx == 0 and idj == 0:
                transformed_train = transformed_train[:, 1:]
                transformed_test = transformed_test[:, 1:]
            else:
                pass

            test_scores = cross_val_score(test_clf, transformed_train, np.ravel(y_train.to_numpy()), cv=5, scoring='f1')
            rod_scores = cross_val_score(test_clf, transformed_train,
                                           np.ravel(y_train.to_numpy()), cv=5,
                                           scoring=rod_score)

            scaled_f1 = (test_scores.mean() - min_f1) / (max_f1 - min_f1)
            scaled_rod = (rod_scores.mean() - min_rod) / (max_rod - min_rod)

            if test_scores.mean() > max_f1:
                max_f1 = test_scores.mean()
            else:
                pass

            if test_scores.mean() < min_f1:
                min_f1 =test_scores.mean()
            else:
                pass

            if rod_scores.mean() > max_rod:
                max_rod = rod_scores.mean()
            else:
                pass

            if rod_scores.mean() < min_rod:
                min_rod = rod_scores.mean()
            else:
                pass

            join_score = scaled_f1 + scaled_rod

            unique_representations.append(accepted_features.copy())
            test_clf.fit(transformed_train, np.ravel(y_train.to_numpy()))
            predicted_ff = test_clf.predict(transformed_test)
            predicted_ff_proba = test_clf.predict_proba(transformed_test)[:, 1]
            rod_ff = ROD.ROD(y_pred=predicted_ff_proba, sensitive=X_test.loc[:, ['sex']],
                             admissible=X_test.loc[:, admissible_features],
                             protected=' Female', name='backward_adult')
            f1_ff = f1_score(np.ravel(y_test.to_numpy()), predicted_ff)
            registered_representations.append(
                [accepted_features.copy(), len(accepted_features.copy()), f1_ff, rod_ff])

            print(transformed_train.shape, f1_ff, rod_ff, accepted_features)

            if join_score > global_join_score:
                global_join_score = join_score

                ##### Step 2: Try to remove a feature:

                selected_ids = []
                for idd, d in enumerate(range(transformed_train.shape[1])):
                    if transformed_train.shape[1] > len(selected_ids) + 1:
                        selected_ids.extend([idd])
                        transformed_train_r = np.delete(transformed_train, selected_ids, 1)
                        transformed_test_r = np.delete(transformed_test, selected_ids, 1)
                        accepted_features_r = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids]
                        accepted_features_r.sort()
                        if accepted_features_r not in unique_representations:
                            unique_representations.append(accepted_features_r.copy())
                            #print(idx, idj, idd, unique_representations)
                            test_scores_r = cross_val_score(test_clf, transformed_train_r, np.ravel(y_train.to_numpy()),
                                                            cv=5, scoring='f1')

                            rod_scores_r = cross_val_score(test_clf, transformed_train_r,
                                                            np.ravel(y_train.to_numpy()), cv=5,
                                                            scoring=rod_score)

                            scaled_f1_r = (test_scores_r.mean() - min_f1) / (max_f1 - min_f1)
                            scaled_rod_r = (rod_scores_r.mean() - min_rod) / (max_rod - min_rod)

                            if test_scores_r.mean() > max_f1:
                                max_f1 = test_scores_r.mean()
                            else:
                                pass

                            if test_scores_r.mean() < min_f1:
                                min_f1 = test_scores_r.mean()
                            else:
                                pass

                            if rod_scores_r.mean() > max_rod:
                                max_rod = rod_scores_r.mean()
                            else:
                                pass

                            if rod_scores_r.mean() < min_rod:
                                min_rod = rod_scores_r.mean()
                            else:
                                pass

                            join_score_r = scaled_f1_r + scaled_rod_r

                            test_clf.fit(transformed_train_r, np.ravel(y_train.to_numpy()))
                            predicted_ff_r = test_clf.predict(transformed_test_r)
                            predicted_ff_r_proba = test_clf.predict_proba(transformed_test_r)[:, 1]
                            rod_ff_r = ROD.ROD(y_pred=predicted_ff_r_proba, sensitive=X_test.loc[:, ['sex']],
                                               admissible=X_test.loc[:, admissible_features],
                                               protected=' Female', name='backward_adult')
                            f1_ff_r = f1_score(np.ravel(y_test.to_numpy()), predicted_ff_r)
                            registered_representations.append(
                                [accepted_features_r.copy(), len(accepted_features_r.copy()), f1_ff_r, rod_ff_r])

                            if join_score_r > global_join_score:
                                global_join_score = join_score_r
                            else:
                                selected_ids.remove(idd)
                        else:
                            selected_ids.remove(idd)
                    else:
                        pass

                if len(selected_ids) > 0:
                    transformed_train = np.delete(transformed_train, selected_ids, 1)
                    transformed_test = np.delete(transformed_test, selected_ids, 1)
                    accepted_features = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids]
                else:
                    pass
            else:
                if idx == 0 and idj == 0:
                    pass
                else:
                    transformed_train = np.delete(transformed_train, -1, 1)
                    transformed_test = np.delete(transformed_test, -1, 1)
                    del accepted_features[-1]

                    #print(unique_representations)

                    dropped_features.extend([j])

        end_time_FR = time.time() - start_time_FR

        time_2_FR += end_time_FR

    print(transformed_train.shape, transformed_test.shape, str(global_join_score))

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    complete_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                       max_iter=100000, multi_class='auto')

    complete_clf.fit(transformed_train, np.ravel(y_train.to_numpy()))
    predicted_backward = complete_clf.predict(transformed_test)
    predicted_backward_proba = complete_clf.predict_proba(transformed_test)[:, 1]
    rod_backward = ROD.ROD(y_pred=predicted_backward_proba, sensitive=X_test.loc[:, ['sex']],
                           admissible=X_test.loc[:, admissible_features],
                           protected=' Female', name='backward_adult')

    f1_backward = f1_score(np.ravel(y_test.to_numpy()), predicted_backward)

    method_list.append(['FC_SFFS_backward', rod_backward, f1_backward, accepted_features, count + 1])
    runtimes.extend([count_transformations, time_2_create_transformations, filtered_transformations, time_2_CF, time_2_FR,
                     time_2_create_transformations+time_2_CF+time_2_FR, end_time_capuchin])

    print(runtimes)

    visited_representations = pd.DataFrame(registered_representations, columns=['Representation', 'Size', 'F1', 'ROD'])
    visited_representations['Fold'] = count + 1

    if CF:
        visited_representations.to_csv(path_or_buf=results_path + '/adult_complete_visited_representations_joinscore_complexity_' + str(complexity)
                                                   + '_CF_' + str(count+1) + '.csv', index=False)
    else:
        visited_representations.to_csv(
            path_or_buf=results_path + '/adult_complete_visited_representations_joinscore_complexity_' + str(complexity) + '_' + str(count + 1) + '.csv', index=False)

    print('ROD join ' + ': ' + str(rod_backward))
    print('F1 join ' + ': ' + str(f1_backward))

    count += 1

    runtimes_array = np.asarray(runtimes)

    runtimes_array = np.reshape(runtimes_array, (1, 10))

    runtimes_df = pd.DataFrame(runtimes_array, columns=['Combinations', 'Complexity', 'Rows', 'Transformations',
                                                  'Time_2_transformations', 'Filtered_transformations',
                                                  'Time_2_CF', 'Time_2_FR', 'Total_runtime_SFFS_BF', 'Repair_time_capuchin'])
    runtimes_df['Fold'] = count

    if CF:
        runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_joinscore_complexity_' + str(complexity) + '_CF_' + str(count) + '.csv',
                       index=False)
    else:
        runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_joinscore_complexity_' + str(complexity) + '_' + str(count) + '.csv',
                           index=False)

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Fold'])


print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

if CF:
    summary_df.to_csv(path_or_buf=results_path + '/complete_adult_results_joinscore_complexity_' + str(complexity) + '_CF.csv', index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/complete_adult_results_joinscore_complexity_' + str(complexity) + '.csv', index=False)