import pandas as pd
from pathlib import Path
import itertools
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from causality.d_separation import d_separation
import multiprocessing as mp
from fastsklearnfeature.candidate_generation.feature_space.division import get_transformation_for_division
from new_project.explore_transformations.considered_transformations import get_transformation_without_oh
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy.linalg import norm
import time
import sys
sys.path.insert(0, '/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from capuchin import repair_dataset

home = str(Path.home())

adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

adult_df = pd.read_csv(adult_path + '/adult.csv', sep=',', header=0)


def label(row):
    if row['class'] == ' <=50K':
        return 0
    else:
        return 1


def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in ['target', 'outcome'] and (df_[i].dtype != object and len(df_[i].unique()) > 4):
            out = pd.cut(df_[i], bins=2)
            df_.loc[:, i] = out.astype(str)

    return df_


sensitive_feature = 'sex'
target = 'target'
inadmissible_features = ['marital-status']
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class', 'relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)
admissible_features = [i for i in list(adult_df) if
                       i not in inadmissible_features and i != sensitive_feature and i != target]


complexity = 4
CF = False
count = 0
method_list = []
kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf1.split(adult_df):

    all_features = list(adult_df)
    all_features.remove(target)



    train_df = adult_df.iloc[train_index]
    test_df = adult_df.iloc[test_index]

    X_train = train_df.loc[:, all_features]

    y_train = train_df.loc[:, 'target']

    X_test = test_df.loc[:, all_features]

    y_test = test_df.loc[:, 'target']

    outcome = pd.DataFrame(y_train.values, columns=['outcome'])
    outcome.reset_index(drop=True, inplace=True)
    admissible_data = X_train.loc[:, admissible_features]
    admissible_data.reset_index(drop=True, inplace=True)

    df = pd.concat([admissible_data, outcome], axis=1)

    mb = ROD.learn_MB(df=df, name='filter')
    all_2_combinations = list(itertools.combinations(mb, 2))

    print(mb)

    runtimes = [len(mb), complexity]
    count_transformations = 0
    filtered_transformations = 0
    time_2_create_transformations = 0
    time_2_CF = 0
    time_2_FR = 0
    time_2_SR = 0

    rod_score = make_scorer(ROD.ROD, greater_is_better=True, needs_proba=True,
                            sensitive=X_train.loc[:, sensitive_feature],
                            admissible=X_train.loc[:, admissible_features],
                            protected=' Female', name='train_adult')

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

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
    registered_representations_train = []
    registered_representations_test = []
    dropped_features = []
    join_score = 0
    for idx, i in enumerate(all_2_combinations):
        transformation = get_transformation_for_division
        if not (sensitive_feature==i[0] and inadmissible_features[0] == i[1]) or (sensitive_feature==i[1] and inadmissible_features[0] == i[0]):

            if sensitive_feature in i or inadmissible_features[0] in i:
                if adult_df[i[0]].dtype == object and adult_df[i[1]].dtype == object:
                    continue
                else:
                    transformation = get_transformation_without_oh
            else:
                pass

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
                                               ConstructionTransformer(c_max=complexity, max_time_secs=1000000, scoring=f1, n_jobs=10,
                                                                       model=LogisticRegression(),
                                                                       parameter_grid={'penalty': ['l2'], 'C': [1],
                                                                                       'solver': ['lbfgs'],
                                                                                       'class_weight': ['balanced'],
                                                                                       'max_iter': [100000],
                                                                                       'multi_class': ['auto']}, cv=5,
                                                                       epsilon=-np.inf,
                                                                       feature_names=new_order,
                                                                       feature_is_categorical=features2_build_mask,
                                                                       transformation_producer=transformation))])

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

            outcome_df = train_df.loc[:, ['target']]
            outcome_df.reset_index(inplace=True, drop=True)
            outcome_df.rename(columns={'target': 'outcome'}, inplace=True)
            sensitive_df = pd.DataFrame(data=X_train.loc[:, sensitive_feature].to_numpy(),
                                        columns=[sensitive_feature])

            def causal_filter(candidate):

                result = False

                j = (candidate.get_name()).strip()

                transformed_train_c = candidate.pipeline.transform(preprocessor.fit_transform(X_train_t))

                if np.isnan(transformed_train_c).sum() == 0 and np.isinf(transformed_train_c).sum() == 0 \
                            and np.unique(transformed_train_c).shape[0] > 1 \
                            and np.unique(transformed_train_c).shape[0] > 1:

                    candidate_df = pd.DataFrame(transformed_train_c, columns=[j])

                    test_df_causal = pd.concat([sensitive_df, candidate_df, outcome_df], axis=1)

                    if re.search(sensitive_feature, j) or re.search(inadmissible_features[0], j):
                        test_sensitive = pd.concat([sensitive_df, candidate_df], axis=1)
                        test_sensitive.rename(columns={j:'candidate'}, inplace=True)
                        if d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome') and \
                                d_separation(test_sensitive, sensitive=sensitive_feature, target='candidate'):
                            result = True
                    elif d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):
                        result = True
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

                pool = mp.Pool(10)
                results = pool.map(causal_filter, transformations2_generate)
                pool.close()

                end_time_CF = time.time()-start_time_CF

                accepted_list = list(itertools.chain(*[results]))
                accepted_idx = np.argwhere(np.array(accepted_list))

                mask = list(itertools.compress(transformations2_generate_idx, accepted_list))

                time_2_CF += end_time_CF

                filtered_transformations += len(mask)

                print('Filtered Transformations: ' + str(len(transformations2_generate_idx)-len(mask)))
            else:
                time_2_CF += 0
                mask = [x for x in transformations2_generate_idx]

            test_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                          max_iter=100000, multi_class='auto', n_jobs=-1)

            print('round 1: Try to improve objective in 1 direction : ')

            start_time_FR = time.time()
            for idj, j in enumerate(mask):

                ##### step 1: Try to add a feature :

                if np.isnan(transformed_train_i[:, [j]]).sum() == 0 and np.isinf(transformed_test_i[:, [j]]).sum() == 0 and \
                        np.unique(transformed_train_i[:, [j]]).shape[0] > 1 and np.unique(transformed_test_i[:, [j]]).shape[0] > 1:

                    transformed_train = np.concatenate((transformed_train, transformed_train_i[:, [j]]), axis=1)
                    transformed_test = np.concatenate((transformed_test, transformed_test_i[:, [j]]), axis=1)
                    all_allowed_train = np.concatenate((all_allowed_train, transformed_train_i[:, [j]]), axis=1)
                    all_allowed_test = np.concatenate((all_allowed_test, transformed_test_i[:, [j]]), axis=1)
                    allowed_names.extend([all_names[j]])
                    accepted_features.extend([all_names[j]])
                    accepted_features_photo = accepted_features.copy()
                    accepted_features_photo.sort()

                    if idx == 0 and idj == 0:
                        transformed_train = transformed_train[:, 1:]
                        transformed_test = transformed_test[:, 1:]
                    else:
                        pass

                    cv_scores = GridSearchCV(LogisticRegression(), param_grid={
                        'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'], 'class_weight': ['balanced'],
                        'max_iter': [100000], 'multi_class': ['auto']
                    },
                                             n_jobs=-1,
                                             scoring={'F1': f1, 'ROD': rod_score}, refit='F1', cv=3)

                    cv_scores.fit(transformed_train, np.ravel(y_train.to_numpy()))
                    test_scores = cv_scores.cv_results_['mean_test_F1'][0]
                    rod_scores = cv_scores.cv_results_['mean_test_ROD'][0]

                    unique_representations.append(accepted_features_photo.copy())
                    #test_clf.fit(transformed_train, np.ravel(y_train.to_numpy()))
                    predicted_ff = cv_scores.predict(transformed_test)
                    predicted_ff_proba = cv_scores.predict_proba(transformed_test)[:, 1]
                    rod_ff = ROD.ROD(y_pred=predicted_ff_proba, sensitive=X_test.loc[:, ['sex']],
                                     admissible=X_test.loc[:, admissible_features],
                                     protected=' Female', name='backward_adult')
                    f1_ff = f1_score(np.ravel(y_test.to_numpy()), predicted_ff)

                    registered_representations_test.append(
                        [accepted_features.copy(), len(accepted_features.copy()), f1_ff, rod_ff])
                    registered_representations_train.append(
                        [accepted_features.copy(), len(accepted_features.copy()), test_scores, rod_scores])

                    print(transformed_train.shape, f1_ff, rod_ff, accepted_features)

                    if test_scores > join_score:
                        join_score = test_scores

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
                                    cv_scores.fit(transformed_train_r, np.ravel(y_train.to_numpy()))
                                    test_scores_r = cv_scores.cv_results_['mean_test_F1'][0]
                                    rod_scores_r = cv_scores.cv_results_['mean_test_ROD'][0]

                                    #test_clf.fit(transformed_train_r, np.ravel(y_train.to_numpy()))
                                    predicted_ff_r = cv_scores.predict(transformed_test_r)
                                    predicted_ff_r_proba = cv_scores.predict_proba(transformed_test_r)[:, 1]
                                    rod_ff_r = ROD.ROD(y_pred=predicted_ff_r_proba, sensitive=X_test.loc[:, ['sex']],
                                                       admissible=X_test.loc[:, admissible_features],
                                                       protected=' Female', name='backward_adult')
                                    f1_ff_r = f1_score(np.ravel(y_test.to_numpy()), predicted_ff_r)

                                    registered_representations_test.append(
                                        [accepted_features_r.copy(), len(accepted_features_r.copy()), f1_ff_r, rod_ff_r])
                                    registered_representations_train.append(
                                        [accepted_features_r.copy(), len(accepted_features_r.copy()), test_scores_r, rod_scores_r])

                                    if test_scores_r > join_score:
                                        join_score = test_scores_r
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
                else:
                    pass

            end_time_FR = time.time() - start_time_FR

            time_2_FR += end_time_FR
        else:
            pass

    start_time_SR = time.time()

    print(transformed_train.shape, transformed_test.shape, str(join_score))

    cv_scores = GridSearchCV(LogisticRegression(), param_grid={
        'penalty': ['l2'], 'C': [0.5, 1, 1.5], 'solver': ['lbfgs', 'newton-cg', 'sag'],
        'class_weight': [None, 'balanced'],
        'max_iter': [100000], 'multi_class': ['auto']
    },
                             n_jobs=-1,
                             scoring={'F1': f1, 'ROD': rod_score}, refit='F1', cv=3)

    cv_scores.fit(transformed_train, np.ravel(y_train.to_numpy()))

    predicted = cv_scores.predict(transformed_test)
    predicted_proba = cv_scores.predict_proba(transformed_test)[:, 1]

    rod_complete = ROD.ROD(y_pred=predicted_proba, sensitive=X_test.loc[:, ['sex']],
                       admissible=X_test.loc[:, admissible_features],
                       protected=' Female', name='backward_adult')
    f1_complete = f1_score(np.ravel(y_test.to_numpy()), predicted)

    selected_representation = [accepted_features, len(accepted_features), f1_complete, rod_complete]

    print('__________________________________________')
    print('Round 2 : Improving in the other direction. Start with backward floating elimination: ')

    print('F1 complete: {:.4f}'.format(f1_complete))
    print('ROD complete {:.4f}'.format(rod_complete))

    method_list.append(['FC_filter_SFFS', selected_representation[3], selected_representation[2], selected_representation[0],
                       len(selected_representation[0]), count + 1])
    runtimes.extend([count_transformations, time_2_create_transformations, filtered_transformations, time_2_CF, time_2_FR, time_2_SR,
                     time_2_create_transformations+time_2_CF+time_2_FR+time_2_SR])

    print(runtimes)

    visited_representations_train = pd.DataFrame(registered_representations_train, columns=['Representation', 'Size', 'F1', 'ROD'])
    visited_representations_train['Fold'] = count + 1

    visited_representations_test = pd.DataFrame(registered_representations_test,
                                                 columns=['Representation', 'Size', 'F1', 'ROD'])
    visited_representations_test['Fold'] = count + 1

    if CF:
        visited_representations_train.to_csv(path_or_buf=results_path + '/adult_complete_visited_representations_train_complexity_' + str(complexity)
                                                   + '_CF_' + str(count+1) + '.csv', index=False)
        visited_representations_test.to_csv(
            path_or_buf=results_path + '/adult_complete_visited_representations_test_complexity_' + str(complexity)
                        + '_CF_' + str(count + 1) + '.csv', index=False)
    else:
        visited_representations_train.to_csv(
            path_or_buf=results_path + '/adult_complete_visited_representations_train_complexity_' + str(complexity) + '_' + str(count + 1) + '.csv',
            index=False)
        visited_representations_test.to_csv(
            path_or_buf=results_path + '/adult_complete_visited_representations_test_complexity_' + str(
                complexity) + '_' + str(count + 1) + '.csv',
            index=False)

    print('ROD backward ' + ': ' + str(selected_representation[3]))
    print('F1 backward ' + ': ' + str(selected_representation[2]))

    count += 1

    runtimes_array = np.asarray(runtimes)

    runtimes_array = np.reshape(runtimes_array, (1, runtimes_array.shape[0]))

    runtimes_df = pd.DataFrame(runtimes_array, columns=['Combinations', 'Complexity', 'Rows', 'Transformations',
                                                  'Time_2_transformations', 'Filtered_transformations',
                                                  'Time_2_CF', 'Time_2_FR', 'Time_2_SR', 'Total_runtime_SFFS_BF'])

    method_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])
    runtimes_df['Fold'] = count

    if CF:
        runtimes_df.to_csv(path_or_buf=results_path + '/filter_runtimes_complexity_' + str(complexity) + '_CF_' + str(count) + '.csv',
                       index=False)
        method_df.to_csv(
            path_or_buf=results_path + '/filter_adult_complexity_' + str(complexity) + '_CF_' + str(count) + '.csv',
            index=False)
    else:
        runtimes_df.to_csv(path_or_buf=results_path + '/filter_runtimes_complexity_' + str(complexity) + '_' + str(count) + '.csv',
                           index=False)
        method_df.to_csv(
            path_or_buf=results_path + '/filter_adult_complexity_' + str(complexity) + str(count) + '.csv',
            index=False)

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])


print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

if CF:
    summary_df.to_csv(path_or_buf=results_path + '/filter_complete_adult_results_complexity_' + str(complexity) + '_CF.csv', index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/filter_complete_adult_results_complexity_' + str(complexity) + '.csv', index=False)