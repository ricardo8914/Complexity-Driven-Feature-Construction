import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from causality.d_separation import d_separation
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
from sklearn.preprocessing import Normalizer
import ROD
from numpy.linalg import norm
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import time

home = str(Path.home())

adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'
adult_df = pd.read_csv(adult_path + '/adult.csv', sep=',', header=0)


def label(row):
    if row['class'] == ' <=50K':
        return 0
    else:
        return 1


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

count = 0
complexity = 3
method_list = []
runtimes = []
CF = False
kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf1.split(adult_df):

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


    unique_features = []
    join_score = 0
    all_allowed_train = np.empty((X_train.shape[0], 1))
    all_allowed_test = np.empty((X_test.shape[0], 1))
    allowed_names = []
    start_time = time.time()
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

        f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

        column_transformation = Pipeline([('new_construction',
                                           ConstructionTransformer(c_max=complexity, max_time_secs=1000000, scoring=f1, n_jobs=4,
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



        transformed_train_i = transformed_pipeline.fit_transform(X_train_t, np.ravel(y_train.to_numpy()))
        all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
            'new_construction'].all_features_set
        transformed_test_i = transformed_pipeline.transform(X_test_t)


        transformations2_generate = [t for t in all_transformations if ((t.get_name()).strip() not in unique_features
                                and not (t.get_complexity() == 2 and re.search(sensitive_feature, t.get_name().strip()))
                                and not (t.get_complexity() == 2 and re.search(inadmissible_features[0], t.get_name().strip())))]
        transformations2_generate_idx = [idx for idx, t in enumerate(all_transformations) if
                                         (t.get_name()).strip() not in unique_features
                                         and not (t.get_complexity() == 2 and re.search(sensitive_feature, t.get_name().strip()))
                                         and not (t.get_complexity() == 2 and re.search(inadmissible_features[0], t.get_name().strip()))]

        all_names = [(t.get_name()).strip() for t in all_transformations]

        unique_features.extend([(t.get_name()).strip() for t in transformations2_generate])

        outcome_df = train_df.loc[:, ['target']]
        outcome_df.reset_index(inplace=True, drop=True)
        outcome_df.rename(columns={'target': 'outcome'}, inplace=True)


        # selected_df_causal = pd.DataFrame(data=all_allowed_train, columns=allowed_names)
        # test_df_causal = pd.concat([selected_df_causal, outcome_df], axis=1)

        # result = False
        # if d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):
        # mb = ROD.learn_MB(test_df_causal, 'markov_fs')
        # selected = [allowed_names.index(x) for x in mb]

        def causal_filter(candidate):

            result = False

            candidate_df = all_allowed_train[:, candidate]

            sensitive_df = pd.DataFrame(data=X_train.loc[:, sensitive_feature].to_numpy(),
                                        columns=[sensitive_feature])
            selected_df_causal = pd.DataFrame(data=candidate_df, columns=[allowed_names[candidate]])
            test_df_causal = pd.concat([sensitive_df, selected_df_causal, outcome_df], axis=1)

            if d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):
                result = True
            else:
                pass

            return result

        if CF:
            pool = mp.Pool(100)
            results = pool.map(causal_filter, transformations2_generate_idx)
            pool.close()

            accepted_list = list(itertools.chain(*[results]))
            accepted_idx = np.argwhere(np.array(accepted_list))

            new_selected = list(itertools.compress(transformations2_generate_idx, accepted_list))

            new_allowed_names = [x for idx, x in enumerate(allowed_names) if idx in new_selected]


            mask = [x for x in new_selected]
        else:
            mask = [x for x in transformations2_generate_idx]


        for idj, j in enumerate(mask):

            if np.isnan(transformed_train_i[:, [j]]).sum() == 0 and np.isinf(transformed_test_i[:, [j]]).sum() == 0 and \
                np.unique(transformed_train_i[:, [j]]).shape[0] > 1 and np.unique(transformed_test_i[:, [j]]).shape[0] > 1:

                all_allowed_train = np.concatenate((all_allowed_train, transformed_train_i[:, [j]]), axis=1)
                all_allowed_test = np.concatenate((all_allowed_test, transformed_test_i[:, [j]]), axis=1)
                allowed_names.extend([all_names[j]])
            else:
                pass

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    ############ Genetic: Try to move it to before SFFS

    all_allowed_train = all_allowed_train[:, 1:]
    all_allowed_test = all_allowed_test[:, 1:]


    #else:
        #print(ROD.learn_MB(test_df_causal, 'markov_fs'))

    cv_scores = GridSearchCV(LogisticRegression(), param_grid={
        'penalty': ['l2'], 'C': [0.5, 1, 1.5], 'solver': ['lbfgs'],
        'class_weight': ['balanced'],
        'max_iter': [100000], 'multi_class': ['auto']
    },
                             n_jobs=-1,
                             scoring={'F1': f1, 'ROD': rod_score}, refit='ROD', cv=5)

    ###### Learn feature importances

    sample_idx_all = np.random.randint(all_allowed_train.shape[0], size=round(all_allowed_train.shape[0] * 0.2))
    sample_x_all = all_allowed_train[sample_idx_all, :]
    sample_y_all = np.ravel(y_train.to_numpy())
    sample_y_all = sample_y_all[sample_idx_all]
    estimator = LogisticRegression(penalty= 'l2', C= 1, solver= 'lbfgs',
        class_weight= 'balanced',
        max_iter= 100000, multi_class= 'auto')

    selector = RFECV(estimator, step=0.05, cv=5, n_jobs=-1, scoring='f1')
    selector = selector.fit(sample_x_all, sample_y_all)
    support = selector.support_
    where_support = np.argwhere(support)
    where_support = [i.item() for i in where_support]
    print('support size: ' + str(len(where_support)))
    
    def eliminate(candidate):

        accepted_features_remove = accepted_features.copy()
        accepted_features_remove.remove(allowed_names[candidate])
        accepted_features_remove.sort()
        r = []
        u = unique_representations.copy()
        if accepted_features_remove not in u:
            temp_deleted = deleted_set.copy()
            temp_deleted.extend([candidate])
            intersection = [f for f in where_support if f not in temp_deleted]
            sample_idx = np.random.randint(all_set.shape[0], size=round(all_set.shape[0]*0.2))
            sample_x = all_set[sample_idx, :]
            transformed_train_remove = sample_x[:, intersection]
            sample_y = np.ravel(y_train.to_numpy())
            sample_y = sample_y[sample_idx]
            cv_scores.fit(transformed_train_remove, sample_y)
            rod_scores_cr = cv_scores.cv_results_['mean_test_ROD'][0]
            test_scores_cr = cv_scores.cv_results_['mean_test_F1'][0]

            r.append(
                [accepted_features_remove, len(accepted_features_remove), test_scores_cr, rod_scores_cr, intersection])
        else:
            rod_scores_cr = np.NAN

        return [rod_scores_cr, r]

    def add(candidate):

        accepted_features_add = accepted_features.copy()
        accepted_features_add.extend([allowed_names[candidate]])
        accepted_features_add.sort()
        r = []
        u = unique_representations.copy()
        if accepted_features_add not in u:
            temp_added = deleted_set.copy()
            temp_added.remove(candidate)
            intersection = [f for f in where_support if f not in temp_added]
            sample_idx = np.random.randint(all_set.shape[0], size=round(all_set.shape[0]*0.2))
            sample_x = all_set[sample_idx, :]
            transformed_train_add = sample_x[:, intersection]
            sample_y = np.ravel(y_train.to_numpy())
            sample_y = sample_y[sample_idx]
            cv_scores.fit(transformed_train_add, sample_y)
            rod_scores_cr = cv_scores.cv_results_['mean_test_ROD'][0]
            test_scores_cr = cv_scores.cv_results_['mean_test_F1'][0]
            r.append(
                [accepted_features_add, len(accepted_features_add), test_scores_cr, rod_scores_cr, intersection])
        else:
            rod_scores_cr = np.NAN

        return [rod_scores_cr, r]

    all_set = all_allowed_train.copy()
    registered_representations_train = []
    deleted_set = []
    unique_representations = []
    global_rod = -np.inf
    while True:

        accepted_features = [allowed_names[idf] for idf in where_support if idf not in deleted_set]
        alive = [idx for idx in where_support if idx not in deleted_set]
        print('current size: ' + str(len(alive)))
        print('deleted: ' + str(len(deleted_set)))

        if len(alive) == 2:
            break
        else:
            pass

        #first phase
        pool = mp.Pool(mp.cpu_count())
        results_back = pool.map(eliminate, alive)
        pool.close()

        evaluation_back = list(itertools.chain(*[results_back]))

        rod_evaluation_back = [item[0] for item in evaluation_back]

        registered_back = []
        for item in evaluation_back:
            try:
                i = item[1]
                registered_back.extend(i)
            except IndexError:
                pass

        registered_representations_train.extend(registered_back)
        print('length visited representations: ' + str(len(registered_representations_train)))
        #if np.amax(np.array(rod_evaluation_back)) < global_rod:
        #    break
        #else:
        max_idx_back = np.nanargmax(np.array(rod_evaluation_back))
            #if np.amax(np.array(rod_evaluation_back)) > global_rod:
        print('deleted: ' + allowed_names[alive[max_idx_back.item()]])
        deleted_set.extend([alive[max_idx_back.item()]])
        accepted_features = [allowed_names[idf] for idf in where_support if idf not in deleted_set]
        accepted_features.sort()
        unique_representations.append(accepted_features)
        global_rod = np.amax(np.array(rod_evaluation_back))

        #Second phase

        # death = deleted_set.copy()
        # pool = mp.Pool(mp.cpu_count())
        # results_ff = pool.map(add, death)
        # pool.close()
        #
        # evaluation_ff = list(itertools.chain(*[results_ff]))
        # rod_evaluation_ff = [item[0] for item in evaluation_ff]
        # registered_forward = []
        # for item in evaluation_ff:
        #     try:
        #         i = item[1]
        #         registered_forward.extend(i)
        #     except IndexError:
        #         pass
        # registered_representations_train.extend(registered_forward)
        #
        # print('length visited representations: ' + str(len(registered_representations_train)))
        #
        # max_idx_ff = np.nanargmax(np.array(rod_evaluation_ff))
        #
        # if np.amax(np.array(rod_evaluation_ff)) > global_rod:
        #     print('added: ' + allowed_names[death[max_idx_ff.item()]])
        #     deleted_set.remove(death[max_idx_ff.item()])
        #     accepted_features = [allowed_names[idf] for idf in where_support if idf not in deleted_set]
        #     accepted_features.sort()
        #     unique_representations.append(accepted_features)
        #     global_rod = np.amax(np.array(rod_evaluation_ff))
        # else:
        #     print('Failed to add: ' + allowed_names[death[max_idx_ff.item()]])


    def is_pareto_efficient_simple(costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    all_visited = np.asarray(registered_representations_train)
    scores = all_visited[:, [2, 3]]

    #v = scores[:, 1]
    #normalized = (v - v.min()) / (0 - v.min())
    #scores[:, 1] = normalized

    #v_1 = scores[:, 1]
    #n = Normalizer().fit_transform(v_1.reshape(1, -1))
    #norm = norm.reshape(v_1.shape[0], 1)

    #normalized = (scores[:, 1] - scores[:, 1].min()) / (0 - scores[:, 1].min())
    #scores[:, 1] = normalized

    pareto = is_pareto_efficient_simple(scores)
    where_pareto = np.argwhere(pareto)
    #pareto_front = scores
    pareto_front = scores[pareto]

    ideal_point = np.asarray([1, 0])
    dist = np.empty((pareto_front.shape[0], 1))

    for idx, i in enumerate(pareto_front):
        dist[idx] = norm(i - ideal_point)

    min_dist = np.argmin(dist)
    selected_indices = all_visited[where_pareto[min_dist].item()][4]
    print('selected representation train, Index=' + str(where_pareto[min_dist].item()) + ' ,F1=' + str(all_visited[where_pareto[min_dist].item()][2]) + ', ROD =' + str(all_visited[where_pareto[min_dist].item()][3]))
    #selected_indices = all_visited[where_pareto[min_dist].item()][4]

    test_clf = GridSearchCV(LogisticRegression(), param_grid={
        'penalty': ['l2'], 'C': [0.5, 1, 1.5], 'solver': ['lbfgs'],
        'class_weight': ['balanced'],
        'max_iter': [100000], 'multi_class': ['auto']
    },
                             n_jobs=-1,
                             scoring=rod_score, cv=5)

    transformed_train_selected = all_set[:, selected_indices]
    test_clf.fit(transformed_train_selected, np.ravel(y_train.to_numpy()))

    predicted_test = test_clf.predict(all_allowed_test[:, selected_indices])
    predicted_test_proba = test_clf.predict_proba(all_allowed_test[:, selected_indices])[:, 1]

    rod_test = ROD.ROD(y_pred=predicted_test_proba, sensitive=X_test.loc[:, ['sex']],
                    admissible=X_test.loc[:, admissible_features],
                    protected=' Female', name='backward_adult')
    f1_test = f1_score(np.ravel(y_test.to_numpy()), predicted_test)

    representation = [x for idx, x in enumerate(allowed_names) if idx in selected_indices]

    selected_representation = [representation, len(representation), f1_test, rod_test]

    end_time = time.time() - start_time

    runtimes.append(['FC_SBFS', complexity, all_allowed_train.shape[0], all_allowed_train.shape[1], end_time, count + 1])

    method_list.append(['FC_SBFS', selected_representation[3], selected_representation[2], selected_representation[0],
                       selected_representation[1], count + 1])

    visited_representations_train = pd.DataFrame(registered_representations_train,
                                                 columns=['Representation', 'Size', 'F1', 'ROD', 'Indices'])
    visited_representations_train['Fold'] = count + 1
    visited_representations_train = visited_representations_train.loc[:, ['Representation', 'Size', 'F1', 'ROD', 'Fold']]

    if CF:
        visited_representations_train.to_csv(path_or_buf=results_path + '/adult_FC_SBFS_visited_representations_train_complexity_' + str(complexity)
                                                   + '_CF_' + str(count+1) + '.csv', index=False)
    else:
        visited_representations_train.to_csv(
            path_or_buf=results_path + '/adult_SBFS_visited_representations_train_complexity_' + str(complexity) + '_' + str(count + 1) + '.csv',
            index=False)

    print('ROD selected: ' + str(selected_representation[3]) + ' F1: ' + str(selected_representation[2]))

    count += 1

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])
runtimes_df = pd.DataFrame(runtimes, columns=['Method', 'Complexity', 'Rows', 'Features', 'Runtime', 'Fold'])

print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

if CF:
    summary_df.to_csv(path_or_buf=results_path + '/adult_FC_SBFS_complexity_' + str(complexity) + '_CF' + '.csv', index=False)
    runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_adult_FC_SBFS_complexity_' + str(complexity) + '_CF' + '.csv',
                          index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/adult_FC_SBFS_complexity_' + str(complexity) + '.csv',
                      index=False)
    runtimes_df.to_csv(
        path_or_buf=results_path + '/runtimes_adult_FC_SBFS_complexity_' + str(complexity) + '.csv',
        index=False)
