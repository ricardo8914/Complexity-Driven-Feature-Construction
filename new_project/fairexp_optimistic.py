from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.metrics import f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import multiprocessing as mp
from functools import partial
import pandas as pd
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from fmeasures import ROD
from sklearn.preprocessing import MinMaxScaler
import time
from numpy.linalg import norm
from causality.causal_filter import causal_filter
from sklearn.feature_selection import mutual_info_classif
from fmeasures import CDP
from filters.bloom_filter import BloomFilter
from fairlearn.metrics import demographic_parity_difference, MetricFrame, true_positive_rate, true_negative_rate
from random import randrange
import parallel_variables


#c = Config.Config

#print(c.load())


def construct_features(df=None, complexity=None, scoring=None, target=None):
    X = df.loc[:, [i for i in list(df) if i != target]]
    y = np.ravel(df.loc[:, target].to_numpy())

    features2_scale = []
    for idx, i in enumerate(list(X)):
        if X.loc[:, i].dtype in (int, float):
            features2_scale.append(idx)
        else:
            pass

    features2_build_cat = []
    features2_build_num = []

    for i in list(X):
        if X.loc[:, i].dtype in (int, float):
            features2_build_num.append(i)
        else:
            features2_build_cat.append(i)
            X.loc[:, i] = X.loc[:, i].astype('object')

    new_order = features2_build_num + features2_build_cat

    features2_build_mask = ([False] * len(features2_build_num)) + ([True] * len(features2_build_cat))


    numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, features2_scale)], remainder='passthrough')

    column_transformation = Pipeline([('new_construction',
                                           ConstructionTransformer(c_max=complexity, max_time_secs=1000000, scoring=scoring, n_jobs=mp.cpu_count(),
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

    Z = transformed_pipeline.fit_transform(X.to_numpy(), y)
    all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
        'new_construction'].all_features_set
    all_names = [(t.get_name()).strip() for t in all_transformations]
    complexities = [t.get_complexity() for t in all_transformations]

    return Z, all_names, complexities, all_transformations


def extend_dataframe_complete(df=None, complexity=None, scoring=None, target=None, sampling=None, train_indices=None, prefiltering=True):

    train_df = df.iloc[train_indices]
    train_df.reset_index(inplace=True, drop=True)


    if sampling == 1.0:
        sample_idx_all = np.random.randint(train_df.shape[0], size=round(train_df.shape[0] * 0.5))
        y_mi_sample = np.ravel(train_df.iloc[sample_idx_all, list(train_df).index(target)].to_numpy())
    else:
        sample_idx_all = np.random.randint(train_df.shape[0], size=round(train_df.shape[0] * sampling))
        y_mi_sample = np.ravel(df.iloc[sample_idx_all, list(train_df).index(target)].to_numpy())

    df_ = train_df.iloc[sample_idx_all]
    df_.reset_index(inplace=True, drop=True)

    features, features_names, complexities, transformations = construct_features(df_, complexity, scoring, target)

    if prefiltering:

        bloomf = BloomFilter(len(features_names), 0.1)

        bloom_selected = []
        for i in range(features.shape[1]):
            if bloomf.check(features[:, [i]]):
                continue
            else:
                bloomf.add(features[:, [i]])
                bloom_selected.append(i)

        def nunique(a, axis):
            return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1

        unique_values = nunique(features[:, bloom_selected], axis=0)
        discrete_indices = np.argwhere(unique_values <= 2)

        mi = mutual_info_classif(features[:, bloom_selected], y_mi_sample, discrete_features=discrete_indices)
        mi_filter = [f[0] for f in np.argwhere(mi > 0)]

        selected_indices = [i for idt, i in enumerate(bloom_selected) if idt in mi_filter]

    else:
        selected_indices = [i for i in range(len(transformations))]

    if prefiltering:
        print('Total constructed features: ' + str(len(features_names)))
        print('Total constructed features after bloom filter: ' + str(len(bloom_selected)))
        print('Total constructed features after mi filter: ' + str(len(selected_indices)))
    else:
        pass

    X = df.loc[:, [f for f in list(df) if f != target]]

    features2_scale = []
    for idx, i in enumerate(list(X)):
        if df.loc[:, i].dtype in (int, float):
            features2_scale.append(idx)
        else:
            pass

    numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, features2_scale)], remainder='passthrough')

    X = preprocessor.fit_transform(X)

    transformed_set = np.empty((1, len(selected_indices)))

    def chunks(array, n):
        """Yield successive n-sized chunks from array."""
        for i in range(0, array.shape[0], n):
            yield array[i:i + n]

    array_chunks = chunks(X, 15000)

    for chunk in array_chunks:

        pool = mp.Pool(mp.cpu_count())
        func = partial(transform, chunk)
        results_back = pool.map(func, [transformations[t] for t in selected_indices])
        pool.close()

        transformation_list = list(itertools.chain(*[results_back]))

        transformed_set_c = np.empty((chunk.shape[0], 1))

        for f in transformation_list:
            transformed_set_c = np.hstack((transformed_set_c, f[1]))

        transformed_set_c = transformed_set_c[:, 1:]

        transformed_set = np.vstack((transformed_set, transformed_set_c))

    transformed_set = transformed_set[1:]

    ### Feature Ordering

    final_names = [features_names[i] for i in selected_indices]
    complexities_idx = np.argsort(np.asarray(complexities)[selected_indices])
    features = transformed_set[:, complexities_idx]
    sorted_names = [final_names[i] for i in complexities_idx]


    drop_indices = np.nonzero(~np.logical_or(np.isnan(features).any(axis=1), np.isinf(features).any(axis=1)))[0]
    retrain_indices = np.intersect1d(train_indices, drop_indices)
    #features = features[drop_indices]
    #y = np.ravel(df.iloc[drop_indices, list(df).index(target)].to_numpy())

    return features, sorted_names, retrain_indices, drop_indices


def eliminate(X, y, df_train, current_names, sensitive_feature,sensitive_features, target, protected,  clf, candidate):
    admissible_features = [f for f in list(df_train) if f not in sensitive_features and f != target]
    d = current_names.copy()
    del d[candidate]
    floating_r = np.delete(X, candidate, 1)

    acc_scores_cr, fair_scores_cr, JCIT = run_evaluation(floating_r, y, df_train,
                                                                        sensitive_feature, sensitive_features,
                                                                        protected, d.copy(),
                                                                        admissible_features, clf)

    return [d, fair_scores_cr, acc_scores_cr, JCIT, candidate]


def add(X, y, df_train, current_names, deleted_idx, sensitive_feature, sensitive_features, target, protected, clf, candidate):
    admissible_features = [f for f in list(df_train) if f not in sensitive_features and f != target]
    deleted_idx.remove(candidate)
    current_names_b = [current_names[f] for f in range(len(current_names)) if f not in deleted_idx]
    floating_b = np.delete(X, deleted_idx, 1)

    acc_scores_cr, fair_scores_cr, JCIT = run_evaluation(floating_b, y, df_train,
                                                                          sensitive_feature, sensitive_features,
                                                                          protected, current_names_b.copy(),
                                                                          admissible_features, clf)

    return [current_names_b, fair_scores_cr, acc_scores_cr, JCIT, candidate]


def transform(X, candidate):
    x_t = candidate.transform(X)
    name = candidate.get_name().strip()

    return [name, x_t]


def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]



def run_evaluation_parallel(i):
    current_representation_train = parallel_variables.current_representation_train
    names = parallel_variables.names
    current_names = parallel_variables.current_names
    train = parallel_variables.train
    y_train = parallel_variables.y_train
    df_train = parallel_variables.df_train
    sensitive_feature = parallel_variables.sensitive_feature
    sensitive_features = parallel_variables.sensitive_features
    protected = parallel_variables.protected
    admissible = parallel_variables.admissible
    clf = parallel_variables.clf

    if type(current_representation_train) == type(None):
        current_representation_train = train[:, [i]]
    else:
        current_representation_train = np.hstack((current_representation_train, train[:, [i]]))

    current_names.append(names[i])

    temp_acc_score, temp_fair_score, JCIT = run_evaluation(current_representation_train,
                                                           y_train, df_train,
                                                           sensitive_feature, sensitive_features, protected,
                                                           current_names.copy(), admissible, clf)

    return {'temp_acc_score': temp_acc_score, 'temp_fair_score': temp_fair_score, 'JCIT': JCIT, 'i': i}


def run_evaluation(x_train, y_train, df_train, sensitive_feature, sensitive_features, protected, current_names, admissible, clf):
    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
    dict = {}
    for x in clf.get_params():
        dict[x] = list([clf.get_params()[x]])
    cv_scores = GridSearchCV(clf,
                             param_grid=dict,
                             n_jobs=5,
                             scoring=f1,
                             cv=KFold(n_splits=5, random_state=66, shuffle=True))
    cv_scores.fit(x_train, y_train)

    cv_folds = KFold(n_splits=5, random_state=66, shuffle=True).split(x_train)
    train_indices, test_indices = list(cv_folds)[cv_scores.best_index_]
    best_f1_train = cv_scores.cv_results_['mean_test_score'][cv_scores.best_index_]
    y_pred_proba_train = cv_scores.predict_proba(x_train[test_indices])[:, 1]

    if np.unique(y_pred_proba_train).shape[0] > 1:

        join_features = [f for f in sensitive_features if f not in current_names.copy()]
        df_train_ = df_train.reset_index(drop=True)
        df_train_ = df_train_.iloc[test_indices]
        df_train_ = df_train_.reset_index(drop=True)
        sensitive_df = df_train_.loc[:, join_features]
        outcomes_df = pd.DataFrame(y_pred_proba_train, columns=['outcome'])
        features_df = df_train_.loc[:, admissible]

        candidate_df = pd.concat([sensitive_df, features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

    else:
        JCIT = False
        mb = []


    rod_train = ROD.ROD(y_pred=y_pred_proba_train, df=df_train, sensitive=sensitive_feature,
                        admissible=admissible, protected=protected, test_idx=test_indices, mb=mb)

    return best_f1_train, rod_train, JCIT


def repair_algorithm(train, names, df_train, y_train, sensitive_feature, sensitive_features, protected, admissible_features,
                     target, clf, sampling):

    if sampling < 1.0:
        sample_idx_all = np.random.randint(df_train.shape[0], size=round(df_train.shape[0] * sampling))
        df_train.reset_index(inplace=True, drop=True)
        df_train = df_train.iloc[sample_idx_all]
        train = train[sample_idx_all]
        y_train = np.ravel(df_train.loc[:, target].to_numpy())
    else:
        pass

    registered_representations_train = []
    current_representation_train = None
    current_names = []

    global_acc_score = 0
    global_fair_score = 0
    explored_representations = []
    start_time = time.time()

    ## First Phase
    i = 0
    number_of_paralllelism = 4
    while i < train.shape[1]:

        parallel_variables.current_representation_train = current_representation_train
        parallel_variables.names = names
        parallel_variables.current_names = current_names
        parallel_variables.train = train
        parallel_variables.y_train = y_train
        parallel_variables.df_train = df_train
        parallel_variables.sensitive_feature = sensitive_feature
        parallel_variables.sensitive_features = sensitive_features
        parallel_variables.protected = protected
        parallel_variables.admissible = admissible_features
        parallel_variables.clf = clf

        to_be_investigated_optimistically = []
        for ii in range(number_of_paralllelism):
            if i + ii < train.shape[1]:
                print(i+ii)
                to_be_investigated_optimistically.append(i + ii)

        pool = mp.Pool(min([number_of_paralllelism, len(to_be_investigated_optimistically)]))
        results_back = pool.map(run_evaluation_parallel, to_be_investigated_optimistically)
        pool.close()

        print(results_back)
        print(len(names))
        max_result_index = -1
        for ii in range(len(results_back)):
            if max_result_index == -1 and results_back[ii]['temp_acc_score'] > global_acc_score:
                max_result_index = ii

            current_names_new = current_names.copy()
            current_names_new.append(names[results_back[ii]['i']])
            registered_representations_train.append([current_names_new.copy(), len(current_names_new.copy()), results_back[ii]['temp_acc_score'], results_back[ii]['temp_fair_score'], results_back[ii]['JCIT'], 'Phase 1'])
            explored_representations.append(current_names_new.copy())

        if max_result_index == -1: #no result was better than the feature set that we already have, so we can skip all 
            i = results_back[-1]['i'] + 1
            continue

        if max_result_index >= 0 and results_back[max_result_index]['temp_acc_score'] > global_acc_score:
            global_acc_score = results_back[max_result_index]['temp_acc_score']
            global_fair_score = results_back[max_result_index]['temp_fair_score']

            if type(current_representation_train) == type(None):
                current_representation_train = train[:, [results_back[max_result_index]['i']]]
            else:
                current_representation_train = np.hstack((current_representation_train, train[:, [results_back[max_result_index]['i']]]))
            current_names.append(names[results_back[max_result_index]['i']])

            i = results_back[max_result_index]['i'] + 1


            candidates_f = []
            if current_representation_train.shape[1] > 1:

                for idz, z in enumerate(current_names):
                    c = current_names.copy()
                    c.remove(z)
                    if c not in explored_representations:
                        candidates_f.append(idz)
                    else:
                        continue

                if len(candidates_f) > 0:


                    pool = mp.Pool(mp.cpu_count())
                    func = partial(eliminate, current_representation_train, y_train, df_train,
                                          current_names.copy(),
                                          sensitive_feature, sensitive_features, target, protected, clf)
                    results_back = pool.map(func, candidates_f)
                    pool.close()



                    floating_evaluation = list(itertools.chain(*[results_back]))

                    for f in floating_evaluation:
                        registered_representations_train.append(
                            [f[0], len(f[0]), f[2], f[1], f[3], 'Phase 1 - floating'])
                        explored_representations.append(f[0])

                    acc_floating_scores = [y[2] for y in floating_evaluation]
                    max_index = acc_floating_scores.index(max(acc_floating_scores))
                    max_acc_floating = max(acc_floating_scores)

                    if max_acc_floating >= global_acc_score:
                        global_acc_score = max_acc_floating
                        global_fair_score = floating_evaluation[max_index][1]
                        current_representation_train = np.delete(current_representation_train,
                                                                 floating_evaluation[max_index][4], 1)
                        current_names = floating_evaluation[max_index][0]

                    else:
                        pass
                else:
                    pass
            else:
                pass
        else:
            i += 1

    first_phase_names = current_names.copy()
    selected_indices_first_phase = [names.index(f) for f in first_phase_names]

    ## Second Phase

    # deleted_idx = []
    # candidates = []
    # for idz, z in enumerate(current_names):
    #     c = current_names.copy()
    #     c.remove(z)
    #     if c not in explored_representations:
    #         candidates.append(idz)
    #     else:
    #         continue
    #
    # complexity_order = []
    # for s in candidates:
    #     ids = names.index(current_names[s])
    #     complexity_order.append((s, ids))
    #
    # complexity_order.sort(key=lambda x: x[1])
    #
    # sorted_candidates = [t[0] for t in complexity_order]
    #
    # current_names_b = current_names.copy()
    #
    # for f in sorted_candidates:
    #     deleted_idx.append(f)
    #
    #     if current_representation_train.shape[1] > len(deleted_idx):
    #         current_representation_train_b = np.delete(current_representation_train, deleted_idx, 1)
    #         current_names_b.remove(current_names[f])
    #         explored_representations.append(current_names_b.copy())
    #
    #         temp_acc_score, temp_fair_score, JCIT = run_evaluation(current_representation_train_b,
    #                                                                y_train,
    #                                                                df_train,
    #                                                                sensitive_feature,
    #                                                                sensitive_features, protected,
    #                                                                current_names_b.copy(),
    #                                                                admissible_features, clf)
    #
    #         if temp_fair_score >= global_fair_score:
    #             global_fair_score = temp_fair_score
    #             global_acc_score = temp_acc_score
    #
    #             registered_representations_train.append(
    #                 [current_names_b.copy(), len(current_names_b.copy()), temp_acc_score, temp_fair_score, JCIT,
    #                  'Phase 2'])
    #
    #             candidates_b = []
    #             if len(deleted_idx) > 0:
    #                 for idz, z in enumerate([current_names[f] for f in deleted_idx]):
    #                     c = current_names_b.copy()
    #                     c.append(z)
    #                     if c not in explored_representations:
    #                         candidates_b.append(deleted_idx[idz])
    #                     else:
    #                         continue
    #
    #                 if len(candidates_b) > 0:
    #
    #                     pool = mp.Pool(mp.cpu_count())
    #                     func = partial(add, current_representation_train, y_train,
    #                                    df_train, current_names.copy(), deleted_idx, sensitive_feature,
    #                                    sensitive_features, target, protected, clf)
    #                     results_back = pool.map(func, candidates_b)
    #                     pool.close()
    #
    #                     floating_evaluation = list(itertools.chain(*[results_back]))
    #
    #                     for f in floating_evaluation:
    #                         registered_representations_train.append(
    #                             [f[0], len(f[0]), f[2], f[1], f[3], 'Phase 2 - floating'])
    #                         explored_representations.extend(f[0])
    #
    #                     fair_floating_scores = [y[1] for y in floating_evaluation]
    #                     max_index = fair_floating_scores.index(max(fair_floating_scores))
    #                     max_fair_floating = max(fair_floating_scores)
    #
    #                     if max_fair_floating > global_fair_score:
    #                         global_fair_score = max_fair_floating
    #                         global_acc_score = floating_evaluation[max_index][2]
    #                         deleted_idx.remove(floating_evaluation[max_index][4])
    #                         current_names_b = floating_evaluation[max_index][0]
    #                         current_names_b.sort()
    #
    #                     else:
    #                         pass
    #
    #                 else:
    #                     pass
    #     else:
    #         pass
    #
    # scores = np.asarray([[f[2], f[3]] for f in registered_representations_train])
    # scores[:, 0] = np.ravel(MinMaxScaler().fit_transform(scores[:, 0].reshape((scores.shape[0], 1))))
    #
    # pareto = identify_pareto(scores)
    # pareto_front = scores[pareto]
    #
    # ideal_point = np.asarray([1, 0])
    # dist = np.empty((pareto_front.shape[0], 1))
    #
    # for idx, i in enumerate(pareto_front):
    #     dist[idx] = norm(i - ideal_point)
    #
    # min_dist = np.argmin(dist)
    #
    # selected_representation = registered_representations_train[pareto[min_dist]][0]
    # selected_indices = [names.index(f) for f in selected_representation]

    return selected_indices_first_phase


def repair_algorithm_original(train, names, df_train, y_train, sensitive_feature, sensitive_features, protected, admissible_features,
                     target, clf, sampling, results_path, fold):

    if sampling < 1.0:
        sample_idx_all = np.random.randint(df_train.shape[0], size=round(df_train.shape[0] * sampling))
        df_train.reset_index(inplace=True, drop=True)
        df_train = df_train.iloc[sample_idx_all]
        train = train[sample_idx_all]
        y_train = np.ravel(df_train.loc[:, target].to_numpy())
    else:
        pass

    registered_representations_train = []
    current_representation_train = np.empty((train.shape[0], 1))
    current_names = []

    global_acc_score = 0
    global_fair_score = 0
    explored_representations = []
    start_time = time.time()

    ## First Phase
    for idx, i in enumerate(range(train.shape[1])):

        current_representation_train = np.hstack((current_representation_train, train[:, [i]]))

        if idx == 0:
            current_representation_train = current_representation_train[:, 1:]
        else:
            pass

        current_names.append(names[i])
        sort_idx = sorted(range(len(current_names)), key=lambda k: current_names[k])
        current_representation_train = current_representation_train[:, sort_idx]
        current_names.sort()
        explored_representations.append(current_names.copy())

        temp_acc_score, temp_fair_score, JCIT = run_evaluation(current_representation_train,
                                                               y_train, df_train,
                                                               sensitive_feature, sensitive_features, protected,
                                                               current_names.copy(), admissible_features, clf)
        registered_representations_train.append(
            [current_names.copy(), len(current_names.copy()), temp_acc_score, temp_fair_score, JCIT, 'Phase 1'])

        if temp_acc_score > global_acc_score:
            global_acc_score = temp_acc_score
            global_fair_score = temp_fair_score

            candidates_f = []
            if current_representation_train.shape[1] > 1:

                for idz, z in enumerate(current_names):
                    c = current_names.copy()
                    c.remove(z)
                    if c not in explored_representations:
                        candidates_f.append(idz)
                    else:
                        continue

                if len(candidates_f) > 0:

                    # pool = mp.Pool(mp.cpu_count())
                    # func = partial(eliminate, current_representation_train, y_train, df_train, current_names.copy(),
                    #                sensitive_feature, sensitive_features, target, protected, clf)
                    # results_back = pool.map(func, candidates_f)
                    # pool.close()

                    results_back = []
                    for candidates_fi in candidates_f:
                        results_back.append(
                            eliminate(current_representation_train, y_train, df_train, current_names.copy(),
                                      sensitive_feature, sensitive_features, target, protected, clf, candidates_fi))

                    floating_evaluation = list(itertools.chain(*[results_back]))

                    for f in floating_evaluation:
                        registered_representations_train.append(
                            [f[0], len(f[0]), f[2], f[1], f[3], 'Phase 1 - floating'])
                        explored_representations.append(f[0])

                    acc_floating_scores = [y[2] for y in floating_evaluation]
                    max_index = acc_floating_scores.index(max(acc_floating_scores))
                    max_acc_floating = max(acc_floating_scores)

                    if max_acc_floating >= global_acc_score:
                        global_acc_score = max_acc_floating
                        global_fair_score = floating_evaluation[max_index][1]
                        current_representation_train = np.delete(current_representation_train,
                                                                 floating_evaluation[max_index][4], 1)
                        current_names = floating_evaluation[max_index][0]

                    else:
                        pass
                else:
                    pass
            else:
                pass
        else:
            remove_idx = current_names.index(names[i])
            current_names.remove(names[i])
            current_representation_train = np.delete(current_representation_train, remove_idx, 1)

    # Second Phase

    deleted_idx = []
    candidates = []
    for idz, z in enumerate(current_names):
        c = current_names.copy()
        c.remove(z)
        if c not in explored_representations:
            candidates.append(idz)
        else:
            continue

    complexity_order = []
    for s in candidates:
        ids = names.index(current_names[s])
        complexity_order.append((s, ids))

    complexity_order.sort(key=lambda x: x[1])

    sorted_candidates = [t[0] for t in complexity_order]

    current_names_b = current_names.copy()

    for f in sorted_candidates:
        deleted_idx.append(f)

        if current_representation_train.shape[1] > len(deleted_idx):
            current_representation_train_b = np.delete(current_representation_train, deleted_idx, 1)
            current_names_b.remove(current_names[f])
            explored_representations.append(current_names_b.copy())

            temp_acc_score, temp_fair_score, JCIT = run_evaluation(current_representation_train_b,
                                                                   y_train,
                                                                   df_train,
                                                                   sensitive_feature,
                                                                   sensitive_features, protected,
                                                                   current_names_b.copy(),
                                                                   admissible_features, clf)

            if temp_fair_score >= global_fair_score:
                global_fair_score = temp_fair_score
                global_acc_score = temp_acc_score

                registered_representations_train.append(
                    [current_names_b.copy(), len(current_names_b.copy()), temp_acc_score, temp_fair_score, JCIT,
                     'Phase 2'])

                candidates_b = []
                if len(deleted_idx) > 0:
                    for idz, z in enumerate([current_names[f] for f in deleted_idx]):
                        c = current_names_b.copy()
                        c.append(z)
                        if c not in explored_representations:
                            candidates_b.append(deleted_idx[idz])
                        else:
                            continue

                    if len(candidates_b) > 0:

                        # pool = mp.Pool(mp.cpu_count())
                        # func = partial(add, current_representation_train, y_train,
                        #                df_train, current_names.copy(), deleted_idx, sensitive_feature,
                        #                sensitive_features, target, protected, clf)
                        # results_back = pool.map(func, candidates_b)
                        # pool.close()

                        results_back = []
                        for candidates_fi in candidates_b:
                            results_back.append(
                                add(current_representation_train, y_train,
                                       df_train, current_names.copy(), deleted_idx.copy(), sensitive_feature,
                                       sensitive_features, target, protected, clf, candidates_fi))

                        floating_evaluation = list(itertools.chain(*[results_back]))

                        for f in floating_evaluation:
                            registered_representations_train.append(
                                [f[0], len(f[0]), f[2], f[1], f[3], 'Phase 2 - floating'])
                            explored_representations.append(f[0])

                        fair_floating_scores = [y[1] for y in floating_evaluation]
                        max_index = fair_floating_scores.index(max(fair_floating_scores))
                        max_fair_floating = max(fair_floating_scores)

                        if max_fair_floating > global_fair_score:
                            global_fair_score = max_fair_floating
                            global_acc_score = floating_evaluation[max_index][2]
                            deleted_idx.remove(floating_evaluation[max_index][4])
                            current_names_b = floating_evaluation[max_index][0]
                            current_names_b.sort()

                        else:
                            pass

                    else:
                        pass
        else:
            pass

    scores = np.asarray([[f[2], f[3]] for f in registered_representations_train])
    #scores[:, 0] = np.ravel(MinMaxScaler().fit_transform(scores[:, 0].reshape((scores.shape[0], 1))))
    scores = MinMaxScaler().fit_transform(scores)

    pareto = identify_pareto(scores)
    pareto_front = scores[pareto]

    ideal_point = np.asarray([1, 1])
    dist = np.empty((pareto_front.shape[0], 1))

    for idx, i in enumerate(pareto_front):
        dist[idx] = norm(i - ideal_point)

    min_dist = np.argmin(dist)

    selected_representation = registered_representations_train[pareto[min_dist]][0]
    selected_indices = [names.index(f) for f in selected_representation]

    train_representation_summary = pd.DataFrame(registered_representations_train,
                                                columns=['Representation', 'Size', 'F1', 'ROD', 'CIT', 'Phase'])

    train_representation_summary.to_csv(
        path_or_buf=results_path + '/registered_representations_' + protected + '_' + str(sampling) + '_' + str(fold) + '.csv',
        index=False)

    return selected_indices


def evaluate(df, complexity, clf, acc_score, sensitive_feature, inadmissible_features, protected, target, sampling, output_path):
    sensitive_features = inadmissible_features + [sensitive_feature]

    admissible_features = [f for f in list(df) if f not in sensitive_features and f != target]
    kf1 = KFold(n_splits=5, random_state=42, shuffle=True)

    start_construction_time = time.time()
    X, names, y = extend_dataframe_complete(df, complexity, acc_score, target, sampling)
    end_construction_time = time.time() - start_construction_time

    selected_representations = []
    all_representations_train = []
    fold = 1
    for train_index, test_index in kf1.split(X):

        registered_representations_train = []
        train = X[train_index]
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        test = X[test_index]

        if sampling < 1.0:
            sample_idx_all = np.random.randint(df_train.shape[0], size=round(df_train.shape[0] * sampling))
            df_train.reset_index(inplace=True, drop=True)
            df_train = df_train.iloc[sample_idx_all]
            train = train[sample_idx_all]
            y_train = np.ravel(df_train.loc[:, target].to_numpy())
        else:
            pass

        current_representation_train = np.empty((train.shape[0], 1))
        current_names = []

        global_acc_score = 0
        global_fair_score = 0
        explored_representations = []
        start_time = time.time()

        ## First Phase
        for idx, i in enumerate(range(train.shape[1])):

            current_representation_train = np.hstack((current_representation_train, train[:, [i]]))

            if idx == 0:
                current_representation_train = current_representation_train[:, 1:]
            else:
                pass

            current_names.append(names[i])
            sort_idx = sorted(range(len(current_names)), key=lambda k: current_names[k])
            current_representation_train = current_representation_train[:, sort_idx]
            current_names.sort()
            explored_representations.append(current_names.copy())

            temp_acc_score, temp_fair_score, JCIT = run_evaluation(current_representation_train,
                                                                    y_train, df_train,
                                                                 sensitive_feature, sensitive_features, protected, current_names.copy(), admissible_features, clf)
            registered_representations_train.append(
                [current_names.copy(), len(current_names.copy()), temp_acc_score, temp_fair_score, fold, JCIT, 'Phase 1'])

            if temp_acc_score > global_acc_score:
                global_acc_score = temp_acc_score
                global_fair_score = temp_fair_score

                candidates_f = []
                if current_representation_train.shape[1] > 1:

                    for idz, z in enumerate(current_names):
                        c = current_names.copy()
                        c.remove(z)
                        if c not in explored_representations:
                            candidates_f.append(idz)
                        else:
                            continue

                    if len(candidates_f) > 0:

                        pool = mp.Pool(mp.cpu_count())
                        func = partial(eliminate, current_representation_train, y_train, df_train, current_names.copy(),
                                       sensitive_features, target, protected, clf)
                        results_back = pool.map(func, candidates_f)
                        pool.close()

                        floating_evaluation = list(itertools.chain(*[results_back]))

                        for f in floating_evaluation:
                            registered_representations_train.append(
                                [f[0], len(f[0]), f[2], f[1], fold, f[3], 'Phase 1 - floating'])
                            explored_representations.extend(f[0])

                        acc_floating_scores = [y[2] for y in floating_evaluation]
                        max_index = acc_floating_scores.index(max(acc_floating_scores))
                        max_acc_floating = max(acc_floating_scores)

                        if max_acc_floating >= global_acc_score:
                            global_acc_score = max_acc_floating
                            global_fair_score = floating_evaluation[max_index][1]
                            current_representation_train = np.delete(current_representation_train, floating_evaluation[max_index][4], 1)
                            current_names = floating_evaluation[max_index][0]

                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                remove_idx = current_names.index(names[i])
                current_names.remove(names[i])
                current_representation_train = np.delete(current_representation_train, remove_idx, 1)

        ## Second Phase

        deleted_idx = []
        candidates = []
        for idz, z in enumerate(current_names):
            c = current_names.copy()
            c.remove(z)
            if c not in explored_representations:
                candidates.append(idz)
            else:
                continue

        complexity_order = []
        for s in candidates:
            ids = names.index(current_names[s])
            complexity_order.append((s, ids))

        complexity_order.sort(key=lambda x: x[1])

        sorted_candidates = [t[0] for t in complexity_order]

        current_names_b = current_names.copy()

        for f in sorted_candidates:
            deleted_idx.append(f)

            if current_representation_train.shape[1] > len(deleted_idx):
                current_representation_train_b = np.delete(current_representation_train, deleted_idx, 1)
                current_names_b.remove(current_names[f])
                explored_representations.append(current_names_b.copy())

                temp_acc_score, temp_fair_score, JCIT = run_evaluation(current_representation_train_b,
                                                                                      y_train,
                                                                                      df_train,
                                                                                      sensitive_feature,
                                                                                      sensitive_features, protected,
                                                                                      current_names_b.copy(),
                                                                                      admissible_features, clf)

                if temp_fair_score >= global_fair_score:
                    global_fair_score = temp_fair_score
                    global_acc_score = temp_acc_score

                    registered_representations_train.append(
                        [current_names_b.copy(), len(current_names_b.copy()), temp_acc_score, temp_fair_score, fold, JCIT,
                         'Phase 2'])

                    candidates_b = []
                    if len(deleted_idx) > 0:
                        for idz, z in enumerate([current_names[f] for f in deleted_idx]):
                            c = current_names_b.copy()
                            c.append(z)
                            if c not in explored_representations:
                                candidates_b.append(deleted_idx[idz])
                            else:
                                continue

                        if len(candidates_b) > 0:

                            pool = mp.Pool(mp.cpu_count())
                            func = partial(add, current_representation_train, y_train,
                                           df_train, current_names.copy(), deleted_idx, sensitive_feature,
                                           sensitive_features, target, protected, clf)
                            results_back = pool.map(func, candidates_b)
                            pool.close()

                            floating_evaluation = list(itertools.chain(*[results_back]))

                            for f in floating_evaluation:

                                registered_representations_train.append([f[0], len(f[0]), f[2], f[1], fold, f[3], 'Phase 2 - floating'])
                                explored_representations.extend(f[0])

                            fair_floating_scores = [y[1] for y in floating_evaluation]
                            max_index = fair_floating_scores.index(max(fair_floating_scores))
                            max_fair_floating = max(fair_floating_scores)

                            if max_fair_floating > global_fair_score:
                                global_fair_score = max_fair_floating
                                global_acc_score = floating_evaluation[max_index][2]
                                deleted_idx.remove(floating_evaluation[max_index][4])
                                current_names_b = floating_evaluation[max_index][0]
                                current_names_b.sort()

                            else:
                                pass

                        else:
                            pass
            else:
                pass

        scores = np.asarray([[f[2], f[3]] for f in registered_representations_train])
        scores[:, 0] = np.ravel(MinMaxScaler().fit_transform(scores[:, 0].reshape((scores.shape[0], 1))))

        pareto = identify_pareto(scores)
        pareto_front = scores[pareto]

        ideal_point = np.asarray([1, 0])
        dist = np.empty((pareto_front.shape[0], 1))

        for idx, i in enumerate(pareto_front):
            dist[idx] = norm(i - ideal_point)

        min_dist = np.argmin(dist)

        selected_representation = registered_representations_train[pareto[min_dist]][0]
        selected_indices = [names.index(f) for f in selected_representation]


        selected_trained = clf.fit(X[np.ix_(train_index, selected_indices)], y[train_index])

        y_pred = selected_trained.predict(test[:, selected_indices])
        y_pred_proba = selected_trained.predict_proba(test[:, selected_indices])[:, 1]

        f1_test = f1_score(y_test, y_pred)

        sensitive_df = df_test.loc[:, sensitive_features]
        sensitive_df.reset_index(inplace=True, drop=True)
        outcomes_df = pd.DataFrame(y_pred_proba, columns=['outcome'])
        features_df = df_test.loc[:, admissible_features]
        features_df.reset_index(inplace=True, drop=True)

        candidate_df = pd.concat([sensitive_df, features_df, outcomes_df], axis=1)

        JCIT, mb = causal_filter(candidate_df, sensitive_features)

        ### Fairness measures

        rod_test = ROD.ROD(y_pred=y_pred_proba, df=df_test, sensitive=sensitive_feature,
                           admissible=admissible_features, protected=protected, mb=mb)

        #binned_test_df = generate_binned_df(df_test)

        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=df_test.loc[:, sensitive_feature])
        tpr = MetricFrame(true_positive_rate, y_test, y_pred,
                         sensitive_features=df_test.loc[:, sensitive_feature])
        tpb = tpr.difference()
        tnr = MetricFrame(true_negative_rate, y_test, y_pred,
                          sensitive_features=df_test.loc[:, sensitive_feature])
        tnb = tnr.difference()

        cdp = CDP(y_test, y_pred, df_test, sensitive_feature, admissible_features)


        end_time = time.time() - start_time + end_construction_time

        print('ROD train: ' + str(registered_representations_train[pareto[min_dist]][3]))
        print('F1 train: ' + str(registered_representations_train[pareto[min_dist]][2]))
        print('ROD test: ' + str(rod_test))
        print('F1 test: ' + str(f1_test))
        print('DP test: ' + str(dp))
        print('TPB test: ' + str(tpb))
        print('TNB test: ' + str(tnb))
        print('CDP test: ' + str(cdp))

        selected_representations.append(
            ['FairExp', clf.__class__.__name__, rod_test, dp, tpb, tnb, f1_test, selected_representation,
             len(selected_representation), fold, JCIT, end_time])

        all_representations_train.extend(registered_representations_train.copy())

        fold += 1

    summary_df = pd.DataFrame(selected_representations,
                              columns=['Method', 'Classifier', 'ROD', 'DP', 'TPB', 'TNB', 'F1', 'Representation', 'Size', 'Fold',
                                       'CIT', 'Runtime'])

    train_representation_summary = pd.DataFrame(all_representations_train, columns=['Representation', 'Size', 'F1', 'ROD', 'Fold', 'CIT', 'Phase'])

    train_representation_summary.to_csv(path_or_buf=output_path + '/complete_train_' + protected + '_results_complexity_' + str(complexity) + '_' + str(sampling) + '.csv',
            index=False)
    summary_df.to_csv(
        path_or_buf=output_path + '/selected_' + protected + '_complexity_' + str(complexity) + '_' + str(sampling) + '.csv',
        index=False)

