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
from causality.d_separation import d_separation
import multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy.linalg import norm
import time
from capuchin import repair_dataset

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'

COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
                    (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
                    & (COMPAS['is_recid'] != -1)
                    & (COMPAS['race'].isin(['African-American','Caucasian']))
                    & (COMPAS['c_charge_degree'].isin(['F','M']))
                    , ['race', 'age', 'age_cat', 'priors_count', 'is_recid', 'c_charge_degree']]

sensitive_feature = 'race'
inadmissible_features = []
target = 'is_recid'

def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in [target, 'outcome'] and (df_[i].dtype != object and len(df_[i].unique()) > 4):
            out = pd.cut(df_[i], bins=2)
            df_.loc[:, i] = out.astype(str)

    return df_

admissible_features = [i for i in list(COMPAS) if
                       i not in inadmissible_features and i != sensitive_feature and i != target and i != 'age']

all_features = list(COMPAS)
all_features.remove(target)
all_features_2_build = all_features.copy()
all_features_2_build.remove('age_cat')
all_2_combinations = list(itertools.combinations(all_features_2_build, 2))

complexity = 4
CF = False
count = 0
method_list = []
kf1 = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf1.split(COMPAS):

    runtimes = [len(all_2_combinations), complexity]
    count_transformations = 0
    filtered_transformations = 0
    time_2_create_transformations = 0
    time_2_CF = 0
    time_2_FR = 0
    time_2_SR = 0

    train_df = COMPAS.iloc[train_index]
    test_df = COMPAS.iloc[test_index]

    X_train = train_df.loc[:, all_features]

    y_train = train_df.loc[:, target]

    X_test = test_df.loc[:, all_features]

    y_test = test_df.loc[:, target]

    rod_score = make_scorer(ROD.ROD, greater_is_better=True, needs_proba=True,
                            sensitive=X_train.loc[:, sensitive_feature],
                            admissible=X_train.loc[:, admissible_features],
                            protected='African-American', name='train_COMPAS')

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

        features2_build = []
        features2_build.extend(i)

        features2_build_cat = []
        features2_build_num = []

        for x in features2_build:
            if COMPAS[x].dtype != np.dtype('O'):
                features2_build_num.extend([x])
            else:
                features2_build_cat.extend([x])

        features2_scale = []
        for x in features2_build:
            if COMPAS[x].dtype != np.dtype('O'):
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
                                             max_iter=100000, multi_class='auto', n_jobs=-1)

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

            pool = mp.Pool(10)
            results = pool.map(causal_filter, transformations2_generate)
            pool.close()

            end_time_CF = time.time()-start_time_CF

            accepted_list = list(itertools.chain(*[results]))

            accepted_idx = np.argwhere(np.array(accepted_list))

            time_2_CF += end_time_CF

            mask = [x for idx, x in enumerate(transformations2_generate_idx) if accepted_list[idx]]
            filtered_transformations += len(mask)

            print('Filtered Transformations: ' + str(len(transformations2_generate_idx)-len(mask)))
        else:
            time_2_CF += 0
            mask = [x for x in transformations2_generate_idx]

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
            accepted_features_photo = accepted_features.copy()
            accepted_features_photo.sort()

            if idx == 0 and idj == 0:
                transformed_train = transformed_train[:, 1:]
                transformed_test = transformed_test[:, 1:]
            else:
                pass

            cv_scores = GridSearchCV(LogisticRegression(), param_grid={
                'penalty': ['l2'], 'C': [0.5, 1, 1.5], 'solver': ['lbfgs', 'newton-cg', 'sag'], 'class_weight': [None, 'balanced'],
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
            rod_ff = ROD.ROD(y_pred=predicted_ff_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                             admissible=X_test.loc[:, admissible_features],
                             protected='African-American', name='forward_COMPAS')
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
                            rod_ff_r = ROD.ROD(y_pred=predicted_ff_r_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                               admissible=X_test.loc[:, admissible_features],
                                               protected='African-American', name='backward_COMPAS')
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

        end_time_FR = time.time() - start_time_FR

        time_2_FR += end_time_FR

    start_time_SR = time.time()

    print(transformed_train.shape, transformed_test.shape, str(join_score))

    cv_scores = GridSearchCV(LogisticRegression(), param_grid={
        'penalty': ['l2'], 'C': [0.5, 1, 1.5], 'solver': ['lbfgs', 'newton-cg', 'sag'], 'class_weight': [None, 'balanced'],
                'max_iter': [100000], 'multi_class': ['auto']
    },
                             n_jobs=-1,
                             scoring={'F1': f1, 'ROD': rod_score}, refit='ROD', cv=3)

    cv_scores.fit(transformed_train, np.ravel(y_train.to_numpy()))
    test_scores_c = cv_scores.cv_results_['mean_test_F1'][0]
    rod_scores_c = cv_scores.cv_results_['mean_test_ROD'][0]

    rod_complete = rod_scores_c
    f1_complete = test_scores_c

    print('__________________________________________')
    print('Round 2 : Improving in the other direction. Start with backward floating elimination: ')

    print('F1 complete: {:.4f}'.format(f1_complete))
    print('ROD complete {:.4f}'.format(rod_complete))

    # Now SBFS

    selected_ids_r = []
    for d in range(transformed_train.shape[1]):
        if transformed_train.shape[1] > len(selected_ids_r) + 1:
            selected_ids_r.extend([d])
            transformed_train_cr = np.delete(transformed_train, selected_ids_r, 1)
            transformed_test_cr = np.delete(transformed_test, selected_ids_r, 1)
            accepted_features_cr = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids_r]
            accepted_features_cr.sort()

            if accepted_features_cr not in unique_representations:

                cv_scores.fit(transformed_train_cr, np.ravel(y_train.to_numpy()))
                test_scores_cr = cv_scores.cv_results_['mean_test_F1'][0]
                rod_scores_cr = cv_scores.cv_results_['mean_test_ROD'][0]

                #complete_clf.fit(transformed_train_cr, np.ravel(y_train.to_numpy()))
                predicted_b = cv_scores.predict(transformed_test_cr)
                predicted_b_proba = cv_scores.predict_proba(transformed_test_cr)[:, 1]
                rod_b = ROD.ROD(y_pred=predicted_b_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                   admissible=X_test.loc[:, admissible_features],
                                   protected='African-American', name='backward_COMPAS')
                f1_b = f1_score(np.ravel(y_test.to_numpy()), predicted_b)

                registered_representations_test.append(
                    [accepted_features_cr.copy(), len(accepted_features_cr.copy()), f1_b, rod_b])
                registered_representations_train.append(
                    [accepted_features_cr.copy(), len(accepted_features_cr.copy()), test_scores_cr, rod_scores_cr])
                unique_representations.append(accepted_features_cr.copy())

                print(transformed_train_cr.shape, f1_b, rod_b, accepted_features_cr)

                if rod_scores_cr > rod_complete:
                    rod_complete = rod_scores_cr
                    f1_complete = test_scores_cr

                    for ida in selected_ids_r:
                        selected_ids_r.remove(ida)
                        transformed_train_a = np.delete(transformed_train, selected_ids_r, 1)
                        transformed_test_a = np.delete(transformed_test, selected_ids_r, 1)
                        accepted_features_a = [f for idf, f in enumerate(accepted_features) if
                                               idf not in selected_ids_r]
                        accepted_features_a.sort()
                        if accepted_features_a not in unique_representations:
                            cv_scores.fit(transformed_train_a, np.ravel(y_train.to_numpy()))
                            test_scores_a = cv_scores.cv_results_['mean_test_F1'][0]
                            rod_scores_a = cv_scores.cv_results_['mean_test_ROD'][0]

                            #complete_clf.fit(transformed_train_a, np.ravel(y_train.to_numpy()))
                            predicted_a = cv_scores.predict(transformed_test_a)
                            predicted_a_proba = cv_scores.predict_proba(transformed_test_a)[:, 1]
                            rod_a = ROD.ROD(y_pred=predicted_a_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                            admissible=X_test.loc[:, admissible_features],
                                            protected='African-American', name='backward_COMPAS')
                            f1_a = f1_score(np.ravel(y_test.to_numpy()), predicted_a)

                            registered_representations_test.append(
                                [accepted_features_a.copy(), len(accepted_features_a.copy()), f1_a, rod_a])
                            registered_representations_train.append(
                                [accepted_features_a.copy(), len(accepted_features_a.copy()), test_scores_a, rod_scores_a])
                            unique_representations.append(accepted_features_a.copy())

                            if rod_scores_a > rod_complete:
                                rod_complete = rod_scores_a
                                f1_complete = test_scores_a
                            else:
                                selected_ids_r.extend([ida])
                        else:
                            selected_ids_r.extend([ida])
                else:
                    selected_ids_r.remove(d)
            else:
                selected_ids_r.remove(d)
        else:
            pass

        print('representation size: ' + str(transformed_train.shape[1] - len(selected_ids_r)),
              'ROD: ' + str(rod_complete), 'F1' + str(f1_complete))

    if len(selected_ids_r) > 0:
        transformed_train = np.delete(transformed_train, selected_ids_r, 1)
        transformed_test = np.delete(transformed_test, selected_ids_r, 1)
        accepted_features = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids_r]
    else:
        pass

    end_time_SR = time.time() - start_time_SR

    time_2_SR += end_time_SR

    # Compute pareto front

    all_visited = np.asarray(registered_representations_train)
    all_visited_test = np.asarray(registered_representations_test)
    scores = all_visited[:, [2, 3]]

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

    normalized_ROD = (scores[:, 1] - scores[:, 1].min()) / (0 - scores[:, 1].min())
    scores[:, 1] = normalized_ROD

    normalized_F1 = (scores[:, 0] - scores[:, 0].min()) / (scores[:, 0].max() - scores[:, 0].min())
    scores[:, 0] = normalized_F1

    pareto = identify_pareto(scores)
    pareto_front = scores[pareto]

    ideal_point = np.asarray([1, 1])
    dist = np.empty((pareto_front.shape[0], 1))

    for idx, i in enumerate(pareto_front):
        dist[idx] = norm(i - ideal_point)

    min_dist = np.argmin(dist)
    selected_representation = all_visited_test[pareto[min_dist]]

    # ########## NSGAII ##############
    # all_allowed_train = all_allowed_train[:, 1:]
    # all_allowed_test = all_allowed_test[:, 1:]
    #
    # start_time_genetic = time.time()
    # selected_array = evolution(all_allowed_train, np.ravel(y_train.to_numpy()),
    #                            scorers=[f1, rod_score], cv_splitter=5,
    #                            max_search_time=60)
    #
    # end_time_genetic = time.time() - start_time_genetic
    #
    # for idg, g in enumerate(selected_array):
    #     if all_allowed_train[:, g].shape[1] > 0:
    #         complete_clf.fit(all_allowed_train[:, g], np.ravel(y_train.to_numpy()))
    #         predicted_genetic = complete_clf.predict(all_allowed_test[:, g])
    #         predicted_genetic_proba = complete_clf.predict_proba(all_allowed_test[:, g])[:, 1]
    #         rod_genetic = ROD.ROD(y_pred=predicted_genetic_proba, sensitive=X_test.loc[:, [sensitive_feature]],
    #                               admissible=X_test.loc[:, admissible_features],
    #                               protected='African-American', name='genetic_COMPAS')
    #         f1_genetic = f1_score(np.ravel(y_test.to_numpy()), predicted_genetic)
    #
    #         my_list = []
    #         x = np.argwhere(selected_array[idg])
    #         for idj, j in enumerate(x):
    #             my_list.extend([x.item(idj)])
    #         representation = [allowed_names[i] for i in my_list]
    #         method_list.append(['FC_genetic_' + str(idg), rod_genetic, f1_genetic, representation, count + 1])
    #
    #         print('ROD genetic ' + str(idg) + ': ' + str(rod_genetic))
    #         print('F1 genetic ' + str(idg) + ': ' + str(f1_genetic))
    #     else:
    #         pass

    ##########################


    method_list.append(['FC_SFFS_backward', round(selected_representation[3],2), round(selected_representation[2],2), selected_representation[0],
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
        visited_representations_train.to_csv(path_or_buf=results_path + '/COMPAS_complete_visited_representations_train_complexity_' + str(complexity)
                                                   + '_CF_' + str(count+1) + '.csv', index=False)
        visited_representations_test.to_csv(
            path_or_buf=results_path + '/COMPAS_complete_visited_representations_test_complexity_' + str(complexity)
                        + '_CF_' + str(count + 1) + '.csv', index=False)
    else:
        visited_representations_train.to_csv(
            path_or_buf=results_path + '/COMPAS_complete_visited_representations_train_complexity_' + str(complexity) + '_' + str(count + 1) + '.csv',
            index=False)
        visited_representations_test.to_csv(
            path_or_buf=results_path + '/COMPAS_complete_visited_representations_test_complexity_' + str(
                complexity) + '_' + str(count + 1) + '.csv',
            index=False)

    print('ROD backward ' + ': ' + str(selected_representation[3]))
    print('F1 backward ' + ': ' + str(selected_representation[2]))

    ########### Dropped

    categorical_features_2 = []
    numerical_features_2 = []

    COMPAS_dropped = COMPAS.copy()
    dropped_columns = list(COMPAS_dropped)
    dropped_columns.remove('age')

    for i in dropped_columns:
        if i != target and i not in inadmissible_features and i != sensitive_feature and COMPAS[i].dtype == np.dtype(
                'O'):
            categorical_features_2.extend([i])
        elif i != target and i not in inadmissible_features and i != sensitive_feature and COMPAS[
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
    if len(inadmissible_features) > 0:
        X_train_dropped = X_train.drop(columns=[sensitive_feature, inadmissible_features, 'age'])
        X_test_dropped = X_test.drop(columns=[sensitive_feature, inadmissible_features, 'age'])
    else:
        X_train_dropped = X_train.drop(columns=[sensitive_feature, 'age'])
        X_test_dropped = X_test.drop(columns=[sensitive_feature, 'age'])


    dropped_pipeline.fit(X_train_dropped, np.ravel(y_train.to_numpy()))
    predicted_dropped = dropped_pipeline.predict(X_test_dropped)
    predicted_dropped_proba = dropped_pipeline.predict_proba(X_test)[:, 1]
    rod_dropped = ROD.ROD(y_pred=predicted_dropped_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                          admissible=X_test.loc[:, admissible_features],
                          protected='African-American', name='dropped_COMPAS')

    f1_dropped = f1_score(np.ravel(y_test.to_numpy()), predicted_dropped)

    method_list.append(['dropped', rod_dropped, f1_dropped, admissible_features, len(admissible_features), count + 1])

    print('ROD dropped ' + ': ' + str(rod_dropped))
    print('F1 dropped ' + ': ' + str(f1_dropped))

    ############################## Capuchin ####################################
    # Remove the sensitive when training and check results --> does ROD decrease variance? : No, bad results, go back

    capuchin_df = COMPAS.copy()
    capuchin_columns = list(capuchin_df)
    capuchin_columns.remove('age')
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
    to_repair = pd.concat([X_train.loc[:, capuchin_columns], y_train], axis=1)
    train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)
    end_time_capuchin = time.time() - start_time_capuchin

    runtimes.extend([end_time_capuchin])

    print(runtimes)

    print('Finished repairing training set with capuchin')
    y_train_repaired = train_repaired.loc[:, [target]].to_numpy()
    X_train_repaired = train_repaired.loc[:, capuchin_columns]

    X_test_capuchin = generate_binned_df(X_test.loc[:, capuchin_columns])

    capuchin_pipeline.fit(X_train_repaired, np.ravel(y_train_repaired))
    predicted_capuchin = capuchin_pipeline.predict(X_test_capuchin)
    predicted_capuchin_proba = capuchin_pipeline.predict_proba(X_test_capuchin)[:, 1]
    rod_capuchin = ROD.ROD(y_pred=predicted_capuchin_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                           admissible=X_test.loc[:, admissible_features],
                           protected='African-American', name='capuchin_COMPAS')

    f1_capuchin = f1_score(np.ravel(y_test.to_numpy()), predicted_capuchin)

    method_list.append(['capuchin', rod_capuchin, f1_capuchin, capuchin_columns, len(capuchin_columns), count + 1])

    print('ROD capuchin ' + ': ' + str(rod_capuchin))
    print('F1 capuchin ' + ': ' + str(f1_capuchin))

    ##################### Original

    categorical_features = []
    numerical_features = []
    for i in capuchin_columns:
        if i != target and COMPAS[i].dtype == np.dtype('O'):
            categorical_features.extend([i])
        elif i != target and COMPAS[i].dtype != np.dtype('O'):
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

    original_pipeline.fit(X_train.loc[:, capuchin_columns], np.ravel(y_train.to_numpy()))
    predicted_original = original_pipeline.predict(X_test.loc[:, capuchin_columns])
    predicted_original_proba = original_pipeline.predict_proba(X_test.loc[:, capuchin_columns])[:, 1]
    rod_original = ROD.ROD(y_pred=predicted_original_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                           admissible=X_test.loc[:, admissible_features],
                           protected='African-American', name='original_COMPAS')

    f1_original = f1_score(np.ravel(y_test.to_numpy()), predicted_original)

    method_list.append(['original', rod_original, f1_original, capuchin_columns, len(capuchin_columns), count + 1])

    method_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation','Size', 'Fold'])
    if CF:
        method_df.to_csv(
            path_or_buf=results_path + '/complete_COMPAS_results_complexity_' + str(complexity) + '_CF_' + str(count+1) + '.csv', index=False)
    else:
        method_df.to_csv(path_or_buf=results_path + '/complete_COMPAS_results_complexity_' + str(complexity) + '_' + str(count+1) + '.csv',
                          index=False)

    print('ROD original ' + ': ' + str(rod_original))
    print('F1 original ' + ': ' + str(f1_original))

    count += 1

    runtimes_array = np.asarray(runtimes)

    runtimes_array = np.reshape(runtimes_array, (1, runtimes_array.shape[0]))

    runtimes_df = pd.DataFrame(runtimes_array, columns=['Combinations', 'Complexity', 'Rows', 'Transformations',
                                                  'Time_2_transformations', 'Filtered_transformations',
                                                  'Time_2_CF', 'Time_2_FR', 'Time_2_SR', 'Total_runtime_SFFS_BF', 'Repair_time_capuchin'])
    runtimes_df['Fold'] = count

    if CF:
        runtimes_df.to_csv(path_or_buf=results_path + '/COMPAS_runtimes_complexity_' + str(complexity) + '_CF_' + str(count) + '.csv',
                       index=False)
    else:
        runtimes_df.to_csv(path_or_buf=results_path + '/COMPAS_runtimes_complexity_' + str(complexity) + '_' + str(count) + '.csv',
                           index=False)

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])


print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

if CF:
    summary_df.to_csv(path_or_buf=results_path + '/complete_COMPAS_results_complexity_' + str(complexity) + '_CF.csv', index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/complete_COMPAS_results_complexity_' + str(complexity) + '.csv', index=False)