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
import ROD
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy.linalg import norm
import time

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
admissible_features = [i for i in list(COMPAS) if
                       i not in inadmissible_features and i != sensitive_feature and i != target]

protected = 'African-American'
dataset = 'COMPAS'

all_features = list(COMPAS)
all_features.remove(target)
all_2_combinations = list(itertools.combinations(all_features, 2))

def evaluation():

    complexity = 4
    CF = False
    count = 0
    method_list = []
    runtimes_list = []
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
                                protected=protected, name=dataset)

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

            start_time_transform = time.time()
            transformed_train_i = transformed_pipeline.fit_transform(X_train_t, np.ravel(y_train.to_numpy()))
            time_2_create = (time.time() - start_time_transform)
            all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
                'new_construction'].all_features_set
            transformed_test_i = transformed_pipeline.transform(X_test_t)
            count_transformations += len(all_transformations)
            time_2_create_transformations += time_2_create

            #########Paralelize!!!!

            outcome_df = train_df.loc[:, [target]]
            outcome_df.reset_index(inplace=True, drop=True)
            outcome_df.rename(columns={target: 'outcome'}, inplace=True)


            def causal_filter(candidate):

                result = False

                candidate_df = pd.DataFrame(data=transformed_train_i[:, [candidate]], columns=[all_names[candidate]])

                sensitive_df = pd.DataFrame(data=X_train.loc[:, sensitive_feature].to_numpy(),
                                            columns=[sensitive_feature])
                test_df_causal = pd.concat([sensitive_df, candidate_df, outcome_df], axis=1)

                if d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):
                    result = True
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

                pool = mp.Pool(mp.cpu_count())
                results = pool.map(causal_filter, transformations2_generate_idx)
                pool.close()

                end_time_CF = time.time() - start_time_CF

                accepted_list = list(itertools.chain(*[results]))
                accepted_idx = np.argwhere(np.array(accepted_list))

                new_selected = list(itertools.compress(transformations2_generate_idx, accepted_list))

                new_allowed_names = [x for idx, x in enumerate(allowed_names) if idx in new_selected]

                mask = [x for x in new_selected]

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
                                             scoring={'F1': f1, 'ROD': rod_score}, refit='F1', cv=5)

                    cv_scores.fit(transformed_train, np.ravel(y_train.to_numpy()))
                    test_scores = cv_scores.cv_results_['mean_test_F1'][cv_scores.best_index_]
                    rod_scores = cv_scores.cv_results_['mean_test_ROD'][cv_scores.best_index_]

                    unique_representations.append(accepted_features_photo.copy())
                    #test_clf.fit(transformed_train, np.ravel(y_train.to_numpy()))
                    predicted_ff = cv_scores.predict(transformed_test)
                    predicted_ff_proba = cv_scores.predict_proba(transformed_test)[:, 1]
                    rod_ff = ROD.ROD(y_pred=predicted_ff_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                     admissible=X_test.loc[:, admissible_features],
                                     protected=protected, name=dataset)
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
                                    test_scores_r = cv_scores.cv_results_['mean_test_F1'][cv_scores.best_index_]
                                    rod_scores_r = cv_scores.cv_results_['mean_test_ROD'][cv_scores.best_index_]

                                    #test_clf.fit(transformed_train_r, np.ravel(y_train.to_numpy()))
                                    predicted_ff_r = cv_scores.predict(transformed_test_r)
                                    predicted_ff_r_proba = cv_scores.predict_proba(transformed_test_r)[:, 1]
                                    rod_ff_r = ROD.ROD(y_pred=predicted_ff_r_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                                       admissible=X_test.loc[:, admissible_features],
                                                       protected=protected, name=dataset)
                                    f1_ff_r = f1_score(np.ravel(y_test.to_numpy()), predicted_ff_r)

                                    registered_representations_test.append(
                                        [accepted_features_r.copy(), len(accepted_features_r.copy()), f1_ff_r, rod_ff_r])
                                    registered_representations_train.append(
                                        [accepted_features_r.copy(), len(accepted_features_r.copy()), test_scores_r, rod_scores_r])

                                    if test_scores_r >= join_score:
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

        start_time_SR = time.time()

        print(transformed_train.shape, transformed_test.shape, str(join_score))

        cv_scores = GridSearchCV(LogisticRegression(), param_grid={
            'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'], 'class_weight': ['balanced'],
            'max_iter': [100000], 'multi_class': ['auto']
        },
                                 n_jobs=-1,
                                 scoring={'F1': f1, 'ROD': rod_score}, refit='ROD', cv=5)

        cv_scores.fit(transformed_train, np.ravel(y_train.to_numpy()))
        test_scores_c = cv_scores.cv_results_['mean_test_F1'][cv_scores.best_index_]
        rod_scores_c = cv_scores.cv_results_['mean_test_ROD'][cv_scores.best_index_]

        rod_complete = rod_scores_c
        f1_complete = test_scores_c

        print('__________________________________________')
        print('Round 2 : Improving in the other direction. Start with backward floating elimination: ')

        print('F1 complete: {:.4f}'.format(f1_complete))
        print('ROD complete {:.4f}'.format(rod_complete))

        # Now SBFS

        selected_ids_r = []
        for d in reversed(range(transformed_train.shape[1])):
            if transformed_train.shape[1] > len(selected_ids_r) + 1:
                selected_ids_r.extend([d])
                transformed_train_cr = np.delete(transformed_train, selected_ids_r, 1)
                transformed_test_cr = np.delete(transformed_test, selected_ids_r, 1)
                accepted_features_cr = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids_r]
                accepted_features_cr.sort()

                if accepted_features_cr not in unique_representations:

                    cv_scores.fit(transformed_train_cr, np.ravel(y_train.to_numpy()))
                    test_scores_cr = cv_scores.cv_results_['mean_test_F1'][cv_scores.best_index_]
                    rod_scores_cr = cv_scores.cv_results_['mean_test_ROD'][cv_scores.best_index_]

                    #complete_clf.fit(transformed_train_cr, np.ravel(y_train.to_numpy()))
                    predicted_b = cv_scores.predict(transformed_test_cr)
                    predicted_b_proba = cv_scores.predict_proba(transformed_test_cr)[:, 1]
                    rod_b = ROD.ROD(y_pred=predicted_b_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                       admissible=X_test.loc[:, admissible_features],
                                       protected=protected, name=dataset)
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
                                test_scores_a = cv_scores.cv_results_['mean_test_F1'][cv_scores.best_index_]
                                rod_scores_a = cv_scores.cv_results_['mean_test_ROD'][cv_scores.best_index_]

                                #complete_clf.fit(transformed_train_a, np.ravel(y_train.to_numpy()))
                                predicted_a = cv_scores.predict(transformed_test_a)
                                predicted_a_proba = cv_scores.predict_proba(transformed_test_a)[:, 1]
                                rod_a = ROD.ROD(y_pred=predicted_a_proba, sensitive=X_test.loc[:, [sensitive_feature]],
                                                admissible=X_test.loc[:, admissible_features],
                                                protected=protected, name=dataset)
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

        normalized = (scores[:, 1] - scores[:, 1].min()) / (0 - scores[:, 1].min())
        scores[:, 1] = normalized

        pareto = identify_pareto(scores)
        pareto_front = scores[pareto]

        ideal_point = np.asarray([1, 1])
        dist = np.empty((pareto_front.shape[0], 1))

        for idx, i in enumerate(pareto_front):
            dist[idx] = norm(i - ideal_point)

        min_dist = np.argmin(dist)
        selected_representation = all_visited_test[min_dist]

        method_list.append(['FC_FS_BS', selected_representation[3], selected_representation[2], selected_representation[0],
                           len(selected_representation[0]), count + 1])
        runtimes.extend([count_transformations, time_2_create_transformations, filtered_transformations, time_2_CF, time_2_FR, time_2_SR,
                         time_2_create_transformations+time_2_CF+time_2_FR+time_2_SR, count +1])

        runtimes_list.append(runtimes)

        print(runtimes)

        visited_representations_train = pd.DataFrame(registered_representations_train, columns=['Representation', 'Size', 'F1', 'ROD'])
        visited_representations_train['Fold'] = count + 1

        visited_representations_test = pd.DataFrame(registered_representations_test,
                                                     columns=['Representation', 'Size', 'F1', 'ROD'])
        visited_representations_test['Fold'] = count + 1

        if CF:
            visited_representations_train.to_csv(path_or_buf=results_path + '/' + dataset +'_complete_visited_representations_train_complexity_' + str(complexity)
                                                       + '_CF_' + str(count+1) + '.csv', index=False)
            visited_representations_test.to_csv(
                path_or_buf=results_path + '/' + dataset + '_complete_visited_representations_test_complexity_' + str(complexity)
                            + '_CF_' + str(count + 1) + '.csv', index=False)
        else:
            visited_representations_train.to_csv(
                path_or_buf=results_path + '/' + dataset + '_complete_visited_representations_train_complexity_' + str(complexity) + '_' + str(count + 1) + '.csv',
                index=False)
            visited_representations_test.to_csv(
                path_or_buf=results_path + '/' + dataset + '_complete_visited_representations_test_complexity_' + str(
                    complexity) + '_' + str(count + 1) + '.csv',
                index=False)

        print('ROD backward ' + ': ' + str(selected_representation[3]))
        print('F1 backward ' + ': ' + str(selected_representation[2]))

        count += 1

    summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])
    runtimes_df = pd.DataFrame(runtimes_list, columns=['Combinations', 'Complexity', 'Rows', 'Transformations',
                                                      'Time_2_transformations', 'Filtered_transformations',
                                                      'Time_2_CF', 'Time_2_FR', 'Time_2_SR', 'Total_runtime_SFFS_BF', 'Fold'])


    print(summary_df.groupby('Method')['ROD'].mean())
    print(summary_df.groupby('Method')['F1'].mean())

    if CF:
        summary_df.to_csv(path_or_buf=results_path + '/complete_' + dataset +'_results_complexity_' + str(complexity) + '_CF.csv', index=False)
        runtimes_df.to_csv(
            path_or_buf=results_path + '/' + dataset + '_runtimes_complexity_' + str(complexity) + '_CF_' + str(count) + '.csv',
            index=False)
    else:
        summary_df.to_csv(path_or_buf=results_path + '/complete_' + dataset + '_results_complexity_' + str(complexity) + '.csv', index=False)
        runtimes_df.to_csv(path_or_buf=results_path + '/' + dataset + '_runtimes_complexity_' + str(complexity) + '_' + str(count) + '.csv',
                           index=False)

if __name__ == '__main__':
    mp.set_start_method('fork')
    evaluation()