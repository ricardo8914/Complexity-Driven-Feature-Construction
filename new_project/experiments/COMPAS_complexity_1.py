import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
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
                    , ['race', 'age', 'priors_count', 'is_recid', 'c_charge_degree']]

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

complexity = 4
CF = True
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
    registered_representations_train = []
    registered_representations_test = []
    dropped_features = []
    join_score = 0

    mask = [idx for idx, i in enumerate(all_features) if COMPAS[i].dtype != np.dtype('O')]
    all_names = all_features

    transformed_train_i = X_train.to_numpy()
    transformed_test_i = X_test.to_numpy()

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

        if idj == 0:
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
                        test_scores_r = cv_scores.cv_results_['mean_test_F1'][0]
                        rod_scores_r = cv_scores.cv_results_['mean_test_ROD'][0]

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
            if idj == 0:
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
        'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'], 'class_weight': ['balanced'],
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
                            test_scores_a = cv_scores.cv_results_['mean_test_F1'][0]
                            rod_scores_a = cv_scores.cv_results_['mean_test_ROD'][0]

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
    selected_representation = all_visited_test[pareto[min_dist]]

    method_list.append(['FS_BS', selected_representation[3], selected_representation[2], selected_representation[0],
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
        visited_representations_train.to_csv(path_or_buf=results_path + '/' + dataset +'_complexity_1_visited_representations_train_complexity_' + str(complexity)
                                                   + '_CF_' + str(count+1) + '.csv', index=False)
        visited_representations_test.to_csv(
            path_or_buf=results_path + '/' + dataset + '_complexity_1_visited_representations_test_complexity_' + str(complexity)
                        + '_CF_' + str(count + 1) + '.csv', index=False)
    else:
        visited_representations_train.to_csv(
            path_or_buf=results_path + '/' + dataset + '_complexity_1_visited_representations_train_complexity_' + str(complexity) + '_' + str(count + 1) + '.csv',
            index=False)
        visited_representations_test.to_csv(
            path_or_buf=results_path + '/' + dataset + '_complexity_1_visited_representations_test_complexity_' + str(
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
    summary_df.to_csv(path_or_buf=results_path + '/complexity_1_' + dataset + '_CF.csv', index=False)
    runtimes_df.to_csv(
        path_or_buf=results_path + '/' + dataset + '_runtimes_complexity_1_' + '_CF.csv',
        index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/complete_' + dataset + '_results_complexity_1.csv', index=False)
    runtimes_df.to_csv(path_or_buf=results_path + '/' + dataset + '_runtimes_complexity_1.csv',
                       index=False)
