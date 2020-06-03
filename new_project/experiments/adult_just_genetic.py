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
from numpy.linalg import norm
from sklearn.model_selection import KFold
from new_project.tests.test_evolutionary import evolution
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
complexity = 4
CF = False
method_list = []
runtimes = []
visited_train = []
visited_test = []
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

    accepted_features = []
    unique_features = []
    transformed_train = np.empty((X_train.shape[0], 1))
    transformed_test = np.empty((X_test.shape[0], 1))
    registered_representations = []
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



        transformed_train_i = transformed_pipeline.fit_transform(X_train_t, np.ravel(y_train.to_numpy()))
        all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
            'new_construction'].all_features_set
        transformed_test_i = transformed_pipeline.transform(X_test_t)


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

            pool = mp.Pool(7)
            results = pool.map(causal_filter, transformations2_generate)
            pool.close()

            accepted_list = list(itertools.chain(*[results]))

            accepted_idx = np.argwhere(np.array(accepted_list))

            mask = [x for idx, x in enumerate(transformations2_generate_idx) if accepted_list[idx]]
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


    selected_array = evolution(all_allowed_train, np.ravel(y_train.to_numpy()),
                               scorers=[f1, rod_score], cv_splitter=3)

    complete_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                      max_iter=100000, multi_class='auto')

    registered_representations_train = []
    registered_representations_test = []
    for idg, g in enumerate(selected_array):
        if all_allowed_train[:, g].shape[1] > 0:
            complete_clf.fit(all_allowed_train[:, g], np.ravel(y_train.to_numpy()))
            predicted_genetic = complete_clf.predict(all_allowed_test[:, g])
            predicted_genetic_proba = complete_clf.predict_proba(all_allowed_test[:, g])[:, 1]
            rod_genetic = ROD.ROD(y_pred=predicted_genetic_proba, sensitive=X_test.loc[:, ['sex']],
                                  admissible=X_test.loc[:, admissible_features],
                                  protected=' Female', name='genetic_adult')
            f1_genetic = f1_score(np.ravel(y_test.to_numpy()), predicted_genetic)

            predicted_genetic_train = complete_clf.predict(all_allowed_train[:, g])
            predicted_genetic_proba_train = complete_clf.predict_proba(all_allowed_train[:, g])[:, 1]
            rod_genetic_train = ROD.ROD(y_pred=predicted_genetic_proba_train, sensitive=X_train.loc[:, ['sex']],
                                  admissible=X_train.loc[:, admissible_features],
                                  protected=' Female', name='genetic_adult')
            f1_genetic_train = f1_score(np.ravel(y_train.to_numpy()), predicted_genetic_train)

            my_list = []

            x = np.argwhere(selected_array[idg])
            for idj, j in enumerate(x):
                my_list.extend([x.item(idj)])
            representation = [allowed_names[i] for i in my_list]
            registered_representations_train.append(
                [representation, len(representation), f1_genetic_train, rod_genetic_train, count +1])
            registered_representations_test.append(
                [representation, len(representation), f1_genetic, rod_genetic, count +1])
        else:
            pass

    all_visited = np.asarray(registered_representations_train)
    all_visited_test = np.asarray(registered_representations_test)
    scores = all_visited[:, [2, 3]]

    normalized_ROD = (scores[:, 1] - scores[:, 1].min()) / (0 - scores[:, 1].min())
    scores[:, 1] = normalized_ROD

    normalized_F1 = (scores[:, 0] - scores[:, 0].min()) / (scores[:, 0].max() - scores[:, 0].min())
    scores[:, 0] = normalized_F1

    pareto_front = scores

    ideal_point = np.asarray([1, 1])
    dist = np.empty((pareto_front.shape[0], 1))

    for idx, i in enumerate(pareto_front):
        dist[idx] = norm(i - ideal_point)

    min_dist = np.argmin(dist)
    selected_representation = all_visited_test[min_dist]

    end_time = time.time() - start_time

    runtimes.append(['FC_NSGAII', complexity, all_allowed_train.shape[0], all_allowed_train.shape[1], end_time, count + 1])

    method_list.append(['FC_NSGAII', round(selected_representation[3], 2), round(selected_representation[2], 2),
                        selected_representation[0],
                        selected_representation[1], count + 1])

    visited_train.extend(registered_representations_train)
    visited_test.extend(registered_representations_test)

    count += 1

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Size', 'Fold'])
runtimes_df = pd.DataFrame(runtimes, columns=['Method', 'Complexity', 'Rows', 'Features', 'Runtime', 'Fold'])
visited_train_df = pd.DataFrame(visited_train, columns=['Representation', 'Size', 'F1', 'ROD', 'Fold'])
visited_test_df = pd.DataFrame(visited_test, columns=['Representation', 'Size', 'F1', 'ROD', 'Fold'])


print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

if CF:
    summary_df.to_csv(path_or_buf=results_path + '/adult_genetic_complexity_' + str(complexity) + '_CF' + '.csv', index=False)
    runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_adult_genetic_complexity_' + str(complexity) + '_CF' + '.csv',
                      index=False)
    visited_train_df.to_csv(path_or_buf=results_path + '/visited_train_adult_genetic_complexity_' + str(complexity) + '_CF' + '.csv',
                      index=False)
    visited_test_df.to_csv(
        path_or_buf=results_path + '/visited_test_adult_genetic_complexity_' + str(complexity) + '_CF' + '.csv',
        index=False)
else:
    summary_df.to_csv(path_or_buf=results_path + '/adult_genetic_complexity_' + str(complexity) + '.csv', index=False)
    runtimes_df.to_csv(path_or_buf=results_path + '/runtimes_adult_genetic_complexity_' + str(complexity) + '.csv', index=False)
    visited_train_df.to_csv(
        path_or_buf=results_path + '/visited_train_adult_genetic_complexity_' + str(complexity) + '.csv',
        index=False)
    visited_test_df.to_csv(
        path_or_buf=results_path + '/visited_test_adult_genetic_complexity_' + str(complexity) + '.csv',
        index=False)