import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from causality.d_separation import d_separation
import multiprocessing as mp


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
inadmissible_features = ['marital-status']
target = 'is_recid'

all_features = list(COMPAS)
all_features.remove(target)
all_2_combinations = list(itertools.combinations(all_features, 2))

all_instances = []
unique_features = []
for i in all_2_combinations:

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

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    column_transformation = Pipeline([('new_construction',
                                       ConstructionTransformer(c_max=5, max_time_secs=1000000, scoring=f1, n_jobs=10,
                                                               model=LogisticRegression(),
                                                                parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                               'class_weight': ['balanced'], 'max_iter': [100000],
                                                               'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                             feature_names=new_order,
                                                            feature_is_categorical=features2_build_mask))])

    transformed_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('feature_construction', column_transformation)])

    X = COMPAS.loc[:, all_features]
    y = COMPAS.loc[:, target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_train_t = X_train.loc[:, features2_build].to_numpy()
    X_test_t = X_test.loc[:, features2_build].to_numpy()

    transformed_pipeline.fit_transform(X_train_t, np.ravel(y_train))
    all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
        'new_construction'].all_features_set
    #transformed_test = transformed_pipeline.transform(X_test_t)


    #########Paralelize!!!!

    def generate_instance(candidate, root=i):

        instance = []
        j = (candidate.get_name()).strip()

        if candidate.transformation != None:
            transformations_applied = []
            try:
                for p in candidate.parents:
                    if p.transformation != None:
                        transformations_applied.extend([p.transformation.name])
                    else:
                        pass

                transformations_applied.extend([candidate.transformation.name])

                instance.extend([root, j, tuple(transformations_applied),
                                 len(candidate.parents), candidate.get_complexity(),
                                 candidate.get_number_of_transformations(),
                                 candidate.get_number_of_raw_attributes(), candidate.get_transformation_depth()])

            except KeyError:
                for p in candidate.parents:
                    if p.transformation != None:
                        transformations_applied.extend([p.transformation.name])
                    else:
                        pass

                transformations_applied.extend([candidate.transformation.name])

                instance.extend([root, j, tuple(transformations_applied),
                                 len(candidate.parents), candidate.get_complexity(),
                                 candidate.get_number_of_transformations(),
                                 candidate.get_number_of_raw_attributes(), candidate.get_transformation_depth()])

        else:
            transformations_applied = []
            for p in candidate.parents:
                if p.transformation != None:
                    transformations_applied.extend([p.transformation.name])
                else:
                    pass

            transformations_applied.extend(['None'])
            instance.extend([root, j, tuple(transformations_applied),
                              len(candidate.parents), candidate.get_complexity(), 0,
                              candidate.get_number_of_raw_attributes(), 0])

        feature_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto')

        parents_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto')

        parents_type = []
        parents_dtype = []
        if len(candidate.parents) >= 1:

            for p in candidate.parents:
                if (p.get_name()).strip() in features2_build_cat:
                    parents_dtype.extend(['categorical'])
                elif (p.get_name()).strip() in features2_build_num:
                    parents_dtype.extend(['numerical'])
                else:
                    parents_dtype.extend([p.properties['type']])

                if (p.get_name()).strip() == sensitive_feature:
                    parents_type.extend(['inadmissible'])
                elif (p.get_name()).strip() in features2_build_cat:
                    categorical_transformer = Pipeline(steps=[
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

                    preprocessor_pc = ColumnTransformer(
                        transformers=[
                            ('cat', categorical_transformer, [(p.get_name()).strip()])], remainder='passthrough')

                    pipeline_pc = Pipeline(steps=[('preprocessor', preprocessor_pc),
                                                  ('clf', parents_clf)])

                    X_train_pc = X_train.loc[:, [(p.get_name()).strip()]]
                    X_test_pc = X_test.loc[:, [(p.get_name()).strip()]]

                    pipeline_pc.fit(X_train_pc, np.ravel(y_train))

                    #y_pred_proba_parent = pipeline_pc.predict_proba(X_test_pc)[:, 1]
                    outcome_parent = pipeline_pc.predict(X_test_pc)

                    outcome_p_df = pd.DataFrame(data=outcome_parent, columns=['outcome'])
                    sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                                columns=[sensitive_feature])
                    parent_df_causal = pd.DataFrame(data=X_test.loc[:, [(p.get_name()).strip()]].to_numpy(),
                                                    columns=[(p.get_name()).strip()])
                    test_p_df_causal = pd.concat([sensitive_df, parent_df_causal, outcome_p_df], axis=1)

                    if d_separation(test_p_df_causal, sensitive=sensitive_feature, target='outcome'):
                        parents_type.extend(['admissible'])
                    else:
                        parents_type.extend(['inadmissible'])

                else:
                    transformed_train_p = p.pipeline.transform(preprocessor.fit_transform(X_train_t))
                    transformed_test_p = p.pipeline.transform(preprocessor.transform(X_test_t))

                    parents_clf.fit(transformed_train_p, np.ravel(y_train))

                    #y_pred_proba_parent = parents_clf.predict_proba(transformed_test_p)[:, 1]
                    outcome_parent = parents_clf.predict(transformed_test_p)

                    outcome_p_df = pd.DataFrame(data=outcome_parent, columns=['outcome'])
                    sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                                columns=[sensitive_feature])
                    parent_df_causal = pd.DataFrame(data=transformed_test_p, columns=[(p.get_name()).strip()])
                    test_p_df_causal = pd.concat([sensitive_df, parent_df_causal, outcome_p_df], axis=1)

                    if np.unique(transformed_test_p).shape[0] == 1:
                        parents_type.extend(['admissible'])
                    elif d_separation(test_p_df_causal, sensitive=sensitive_feature, target='outcome'):
                        parents_type.extend(['admissible'])
                    else:
                        parents_type.extend(['inadmissible'])
        else:
            parents_type.extend(['no parents'])
            parents_dtype.extend(['no parents'])

        instance.extend([tuple(parents_type)])
        instance.extend([tuple(parents_dtype)])
        instance.extend([candidate.properties['type']])

        if j != sensitive_feature:

            transformed_train_c = candidate.pipeline.transform(preprocessor.fit_transform(X_train_t))
            transformed_test_c = candidate.pipeline.transform(preprocessor.transform(X_test_t))

            feature_clf.fit(transformed_train_c, np.ravel(y_train))
            #y_pred_proba_candidate = feature_clf.predict_proba(transformed_test_c)[:, 1]
            outcome_candidate = feature_clf.predict(transformed_test_c)

            outcome_df = pd.DataFrame(data=outcome_candidate, columns=['outcome'])
            sensitive_df = pd.DataFrame(data=X_test.loc[:, sensitive_feature].to_numpy(),
                                        columns=[sensitive_feature])
            selected_df_causal = pd.DataFrame(data=transformed_test_c, columns=[j])
            test_df_causal = pd.concat([sensitive_df, selected_df_causal, outcome_df], axis=1)

            if np.unique(transformed_test_c).shape[0] == 1:
                instance.extend([1])
            elif d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):

                instance.extend([1])
            else:
                instance.extend([0])
        else:
            instance.extend(['NA'])

        return instance

    transformations2_generate = [t for t in all_transformations if (t.get_name()).strip() not in unique_features]

    pool = mp.Pool(7)
    results = pool.map(generate_instance, transformations2_generate)
    pool.close()

    can = list(itertools.chain(*[results]))
    pool.close()

    #print(can)

    print('done with parallelization')

    all_instances.extend(results)
    unique_features.extend([(t.get_name()).strip() for t in transformations2_generate])

instances_df = pd.DataFrame(all_instances, columns=['root', 'name', 'transformations', 'number_parents', 'complexity',
                                                    'number_transformations', 'number_raw_features', 'transformation_depth',
                                                    'parents_types', 'parents_dtypes', 'dtype', 'label'])

print(instances_df.shape)
print(instances_df.groupby('transformations')['label'].mean())

instances_df.to_csv(path_or_buf=results_path + '/feature_analysis_COMPAS.csv', index=False)
