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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import ROD
from sklearn.model_selection import KFold
from tests.test_evolutionary import evolution
import sys
sys.path.insert(0, '/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from methods.capuchin import repair_dataset

home = str(Path.home())

adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

adult_df = pd.read_csv(adult_path + '/adult.csv', sep=';', header=0)

def label(row):
   if row['class'] == ' <=50K' :
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
admissible_features = [i for i in list(adult_df) if i not in inadmissible_features and i != sensitive_feature and i != target]

all_features = list(adult_df)
all_features.remove(target)
all_2_combinations = list(itertools.combinations(all_features, 2))



count = 0
method_list = []
kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf1.split(adult_df):

    train_df = adult_df.iloc[train_index]
    test_df = adult_df.iloc[test_index]

    X_train = train_df.loc[:, all_features]

    y_train = train_df.loc[:, 'target']

    X_test = test_df.loc[:, all_features]

    y_test = test_df.loc[:, 'target']

    rod_score = make_scorer(ROD.ROD, greater_is_better=True, needs_proba=True, sensitive=X_train.loc[:, sensitive_feature],
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
                                           ConstructionTransformer(c_max=4, max_time_secs=1000000, scoring=f1, n_jobs=7,
                                                                   model=LogisticRegression(),
                                                                    parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                   'class_weight': ['balanced'], 'max_iter': [100000],
                                                                   'multi_class':['auto']}, cv=5, epsilon=-np.inf,
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
        transformations2_generate_idx = [idx for idx, t in enumerate(all_transformations) if (t.get_name()).strip() not in unique_features]
        all_names = [(t.get_name()).strip() for t in all_transformations]

        unique_features.extend([(t.get_name()).strip() for t in transformations2_generate])

        pool = mp.Pool(7)
        results = pool.map(causal_filter, transformations2_generate)
        pool.close()

        accepted_list = list(itertools.chain(*[results]))

        accepted_idx = np.argwhere(np.array(accepted_list))

        mask = [x for idx, x in enumerate(transformations2_generate_idx) if accepted_list[idx]]

        test_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto')

        for idj, j in enumerate(mask):

            ##### Step 1: Try to add a feature:

            transformed_train = np.concatenate((transformed_train, transformed_train_i[:, [j]]), axis=1)
            transformed_test = np.concatenate((transformed_test, transformed_test_i[:, [j]]), axis=1)
            all_allowed_train = np.concatenate((all_allowed_train, transformed_train_i[:, [j]]), axis=1)
            all_allowed_test = np.concatenate((all_allowed_test, transformed_test_i[:, [j]]), axis=1)
            accepted_features.extend([all_names[j]])
            allowed_names.extend([all_names[j]])

            if idx == 0 and idj == 0:
                transformed_train = transformed_train[:, 1:]
                transformed_test = transformed_test[:, 1:]
            else:
                pass

            test_scores = cross_val_score(test_clf, transformed_train, np.ravel(y_train.to_numpy()), cv=5, scoring='f1')
            #rod_scores = cross_val_score(test_clf, transformed_train, np.ravel(y_train), cv=5, scoring=rod_score)

            if test_scores.mean() > join_score:
                join_score = test_scores.mean()
                registered_representations.append(accepted_features)

                ##### Step 2: Try to remove a feature:

                selected_ids = []
                for idd, d in enumerate(range(transformed_train.shape[1])):
                    if transformed_train.shape[1] > len(selected_ids) + 1:
                        selected_ids.extend([idd])
                        transformed_train_r = np.delete(transformed_train, selected_ids, 1)
                        transformed_test_r = np.delete(transformed_test, selected_ids, 1)
                        accepted_features_r = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids]

                        test_scores_r = cross_val_score(test_clf, transformed_train_r, np.ravel(y_train.to_numpy()), cv=5, scoring='f1')
                        #rod_scores_r = cross_val_score(test_clf, transformed_train_r, np.ravel(y_train), cv=5,
                                                     #scoring=rod_score)

                        if test_scores_r.mean() > join_score:
                            if accepted_features_r not in registered_representations:
                                join_score = test_scores_r.mean()
                                registered_representations.append(accepted_features_r)
                            else:
                                selected_ids.remove(idd)
                        else:
                            if accepted_features_r not in registered_representations:
                                registered_representations.append(accepted_features_r)
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

        print(transformed_train.shape, transformed_test.shape, str(join_score))

    f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    ############ Genetic: Try to move it to before SFFS

    all_allowed_train = all_allowed_train[:, 1:]
    all_allowed_test = all_allowed_test[:, 1:]

    selected_array = evolution(all_allowed_train, np.ravel(y_train.to_numpy()),
                                   scorers=[f1, rod_score], cv_splitter=5,
                                   max_search_time=60)

    complete_clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto')

    for idg, g in enumerate(selected_array):
        if all_allowed_train[:, g].shape[1] > 0:
            complete_clf.fit(all_allowed_train[:, g], np.ravel(y_train.to_numpy()))
            predicted_genetic = complete_clf.predict(all_allowed_test[:, g])
            predicted_genetic_proba = complete_clf.predict_proba(all_allowed_test[:, g])[:, 1]
            rod_genetic = ROD.ROD(y_pred=predicted_genetic_proba, sensitive=X_test.loc[:, ['sex']], admissible = X_test.loc[:, admissible_features],
                              protected=' Female', name='genetic_adult')
            f1_genetic = f1_score(np.ravel(y_test.to_numpy()), predicted_genetic)

            my_list = []
            x = np.argwhere(selected_array[idg])
            for idj, j in enumerate(x):
                my_list.extend([x.item(idj)])
            representation = [allowed_names[i] for i in my_list]
            method_list.append(['FC_genetic_' + str(idg), rod_genetic, f1_genetic, representation, count + 1])

            print('ROD genetic ' + str(idg) + ': ' + str(rod_genetic))
            print('F1 genetic ' + str(idg) + ': ' + str(f1_genetic))
        else:
            pass

    test_scores_c = cross_val_score(complete_clf, transformed_train, np.ravel(y_train.to_numpy()), cv=5, scoring='f1')
    rod_scores_c = cross_val_score(complete_clf, transformed_train, np.ravel(y_train.to_numpy()), cv=5, scoring=rod_score)

    rod_complete = rod_scores_c.mean()
    f1_complete = test_scores_c.mean()

    print('__________________________________________')
    print('Start with backward elimination: ')

    print('F1 complete: {:.4f}'.format(f1_complete))
    print('ROD complete {:.4f}'.format(rod_complete))

    selected_ids = []
    for idd, d in enumerate(range(transformed_train.shape[1])):
        if transformed_train.shape[1] > len(selected_ids) + 1:
            selected_ids.extend([idd])
            transformed_train_cr = np.delete(transformed_train, selected_ids, 1)
            transformed_test_cr = np.delete(transformed_test, selected_ids, 1)

            test_scores_cr = cross_val_score(complete_clf, transformed_train_cr, np.ravel(y_train.to_numpy()), cv=5, scoring='f1')
            rod_scores_cr = cross_val_score(complete_clf, transformed_train_cr, np.ravel(y_train.to_numpy()), cv=5, scoring=rod_score)

            if rod_scores_cr.mean() > rod_complete and test_scores_cr.mean() > f1_complete * 0.99:
                rod_complete = rod_scores_cr.mean()
                f1_complete = test_scores_cr.mean()
            else:
                selected_ids.remove(idd)
        else:
            pass

        print('representation size: ' + str(transformed_train.shape[1] - len(selected_ids)),
              'ROD: ' + str(rod_complete), 'F1' + str(f1_complete))

    if len(selected_ids) > 0:
        transformed_train = np.delete(transformed_train, selected_ids, 1)
        transformed_test = np.delete(transformed_test, selected_ids, 1)
        accepted_features = [f for idf, f in enumerate(accepted_features) if idf not in selected_ids]
    else:
        pass

    complete_clf.fit(transformed_train, np.ravel(y_train.to_numpy()))
    predicted_backward = complete_clf.predict(transformed_test)
    predicted_backward_proba = complete_clf.predict_proba(transformed_test)[:, 1]
    rod_backward = ROD.ROD(y_pred=predicted_backward_proba, sensitive=X_test.loc[:, ['sex']],
                          admissible=X_test.loc[:, admissible_features],
                          protected=' Female', name='backward_adult')
    f1_backward = f1_score(np.ravel(y_test.to_numpy()), predicted_backward)

    method_list.append(['FC_SFFS_backward', rod_backward, f1_backward, accepted_features, count + 1])

    print('ROD backward ' + ': ' + str(rod_backward))
    print('F1 backward ' + ': ' + str(f1_backward))


    ########### Dropped

    categorical_features_2 = []
    numerical_features_2 = []

    for i in list(adult_df):
        if i != target and i not in inadmissible_features and i != sensitive_feature and adult_df[i].dtype == np.dtype('O'):
            categorical_features_2.extend([i])
        elif i != target and i not in inadmissible_features and i != sensitive_feature and adult_df[i].dtype != np.dtype('O'):
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
                          ('clf', LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto'))])

    X_train_dropped = X_train.drop(columns=['sex', 'marital-status'])
    X_test_dropped = X_test.drop(columns=['sex', 'marital-status'])

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
                            ('repair', FunctionTransformer(repair_dataset, kw_args={'admissible_attributes' : admissible_features,
                                                                                    'sensitive_attribute': sensitive_feature,
                                                                                    'target': target}))])

    capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
                            ('clf', LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto'))])

    print('Start repairing training set with capuchin')
    to_repair = pd.concat([X_train, y_train], axis=1)
    train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)
    print('Finished repairing training set with capuchin')
    y_train_repaired = train_repaired.loc[:, ['target']].to_numpy()
    X_train_repaired = train_repaired.loc[:,
                           ['workclass', 'education', 'occupation', 'age', 'sex', 'marital-status', 'capital-gain',
                               'capital-loss', 'hours-per-week']]

    X_test_capuchin = (generate_binned_df(X_test)).loc[:, ['workclass', 'education', 'occupation', 'age', 'sex', 'marital-status',
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
                          ('clf', LogisticRegression(penalty='l2', C=1, solver='lbfgs', class_weight='balanced',
                                         max_iter=100000, multi_class='auto'))])

    original_pipeline.fit(X_train, np.ravel(y_train.to_numpy()))
    predicted_original = original_pipeline.predict(X_test)
    predicted_original_proba = original_pipeline.predict_proba(X_test)[:, 1]
    rod_original = ROD.ROD(y_pred=predicted_original_proba, sensitive=X_test.loc[:, ['sex']],
                          admissible=X_test.loc[:, admissible_features],
                          protected=' Female', name='original_adult')

    f1_original = f1_score(np.ravel(y_test.to_numpy()), predicted_original)

    method_list.append(['original', rod_original, f1_original, all_features, count + 1])

    print('ROD original ' + ': ' + str(rod_original))
    print('F1 original ' + ': ' + str(f1_original))

    count += 1

summary_df = pd.DataFrame(method_list, columns=['Method', 'ROD', 'F1', 'Representation', 'Fold'])

print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['F1'].mean())

summary_df.to_csv(path_or_buf=results_path + '/summary_new_adult_pipeline.csv', index=False)