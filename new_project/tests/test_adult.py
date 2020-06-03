import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from causality.d_separation import d_separation
from sklearn.metrics import log_loss
from tests.test_evolutionary import evolution
from tqdm import tqdm
import ROD
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

sensitive_feature = 'sex'
inadmissible_features = ['marital-status']
target = 'target'
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class', 'relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)
admissible_features = [i for i in list(adult_df) if i not in inadmissible_features and i != sensitive_feature and i != target]

def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in ['target', 'outcome'] and (df_[i].dtype != np.dtype('O') and len(df_[i].unique()) > 4):


            out, bins = pd.cut(df_[i], bins=2, retbins=True, duplicates='drop')
            df_.loc[:, i] = out.astype(str)

    return df_

#a = generate_binned_df(adult_df)
#print(adult_df['sex'].unique())

######################## Dropped #####################################

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
                      ('clf', RandomForestClassifier())])

cv_grid_dropped = GridSearchCV(dropped_pipeline, param_grid = {
    'clf__n_estimators' : [100],#,
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5] #,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='f1')

######################### Original ###########################

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
                      ('clf', RandomForestClassifier())])

cv_grid_original = GridSearchCV(original_pipeline, param_grid = {
    'clf__n_estimators' : [100],#,
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5] #,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='f1')

############################## Capuchin ####################################

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
                        ('clf', RandomForestClassifier())])


cv_grid_capuchin = GridSearchCV(capuchin_pipeline, param_grid = {
    'clf__n_estimators' : [100] #,
    #'clf__criterion' : ['gini', 'entropy'],
    #'clf__class_weight' : [None, 'balanced']#,
    #'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='f1')


############################## Feature Construction #############################################

features2_build = []
for i in list(adult_df):
    if i not in inadmissible_features and i != sensitive_feature and i != target:
    #if i != target:
        features2_build.extend([i])

features2_build_2 = []
for i in list(adult_df):
    if i != target:
    #if i != target:
        features2_build_2.extend([i])

features2_build_cat = []
features2_build_num = []
features2_build_cat_2 = []
features2_build_num_2 = []
for i in features2_build:
    if adult_df[i].dtype == np.dtype('O'):
        features2_build_cat.extend([i])
    else:
        features2_build_num.extend([i])

for i in features2_build_2:
    if adult_df[i].dtype == np.dtype('O'):
        features2_build_cat_2.extend([i])
    else:
        features2_build_num_2.extend([i])

features2_scale = []
for i in features2_build:
    if adult_df[i].dtype != np.dtype('O'):
        features2_scale.extend([features2_build.index(i)])
    else:
        pass

features2_scale_2 = []
for i in features2_build_2:
    if adult_df[i].dtype != np.dtype('O'):
        features2_scale_2.extend([features2_build_2.index(i)])
    else:
        pass

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features2_scale)], remainder='passthrough')

preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features2_scale_2)], remainder='passthrough')

new_order = features2_build_num + features2_build_cat
new_order_2 = features2_build_num_2 + features2_build_cat_2
features2_build_mask = ([False] * len(features2_build_num)) + ([True] * len(features2_build_cat))
features2_build_mask_2 = ([False] * len(features2_build_num_2)) + ([True] * len(features2_build_cat_2))

acc = make_scorer(accuracy_score, greater_is_better=True, needs_threshold=False)
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
column_transformation = Pipeline([('new_construction', ConstructionTransformer(c_max=2, max_time_secs=10000, scoring=f1, n_jobs=7, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=new_order,
                                                    feature_is_categorical=features2_build_mask))])

column_transformation_2 = Pipeline([('new_construction', ConstructionTransformer(c_max=2, max_time_secs=10000, scoring=f1, n_jobs=7, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=new_order_2,
                                                    feature_is_categorical=features2_build_mask_2))])



transformed_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('feature_construction', column_transformation)])#,#,
                                #('clf', RandomForestClassifier())])
                                #('clf', cv_grid_transformed)])

transformed_pipeline_2 = Pipeline(steps=[('preprocessor', preprocessor_2),
                                ('feature_construction', column_transformation_2)])

#transformed_classifier = RandomForestClassifier(n_estimators=250, max_features=1.0, n_jobs=-1)

#########################################

################################################
count = 0
method_list = []
kf1 = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf1.split(adult_df):

    print('Start proccessing fold: {}'.format(count+1))
    train_df = adult_df.iloc[train_index]
    test_df = adult_df.iloc[test_index]

    rod = make_scorer(ROD.ROD, greater_is_better=True, needs_proba=False, sensitive=train_df.loc[:, ['sex']],
                      admissible=train_df.loc[:, admissible_features],
                      protected=' Female', name='dropped_adult')

    X_train = train_df.loc[:, ['workclass', 'education', 'sex', 'marital-status', 'occupation', 'age', 'capital-gain',
                            'capital-loss', 'hours-per-week']]

    y_train = train_df.loc[:, 'target']

    X_test = test_df.loc[:, ['workclass', 'education', 'sex', 'marital-status', 'occupation', 'age', 'capital-gain',
                               'capital-loss', 'hours-per-week']]

    y_test = test_df.loc[:, 'target']

    X_train_t_complete = train_df.loc[:, features2_build_2]
    X_test_t_complete = test_df.loc[:, features2_build_2]

    X_train_t_dropped = X_train_t_complete.drop(columns=['sex', 'marital-status'])
    X_test_t_dropped = X_test_t_complete.drop(columns=['sex', 'marital-status'])

    transformed_train = transformed_pipeline.fit_transform(X_train_t_dropped, np.ravel(y_train.to_numpy()))
    transformed_train_2 = transformed_pipeline_2.fit_transform(X_train_t_complete, np.ravel(y_train.to_numpy()))
    all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps[
        'new_construction'].all_features_set
    all_transformations_2 = transformed_pipeline_2.named_steps['feature_construction'].named_steps[
        'new_construction'].all_features_set
    transformed_test = transformed_pipeline.transform(X_test_t_dropped.to_numpy())
    transformed_test_2 = transformed_pipeline_2.transform(X_test_t_complete.to_numpy())

    transformed_columns_dropped = []
    for i in all_transformations:
        j = (i.get_name()).strip()
        transformed_columns_dropped.extend([j])


    transformed_columns_complete = []
    for i in all_transformations_2:
        j = (i.get_name()).strip()
        transformed_columns_complete.extend([j])

    transformed_train_2_df = pd.DataFrame(data=transformed_train_2, columns=transformed_columns_complete)


    print('Start repairing training set with capuchin')
    to_repair = pd.concat([X_train, y_train], axis=1)
    train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)
    print('Finished repairing training set with capuchin')
    y_train_repaired = train_repaired.loc[:, ['target']].to_numpy()
    X_train_repaired = train_repaired.loc[:,
                           ['workclass', 'education', 'occupation', 'sex', 'marital-status', 'age', 'capital-gain',
                               'capital-loss', 'hours-per-week']]

    X_test_capuchin = (generate_binned_df(X_test)).loc[:,['workclass', 'education', 'occupation', 'sex', 'marital-status',
                                                         'age', 'capital-gain',
                                                        'capital-loss', 'hours-per-week']]

    X_train_dropped = X_train.drop(columns=['sex', 'marital-status'])
    X_test_dropped = X_test.drop(columns=['sex', 'marital-status'])

    print('Training classifiers')
    dropped = dropped_pipeline.fit(X_train_dropped, np.ravel(y_train.to_numpy()))
    original = original_pipeline.fit(X_train, np.ravel(y_train.to_numpy()))
    capuchin = cv_grid_capuchin.fit(generate_binned_df(X_train_repaired), np.ravel(y_train_repaired))

    print('Starting to evaluate genetic feature selection')
    selected_array = evolution(transformed_train_2_df, np.ravel(y_train.to_numpy()), scorers=[f1, rod], cv_splitter=5,
                               max_search_time=60)

    genetic_trained = []
    for i in selected_array:
        cv_grid_evolutionary = GridSearchCV(RandomForestClassifier(), param_grid={
            'n_estimators': [100],  # ,
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced'],
            'max_depth': [None, 3, 5]  # ,
            # 'max_features' : [1.0]
        },
                                            n_jobs=-1,
                                            scoring='f1')

        genetic_trained.append(cv_grid_evolutionary.fit(transformed_train_2[:, i], np.ravel(y_train.to_numpy())))

    print('start processing feature construction training sets')
    
    print('Learning dropped feature construction features')
    selected_idx_dropped = []
    selected_names_dropped = []
    size_dropped = []
    L_dropped = []
    ll_dropped = np.inf
    for idx, i in enumerate(tqdm(transformed_columns_dropped)):
        if i != sensitive_feature:

            test_model = LogisticRegression()

            selected_idx_dropped.extend([idx])
            selected_names_dropped.extend([i])
            test_model.fit(transformed_train[:, selected_idx_dropped], np.ravel(y_train.to_numpy()))
            proba_i_dropped = test_model.predict_proba(transformed_train[:, selected_idx_dropped])[:, 1]
            logLoss_dropped = log_loss(y_true=np.ravel(y_train.to_numpy()), y_pred=proba_i_dropped)
            if ll_dropped - logLoss_dropped >= 0.005:
                ll_dropped = logLoss_dropped
                size_dropped.extend([len(selected_names_dropped)])
                L_dropped.extend([logLoss_dropped])
                print('Selected: {}'.format(selected_names_dropped))
                print('Size: {}'.format(str(len(selected_names_dropped))))
                print('Log Loss: {:.4f}'.format(logLoss_dropped))
            else:
                selected_names_dropped.remove(i)
                selected_idx_dropped.remove(idx)

    print('Learning complete feature construction features')
    selected_idx_complete = []
    selected_names_complete = []
    size_complete = []
    L_complete = []
    ll_complete = np.inf
    for idx, i in enumerate(tqdm(transformed_columns_complete)):
        if i != sensitive_feature:

            test_model = LogisticRegression()

            selected_idx_complete.extend([idx])
            selected_names_complete.extend([i])
            test_model.fit(transformed_train_2[:, selected_idx_complete], np.ravel(y_train.to_numpy()))
            proba_i_complete = test_model.predict_proba(transformed_train_2[:, selected_idx_complete])[:, 1]
            logLoss_complete = log_loss(y_true=np.ravel(y_train.to_numpy()), y_pred=proba_i_complete)
            if ll_complete - logLoss_complete >= 0.005:
                ll_complete = logLoss_complete
                size_complete.extend([len(selected_names_complete)])
                L_complete.extend([logLoss_complete])
                print('Selected: {}'.format(selected_names_complete))
                print('Size: {}'.format(str(len(selected_names_complete))))
                print('Log Loss: {:.4f}'.format(logLoss_complete))
            else:
                selected_names_complete.remove(i)
                selected_idx_complete.remove(idx)

    print('Learning complete feature construction features with causal filter')
    selected_idx_causal = []
    selected_names_causal = []
    size_causal = []
    L_causal = []
    ll_causal = np.inf
    for idx, i in enumerate(tqdm(transformed_columns_complete)):
        if i != sensitive_feature:

            test_model = LogisticRegression()
            test_model.fit(transformed_train_2[:, [idx]], np.ravel(y_train.to_numpy()))

            proba_causal = test_model.predict_proba(transformed_train_2[:, [idx]])[:, 1]
            predictions_causal = test_model.predict(transformed_train_2[:, [idx]])
            outcome_df_causal = pd.DataFrame(data=predictions_causal, columns=['outcome'])
            sensitive_df_causal = pd.DataFrame(data=X_train.loc[:, sensitive_feature].to_numpy(), columns=[sensitive_feature])
            selected_df_causal = pd.DataFrame(data=transformed_train_2[:, [idx]], columns=[i])
            test_df_causal = pd.concat([sensitive_df_causal, selected_df_causal, outcome_df_causal], axis=1)

            if d_separation(test_df_causal, sensitive=sensitive_feature, target='outcome'):
                selected_idx_causal.extend([idx])
                selected_names_causal.extend([i])
                test_model.fit(transformed_train_2[:, selected_idx_causal], np.ravel(y_train.to_numpy()))
                proba_i_causal = test_model.predict_proba(transformed_train_2[:, selected_idx_causal])[:, 1]
                logLoss_causal = log_loss(y_true=np.ravel(y_train.to_numpy()), y_pred=proba_i_causal)
                if ll_causal - logLoss_causal >= 0.005:
                    ll_causal = logLoss_causal
                    size_causal.extend([len(selected_names_causal)])
                    L_causal.extend([logLoss_causal])
                    print('Selected: {}'.format(selected_names_causal))
                    print('Size: {}'.format(str(len(selected_names_causal))))
                    print('Log Loss: {:.4f}'.format(logLoss_causal))
                else:
                    selected_names_causal.remove(i)
                    selected_idx_causal.remove(idx)

    cv_grid_transformed_dropped = GridSearchCV(RandomForestClassifier(), param_grid={
        'n_estimators': [100],  # ,
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'max_depth': [None, 3, 5]  # ,
        # 'max_features' : [1.0]
    },
                                         n_jobs=-1,
                                         scoring='f1')

    cv_grid_transformed_complete = GridSearchCV(RandomForestClassifier(), param_grid={
        'n_estimators': [100],  # ,
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'max_depth': [None, 3, 5]  # ,
        # 'max_features' : [1.0]
    },
                                               n_jobs=-1,
                                               scoring='f1')

    cv_grid_transformed_causal = GridSearchCV(RandomForestClassifier(), param_grid={
        'n_estimators': [100],  # ,
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'max_depth': [None, 3, 5]  # ,
        # 'max_features' : [1.0]
    },
                                               n_jobs=-1,
                                               scoring='f1')
    
    feature_construction_dropped = cv_grid_transformed_dropped.fit(transformed_train[:, selected_idx_dropped], 
                                                                   np.ravel(y_train.to_numpy()))
    feature_construction_complete = cv_grid_transformed_complete.fit(transformed_train_2[:, selected_idx_complete],
                                                                   np.ravel(y_train.to_numpy()))
    feature_construction_causal = cv_grid_transformed_causal.fit(transformed_train_2[:, selected_idx_causal],
                                                                     np.ravel(y_train.to_numpy()))


    print('Classifiers were trained')
    #
    outcome_dropped = dropped.predict(X_test_dropped)
    y_pred_proba_dropped = dropped.predict_proba(X_test_dropped)[:, 1]
    outcome_original = original.predict(X_test)
    y_pred_proba_original = original.predict_proba(X_test)[:, 1]
    outcome_capuchin = capuchin.predict(X_test_capuchin)
    y_pred_proba_capuchin = capuchin.predict_proba(X_test_capuchin)[:, 1]
    outcome_transformed_dropped = feature_construction_dropped.predict(transformed_test[:, selected_idx_dropped])
    y_pred_proba_transformed_dropped = feature_construction_dropped.predict_proba(transformed_test[:, selected_idx_dropped])[:, 1]
    outcome_transformed_complete = feature_construction_complete.predict(transformed_test_2[:, selected_idx_complete])
    y_pred_proba_transformed_complete = feature_construction_complete.predict_proba(transformed_test_2[:, selected_idx_complete])[:, 1]
    outcome_transformed_causal = feature_construction_causal.predict(transformed_test_2[:, selected_idx_causal])
    y_pred_proba_transformed_causal = feature_construction_causal.predict_proba(
        transformed_test_2[:, selected_idx_causal])[:, 1]

    admissible_df = X_test_dropped

    rod_dropped = ROD.ROD(y_pred=y_pred_proba_dropped, sensitive=X_test.loc[:, ['sex']], admissible = admissible_df,
                      protected=' Female', name='dropped_adult')
    rod_original = ROD.ROD(y_pred=y_pred_proba_original, sensitive=X_test.loc[:, ['sex']], admissible = admissible_df,
                      protected=' Female', name='original_adult')
    rod_capuchin = ROD.ROD(y_pred=y_pred_proba_capuchin, sensitive=X_test.loc[:, ['sex']], admissible = admissible_df,
                      protected=' Female',
                      name='capuchin_adult')
    rod_transformed_dropped = ROD.ROD(y_pred=y_pred_proba_transformed_dropped, sensitive=X_test.loc[:, ['sex']], 
                                      admissible=admissible_df,
                  protected=' Female', name='feature_construction_adult_dropped')
    rod_transformed_complete = ROD.ROD(y_pred=y_pred_proba_transformed_complete, sensitive=X_test.loc[:, ['sex']],
                              admissible=admissible_df,
                              protected=' Female', name='feature_construction_adult_complete')
    rod_transformed_causal = ROD.ROD(y_pred=y_pred_proba_transformed_causal, sensitive=X_test.loc[:, ['sex']],
                                       admissible=admissible_df,
                                       protected=' Female', name='feature_construction_adult_causal')

    tpr_dropped = true_positive_rate_score(y_test.to_numpy(), outcome_dropped, sensitive_data=X_test.loc[:, ['sex']])
    tpr_original = true_positive_rate_score(y_test.to_numpy(), outcome_original,
                                           sensitive_data=X_test.loc[:, ['sex']])
    tpr_capuchin = true_positive_rate_score(y_test.to_numpy(), outcome_capuchin,
                                           sensitive_data=X_test.loc[:, ['sex']])
    tpr_transformed_dropped = true_positive_rate_score(y_test.to_numpy(), outcome_transformed_dropped,
                                           sensitive_data=X_test.loc[:, ['sex']])
    tpr_transformed_complete = true_positive_rate_score(y_test.to_numpy(), outcome_transformed_complete,
                                               sensitive_data=X_test.loc[:, ['sex']])
    tpr_transformed_causal = true_positive_rate_score(y_test.to_numpy(), outcome_transformed_causal,
                                                        sensitive_data=X_test.loc[:, ['sex']])

    acc_dropped = accuracy_score(np.ravel(y_test), outcome_dropped)
    acc_original = accuracy_score(np.ravel(y_test), outcome_original)
    acc_capuchin = accuracy_score(np.ravel(y_test), outcome_capuchin)
    acc_transformed_dropped = accuracy_score(np.ravel(y_test), outcome_transformed_dropped)
    acc_transformed_complete = accuracy_score(np.ravel(y_test), outcome_transformed_complete)
    acc_transformed_causal = accuracy_score(np.ravel(y_test), outcome_transformed_causal)

    f1_dropped = f1_score(np.ravel(y_test), outcome_dropped)
    f1_original = f1_score(np.ravel(y_test), outcome_original)
    f1_capuchin = f1_score(np.ravel(y_test), outcome_capuchin)
    f1_transformed_dropped = f1_score(np.ravel(y_test), outcome_transformed_dropped)
    f1_transformed_complete = f1_score(np.ravel(y_test), outcome_transformed_complete)
    f1_transformed_causal = f1_score(np.ravel(y_test), outcome_transformed_causal)

    representation_dropped = list(X_train_dropped)
    representation_original = list(X_train)
    representation_capuchin = list(X_train)
    representation_fc_dropped = selected_names_dropped
    representation_fc_complete = selected_names_complete
    representation_fc_causal = selected_names_causal

    for idx, i in enumerate(genetic_trained):
        outcome_gen = i.predict(transformed_test_2[:, selected_array[idx]])
        y_pred_proba_gen = i.predict_proba(transformed_test_2[:, selected_array[idx]])[:, 1]
        rod_gen = ROD.ROD(y_pred=y_pred_proba_gen, sensitive=X_test.loc[:, ['sex']], admissible=admissible_df,
                              protected=' Female', name='feature_construction_genetic_' + str(idx)+ '_' + str(count+1))
        tpr_gen = true_positive_rate_score(y_test.to_numpy(), outcome_gen,
                                               sensitive_data=X_test.loc[:, ['sex']])
        acc_gen = accuracy_score(np.ravel(y_test), outcome_gen)
        f1_gen = f1_score(np.ravel(y_test), outcome_gen)
        my_list = []
        x = np.argwhere(selected_array[idx])
        for idj, j in enumerate(x):
            my_list.extend([x.item(idj)])
        representation = [transformed_columns_complete[i] for i in my_list]
        method_list.append(['feature_construction_genetic_' + str(idx), acc_gen, rod_gen, tpr_gen,
                            f1_gen, representation, count + 1])

    method_list.extend([['feature_construction_dropped', acc_transformed_dropped, rod_transformed_dropped, 
                         tpr_transformed_dropped, f1_transformed_dropped, representation_fc_dropped, count + 1],
                        ['feature_construction_complete', acc_transformed_complete, rod_transformed_complete, 
                         tpr_transformed_complete, f1_transformed_complete, representation_fc_complete, count + 1],
                        ['feature_construction_causal', acc_transformed_causal, rod_transformed_causal,
                         tpr_transformed_causal, f1_transformed_causal, representation_fc_causal, count + 1],
                        ['original', acc_original, rod_original, tpr_original, f1_original, representation_original,
                         count + 1],
                        ['dropped', acc_dropped, rod_dropped, tpr_dropped, f1_dropped, representation_dropped, count + 1],
                        ['capuchin', acc_capuchin, rod_capuchin, tpr_capuchin, f1_capuchin, representation_capuchin,
                         count + 1]])

    count += 1

    print('Fold: {}'.format(count))
    print('ROD dropped: {:.4f}'.format(rod_dropped))
    print('ROD orginal: {:.4f}'.format(rod_original))
    print('ROD capuchin: {:.4f}'.format(rod_capuchin))
    print('ROD transformed dropped: {:.4f}'.format(rod_transformed_dropped))
    print('ROD transformed complete: {:.4f}'.format(rod_transformed_complete))
    print('ROD transformed causal: {:.4f}'.format(rod_transformed_causal))
    print('_______')
    print('TPR dropped: {:.4f}'.format(tpr_dropped))
    print('TPR orginal: {:.4f}'.format(tpr_original))
    print('TPR capuchin: {:.4f}'.format(tpr_capuchin))
    print('TPR transformed dropped: {:.4f}'.format(tpr_transformed_dropped))
    print('TPR transformed complete: {:.4f}'.format(tpr_transformed_complete))
    print('TPR transformed causal: {:.4f}'.format(tpr_transformed_causal))
    print('_______')
    print('ACC dropped: {:.4f}'.format(acc_dropped))
    print('ACC orginal: {:.4f}'.format(acc_original))
    print('ACC capuchin: {:.4f}'.format(acc_capuchin))
    print('ACC transformed dropped: {:.4f}'.format(acc_transformed_dropped))
    print('ACC transformed complete: {:.4f}'.format(acc_transformed_complete))
    print('ACC transformed causal: {:.4f}'.format(acc_transformed_causal))
    print('_______')
    print('f1 dropped: {:.4f}'.format(f1_dropped))
    print('f1 orginal: {:.4f}'.format(f1_original))
    print('f1 capuchin: {:.4f}'.format(f1_capuchin))
    print('f1 transformed dropped: {:.4f}'.format(f1_transformed_dropped))
    print('f1 transformed complete: {:.4f}'.format(f1_transformed_complete))
    print('f1 transformed causal: {:.4f}'.format(f1_transformed_causal))

summary_df = pd.DataFrame(method_list, columns=['Method', 'Accuracy', 'ROD', 'Equal_Oportunity', 'F1', 'Representation', 'Fold'])

print(summary_df.groupby('Method')['Accuracy'].mean())
print(summary_df.groupby('Method')['ROD'].mean())
print(summary_df.groupby('Method')['Equal_Oportunity'].mean())
print(summary_df.groupby('Method')['F1'].mean())

summary_df.to_csv(path_or_buf=results_path + '/summary_adult_rfACCF1_causal_df_complete_genetic_50it.csv', index=False)

#print(mb_original)
#print(mb_dropped)

#binned_X_test_original.to_csv(path_or_buf=adult_path + '/original_adult.csv', index=False)
#binned_X_test_dropped.to_csv(path_or_buf=adult_path + '/dropped_adult.csv', index=False)

