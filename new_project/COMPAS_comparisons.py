import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from typing import List, Dict
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_selection import RFECV, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import sys
from pathlib import Path
sys.path.insert(0,'/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from ROD import ROD
from measures.logROD import LROD
from methods.capuchin import repair_dataset
home = str(Path.home())


path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/feature_construction/tmp'
COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

cost_2_raw_features: Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed : Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed: Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination : Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))


acc = make_scorer(accuracy_score, greater_is_better=True, needs_threshold=False)
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
my_pipeline = Pipeline([('new_construction', ConstructionTransformer(c_max=5, scoring=f1, n_jobs=4, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=['age', 'age_cat', 'priors_count', 'c_charge_degree'],
                                                    feature_is_categorical=[False, True, False, True]))])

COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
                    (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
                    & (COMPAS['is_recid'] != -1)
                    & (COMPAS['race'].isin(['African-American','Caucasian']))
                    & (COMPAS['c_charge_degree'].isin(['F','M']))
                    , ['race', 'age', 'age_cat', 'priors_count', 'is_recid', 'c_charge_degree']]

# cv_grid_transformed = GridSearchCV(LogisticRegression(), param_grid = {
# #     'penalty' : ['l2'],
# #     'C' : [0.5, 1, 1.5],
# #     'class_weight' : [None, 'balanced'],
# #     'max_iter' : [100000]},
# #     n_jobs=-1,
# #     scoring='accuracy')

#rod = make_scorer(ROD, greater_is_better=False, needs_proba=True)



features2_drop = ['race', 'age_cat', 'c_charge_degree']
preprocessor_transformed = ColumnTransformer(
    transformers=[
        ('drop_cat', 'drop', features2_drop,)], remainder='passthrough')

transformed_pipeline = Pipeline(steps=[('preprocessor', preprocessor_transformed),
                                ('scaler', MinMaxScaler()),
                                #('feature_selection',
                                #RFECV(estimator=LogisticRegression(penalty = 'l2', C = 1, solver = 'lbfgs',
                                #                                        #class_weight = 'balanced',
                                #                                    max_iter = 100000), step=0.3, cv=2, scoring='accuracy',
                                #                                     n_jobs=-1, min_features_to_select=10)),
                                 #SelectKBest(chi2, k=1000)),
                                #,('clf', cv_grid_transformed)
                                ('clf', RandomForestClassifier())])

cv_grid_transformed = GridSearchCV(transformed_pipeline, param_grid = {
    'clf__n_estimators' : [500],#,
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5], #,
    #'clf__ccp_alpha' : [None, 0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

categorical_features = ['race', 'age_cat', 'c_charge_degree']
numerical_features = ['priors_count']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)])

original_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('clf', RandomForestClassifier())])

# cv_grid_original = GridSearchCV(original_pipeline, param_grid = {
#     'clf__penalty' : ['l2'],
#     'clf__C' : [0.5, 1, 1.5],
#     'clf__class_weight' : [None, 'balanced'],
#     'clf__max_iter' : [100000]},
#     n_jobs=-1,
#     scoring='accuracy')

cv_grid_original = GridSearchCV(original_pipeline, param_grid = {
    'clf__n_estimators' : [100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

categorical_features_2 = ['age_cat', 'c_charge_degree']
numerical_features_2 = ['priors_count']

categorical_transformer_2 = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer_2 = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor_2 = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer_2, categorical_features_2),
        ('num', numerical_transformer_2, numerical_features_2)])

dropped_pipeline = Pipeline(steps=[('preprocessor', preprocessor_2),
                      ('clf', RandomForestClassifier())])

# cv_grid_dropped = GridSearchCV(dropped_pipeline, param_grid = {
#     'clf__penalty' : ['l2'],
#     'clf__C' : [0.5, 1, 1.5],
#     'clf__class_weight' : [None, 'balanced'],
#     'clf__max_iter' : [100000]},
#     n_jobs=-1,
#     scoring='accuracy')

cv_grid_dropped = GridSearchCV(dropped_pipeline, param_grid = {
    'clf__n_estimators' : [100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

categorical_features_3 = ['race', 'age_cat', 'c_charge_degree']
numerical_features_3 = ['priors_count']

categorical_transformer_3 = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer_3 = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor_3 = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer_3, categorical_features_3),
        ('num', numerical_transformer_3, numerical_features_3)])

capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
                      ('clf', RandomForestClassifier())])

# cv_grid_capuchin = GridSearchCV(capuchin_pipeline, param_grid = {
# #     'clf__penalty' : ['l2'],
# #     'clf__C' : [0.5, 1, 1.5],
# #     'clf__class_weight' : [None, 'balanced'],
# #     'clf__max_iter' : [100000]},
# #     n_jobs=-1,
# #     scoring='accuracy')

cv_grid_capuchin = GridSearchCV(capuchin_pipeline, param_grid = {
    'clf__n_estimators' : [100],#,
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

kf1 = KFold(n_splits=5, shuffle=True)

count = 0
method_list = []
for train_index, test_index in kf1.split(COMPAS):
    print('start with fold: %s' % str(count+1))
    train_df = COMPAS.iloc[train_index]
    test_df = COMPAS.iloc[test_index]

    COMPAS_train_transformed = train_df.loc[:, ['race', 'age', 'age_cat', 'priors_count', 'c_charge_degree']]
    COMPAS_test_transformed = test_df.loc[:, ['race', 'age', 'age_cat', 'priors_count', 'c_charge_degree']]
    COMPAS_train_original = train_df.loc[:, ['race', 'age_cat', 'priors_count', 'c_charge_degree']]
    COMPAS_test_original = test_df.loc[:, ['race', 'age_cat', 'priors_count', 'c_charge_degree']]
    COMPAS_train_dropped = train_df.loc[:, ['age_cat', 'priors_count', 'c_charge_degree']]
    COMPAS_test_dropped = test_df.loc[:, ['age_cat', 'priors_count', 'c_charge_degree']]

    train_df_capuchin = repair_dataset(train_df, sensitive_attribute='race',
                         admissible_attributes=['age_cat', 'priors_count', 'c_charge_degree'], target='is_recid')

    COMPAS_train_capuchin = train_df_capuchin.loc[:, ['race', 'age_cat', 'priors_count', 'c_charge_degree']]
    y_train_capuchin = train_df_capuchin.loc[:, ['is_recid']].to_numpy()

    X_train_t = train_df.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']].to_numpy()
    y_train = train_df.loc[:, ['is_recid']].to_numpy()
    X_test_t = test_df.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']].to_numpy()
    y_test = test_df.loc[:, ['is_recid']].to_numpy()

    for k, v in cost_2_unary_transformed.items():
        for c in v:
            COMPAS_train_transformed[c.get_name()] = c.pipeline.transform(X_train_t)
            COMPAS_test_transformed[c.get_name()] = c.pipeline.transform(X_test_t)

    # COMPAS_train_transformed = my_pipeline.fit_transform(X_train, y_train)
    # COMPAS_test_transformed = my_pipeline.fit_transform(X_test, y_test)

    for k, v in cost_2_binary_transformed.items():
        for c in v:
            COMPAS_train_transformed[c.get_name()] = c.pipeline.transform(X_train_t)
            COMPAS_test_transformed[c.get_name()] = c.pipeline.transform(X_test_t)

    context_features = ['age_cat', 'priors_count', 'c_charge_degree']
    sensitive_feature = 'race'
    context_indices = []
    for i in context_features:
        context_indices.extend([list(COMPAS_train_transformed).index(i)])

    sensitive_index = list(COMPAS_train_transformed).index(sensitive_feature)

    X_train = COMPAS_train_transformed

    fair_train = make_scorer(ROD, greater_is_better=False, needs_proba=True,
                            sensitive_data=(X_train.loc[:, 'race']).to_numpy(),
                            contexts=(X_train.loc[:, ['age_cat', 'priors_count', 'c_charge_degree']]).to_numpy(),
                            protected='African-American')

    f1_train = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

    #cv_grid_transformed.scoring = fair_train
    #cv_grid_transformed.scoring = f1_train
    transformed = cv_grid_transformed.fit(COMPAS_train_transformed, np.ravel(y_train)) ###### PROBLEM WITH THE INDICES WHEN PASSED TO ROD!!!!!
    # select_indices = transformed.named_steps['feature_selection'].transform(
    #     np.arange(len(COMPAS_train_transformed.columns)).reshape(1, -1)
    # )
    # feature_names = COMPAS_train_transformed.columns[select_indices]

    #cv_grid_original.scoring = fair_train
    #cv_grid_original.scoring = f1_train
    original = cv_grid_original.fit(COMPAS_train_original, np.ravel(y_train))
    #cv_grid_dropped.scoring = fair_train
    #cv_grid_dropped.scoring = f1_train
    dropped = cv_grid_dropped.fit(COMPAS_train_dropped, np.ravel(y_train))
    #cv_grid_capuchin.scoring = fair_train
    #cv_grid_capuchin.scoring = f1_train
    capuchin = cv_grid_capuchin.fit(COMPAS_train_capuchin, np.ravel(y_train_capuchin))

    y_pred_original = original.predict(COMPAS_test_original)
    y_pred_proba_original = original.predict_proba(COMPAS_test_original)[:,1]
    y_pred_dropped = dropped.predict(COMPAS_test_dropped)
    y_pred_proba_dropped = dropped.predict_proba(COMPAS_test_dropped)[:,1]
    y_pred_transformed = transformed.predict(COMPAS_test_transformed)
    y_pred_proba_transformed = transformed.predict_proba(COMPAS_test_transformed)[:,1]
    y_pred_capuchin = capuchin.predict(COMPAS_test_original)
    y_pred_proba_capuchin = capuchin.predict_proba(COMPAS_test_original)[:,1]

    contexts = test_df.loc[:, ['age_cat', 'priors_count', 'c_charge_degree']]
    sensitive = np.squeeze(test_df['race'].to_numpy())

    rod_transformed = ROD(y_pred=y_pred_proba_transformed, sensitive=test_df.loc[:, ['race']], admissible = contexts,
                      protected='African-American', name='transformed_COMPAS')
    rod_original =ROD(y_pred=y_pred_proba_original, sensitive=test_df.loc[:, ['race']], admissible = contexts,
                      protected='African-American', name='transformed_COMPAS')
    rod_dropped = ROD(y_pred=y_pred_proba_dropped, sensitive=test_df.loc[:, ['race']], admissible = contexts,
                      protected='African-American', name='transformed_COMPAS')
    rod_capuchin = ROD(y_pred=y_pred_proba_capuchin, sensitive=test_df.loc[:, ['race']], admissible = contexts,
                      protected='African-American', name='transformed_COMPAS')

    acc_transformed = accuracy_score(np.ravel(y_test), y_pred_transformed)
    f1_transformed = f1_score(np.ravel(y_test), y_pred_transformed)
    acc_original = accuracy_score(np.ravel(y_test), y_pred_original)
    f1_original = f1_score(np.ravel(y_test), y_pred_original)
    acc_dropped = accuracy_score(np.ravel(y_test), y_pred_dropped)
    f1_dropped = f1_score(np.ravel(y_test), y_pred_dropped)
    acc_capuchin = accuracy_score(np.ravel(y_test), y_pred_capuchin)
    f1_capuchin = f1_score(np.ravel(y_test), y_pred_capuchin)

    method_list.extend([['feature_construction', acc_transformed, f1_transformed, rod_transformed, count + 1],
                        ['original', acc_original, f1_original, rod_original, count +1],
                        ['dropped', acc_dropped, f1_dropped, rod_dropped, count+1],
                        ['capuchin', acc_capuchin, f1_capuchin, rod_capuchin, count+1]])

    count += 1

    #print()
    #print('# of Features: {}'.format(feature_names.shape[1]))
    print('Fold: {}'.format(count))
    print("F1 Transformed: {:.4f}".format(f1_transformed))
    print("F1 Original: {:.4f}".format(f1_original))
    print("F1 Dropped: {:.4f}".format(f1_dropped))
    print("F1 Capuchin: {:.4f}".format(f1_capuchin))
    print('_________________')
    print("Accuracy Transformed: {:.4f}".format(acc_transformed))
    print("Accuracy Original: {:.4f}".format(acc_original))
    print("Accuracy Dropped: {:.4f}".format(acc_dropped))
    print("Accuracy Capuchin: {:.4f}".format(acc_capuchin))
    print('_________________')
    print("ROD Transformed: {:.4f}".format(rod_transformed))
    print("ROD Original: {:.4f}".format(rod_original))
    print("ROD Dropped: {:.4f}".format(rod_dropped))
    print("ROD Capuchin: {:.4f}".format(rod_capuchin))
    print('_________________________________________________')

summary_df = pd.DataFrame(method_list, columns=['Method', 'Accuracy', 'F1', 'ROD', 'fold'])

print(summary_df.groupby('Method')['Accuracy'].mean())
print(summary_df.groupby('Method')['ROD'].mean())

summary_df.to_csv(path_or_buf=results_path + '/summary_rf500F1_df.csv', index=False)