from pathlib import Path
import pandas as pd
import numpy as np
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import statistics
import sys
sys.path.insert(0,'/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from measures.ROD import ROD
from methods.capuchin import repair_dataset

home = str(Path.home())


adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'

adult_df = pd.read_csv(adult_path + '/adult.csv', sep=';', header=0)

def label(row):
   if row['class'] == ' <=50K' :
      return 0
   else:
       return 1

######################### Feature Construction ####################################

sensitive_feature = 'sex'
inadmissible_features = ['marital-status']
target = 'target'
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class','relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)

features2_build = []
for i in list(adult_df):
    if i not in inadmissible_features and i != sensitive_feature and i != target:
        features2_build.extend([i])

features2_build_cat = []
features2_build_num = []
for i in features2_build:
    if adult_df[i].dtype == np.dtype('O'):
        features2_build_cat.extend([i])
    else:
        features2_build_num.extend([i])

features2_scale = []
for i in features2_build:
    if adult_df[i].dtype != np.dtype('O'):
        features2_scale.extend([features2_build.index(i)])
    else:
        pass

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features2_scale)], remainder='passthrough')

#print(preprocessor.fit_transform(adult_df.loc[:, features2_build].to_numpy())[0])

new_order = features2_build_num + features2_build_cat
features2_build_mask = ([False] * len(features2_build_num)) + ([True] * len(features2_build_cat))

acc = make_scorer(accuracy_score, greater_is_better=True, needs_threshold=False)
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
column_transformation = Pipeline([('new_construction', ConstructionTransformer(c_max=3,max_time_secs=10000, scoring=acc, n_jobs=4, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=new_order,
                                                    feature_is_categorical=features2_build_mask))])

cv_grid_transformed = GridSearchCV(RandomForestClassifier(), param_grid = {
    'n_estimators' : [100],#,
    'criterion' : ['gini', 'entropy'],
    'class_weight' : [None, 'balanced'],
    'max_depth' : [None, 3, 5]#,
    #'ccp_alpha' : [None, 0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

transformed_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('feature_construction', column_transformation),
                                #('clf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
                                ('clf', cv_grid_transformed)])

#################### Original#########################

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
    'clf__n_estimators' : [100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

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
    'clf__n_estimators' : [100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

######################### Capuchin ####################################


capuchin_df = adult_df

def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i != target and (df_[i].dtype != np.dtype('O') and len(df_[i].unique()) > 4):

            out, bins = pd.cut(df_[i], bins=4, retbins=True)
            df_['binned_' + i] = out.astype(str)
            columns2_drop.extend([i])

    df_.drop(columns=columns2_drop, inplace=True)

    return df_

categorical_features_3 = []
numerical_features_3 = []

a = generate_binned_df(capuchin_df)

for i in list(a):
    if i != target and i != sensitive_feature and i not in inadmissible_features and (a[i].dtype == np.dtype('O') or a[i].dtype == 'category'):
        categorical_features_3.extend([i])
    elif i != target and i != sensitive_feature and i not in inadmissible_features and (a[i].dtype != np.dtype('O') and a[i].dtype != 'category'):
        numerical_features_3.extend([i])

categorical_transformer_3 = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer_3 = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor_3 = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer_3, categorical_features_3),
        ('num', numerical_transformer_3, numerical_features_3)], remainder='passthrough')

capuchin_repair_pipeline = Pipeline(steps=[('generate_binned_df', FunctionTransformer(generate_binned_df)),
                        ('repair', FunctionTransformer(repair_dataset, kw_args={'admissible_attributes' : ['workclass',
                                                                                                    'education',
                                                                                                    'occupation',
                                                                                                    'binned_age',
                                                                                                    'binned_capital-gain',
                                                                                                    'binned_capital-loss',
                                                                                                    'binned_hours-per-week'],
                                                                                'sensitive_attribute': 'sex',
                                                                                'target': 'target'}))])

capuchin_pipeline = Pipeline(steps=[('preprocessor', preprocessor_3),
                        ('clf', RandomForestClassifier())])


cv_grid_capuchin = GridSearchCV(capuchin_pipeline, param_grid = {
    'clf__n_estimators' : [100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

kf1 = KFold(n_splits=5, shuffle=True)

count = 0
rod = []
acc = []
f1 = []
for train_index, test_index in kf1.split(adult_df):
    print('start with fold: %s' % str(count+1))

    train_df = adult_df.iloc[train_index]
    test_df = adult_df.iloc[test_index]

    X_train_t = train_df.loc[:, features2_build].to_numpy()
    X_test_t = test_df.loc[:, features2_build].to_numpy()
    X_train_original = train_df.loc[:, ['workclass', 'education', 'marital-status', 'occupation', 'sex', 'age', 'capital-gain',
                        'capital-loss', 'hours-per-week']]
    X_test_original = test_df.loc[:,
                       ['workclass', 'education', 'marital-status', 'occupation', 'sex', 'age', 'capital-gain',
                        'capital-loss', 'hours-per-week']]
    X_train_dropped = train_df.loc[:, ['workclass', 'education', 'occupation', 'age', 'capital-gain',
                        'capital-loss', 'hours-per-week']]
    X_test_dropped = test_df.loc[:,
                       ['workclass', 'education', 'occupation', 'age', 'capital-gain',
                        'capital-loss', 'hours-per-week']]
    y_train = train_df.loc[:, target].to_numpy()
    y_test = test_df.loc[:, target].to_numpy()

    train_repaired = capuchin_repair_pipeline.fit_transform(train_df)
    X_train_repaired = train_repaired.loc[:,
                       ['workclass', 'education', 'occupation', 'binned_age', 'binned_capital-gain',
                        'binned_capital-loss', 'binned_hours-per-week']]
    y_train_repaired = train_repaired.loc[:, ['target']].to_numpy()
    X_test_capuchin = (generate_binned_df(test_df)).loc[:,['workclass', 'education', 'occupation',
                                                     'binned_age', 'binned_capital-gain',
                                                    'binned_capital-loss', 'binned_hours-per-week']]

    transformed = transformed_pipeline.fit(X_train_t, np.ravel(y_train))
    capuchin = cv_grid_capuchin.fit(X_train_repaired, np.ravel(y_train_repaired))
    original = cv_grid_original.fit(X_train_original, np.ravel(y_train))
    dropped = cv_grid_dropped.fit(X_train_dropped, np.ravel(y_train))


    y_pred_transformed = transformed.predict(X_test_t)
    y_pred_proba_transformed = transformed.predict_proba(X_test_t)[:,1]
    y_pred_capuchin = capuchin.predict(X_test_capuchin)
    y_pred_proba_capuchin = capuchin.predict_proba(X_test_capuchin)[:,1]
    y_pred_original = original.predict(X_test_original)
    y_pred_proba_original = original.predict_proba(X_test_original)[:, 1]
    y_pred_dropped = dropped.predict(X_test_dropped)
    y_pred_proba_dropped = dropped.predict_proba(X_test_dropped)[:, 1]

    contexts = (generate_binned_df(test_df)).loc[:, ['workclass', 'education', 'occupation', 'binned_age', 'binned_capital-gain',
                                                     'binned_capital-loss', 'binned_hours-per-week']].to_numpy()
    sensitive = (generate_binned_df(test_df)).loc[:, ['sex']].to_numpy()

    rod_transformed = ROD(np.ravel(y_test), pd.DataFrame(y_pred_proba_transformed), sensitive, contexts, protected='Female')
    rod_capuchin = ROD(np.ravel(y_test), pd.DataFrame(y_pred_proba_capuchin), sensitive, contexts, protected='Female')
    rod_original = ROD(np.ravel(y_test), pd.DataFrame(y_pred_proba_original), sensitive, contexts, protected='Female')
    rod_dropped = ROD(np.ravel(y_test), pd.DataFrame(y_pred_proba_dropped), sensitive, contexts, protected='Female')

    acc_transformed = accuracy_score(np.ravel(y_test), y_pred_transformed)
    f1_transformed = f1_score(np.ravel(y_test), y_pred_transformed)
    acc_capuchin = accuracy_score(np.ravel(y_test), y_pred_capuchin)
    f1_capuchin = f1_score(np.ravel(y_test), y_pred_capuchin)
    acc_original = accuracy_score(np.ravel(y_test), y_pred_original)
    f1_original = f1_score(np.ravel(y_test), y_pred_original)
    acc_dropped = accuracy_score(np.ravel(y_test), y_pred_dropped)
    f1_dropped = f1_score(np.ravel(y_test), y_pred_dropped)

    # rod.extend([rod_transformed])
    # acc.extend([acc_transformed])
    # f1.extend(([f1_transformed]))

    count += 1

    print('Fold: {}'.format(count))
    print("F1 Transformed: {:.4f}".format(f1_transformed))
    print("F1 Capuchin: {:.4f}".format(f1_capuchin))
    print("F1 Original: {:.4f}".format(f1_original))
    print("F1 Dropped: {:.4f}".format(f1_dropped))
    print('_________________')
    print("Accuracy Transformed: {:.4f}".format(acc_transformed))
    print("Accuracy Capuchin: {:.4f}".format(acc_capuchin))
    print("Accuracy Original: {:.4f}".format(acc_original))
    print("Accuracy Dropped: {:.4f}".format(acc_dropped))
    print('_________________')
    print("ROD Transformed: {:.4f}".format(rod_transformed))
    print("ROD Capuchin: {:.4f}".format(rod_capuchin))
    print("ROD Original: {:.4f}".format(rod_original))
    print("ROD Dropped: {:.4f}".format(rod_dropped))
    print('_________________________________________________')
