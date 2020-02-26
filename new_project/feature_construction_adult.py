import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from fastsklearnfeature.configuration import Config
import ROD

home = str(Path.home())

# c = Config.Config
#
# print(c.load())

#data_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'

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

            out, bins = pd.cut(df_[i], bins=4, retbins=True)
            df_.loc[:, 'binned_' + i] = out.astype(str)
            columns2_drop.extend([i])

    df_.drop(columns=columns2_drop, inplace=True)

    return df_

sensitive_feature = 'sex'
inadmissible_features = ['marital-status']
target = 'target'
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class','relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)

############################## Feature Construction #############################################

features2_build = []
for i in list(adult_df):
    if i not in inadmissible_features and i != sensitive_feature and i != target:
    #if i != target:
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

cv_grid_transformed_train = GridSearchCV(RandomForestClassifier(), param_grid = {
    'n_estimators' : [100],#,
    'criterion' : ['gini', 'entropy'],
    'class_weight' : [None, 'balanced'],
    'max_depth' : [None, 3, 5],#,
    'max_features' : [1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

transformed_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('feature_construction', column_transformation)])#,
                                #('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
                                #('clf', cv_grid_transformed)])

#########################################

X = adult_df.loc[:, ['workclass', 'education', 'sex', 'marital-status', 'occupation', 'age', 'capital-gain',
                            'capital-loss', 'hours-per-week']]
y = adult_df.loc[:, 'target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train_t = X_train.loc[:, features2_build].to_numpy()
X_test_t = X_test.loc[:, features2_build].to_numpy()

transformed_train = transformed_pipeline.fit_transform(X_train_t, np.ravel(y_train))
all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps['new_construction'].all_features_set
transformed_test = transformed_pipeline.transform(X_test_t)

transformed_columns = []
for i in all_transformations:
    j = (i.get_name()).strip()
    transformed_columns.extend([j])

admissible = pd.DataFrame(transformed_train, columns=transformed_columns)

cv_grid_transformed_train.fit(transformed_train, np.ravel(y_train))

fair_train = make_scorer(ROD.ROD, greater_is_better=False, needs_proba=True,
                         sensitive=X_train.loc[:, ['sex']], admissible=admissible,
                         protected=' Female', name='feature_construction_adult')

result = permutation_importance(cv_grid_transformed_train, transformed_train, np.ravel(y_train), n_repeats=5, scoring='accuracy', n_jobs=-1)

sorted_idx = result.importances_mean.argsort()

best_10 = [all_transformations[i] for i in sorted_idx][-10:]
best_10_idx = sorted_idx[-10:]
#print(best_10_idx)


cv_grid_transformed = GridSearchCV(RandomForestClassifier(), param_grid = {
    'n_estimators' : [100], #,
    'criterion' : ['gini', 'entropy'],
    'class_weight' : [None, 'balanced'],
    'max_depth' : [None, 3, 5],
    'max_features' : [0.1, 0.5, 1.0] #,
    #'ccp_alpha' : [None, 0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

X_train_test, X_test_test, y_train_test, y_test_test = train_test_split(X_test, y_test, test_size=0.33)

X_train_test_t = X_train_test.loc[:, features2_build].to_numpy()
X_test_test_t = X_test_test.loc[:, features2_build].to_numpy()
transformed_train_test = transformed_pipeline.transform(X_train_test_t)
transformed_test_test = transformed_pipeline.transform(X_test_test_t)
X_train_test_trunc = transformed_train_test[:, best_10_idx]
X_test_test_trunc = transformed_test_test[:, best_10_idx]

cv_grid_transformed.fit(X_train_test_trunc, np.ravel(y_train_test))

y_pred = cv_grid_transformed.predict(X_test_test_trunc)
y_pred_proba = cv_grid_transformed.predict_proba(X_test_test_trunc)[:, 1]

columns2_df = []
for i in best_10:
    j = (i.get_name()).strip()
    columns2_df.extend([j])

df = pd.DataFrame(X_test_test_trunc, columns=columns2_df)

rod = ROD.ROD(y_pred=y_pred_proba, sensitive=X_test_test.loc[:, ['sex']], admissible = df,
                      protected=' Female', name='feature_construction_adult')

acc = accuracy_score(np.ravel(y_test_test), y_pred)

print('Accuracy for model: {}'.format(best_10))
print('{:.4f}'.format(acc))
print('ROD for model: {:.4f}'.format(rod))


# fig, ax = plt.subplots()
# ax.boxplot(result.importances[sorted_idx].T[:,-10:],
#            vert=False, labels=best_10)
#
# plt.show()