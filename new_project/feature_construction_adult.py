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
import sys
sys.path.insert(0,'/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from measures.ROD import ROD
from methods.capuchin import repair_dataset

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

cv_grid_transformed_train = GridSearchCV(RandomForestClassifier(), param_grid = {
    'n_estimators' : [250],#,
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

X = adult_df.loc[:, features2_build].to_numpy()
y = adult_df.loc[:, 'target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

transformed_train = transformed_pipeline.fit_transform(X_train, np.ravel(y_train))
all_transformations = transformed_pipeline.named_steps['feature_construction'].named_steps['new_construction'].all_features_set
transformed_test = transformed_pipeline.transform(X_test)

print(len(all_transformations))
print(transformed_train.shape, transformed_test.shape)

#rf = RandomForestClassifier(n_estimators=250, n_jobs=-1)
cv_grid_transformed_train.fit(transformed_train, np.ravel(y_train))

result = permutation_importance(cv_grid_transformed_train, transformed_train, np.ravel(y_train), n_repeats=5, scoring='accuracy', n_jobs=-1)
#trunc_result = result.importances_mean[:len(all_transformations)]

print(result.importances_mean)
sorted_idx = result.importances_mean.argsort()
print(sorted_idx)

best_10 = [all_transformations[i] for i in sorted_idx][-20:]
best_10_idx = sorted_idx[-20:]
#print(best_10_idx)
X_test_trunc = transformed_test[:,best_10_idx]

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

X_train_test, X_test_test, y_train_test, y_test_test = train_test_split(X_test_trunc, y_test, test_size=0.33)

cv_grid_transformed.fit(X_train_test, np.ravel(y_train_test))

y_pred = cv_grid_transformed.predict(X_test_test)

acc = accuracy_score(np.ravel(y_test_test), y_pred)

print('Accuracy for model: {}'.format(best_10))
print('{:.4f}'.format(acc))


# fig, ax = plt.subplots()
# ax.boxplot(result.importances[sorted_idx].T[:,-10:],
#            vert=False, labels=best_10)
#
# plt.show()