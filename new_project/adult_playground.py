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
import statistics
import sys
sys.path.insert(0,'/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from measures.ROD import ROD

home = str(Path.home())


adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'

adult_df = pd.read_csv(adult_path + '/adult.csv', sep=';', header=0)

def label(row):
   if row['class'] == ' <=50K' :
      return 0
   else:
       return 1

#print(list(adult_df))

sensitive_feature = 'sex'
inadmissible_features = ['marital-status']
target = 'target'
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)
adult_df.drop(columns=['class'], inplace=True)
print(adult_df.head)

features2_build = []
for i in list(adult_df):
    if i not in inadmissible_features and i != sensitive_feature and i != target:
        features2_build.extend([i])

features2_build_mask = []
for i in features2_build:
    if adult_df[i].dtype == np.dtype('O'):
        features2_build_mask.extend([True])
    else:
        features2_build_mask.extend([False])

acc = make_scorer(accuracy_score, greater_is_better=True, needs_threshold=False)
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
column_transformation = Pipeline([('new_construction', ConstructionTransformer(c_max=2,max_time_secs=10000, scoring=acc, n_jobs=4, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=features2_build,
                                                    feature_is_categorical=features2_build_mask))])

transformed_pipeline = Pipeline(steps=[
                                ('feature_construction', column_transformation),
                                ('clf', RandomForestClassifier())])

cv_grid_transformed = GridSearchCV(transformed_pipeline, param_grid = {
    'clf__n_estimators' : [100],#,
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5], #,
    #'clf__ccp_alpha' : [None, 0.0, 0.5, 1.0]
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

    X_train = train_df.loc[:, features2_build].to_numpy()
    y_train = train_df.loc[:, target].to_numpy()
    X_test = test_df.loc[:, features2_build].to_numpy()
    y_test = test_df.loc[:, target].to_numpy()

    transformed = transformed_pipeline.fit(X_train, y_train)

    y_pred_transformed = transformed.predict(X_test)
    y_pred_proba_transformed = transformed.predict_proba(X_test)[:,1]

    contexts = test_df.loc[:, features2_build].to_numpy()
    sensitive = test_df.loc[:, sensitive_feature].to_numpy()

    rod_transformed = ROD(np.ravel(y_test), pd.DataFrame(y_pred_proba_transformed), sensitive, contexts,
                          protected='Female')

    acc_transformed = accuracy_score(np.ravel(y_test), y_pred_transformed)
    f1_transformed = f1_score(np.ravel(y_test), y_pred_transformed)

    rod.extend([rod_transformed])
    acc.extend([acc_transformed])
    f1.extend(([f1_transformed]))

    print('Fold: {}'.format(count))
    print("F1 Transformed: {:.4f}".format(f1_transformed))
    print('_________________')
    print("Accuracy Transformed: {:.4f}".format(acc_transformed))
    print('_________________')
    print("ROD Transformed: {:.4f}".format(rod_transformed))
    print('_________________________________________________')

print('Mean ROD: {:.4f}'.format(statistics.mean(rod)))
print('Mean ACC: {:.4f}'.format(statistics.mean(acc)))
print('Mean F1: {:.4f}'.format(statistics.mean(f1)))