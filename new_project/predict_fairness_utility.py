import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import re
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import sklearn

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/feature_analysis_p2_c4_dsep.csv')

#results.loc[:, ['name', 'fair_categories', 'types', 'transformations', 'complexity', 'label']].to_html(buf=results_path + '/features_df.html')

transformation_summary = results.groupby(['transformations'])['transformations', 'label'].mean()
transformation_summary.reset_index(inplace=True)


black_list = transformation_summary.loc[transformation_summary['label']==0, 'transformations'].tolist()

def filter(row, black_list=black_list):
    result = False
    if row['transformations'] in black_list:
        result = True
    #elif row['transformations'] == "('-1*', 'GroupByThennanstd')" and row['number_parents'] == 2:
    #    result = True
    elif row['parents_types'] == "('inadmissible',)":
        result = True
    else:
        pass

    return result

results['filter'] = results.apply(filter, axis=1)


X = results.loc[results['filter']==False,
                ['transformations', 'number_parents', 'complexity','number_transformations', 'number_raw_features', 'transformation_depth',
                'parents_types', 'parents_dtypes', 'dtype']]
y = np.ravel(results.loc[results['filter']==False, ['label']].to_numpy())

summary_df = results.loc[:, ['name', 'transformations', 'number_parents', 'complexity',
                                                    'number_transformations', 'number_raw_features', 'transformation_depth',
                                                    'parents_types', 'parents_dtypes', 'dtype', 'label']]

summary_df.to_html(buf=results_path + '/features_df.html')

pd.crosstab(summary_df.transformations, summary_df.number_parents, values=summary_df.label, aggfunc=['count','mean'], margins=True,
                  margins_name='Total').to_html(buf=results_path + '/crosstab_trans_parents.html',
                                                                                        bold_rows=False, na_rep=' ', border=1)

parents_df = pd.crosstab(summary_df.parents_types, summary_df.number_parents, values=summary_df.label,
                  aggfunc=['count', 'mean'], margins=True, margins_name='Total')

parents_df = parents_df.reindex(index=["('no parents',)", "('admissible',)", "('inadmissible',)", "('admissible', 'admissible')",
                          "('inadmissible', 'inadmissible')", "('admissible', 'inadmissible')",
                          "('inadmissible', 'admissible')", 'Total'])

parents_df.to_html(buf=results_path + '/crosstab_fair_pred.html', bold_rows=False, na_rep=' ', border=1)


admissible_explore = summary_df.loc[summary_df['parents_types']=="('admissible',)", ].groupby(['transformations'])['transformations','label'].mean()

admissible_explore.reset_index(inplace=True)

admissible_explore.to_html(buf=results_path + '/admissible_explore.html', bold_rows=False, na_rep=' ', border=1)

inadmissible_explore = summary_df.loc[summary_df['parents_types']=="('inadmissible', 'inadmissible')", ].groupby(['transformations'])['transformations', 'label'].mean()

inadmissible_explore.reset_index(inplace=True)

inadmissible_explore.to_html(buf=results_path + '/inadmissible_explore.html', bold_rows=False, na_rep=' ', border=1)


################### Predict

categorical_features = []
numerical_features = []

for i in list(X):
    if results[i].dtype == np.dtype('O'):
        categorical_features.extend([i])
    elif results[i].dtype != np.dtype('O'):
        numerical_features.extend([i])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)], remainder='passthrough')

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('clf', RandomForestClassifier())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#X_train['transformation'].fillna('None', inplace=True)
#X_test['transformation'].fillna('None', inplace=True)


#model = cv_grid.fit(X_train, y_train)

#predictions = model.predict(X_test)

#f1 = f1_score(y_test, predictions)

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())

for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train, scoring='f1')
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
            print(key, ' mean ', values.mean())
            print(key, ' std ', values.std())

# pipeline.set_params(clf= SVC())
#
# cv_grid_SVC = GridSearchCV(pipeline, param_grid = {
#     'clf__C' : [0.2, 0.5, 1.0, 1.5],
#     'clf__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
#     'clf__shrinking' : [True, False],
#     'clf__class_weight' : [None, 'balanced']
#     },
#     n_jobs=-1,
#     scoring='f1')
#
# cv_grid_SVC.fit(X_train, y_train)
# print(cv_grid_SVC.best_params_)
# print('Best score for SVC: {:.4f}'.format(cv_grid_SVC.best_score_))
# y_predict = cv_grid_SVC.predict(X_test)
# accuracy = accuracy_score(y_test, y_predict)
# f1 = f1_score(y_test, y_predict)
# #print('Accuracy of SVC after CV is %.3f%%' % (accuracy*100))
# print('F1 of SVC after CV is %.3f%%' % (f1*100))
# print('Acc of SVC after CV is %.3f%%' % (accuracy*100))
#print('Support Vectors: {}'.format(cv_grid_SVC.best_estimator_.named_steps['clf'].support_vectors_))

pipeline.set_params(clf= RandomForestClassifier())

cv_grid_RF = GridSearchCV(pipeline, param_grid = {
    'clf__n_estimators': [50, 100, 150],
    'clf__criterion': ['gini', 'entropy'],
    'clf__class_weight': [None, 'balanced'],
    'clf__max_depth': [3, 5, 7, 9, 11],
    'clf__max_features': [0.2, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='f1')

cv_grid_RF.fit(X_train, y_train)
print(cv_grid_RF.best_params_)
print('Best score for RF: {:.4f}'.format(cv_grid_RF.best_score_))
y_predict_RF = cv_grid_RF.predict(X_test)
y_predict_RF_proba = cv_grid_RF.predict_proba(X_test)[:, 1]
accuracy_RF = accuracy_score(y_test, y_predict_RF)
f1_RF = f1_score(y_test, y_predict_RF)
#print('Accuracy of SVC after CV is %.3f%%' % (accuracy*100))
print('F1 of RF after CV is %.3f%%' % (f1_RF*100))
print('Acc of RF after CV is %.3f%%' % (accuracy_RF*100))


feature_importances = cv_grid_RF.best_estimator_.named_steps['clf'].feature_importances_
feature_names = cv_grid_RF.best_estimator_.named_steps['preprocessor'].transformers_[0][1]['onehot'].get_feature_names(categorical_features)
feature_names = np.r_[feature_names, np.array(numerical_features)]

important = feature_importances[feature_importances < 0.001].shape[0]
sorted_idx = feature_importances.argsort()[important:]

fpr, tpr, _ = roc_curve(y_test, y_predict_RF_proba)



for i in _:
    new_y_pred_RF = np.where(y_predict_RF_proba <= i, 0, 1)
    new_F1 = f1_score(y_test, new_y_pred_RF)
    print(i, new_F1)

new_y_pred_RF = np.where(y_predict_RF_proba < 0.51, 0, 1)
new_F1 = f1_score(y_test, new_y_pred_RF)

roc_auc = auc(fpr, tpr)

print(_)
print(fpr)
print(tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.axvline(x=fpr[21], color='red')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for RFC')
plt.legend(loc="lower right")
plt.show()


# y_ticks = np.arange(0, len(feature_names[sorted_idx]))
# fig, ax = plt.subplots()
# ax.barh(y_ticks, feature_importances[sorted_idx])
# ax.set_yticklabels(feature_names[sorted_idx])
# ax.set_yticks(y_ticks)
# ax.set_title("Random Forest Classifier Feature Importances")
# fig.tight_layout()
# plt.show()

pipeline.set_params(clf= GradientBoostingClassifier())

cv_grid_GBC = GridSearchCV(pipeline, param_grid = {
    'clf__loss' : ['deviance'],
    'clf__learning_rate' : [0.05, 0.1, 0.3, 0.5],
    'clf__n_estimators' : [50, 100, 150],
    'clf__max_depth' : [3, 5, 7],
    'clf__max_features' : [0.2, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='f1',
                           cv=10)

cv_grid_GBC.fit(X_train, y_train)
print(cv_grid_GBC.best_params_)
print('Best score for GBC: {:.4f}'.format(cv_grid_GBC.best_score_))

y_predict = cv_grid_GBC.predict(X_test)
accuracy_GBC = accuracy_score(y_test, y_predict)
f1_GBC = f1_score(y_test, y_predict)
print('F1 of GBC after CV is %.3f%%' % (f1_GBC*100))
print('Accuracy of GBC after CV is %.3f%%' % (accuracy_GBC*100))
# feature_importances = cv_grid_GBC.best_estimator_.named_steps['clf'].feature_importances_
# feature_names = cv_grid_GBC.best_estimator_.named_steps['preprocessor'].transformers_[0][1]['onehot'].get_feature_names(categorical_features)
# feature_names = np.r_[feature_names, np.array(numerical_features)]
#
# important = feature_importances[feature_importances < 0.02].shape[0]
# sorted_idx = feature_importances.argsort()[important:]
#
#
# y_ticks = np.arange(0, len(feature_names[sorted_idx]))
# fig, ax = plt.subplots()
# ax.barh(y_ticks, feature_importances[sorted_idx])
# ax.set_yticklabels(feature_names[sorted_idx])
# ax.set_yticks(y_ticks)
# ax.set_title("Gradient Boosting Classifier Feature Importances")
# fig.tight_layout()
# plt.show()