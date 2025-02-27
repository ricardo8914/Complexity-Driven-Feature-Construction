import pandas as pd
from pathlib import Path
from causality.d_separation import d_separation
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from sklearn.metrics import log_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import ROD
import sys
sys.path.insert(0, '/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')

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

X = adult_df.loc[:, ['workclass', 'education', 'sex', 'marital-status', 'occupation', 'age', 'capital-gain',
                            'capital-loss', 'hours-per-week']]
y = adult_df.loc[:, target]

################################## Feature Construction

features2_build_2 = []
for i in list(adult_df):
    if i != target:
    #if i != target:
        features2_build_2.extend([i])

features2_build_cat_2 = []
features2_build_num_2 = []

for i in features2_build_2:
    if adult_df[i].dtype == np.dtype('O'):
        features2_build_cat_2.extend([i])
    else:
        features2_build_num_2.extend([i])

features2_scale_2 = []
for i in features2_build_2:
    if adult_df[i].dtype != np.dtype('O'):
        features2_scale_2.extend([features2_build_2.index(i)])
    else:
        pass

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor_t = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features2_scale_2)], remainder='passthrough')

new_order_2 = features2_build_num_2 + features2_build_cat_2
features2_build_mask_2 = ([False] * len(features2_build_num_2)) + ([True] * len(features2_build_cat_2))

acc = make_scorer(accuracy_score, greater_is_better=True, needs_threshold=False)
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

column_transformation_2 = Pipeline([('new_construction', ConstructionTransformer(c_max=3,max_time_secs=10000, scoring=f1, n_jobs=7, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': [None, 'balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=new_order_2,
                                                    feature_is_categorical=features2_build_mask_2))])

transformed_pipeline_2 = Pipeline(steps=[('preprocessor', preprocessor_t),
                                ('feature_construction', column_transformation_2)])

# cv_grid_transformed_2 = LogisticRegression(
# #     penalty =  'l2', C = 1, solver = 'lbfgs',
# #     max_iter = 100000, class_weight='balanced', n_jobs=-1
# # )

cv_grid_transformed_2 = RandomForestClassifier(class_weight='balanced', max_features=1.0, n_jobs=-1)

#########################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train_t = X_train.loc[:, features2_build_2].to_numpy()
X_test_t = X_test.loc[:, features2_build_2].to_numpy()

transformed_train_2 = transformed_pipeline_2.fit_transform(X_train_t, np.ravel(y_train.to_numpy()))
all_transformations_2 = transformed_pipeline_2.named_steps['feature_construction'].named_steps[
        'new_construction'].all_features_set
transformed_test_2 = transformed_pipeline_2.transform(X_test_t)

transformed_columns_2 = []
for i in all_transformations_2:
    j = (i.get_name()).strip()
    transformed_columns_2.extend([j])

selected_idx = []
selected_names = []
size = []
L = []
ll = np.inf
for idx, i in enumerate(tqdm(transformed_columns_2)):
    if i != sensitive_feature:

        test_model = LogisticRegression()

        test_model.fit(transformed_train_2[:, [idx]], np.ravel(y_train.to_numpy()))

        proba = test_model.predict_proba(transformed_train_2[:, [idx]])[:, 1]
        predictions = test_model.predict(transformed_train_2[:, [idx]])
        outcome_df = pd.DataFrame(data=predictions, columns=['outcome'])
        sensitive_df = pd.DataFrame(data=X_train.loc[:, sensitive_feature].to_numpy(), columns=[sensitive_feature])
        selected_df = pd.DataFrame(data=transformed_train_2[:, [idx]], columns=[i])
        test_df = pd.concat([sensitive_df, selected_df, outcome_df], axis=1)

        if d_separation(test_df, sensitive=sensitive_feature, target='outcome'):
            selected_idx.extend([idx])
            selected_names.extend([i])
            test_model.fit(transformed_train_2[:, selected_idx], np.ravel(y_train.to_numpy()))
            proba_i = test_model.predict_proba(transformed_train_2[:, selected_idx])[:, 1]
            logLoss = log_loss(y_true=np.ravel(y_train.to_numpy()), y_pred=proba_i)
            if ll - logLoss >= 0.005:
                ll = logLoss
                size.extend([len(selected_names)])
                L.extend([logLoss])
                print('Selected: {}'.format(selected_names))
                print('Size: {}'.format(str(len(selected_names))))
                print('Log Loss: {:.4f}'.format(logLoss))
            else:
                selected_names.remove(i)
                selected_idx.remove(idx)
        else:
            pass


print(selected_names)

cv_grid_transformed_3 = GridSearchCV(RandomForestClassifier(), param_grid={
        'n_estimators': [100],  # ,
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'max_depth': [None, 3, 5]  # ,
        # 'max_features' : [1.0]
    },
                             n_jobs=-1,
                             scoring='f1')

cv_grid_transformed_3.fit(transformed_train_2[:, selected_idx], np.ravel(y_train.to_numpy()))
proba = cv_grid_transformed_3.predict_proba(transformed_test_2[:, selected_idx])[:, 1]
predictions = cv_grid_transformed_3.predict(transformed_test_2[:, selected_idx])
admissible_feature_construction = pd.DataFrame(data=transformed_test_2[:, selected_idx], columns=selected_names)

rod = ROD.ROD(y_pred=proba, sensitive=X_test.loc[:, ['sex']],
                              admissible=admissible_feature_construction,
                              protected=' Female', name='d_separation_test')

tpr = true_positive_rate_score(y_test.to_numpy(), predictions,
                                               sensitive_data=X_test.loc[:, ['sex']])

f1_ = f1_score(np.ravel(y_test), predictions)

print('ROD: {:.4f}'.format(rod))
print('TPR: {:.4f}'.format(tpr))
print('f1: {:.4f}'.format(f1_))

plt.plot(size, L)
plt.xlabel('Number of features')
plt.ylabel('Log Loss')
plt.show()
