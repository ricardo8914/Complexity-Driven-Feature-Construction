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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
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
    'clf__n_estimators' : [100],
    'clf__criterion' : ['gini', 'entropy'],
    'clf__class_weight' : [None, 'balanced'],
    'clf__max_depth' : [None, 3, 5]#,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

##############################################

X = adult_df.loc[:, categorical]
y = adult_df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

to_repair = pd.concat([X_train, pd.DataFrame(y_train, columns=['target'])], axis=1)
binned_x_train = generate_binned_df(to_repair)
train_repaired = capuchin_repair_pipeline.fit_transform(to_repair)

print(train_repaired.groupby('sex')['target'].mean())
print(to_repair.groupby('sex')['target'].mean())

X_train_capuchin = train_repaired.loc[:, ['workclass', 'education', 'sex', 'marital-status', 'occupation', 'age', 'capital-gain',
                            'capital-loss', 'hours-per-week']]
y_train_capuchin = train_repaired.loc[:, 'target'].to_numpy()

capuchin = capuchin_pipeline.fit(X_train_capuchin, np.ravel(y_train_capuchin))

outcome = capuchin.predict(generate_binned_df(X_test))
y_pred_proba_capuchin = capuchin.predict_proba(generate_binned_df(X_test))[:, 1]


X_test_dropped = X_test.drop(columns=['sex', 'marital-status'])
admissible_df = generate_binned_df(X_test_dropped)
acc_capuchin = accuracy_score(np.ravel(y_test.to_numpy()), outcome)
rod_capuchin = ROD.ROD(y_pred=y_pred_proba_capuchin, sensitive=X_test.loc[:, ['sex']], admissible = admissible_df,
                      protected=' Female',
                      name='capuchin_adult')


#print(X_train_capuchin.head)
#print(generate_binned_df(X_test).head)
print(str(acc_capuchin), str(rod_capuchin))

