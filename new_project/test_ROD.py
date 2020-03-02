import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import ROD
home = str(Path.home())

###### Define path where dataset lives
adult_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'
adult_df = pd.read_csv(adult_path + '/adult.csv', sep=';', header=0)

######### Some functions needed for data formatting
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


###### Definition of feature categories
sensitive_feature = 'sex'
inadmissible_features = ['marital-status']
target = 'target'
adult_df['target'] = adult_df.apply(lambda row: label(row), axis=1)

#### Other sensitive features are dropped. CAPUCHIN did the same. 'education-num' is a duplicate from education
adult_df.drop(columns=['class', 'relationship', 'race', 'native-country', 'fnlwgt', 'education-num'], inplace=True)

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
    'clf__n_estimators' : [100]#,
    #'clf__criterion' : ['gini', 'entropy'],
    #'clf__class_weight' : [None, 'balanced'],
    #'clf__max_depth' : [None, 3, 5] #,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

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
    'clf__n_estimators' : [100]#,
    #'clf__criterion' : ['gini', 'entropy'],
    #'clf__class_weight' : [None, 'balanced'],
    #'clf__max_depth' : [None, 3, 5] #,
    #'clf__ccp_alpha' : [0.0, 0.5, 1.0]
    },
    n_jobs=-1,
    scoring='accuracy')

#######################

X = adult_df.loc[:, ['workclass', 'education', 'sex', 'marital-status', 'occupation', 'age', 'capital-gain',
                            'capital-loss', 'hours-per-week']]
y = adult_df.loc[:, 'target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train_dropped = X_train.drop(columns = ['sex', 'marital-status'])
X_test_dropped = X_test.drop(columns = ['sex', 'marital-status'])

original = cv_grid_original.fit(X_train, np.ravel(y_train))
dropped = cv_grid_dropped.fit(X_train_dropped, np.ravel(y_train))

outcome_dropped = dropped.predict(X_test_dropped)
y_pred_proba_dropped = dropped.predict_proba(X_test_dropped)[:, 1]
outcome_original = original.predict(X_test)
y_pred_proba_original = original.predict_proba(X_test)[:, 1]

########## Bins are necessary for conditioning
admissible_df = X_test_dropped

######## Compute ROD & Accuracy
rod_dropped = ROD.ROD(y_pred=y_pred_proba_dropped, sensitive=X_test.loc[:, ['sex']], admissible = admissible_df,
                      protected=' Female', name='dropped_adult')
rod_original = ROD.ROD(y_pred=y_pred_proba_original, sensitive=X_test.loc[:, ['sex']], admissible = admissible_df,
                      protected=' Female', name='original_adult')

acc_dropped = accuracy_score(np.ravel(y_test), outcome_dropped)
acc_original = accuracy_score(np.ravel(y_test), outcome_original)

######## Print results

print('ROD dropped: {:.4f}'.format(rod_dropped))
print('ROD orginal: {:.4f}'.format(rod_original))
print('ACC dropped: {:.4f}'.format(acc_dropped))
print('ACC orginal: {:.4f}'.format(acc_original))