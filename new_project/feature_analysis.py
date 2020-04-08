from sklearn.feature_selection import RFE
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0,'/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from measures.ROD import ROD

home = str(Path.home())

COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'
path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/feature_construction/tmp'
results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
                    (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
                    & (COMPAS['is_recid'] != -1)
                    & (COMPAS['race'].isin(['African-American','Caucasian']))
                    & (COMPAS['c_charge_degree'].isin(['F','M']))
                    , ['race','age', 'age_cat', 'priors_count','is_recid','c_charge_degree']]

cost_2_raw_features: Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed : Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed: Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination : Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))

cv_grid_transformed = GridSearchCV(LogisticRegression(), param_grid = {
    'penalty' : ['l2'],
    'C' : [0.5, 1, 1.5],
    'class_weight' : [None, 'balanced'],
    'max_iter' : [100000]},
    n_jobs=-1,
    scoring='accuracy')

feature_estimator = SVR(kernel="linear")

transformed_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                                ('feature_selection',
                                RFE(feature_estimator, 5, step=0.5)),
                                 #SelectKBest(chi2, k=10)),
                                ('clf', cv_grid_transformed)])

# kf1 = KFold(n_splits=5, shuffle=True)

#count = 0
results = []
# for train_index, test_index in kf1.split(COMPAS):

X_train, X_test, y_train, y_test = train_test_split(COMPAS.loc[:, ['race', 'age', 'age_cat', 'priors_count', 'c_charge_degree']],
                                                    COMPAS.loc[:, ['is_recid']], test_size=0.33, random_state=42)

X_train_t = X_train.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']]
X_test_t = X_test.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']]

COMPAS_train_transformed = X_train.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']]
COMPAS_test_transformed = X_test.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']]

for k, v in cost_2_unary_transformed.items():
    for c in v:
        COMPAS_train_transformed[c.get_name()] = c.pipeline.transform(X_train_t.to_numpy())
        COMPAS_test_transformed[c.get_name()] = c.pipeline.transform(X_test_t.to_numpy())


for k, v in cost_2_binary_transformed.items():
    for c in v:
        COMPAS_train_transformed[c.get_name()] = c.pipeline.transform(X_train_t.to_numpy())
        COMPAS_test_transformed[c.get_name()] = c.pipeline.transform(X_test_t.to_numpy())







COMPAS_train_transformed.drop(columns=['age_cat', 'c_charge_degree'], inplace=True)
COMPAS_test_transformed.drop(columns=['age_cat', 'c_charge_degree'], inplace=True)


for i in range(4, COMPAS_train_transformed.shape[1], 50):

    print('Features: %s' % str(i))
    transformed_pipeline.set_params(feature_selection__n_features_to_select = i).fit(COMPAS_train_transformed, np.ravel(y_train.to_numpy()))
    y_pred = transformed_pipeline.predict(COMPAS_test_transformed)
    y_pred_proba = transformed_pipeline.predict_proba(COMPAS_test_transformed)[:, 1]

    contexts = X_test.loc[:, ['age_cat', 'priors_count', 'c_charge_degree']].to_numpy()
    sensitive = np.squeeze(X_test['race'].to_numpy())

    rod = ROD(y_pred_proba, sensitive, contexts, protected='African-American')
    acc = accuracy_score(np.ravel(y_test), y_pred)

    results.extend([[i, rod, acc]])


results_df = pd.DataFrame(results, columns=['number_of_features', 'ROD', 'ACC'])
results_df.to_csv(path_or_buf=results_path + '/feature_analysis.csv', index=False)

print(results_df)