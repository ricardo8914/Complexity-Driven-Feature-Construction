from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from pathlib import Path
import pandas as pd
home = str(Path.home())

auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
recall = make_scorer(recall_score, greater_is_better=True, needs_threshold=False)
precision = make_scorer(precision_score, greater_is_better=True, needs_threshold=False )
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
my_pipeline = Pipeline([('new_construction', ConstructionTransformer(c_max=5, scoring=f1, n_jobs=4, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=10, epsilon=-np.inf,
                                                    feature_names=['race', 'age', 'priors_count', 'c_charge_degree'],
                                                    feature_is_categorical=[True, False, False, True]))])

COMPAS = pd.read_csv(home + '/Finding-Fair-Representations-Through-Feature-Construction/data' + '/compas-analysis/compas-scores.csv')

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) & (COMPAS['is_recid'] != -1) & (COMPAS['race'].isin(['African-American','Caucasian']))
                     & (COMPAS['c_charge_degree'].isin(['F','M']))
                     , ['race', 'age', 'priors_count', 'is_recid', 'c_charge_degree']]

X = COMPAS.loc[:, ['race','age', 'priors_count', 'c_charge_degree']].to_numpy()
y = COMPAS.loc[:, ['is_recid']].to_numpy()

my_pipeline.fit(X, y)