from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,'/Users/ricardosalazar/Finding-Fair-Representations-Through-Feature-Construction/Code')
from measures.ROD import ROD
from fastsklearnfeature.configuration import Config

# c = Config.Config
#
# print(c.load())


home = str(Path.home())

auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
recall = make_scorer(recall_score, greater_is_better=True, needs_threshold=False)
precision = make_scorer(precision_score, greater_is_better=True, needs_threshold=False)
f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)
acc = make_scorer(accuracy_score, greater_is_better=True, needs_threshold=False)
my_pipeline = Pipeline([('new_construction', ConstructionTransformer(c_max=5, scoring=f1, n_jobs=4, model=LogisticRegression(),
                                                       parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'],
                                                                       'class_weight': ['balanced'], 'max_iter': [100000],
                                                                       'multi_class':['auto']}, cv=5, epsilon=-np.inf,
                                                    feature_names=['age', 'age_cat', 'priors_count', 'c_charge_degree'],
                                                    feature_is_categorical=[False, True, False, True]))])

COMPAS = pd.read_csv(home + '/Finding-Fair-Representations-Through-Feature-Construction/data' + '/compas-analysis/compas-scores.csv')

COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
                    (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
                    & (COMPAS['is_recid'] != -1)
                    & (COMPAS['race'].isin(['African-American','Caucasian']))
                    & (COMPAS['c_charge_degree'].isin(['F','M']))
                    , ['race', 'age', 'age_cat', 'priors_count','is_recid','c_charge_degree']]

X = COMPAS.loc[:, ['age', 'age_cat', 'priors_count', 'c_charge_degree']].to_numpy()
y = COMPAS.loc[:, ['is_recid']].to_numpy()

t = my_pipeline.fit_transform(X, y)
print(t.shape)
