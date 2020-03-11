from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.factory import get_crossover, get_sampling, get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from pymoo.visualization.scatter import Scatter
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

class MaskSelection(BaseEstimator, SelectorMixin):
    def __init__(self, mask):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def _get_support_mask(self):
        return self.mask

def evolution(X_train, y_train, scorers=[], cv_splitter=5, max_search_time=60):

	def f_clf1(mask):
		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', LogisticRegression())
		])
		return model

	# define an objective function
	def objective(features):
		if np.sum(features) == 0:
			return [0.0, 0.0, 0.0, 0.0]

		model = f_clf1(features)

		scorer_mapping = {}
		for scorer_i in range(len(scorers)):
			scorer_mapping['scorer' + str(scorer_i)] = scorers[scorer_i]


		cv = GridSearchCV(model, param_grid={'clf__C': [1.0]}, cv=cv_splitter,
						  scoring=scorer_mapping,
						  refit=False)
		cv.fit(X_train, pd.DataFrame(y_train))
		results = []

		for scorer_i in range(len(scorers)):
			results.append(cv.cv_results_['mean_test_' + 'scorer' + str(scorer_i)][0])

		return results

	class MyProblem(Problem):

		def __init__(self):
			number_objectives = len(scorers)

			super().__init__(n_var=X_train.shape[1],
							 n_obj=number_objectives,
							 n_constr=0,
							 xl=0, xu=1, type_var=anp.bool)

		def _evaluate(self, x, out, *args, **kwargs):
			result_list = []
			for i in range(self.n_obj):
				result_list.append([])

			for i in range(len(x)):
				results = objective(x[i])
				for o in range(self.n_obj):
					result_list[o].append(results[o] * -1)

			out["F"] = anp.column_stack(result_list)

	problem = MyProblem()



	population_size = 100
	cross_over_rate = 0.9
	algorithm = NSGA2(pop_size=population_size,
					  sampling=get_sampling("bin_random"),
					  crossover=get_crossover('bin_one_point'),
					  # get_crossover("bin_hux"),#get_crossover("bin_two_point"),
					  mutation=BinaryBitflipMutation(1.0 / X_train.shape[1]),
					  elimate_duplicates=True,
					  # n_offsprings= cross_over_rate * population_size
					  )

	#res = minimize(problem, algorithm, ('n_gen', 10), disp=False)
	res = minimize(problem, algorithm, get_termination("n_eval", 5), disp=False)

	return res.X

	#print(res.F)

	#Scatter().add(res.F).show()
	#plt.show()


from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=1)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
f1_scorer = make_scorer(f1_score)
# # #fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
# #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

a = evolution(X_train, y_train, scorers=[auc_scorer, f1_scorer], cv_splitter=5, max_search_time=1000)

print(a)