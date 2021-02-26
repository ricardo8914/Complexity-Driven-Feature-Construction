from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.factory import get_crossover, get_sampling, get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
import math
from random import randrange

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from fmeasures.ROD import ROD
from causality.causal_filter import causal_filter
from sklearn.metrics import f1_score
import multiprocessing as mp
import itertools
from functools import partial


class MaskSelection(BaseEstimator, SelectorMixin):
	def __init__(self, mask):
		self.mask = mask

	def fit(self, X, y=None):
		return self

	def _get_support_mask(self):
		return self.mask


def objective_(df_train, X_train, y_train, sampling, sensitive_feature, sensitive_features, admissible_features,
			  protected, features):
	if np.sum(features) == 0:
		return [0.0, 0.0, 0.0, 0.0]

	if sampling < 1.0:
		sample_idx_all = np.random.randint(X_train.shape[0], size=round(X_train.shape[0] * sampling))
		X_train_ = X_train[sample_idx_all]
		y_train_ = y_train[sample_idx_all]
	else:
		X_train_ = X_train
		y_train_ = y_train

	results = []

	model_ = Pipeline(steps=[
		('selection', MaskSelection(features)),
		('clf', LogisticRegression(n_jobs=-1))
	])
	model_.fit(X_train_, pd.DataFrame(y_train_))
	y_pred_proba_train = model_.predict_proba(X_train)[:, 1]
	y_pred_train = model_.predict(X_train)
	f1_ = f1_score(y_train, y_pred_train)
	results.append(f1_)

	sensitive_df = df_train.loc[:, sensitive_features]
	sensitive_df.reset_index(inplace=True, drop=True)
	outcomes_df = pd.DataFrame(y_pred_proba_train, columns=['outcome'])
	features_df = df_train.loc[:, admissible_features]
	features_df.reset_index(inplace=True, drop=True)

	candidate_df = pd.concat([sensitive_df, features_df, outcomes_df], axis=1)

	JCIT, mb = causal_filter(candidate_df, sensitive_features)

	rod_train = ROD(y_pred=y_pred_proba_train, df=df_train, sensitive=sensitive_feature,
					admissible=admissible_features, protected=protected, mb=mb)

	results.append(rod_train)
	results.append(-np.sum(features))
	return results


def evolution(X_train, y_train, scorers=[], cv_splitter=3, max_search_time=60, df_train=None, sensitive_feature=None,
			protected=None, sensitive_features=None, admissible_features=None, sampling=1.0):
	def f_clf1(mask):
		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', LogisticRegression(n_jobs=-1))
		])
		return model

	# define an objective function
	def objective(features):
		if np.sum(features) == 0:
			return [0.0, 0.0, 0.0, 0.0]

		if sampling < 1.0:
			sample_idx_all = np.random.randint(X_train.shape[0], size=round(X_train.shape[0] * sampling))
			X_train_ = X_train[sample_idx_all]
			y_train_ = y_train[sample_idx_all]
		else:
			X_train_ = X_train
			y_train_ = y_train

		scorer_mapping = {}
		for scorer_i in range(len(scorers)):
			scorer_mapping['scorer' + str(scorer_i)] = scorers[scorer_i]

		results = []
		if len(scorers) > 1:
			model_ = f_clf1(features)
			cv = GridSearchCV(model_, param_grid={'clf__C': [1.0]}, cv=cv_splitter,
							  scoring=scorer_mapping,
							  refit=False, n_jobs=-1)

			cv.fit(X_train_, y_train_)

			for scorer_i in range(len(scorers)):
				results.append(cv.cv_results_['mean_test_' + 'scorer' + str(scorer_i)][cv.best_index_])

			y_pred_proba_train = cv.predict_proba[X_train][:, 1]
		else:
			model_ = Pipeline(steps=[
				('selection', MaskSelection(features)),
				('clf', LogisticRegression(n_jobs=-1))
			])
			model_.fit(X_train_, pd.DataFrame(y_train_))
			y_pred_proba_train = model_.predict_proba(X_train)[:, 1]
			y_pred_train = model_.predict(X_train)
			f1_ = f1_score(y_train, y_pred_train)
			results.append(f1_)

		sensitive_df = df_train.loc[:, sensitive_features]
		sensitive_df.reset_index(inplace=True, drop=True)
		outcomes_df = pd.DataFrame(y_pred_proba_train, columns=['outcome'])
		features_df = df_train.loc[:, admissible_features]
		features_df.reset_index(inplace=True, drop=True)

		candidate_df = pd.concat([sensitive_df, features_df, outcomes_df], axis=1)

		JCIT, mb = causal_filter(candidate_df, sensitive_features)

		rod_train = ROD(y_pred=y_pred_proba_train, df=df_train, sensitive=sensitive_feature,
						admissible=admissible_features, protected=protected, mb=mb)

		results.append(rod_train)
		results.append(-np.sum(features))
		return results



	class MyProblem(Problem):

		def __init__(self):
			number_objectives = 3

			super().__init__(n_var=X_train.shape[1],
							 n_obj=number_objectives,
							 n_constr=0,
							 xl=0, xu=1, type_var=np.bool_)



		def _evaluate(self, x, out, *args, **kwargs):
			result_list = []
			for i in range(self.n_obj):
				result_list.append([])

			for i in range(len(x)):
				if np.sum(x[i]) > 0:
					continue
				else:
					r = randrange(x[i].shape[0])
					x[i][r] = True

			# pool = mp.Pool(mp.cpu_count())
			# func = partial(objective_, df_train, X_train, y_train, sampling, sensitive_feature, sensitive_features, admissible_features,
			#   protected)
			# results_back = pool.map(func, x)
			# pool.close()

			results_back = []
			for candidates_fi in x:
				results_back.append(
					objective_(df_train, X_train, y_train, sampling, sensitive_feature, sensitive_features, admissible_features,
			  protected, candidates_fi))

			floating_evaluation = list(itertools.chain(*[results_back]))
			for i in floating_evaluation:
				for o in range(self.n_obj):
					result_list[o].append(i[o] * -1)

			out["F"] = anp.column_stack(result_list)

	problem = MyProblem()

	population_size = math.ceil(math.sqrt(X_train.shape[1]))
	#population_size = 30
	cross_over_rate = 0.9
	algorithm = NSGA2(pop_size=population_size,
					  sampling=get_sampling("bin_random"),
					  crossover=get_crossover('bin_one_point'),
					  # get_crossover("bin_hux"),#get_crossover("bin_two_point"),
					  mutation=BinaryBitflipMutation(1.0 / X_train.shape[1]),
					  elimate_duplicates=True  # ,
					  # n_offsprings= cross_over_rate * population_size
					  )

	res = minimize(problem, algorithm, get_termination('n_gen', 100), disp=False)
	# res = minimize(problem, algorithm, get_termination("n_gen", 5), disp=False)

	return res

# print(res.F)

# Scatter().add(res.F).show()
# plt.show()


# from sklearn import datasets
# from sklearn.model_selection import train_test_split
#
# X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=1)
#
# auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
# f1_scorer = make_scorer(f1_score)
# # # #fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
# # #
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# a = evolution(X_train, y_train, scorers=[auc_scorer, f1_scorer], cv_splitter=5, max_search_time=1000)
#
# selected_columns = []
# for i in a:
# 	x = np.argwhere(i)
# 	s = []
# 	for idj, j in enumerate(x):
# 		s.extend([x.item(idj)])
# 	selected_columns.append(s)
# print(selected_columns)
