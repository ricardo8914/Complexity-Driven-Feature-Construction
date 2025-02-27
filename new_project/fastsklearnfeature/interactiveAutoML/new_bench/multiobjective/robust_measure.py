import numpy as np
from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier

from art.attacks import FastGradientMethod
from sklearn.model_selection import GridSearchCV

def robust_score(y_true, y_pred, eps=0.1, X=None, y=None, feature_selector=None):
	all_ids = range(X.shape[0])
	test_ids = y_true.index.values
	train_ids = list(set(all_ids)-set(test_ids))

	X_train = X[train_ids,:]
	y_train = y[train_ids]
	X_test = X[test_ids,:]
	y_test = y[test_ids]

	X_train = feature_selector.fit_transform(X_train)
	X_test = feature_selector.transform(X_test)


	from sklearn.svm import LinearSVC
	model = LinearSVC()
	tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	cv = GridSearchCV(LinearSVC(), tuned_parameters)
	cv.fit(X_train, y_train)
	model = cv.best_estimator_

	classifier = SklearnClassifier(model=model)
	attack = FastGradientMethod(classifier, eps=eps, batch_size=1)

	X_test_adv = attack.generate(X_test)

	diff = model.score(X_test, y_test) - model.score(X_test_adv, y_test)
	return diff

def robust_score_test(y_true, y_pred, eps=0.1, X_train=None, y_train=None, X_test=None, y_test=None, feature_selector=None):
	X_train = feature_selector.fit_transform(X_train)
	X_test = feature_selector.transform(X_test)


	from sklearn.svm import LinearSVC
	model = LinearSVC()
	tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	cv = GridSearchCV(LinearSVC(), tuned_parameters)
	cv.fit(X_train, y_train)
	model = cv.best_estimator_

	classifier = SklearnClassifier(model=model)
	attack = FastGradientMethod(classifier, eps=eps, batch_size=1)

	X_test_adv = attack.generate(X_test)

	diff = model.score(X_test, y_test) - model.score(X_test_adv, y_test)
	return diff


def unit_test_score(y_true, y_pred, unit_x=None, unit_y=None, X=None, y=None, pipeline=None):
	all_ids = range(X.shape[0])
	test_ids = y_true.index.values
	train_ids = list(set(all_ids)-set(test_ids))

	X_train = X[train_ids, :]
	y_train = y[train_ids]

	pipeline.fit(X_train, y_train)

	class_id = -1
	for c_i in range(len(pipeline.classes_)):
		if pipeline.classes_[c_i] == unit_y:
			class_id = c_i
			break
	y_pred = pipeline.predict_proba(np.array([unit_x]))[0, class_id]
	return y_pred