import openml
from sklearn.model_selection import KFold
import multiprocessing as mp
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fairexp import extend_dataframe_complete
import numpy as np

#### Adult dataset
dataset = openml.datasets.get_dataset(1590)

adult, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)

f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

kf1 = KFold(n_splits=5, random_state=42, shuffle=True)

target = 'income'
sensitive_feature = 'sex'
inadmissible_feature = 'marital-status'
protected = 'Female'
sensitive_features = [sensitive_feature] + [inadmissible_feature]
admissible_features = [i for i in list(adult) if i not in sensitive_features and i != target]

## Drop other sensitive information
adult.drop(columns=['fnlwgt', 'relationship', 'race', 'native-country'], inplace=True)

adult['income'] = y.apply(lambda x: 1 if x == '>50K' else 0)

adult.dropna(inplace=True)


if __name__ == '__main__':

    mp.set_start_method('fork')

    fold = 1
    results = []
    for train_index, test_index in kf1.split(adult):

        X, names, retained_indices, valid_indices = extend_dataframe_complete(adult, 3, f1, target, 0.05,
                                                                              train_index)

        test_indices = np.intersect1d(valid_indices, test_index)

        X_train = X[retained_indices]
        y_train = np.ravel(y.iloc[retained_indices].to_numpy())
        X_test = X[test_indices]
        y_test = np.ravel(y.iloc[retained_indices].to_numpy())

        print(names)


