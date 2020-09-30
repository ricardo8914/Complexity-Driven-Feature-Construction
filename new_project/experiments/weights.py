import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
#from kneed import KneeLocator
from numpy.linalg import norm
from sklearn.preprocessing import Normalizer
home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/Final_results/Weights'

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def select_representation(dataset=None, fairness=0.5):

    accuracy = 1-fairness
    results = []
    for i in range(1, 6):
        visited_train = pd.read_csv(
            results_path + '/' + dataset + '/train_' + str(i) + '.csv')
        visited_test = pd.read_csv(
            results_path + '/' + dataset + '/test_' + str(i) + '.csv')

        ROD = visited_train.loc[:, ['ROD']].to_numpy()
        F1 = visited_train.loc[:, ['F1']].to_numpy()

        scores = np.hstack((ROD, F1))

        v = scores[:, 0]
        normalized = (v - v.min()) / (0 - v.min())
        scores[:, 0] = normalized

        pareto = identify_pareto(scores)
        pareto_front = scores[pareto]

        ideal_point = np.asarray([1 * fairness, 1 * accuracy])
        dist = np.empty((pareto_front.shape[0], 1))

        for idx, j in enumerate(pareto_front):
            j[0] = j[0] * fairness
            j[1] = j[1] * accuracy
            dist[idx] = norm(j - ideal_point)

        min_dist = np.argmin(dist)
        selected_representation = visited_test.iloc[pareto[min_dist], :].tolist()
        results.append([selected_representation[2], selected_representation[3], selected_representation[1], i])

    df = pd.DataFrame(results, columns=['F1', 'ROD', 'Size', 'Fold'])
    df['Dataset'] = dataset
    return df


importances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
datasets = ['Adult', 'COMPAS', 'German credit']

complexity_results = pd.DataFrame(columns=['F1', 'ROD', 'Size', 'Fold', 'Dataset', 'Fairness', 'Accuracy'])
for i in datasets:
    for j in importances:
        df = select_representation(dataset=i, fairness=j)
        df['Fairness'] = j
        df['Accuracy'] = 1-j
        complexity_results = complexity_results.append(df, ignore_index=True)
        print(complexity_results)

complexity_results.to_csv(path_or_buf=results_path + '/weights_summary.csv', index=False)


