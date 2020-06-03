import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
#from kneed import KneeLocator
from numpy.linalg import norm
home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'
visited_train = pd.read_csv(results_path + '/COMPAS_complete_visited_representations_train_complexity_5_2.csv')


v = visited_train.loc[:, 'ROD'].to_numpy()
normalized = (v - v.min()) / (0 - v.min())
visited_train.loc[:, 'ROD'] = normalized

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

scores = visited_train.loc[:, ['ROD','F1']].to_numpy()

pareto = identify_pareto(scores)
pareto_front = scores[pareto]

x_pareto = pareto_front[:, 0]
y_pareto = pareto_front[:, 1]

ideal_point = np.asarray([1, 1])
dist = np.empty((pareto_front.shape[0], 1))

for idx, i in enumerate(pareto_front):
   dist[idx] = norm(i-ideal_point)

min_dist=np.argmin(dist)

selected_representation = pareto_front[min_dist]

pareto_colors = np.zeros((pareto_front.shape[0], 1), dtype='object')
for idx,i in enumerate(pareto_front):
   pareto_colors[idx] = '#0000FF'

pareto_colors[min_dist] = '#FF0000'

x_projline = [selected_representation[0], 1]
y_projline = [selected_representation[1], 1]

fig, ax = plt.subplots()

for idx, i in enumerate(pareto_front):
    ax.scatter(x_pareto[idx], y_pareto[idx], color=pareto_colors[idx])

#ax.scatter(scores[:, 0], scores[:, 1])


ax.set(ylim=(0, 1))
ax.set(xlim=(0, 1.01))
ax.set_title('Pareto front SFFS_SBFS')
ax.set(xlabel='ROD', ylabel='F1')
ax.plot(x_projline, y_projline, color='#FF0000')
ax.annotate('Min distance', xy=(selected_representation[0], selected_representation[1]),
            xycoords='data',
            xytext=(selected_representation[0]+0.02, selected_representation[1]+0.02), textcoords='data'
                   )

plt.show()


