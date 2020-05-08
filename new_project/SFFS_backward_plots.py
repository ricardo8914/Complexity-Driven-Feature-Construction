import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

def check_name(row):
    match = re.search('FC_genetic_', row['Method'])
    if match:
        x = 'NSGAII'
    else:
        x = row['Method']
    return x

def set_size(row):
    x = len(row['Representation'].strip('[').strip(']').strip("\'").split(','))
    return x

visited_representations_cf = pd.read_csv(results_path + '/visited_representations_CF_0.csv')
visited_representations = pd.read_csv(results_path + '/visited_representations_0.csv')
genetic_representations = pd.read_csv(results_path + '/just_genetic_C3_0.csv')
#genetic_representations.reset_index(inplace=True)
print(genetic_representations.head(5))
results = pd.read_csv(results_path + '/just_genetic_C3_0.csv')

for i in results['Method'].unique():
    #match = re.search(pattern, i)
    for j in results.loc[results['Method'] == i, 'Representation']:
        results.loc[results['Method'] == i, 'Size'] = len(j.strip('[').strip(']').strip("\'").split(','))
    else:
        pass

genetic_representations = genetic_representations.loc[genetic_representations['Fold'] == 1, ]
#genetic_representations.reset_index(inplace=True, drop=True)
results_e = results.loc[(results['Fold'] == 1) & (results['Method'].isin(['original', 'dropped', 'capuchin'])), ['Method', 'ROD', 'F1', 'Size']]
genetic_representations['Method'] = 'NSGAII'
genetic_representations['Size'] = 10
visited_representations['Method'] = 'FC_SFFS_backward'
visited_representations_cf['Method'] = 'FC_SFFS_backward_CF'


visited_representations = visited_representations.loc[:, ['Method', 'ROD', 'F1', 'Size']]
visited_representations = visited_representations.append(results_e, ignore_index=True)
#visited_representations = visited_representations.append(visited_representations_cf, ignore_index=True)
visited_representations = visited_representations.append(genetic_representations, ignore_index=True)

v = visited_representations.loc[:, 'ROD'].to_numpy()
normalized = (v - v.min()) / (v.max() - v.min())
visited_representations.loc[:, 'ROD'] = normalized

# ax = sns.kdeplot(visited_representations.ROD, visited_representations.F1, cmap="Reds", shade=True, bw=0.5,
#                  cbar=True, shade_lowest=True, gridsize=500)
# ax.set(ylim=(0.4, 1))
# ax.set(xlim=(0, 1))

ax_2 = sns.scatterplot(x="ROD", y="F1", data=visited_representations, style='Method', hue='Method')

ax_2.set(ylim=(0.3, 1))
ax_2.set(xlim=(0, 1))
plt.legend(loc='best')
plt.show()




