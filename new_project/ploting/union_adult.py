import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/complete_adult_results_complexity_4.csv')
results.loc[results['Method'] == 'FC_SFFS_backward', 'Method'] = 'FC_SFFS_SBFS'
results_genetic = pd.read_csv(results_path + '/adult_genetic_complexity_4.csv')

adult_union = pd.concat([results, results_genetic], ignore_index=True)

for i in range(1, 6):
    v_1 = adult_union.loc[adult_union['Fold'] == i, 'ROD'].to_numpy() * -1
    norm = Normalizer().fit_transform(v_1.reshape(1, -1))
    norm = norm.reshape(v_1.shape[0], 1)
    adult_union.loc[adult_union['Fold'] == i, 'ROD'] = 1 - norm

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Adult adult_union - ROD')
ax2.set_title('Adult adult_union - F1')

my_pal = 'bright'
sns.set_style("whitegrid")
sns.boxplot(y='ROD', x='Method',
                 data=adult_union,
                 palette=my_pal, hue='Method', ax=ax1)#,
                 #hue='Method')
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
sns.boxplot(y='F1', x='Method',
                 data=adult_union,
                 palette=my_pal, hue='Method', ax=ax2)#,
                 #hue='Method')

ax1.set(ylim=(0, 1.01))
ax2.set(ylim=(0.5, 1))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
#_ = plt.xticks(rotation=90)
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()