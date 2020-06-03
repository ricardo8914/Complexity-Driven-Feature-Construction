import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

german_credit = pd.read_csv(results_path + '/credit_df_original_FS.csv')
COMPAS = pd.read_csv(results_path + '/COMPAS_original_FS.csv')
adult = pd.read_csv(results_path + '/adult_original_FS.csv')
#german_credit.loc[german_credit['Problem - Set'] == 'FC_SFFS_backward', 'Method'] = 'FC_SFFS_SBFS'

def check_name(row):
    match = re.search('train', row['Problem - Set'])
    if match:
        x = 'train'
    else:
        x = 'test'
    return x

for set in [german_credit, COMPAS, adult]:
    for i in range(1, 6):
        v_1 = set.loc[set['Fold'] == i, 'ROD'].to_numpy()
        set.loc[set['Fold'] == i, 'ROD'] = 1 + v_1

    set['Problem - Set'] = set.apply(check_name, axis=1)


# ROD_mean = results.loc[results['Method'] == 'dropped', 'ROD'].mean()
#
# pattern = 'FC_genetic'
# not_filter = ['dropped', 'original', 'capuchin', 'FC_SFFS_backward']
# for i in results['Method'].unique():
#     match = re.search(pattern, i)
#     if match:
#         if results.loc[results['Method'] == i, 'ROD'].mean() >= ROD_mean:
#             not_filter.extend([i])
#     else:
#         pass

# results['filter'] = results.apply(filter, axis=1)
# results['Method'] = results.apply(check_name, axis=1)
# results['Method'] = results.apply(check_name_2, axis=1)
#
# partial_results = results[results.Method.isin(not_filter)]
#
# for i in range(1, 6):
#     v = partial_results.loc[partial_results['Fold'] == i, 'ROD'].to_numpy()
#     normalized = (v - v.min()) / (v.max() - v.min())
#     partial_results.loc[partial_results['Fold'] == i, 'ROD'] = normalized


#results.loc[:, ['Method', 'Fold', 'Representation']].to_html(open(results_path + '/representations_summary_perm.html', 'w'))
#print(results.head)

#print(results.groupby('Method')['F1'].mean())
#print(results.groupby('Method')['ROD'].mean())

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.set_title('Adult')
ax2.set_title('COMPAS')
ax3.set_title('German credit')

my_pal = 'bright'
sns.set_style("whitegrid")
sns.boxplot(y='F1', x='Problem - Set',
                 data=adult,
                 palette=my_pal, hue='Problem - Set', ax=ax1)#,
                 #hue='Method')
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
sns.boxplot(y='F1', x='Problem - Set',
                 data=COMPAS,
                 palette=my_pal, hue='Problem - Set', ax=ax2)#,
                 #hue='Method')

#ax1.set(ylim=(0, 1.01))
#ax2.set(ylim=(0.4, 1))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
#_ = plt.xticks(rotation=90)
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
sns.boxplot(y='F1', x='Problem - Set',
                 data=german_credit,
                 palette=my_pal, hue='Problem - Set', ax=ax3)#,
                 #hue='Method')

#ax1.set(ylim=(0, 1.01))
#ax2.set(ylim=(0.4, 1))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
#_ = plt.xticks(rotation=90)
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax1.legend(loc='best')
ax2.get_legend().remove()
ax3.get_legend().remove()

ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')
#ax2.legend(loc='best')
#ax3.legend(loc='best')
plt.show()


# for i in results['Method'].unique():
#     match = re.search(pattern, i)
#     if match:
#         for j in results.loc[results['Method'] == i, 'Representation']:
#             print(len(j.strip('[').strip(']').strip("\'").split(',')))
#     else:
#         pass

