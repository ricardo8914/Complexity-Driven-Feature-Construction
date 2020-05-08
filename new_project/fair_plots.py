import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/summary_new_adult_pipeline.csv')
results_gen = pd.read_csv(results_path + '/summary_just_genetic_30pop.csv')

backward = results.loc[results['Method'] == 'FC_SFFS_backward', ]
results = results_gen.append(backward, ignore_index=True)

for i in range(1, 6):
    v = results.loc[results['Fold'] == i, 'ROD'].to_numpy()
    normalized = (v - v.min()) / (v.max() - v.min())
    results.loc[results['Fold'] == i, 'ROD'] = normalized

def check_name(row):
    match = re.search('feature_construction_genetic', row['Method'])
    if match:
        x = 'fcg' + row['Method'][29:31]
    else:
        x = row['Method']
    return x

def check_name_2(row):
    match = re.search('feature_construction', row['Method'])
    if match:
        x = 'fc_' + row['Method'][21:]
    else:
        x = row['Method']
    return x

def filter(row, pattern='FC_genetic'):
    match = re.search(pattern, row['Method'])
    result = False
    if match:
        result = True
    else:
        pass
    return result


ROD_mean = results.loc[results['Method'] == 'dropped', 'ROD'].mean()

pattern = 'FC_genetic'
not_filter = ['dropped', 'original', 'capuchin', 'FC_SFFS_backward']
for i in results['Method'].unique():
    match = re.search(pattern, i)
    if match:
        if results.loc[results['Method'] == i, 'ROD'].mean() >= ROD_mean:
            not_filter.extend([i])
    else:
        pass

results['filter'] = results.apply(filter, axis=1)
results['Method'] = results.apply(check_name, axis=1)
results['Method'] = results.apply(check_name_2, axis=1)

partial_results = results[results.Method.isin(not_filter)]

for i in range(1, 6):
    v = partial_results.loc[partial_results['Fold'] == i, 'ROD'].to_numpy()
    normalized = (v - v.min()) / (v.max() - v.min())
    partial_results.loc[partial_results['Fold'] == i, 'ROD'] = normalized


results.loc[:, ['Method', 'Fold', 'Representation']].to_html(open(results_path + '/representations_summary_perm.html', 'w'))
#print(results.head)

print(results.groupby('Method')['F1'].mean())
print(results.groupby('Method')['ROD'].mean())

my_pal = 'bright'
sns.set_style("whitegrid")
ax = sns.boxplot(y='F1', x='Method',
                 data=partial_results,
                 palette=my_pal)#,
                 #hue='Method')

#ax.set(ylim=(0, 1))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
_ = plt.xticks(rotation=90)
plt.legend(loc='best')
plt.show()


# for i in results['Method'].unique():
#     match = re.search(pattern, i)
#     if match:
#         for j in results.loc[results['Method'] == i, 'Representation']:
#             print(len(j.strip('[').strip(']').strip("\'").split(',')))
#     else:
#         pass

