import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/summary_adult_genetic_50it_permutation_complexity_4_causaljoin.csv')

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

results['Method'] = results.apply(check_name, axis=1)
results['Method'] = results.apply(check_name_2, axis=1)


results.loc[:,['Method', 'Fold', 'Representation']].to_html(open(results_path + '/representations_summary_perm.html', 'w'))
#print(results.head)

print(results.groupby('Method')['F1'].mean())
print(results.groupby('Method')['ROD'].mean())

my_pal = 'bright'
sns.set_style("whitegrid")
ax = sns.boxplot(y='ROD', x='Method',
                 data=results,
                 palette=my_pal)#,
                 #hue='Method')

#ax.set(ylim=(0, 1))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
_ = plt.xticks(rotation=90)
plt.legend(loc='best')
plt.show()



