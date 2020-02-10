import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/summary_rf500f1_df.csv')

array = results['ROD'].to_numpy()
transformer = Normalizer().fit_transform(array.reshape(1, -1))
results['Normalized_ROD'] = transformer.flatten()
#mean_ROD = pd.DataFrame(results.groupby(by=['Method'])['Normalized_ROD'].mean(), columns=['Normalized_ROD'])
mean_ROD = pd.DataFrame(results.groupby(by=['Method'])['ROD'].mean(), columns=['ROD'])
mean_ROD = mean_ROD.round(3)
mean_ROD.columns = ['ROD']
mean_ROD.reset_index(inplace=True)
results.drop(columns=['ROD'], inplace=True)
results = pd.merge(results, mean_ROD, left_on='Method', right_on='Method', how='left')


print(results.groupby('Method')['Accuracy'].mean())
print(results.groupby('Method')['ROD'].mean())

my_pal = {"capuchin": "g", "dropped": "b", "feature_construction":"m", "original":"lightskyblue"}

sns.boxplot(y='Accuracy', x='ROD',
                 data=results,
                 palette=my_pal,
                 hue='Method')

plt.legend(loc='upper right')
plt.show()

# COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'
#
# COMPAS = pd.read_csv(COMPAS_path + '/compas-scores.csv')
#
# COMPAS = COMPAS.loc[(COMPAS['days_b_screening_arrest'] <= 30) &
#                     (COMPAS['priors_count'].isin([1, 2, 3, 4, 5, 6]))
#                     & (COMPAS['is_recid'] != -1)
#                     & (COMPAS['race'].isin(['African-American','Caucasian']))
#                     & (COMPAS['c_charge_degree'].isin(['F','M']))
#                     , ['race','age', 'age_cat', 'priors_count','is_recid','c_charge_degree']]
#
#
# to_html = COMPAS.loc[:, ['race', 'age_cat', 'priors_count', 'c_charge_degree', 'is_recid']]
# to_html.to_html(results_path + '/COMPAS_html.html', index=False)

