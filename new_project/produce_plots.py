import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/summary_df.csv')

array = results['ROD'].to_numpy()
transformer = Normalizer().fit_transform(array.reshape(1, -1))
results['Normalized_ROD'] = transformer.flatten()
mean_ROD = pd.DataFrame(results.groupby(by=['method'])['Normalized_ROD'].mean(), columns=['Normalized_ROD'])
mean_ROD = mean_ROD.round(2)
mean_ROD.columns = ['ROD']
mean_ROD.reset_index(inplace=True)
results.drop(columns=['ROD'], inplace=True)
results = pd.merge(results, mean_ROD, left_on='method', right_on='method', how='left')


print(results.groupby('method')['accuracy'].mean())
print(results.groupby('method')['ROD'].mean())

sns.boxplot(y='accuracy', x='ROD',
                 data=results,
                 palette="colorblind",
                 hue='method')

plt.legend(loc='lower left')
plt.show()