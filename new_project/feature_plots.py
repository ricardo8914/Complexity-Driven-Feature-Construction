import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

results = pd.read_csv(results_path + '/feature_analysis.csv')

results['normalized_ROD'] = results.apply(lambda x: abs(x['ROD']-1), axis=1)

#results = results.round(2)

# ax1 = sns.lineplot(x="number_of_features", y="normalized_ROD", data=results, ci='sd', err_style='band')
# # ax2 = sns.lineplot(x="number_of_features", y="ACC", data=results, ci='sd', err_style='band')

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(results['number_of_features'], results['normalized_ROD'])
ax2.plot(results['number_of_features'], results['ACC'], color="red" )

ax1.set_xlabel('Number of features')
ax1.set_ylabel('Normalized ROD', color='b')
ax2.set_ylabel('Accuracy', color='r')

plt.show()
#print(results)