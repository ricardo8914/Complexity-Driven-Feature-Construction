import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

german_credit = pd.read_csv(results_path + '/complete_credit_df_results_complexity_3.csv')
COMPAS = pd.read_csv(results_path + '/complete_COMPAS_results_complexity_4.csv')
adult = pd.read_csv(results_path + '/complete_adult_results_complexity_4.csv')

german_credit['Problem'] = 'German credit'
COMPAS['Problem'] = 'COMPAS'
adult['Problem'] = 'Adult'

german_credit.loc[german_credit['Method'] == 'FC_SFFS_backward', 'Method'] = 'FC_SFFS_SBFS'
COMPAS.loc[COMPAS['Method'] == 'FC_SFFS_backward', 'Method'] = 'FC_SFFS_SBFS'
adult.loc[adult['Method'] == 'FC_SFFS_backward', 'Method'] = 'FC_SFFS_SBFS'

columns = list(german_credit)
results = pd.DataFrame(columns=columns)
for ids, set in enumerate([german_credit, COMPAS, adult]):
    for i in range(1, 6):
        v_1 = set.loc[set['Fold'] == i, 'ROD'].to_numpy() * -1
        norm = Normalizer().fit_transform(v_1.reshape(1, -1))
        norm = norm.reshape(v_1.shape[0], 1)
        set.loc[set['Fold'] == i, 'ROD'] = 1 - norm

    results = results.append(set, ignore_index=True)

table = pd.pivot_table(results, values=['ROD'], index=['Problem','Method'], aggfunc={'ROD':[np.mean, np.std]})

print(table)

table.to_latex(results_path + '/all_methods_ROD.txt', float_format='%.2f')

# table.to_html(open(results_path + '/original_FC_FS.html', 'w'), float_format=lambda x: '%10.2f' % x,
#               index_names=False, bold_rows=False, border=1)




