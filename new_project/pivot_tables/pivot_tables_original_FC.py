import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/intermediate_results'

german_credit = pd.read_csv(results_path + '/credit_df_original_FS.csv')
COMPAS = pd.read_csv(results_path + '/COMPAS_original_FS.csv')
adult = pd.read_csv(results_path + '/adult_original_FS.csv')

german_credit['Problem'] = 'German credit'
COMPAS['Problem'] = 'COMPAS'
adult['Problem'] = 'Adult'

def check_name(row):
    match = re.search('train', row['Problem - Set'])
    if match:
        x = 'train'
    else:
        x = 'test'
    return x


columns = list(german_credit)
columns.remove('Problem - Set')
columns.extend(['Set'])
results = pd.DataFrame(columns=columns)
for ids, set in enumerate([german_credit, COMPAS, adult]):
    for i in range(1, 6):
        v_1 = set.loc[set['Fold'] == i, 'ROD'].to_numpy()
        set.loc[set['Fold'] == i, 'ROD'] = 1 + v_1

    set['Set'] = set.apply(check_name, axis=1)
    set.drop(columns='Problem - Set', inplace=True)

    results = results.append(set, ignore_index=True)

table = pd.pivot_table(results, values=['ROD'], index=['Problem', 'Set'], aggfunc={'ROD' : [np.mean, np.std]})

table.to_latex(results_path + '/table_original_FC_FS.txt', float_format='%.2f')

# table.to_html(open(results_path + '/original_FC_FS.html', 'w'), float_format=lambda x: '%10.2f' % x,
#               index_names=False, bold_rows=False, border=1)




