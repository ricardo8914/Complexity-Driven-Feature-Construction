import subprocess
from pathlib import Path
from random import randrange
from fastsklearnfeature.configuration.Config import Config
home = str(Config.get('path_to_project'))
#home = str(Path.home())
import os

path = Path(home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp')
path.mkdir(parents=True, exist_ok=True)
tmp_folder = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp'
rscript_path = home + '/Complexity-Driven-Feature-Construction/new_project/R_scripts/test_d_sep.R'

if Path(rscript_path).is_file():
    pass
else:
    print('Please locate the corresponding Rscript in the following path: ' + rscript_path)
    exit()

def test_d_separation(df=None, tmp_path=tmp_folder, sensitive_features=None, admissible=None, target=None):
    r = randrange(1000000)
    df_ = df.copy()

    for i in list(df_):
        if i in sensitive_features:
            df_.rename(columns={i: 's_' + i}, inplace=True)
        elif i in admissible:
            df_.rename(columns={i: 'a_' + i}, inplace=True)
        elif i != target and i not in sensitive_features and i not in admissible:
            df_.rename(columns={i: 'x_' + i}, inplace=True)

    df_.to_csv(path_or_buf=tmp_folder + '/test_d_sep_' + str(r) + '.csv', index=False)
    subprocess.run("Rscript " + rscript_path + ' ' + tmp_path + ' ' + 'test_d_sep_' + str(r), shell=True)

    selected = []
    try:
        file = open(tmp_folder + '/test_d_sep_' + str(r) + '.txt', 'r')
        f1 = file.readlines()
        for line in f1:
            line = line.strip()
            l = line.replace('\n\'', '')
            l = l.replace("\\", "")
            selected.extend([l[2:]])
    except FileNotFoundError:
        pass

    try:
        os.remove(tmp_folder + '/test_d_sep_' + str(r) + '.csv')
        os.remove(tmp_folder + '/test_d_sep_' + str(r) + '.txt')
    except FileNotFoundError:
        pass

    return selected
