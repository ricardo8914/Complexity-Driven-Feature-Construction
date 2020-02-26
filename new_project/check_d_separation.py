import numpy as np
import subprocess
from pathlib import Path
import pandas as pd
home = str(Path.home())

path = Path(home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp')
path.mkdir(parents=True, exist_ok=True)
tmp_folder = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp'
rscript_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/d_separation.R'

def check_d_separation(df=None, name=None, tmp_path=tmp_folder):

    df.to_csv(path_or_buf=tmp_folder + '/' + name + '.csv', index=False)
    subprocess.run("Rscript " + rscript_path + ' ' + name + ' ' + tmp_path, shell=True)

    response = []
    file = open(tmp_folder + '/' + name + '.txt', 'r')
    f1 = file.readlines()
    for line in f1:
        line = line.strip()
        l = line.replace('\n\'', '')
        l = l.replace("\\", "")
        response.extend([l])

    print('Markov blanket for ' + name + ' : {}'.format(response))
    return response