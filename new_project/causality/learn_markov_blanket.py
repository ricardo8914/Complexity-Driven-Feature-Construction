import subprocess
from pathlib import Path
import pandas as pd
home = str(Path.home())

tmp_folder = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp'
rscript_path = home + '/projects/Complexity-Driven-Feature-Construction/new_project/R_scripts/markov_blanket.R'

def learn_MB(df, name):

    df.to_csv(path_or_buf=tmp_folder + '/' + name + '.csv', index=False)
    subprocess.run("Rscript " + rscript_path + ' ' + name, shell=True)

    mb = []
    file = open(tmp_folder + '/' + name + '.txt', 'r')
    f1 = file.readlines()
    for line in f1:
        l = str(line).strip("\n'")
        l = l.replace('.', '-')
        mb.extend([l])

    return mb
