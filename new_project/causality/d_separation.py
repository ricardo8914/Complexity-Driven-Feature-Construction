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
rscript_path = home + '/Complexity-Driven-Feature-Construction/new_project/R_scripts/d_separation.R'

if Path(rscript_path).is_file():
    pass
else:
    print('Please locate the corresponding Rscript in the following path: ' + rscript_path)
    exit()

def d_separation(df=None, sensitive=None, target=None, tmp_path=tmp_folder):
    r = randrange(1000000)

    df.to_csv(path_or_buf=tmp_folder + '/' + sensitive + str(r) + '.csv', index=False)
    subprocess.run("Rscript " + rscript_path + ' ' + sensitive + ' ' + target + ' ' + tmp_path + ' ' + sensitive + str(r), shell=True)

    # file = open(tmp_folder + '/' + sensitive + str(r) + '.txt', 'r')
    # f1 = file.readlines()
    # l = f1[0].strip()
    # l = l.replace('\n\'', '')

    mb = []
    l = 'FALSE'
    try:
        file = open(tmp_folder + '/' + sensitive + str(r) + '_MB' + '.txt', 'r')
        f1 = file.readlines()
        for line in f1:
            line = line.strip()
            l = line.replace('\n\'', '')
            l = l.replace("\\", "")
            mb.extend([l])
    except FileNotFoundError:
        pass

    response = False
    if l == 'TRUE':
        response = True

    try:
        os.remove(tmp_folder + '/' + sensitive + str(r) + '.csv')
        #os.remove(tmp_folder + '/' + sensitive + str(r) + '.txt')
        os.remove(tmp_folder + '/' + sensitive + str(r) + '_MB' + '.txt')
    except FileNotFoundError:
        pass

    return mb
