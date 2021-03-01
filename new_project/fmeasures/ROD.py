import numpy as np
import subprocess
from pathlib import Path
import pandas as pd
from random import randrange
import os

from fastsklearnfeature.configuration.Config import Config
home = str(Config.get('path_to_project'))
#home = str(Path.home())

path = Path(home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp')
path.mkdir(parents=True, exist_ok=True)
tmp_folder = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/tmp'
rscript_path = home + '/Complexity-Driven-Feature-Construction/new_project/R_scripts/markov_blanket.R'

if Path(rscript_path).is_file():
    pass
else:
    print('Please locate the corresponding Rscript in the following path: ' + rscript_path)
    exit()

def learn_MB(df=None, name='_', tmp_path=tmp_folder):

    r = randrange(100000000)
    df.to_csv(path_or_buf=tmp_folder + '/' + name + str(r) + '.csv', index=False)
    subprocess.run("Rscript " + rscript_path + ' ' + name + str(r) + ' ' + tmp_path, shell=True)

    mb = []
    try:
        file = open(tmp_folder + '/' + name + str(r) + '.txt', 'r')
        f1 = file.readlines()
        for line in f1:
            line = line.strip()
            l = line.replace('\n\'', '')
            l = l.replace("\\", "")
            mb.extend([l])
    except FileNotFoundError:
        pass


    try:
        os.remove(tmp_folder + '/' + name + str(r) + '.csv')
        os.remove(tmp_folder + '/' + name + str(r) + '.txt')
    except FileNotFoundError:
        pass


    #print('Markov blanket for ' + name + ' : {}'.format(mb))
    return mb

def generate_binned_df(df):
    columns2_drop = []
    df_ = df.copy()
    for i in list(df_):
        if i not in ['target', 'outcome'] and (df_[i].dtype in (int, float) and len(df_[i].unique()) > 4):
            out, bins = pd.qcut(df_[i], q=4, retbins=True, duplicates='drop')
            df_.loc[:, i] = out.astype(str)
    return df_


def ROD(y_true=None, y_pred=None, df=None, sensitive=None, protected=None, admissible=None, test_idx=None, mb=None):

    df_ = df.reset_index(drop=True)

    if test_idx is not None:
        df_ = df_.iloc[test_idx]
        df_ = df_.reset_index(drop=True)

    sensitive_data = df_.loc[:, sensitive]
    outcome = pd.DataFrame(y_pred, columns=['outcome'])
    admissible_data = df_.loc[:, admissible]

    evaluation_df = pd.concat([admissible_data, outcome], axis=1)

    mb = [f for f in mb if f in admissible]
    binned_df = generate_binned_df(evaluation_df)
    contexts = binned_df.loc[:, mb].to_numpy()

    protected = np.asarray(protected)
    unique_contexts = np.array(list(set([tuple(x) for x in contexts])))

    mb_empty = False
    if len(mb) > 0:
        contexts = np.array(list([tuple(x) for x in contexts]))
    else:
        mb_empty = True


    result = 0
    if mb_empty == False:
        ROD = []
        weights = []
        for z in unique_contexts:

            if z.dtype == 'float64' and unique_contexts.dtype == 'float64':
                test_c = z == contexts
            else:
                try:
                    test_c = np.char.equal(contexts, z)
                except TypeError:
                    test_c = np.equal(contexts, z)

            ids = np.argwhere(np.all(test_c, axis=1))

            test_s = np.not_equal(sensitive_data.to_numpy(), protected)
            s_ids = np.argwhere(test_s)
            cs_ids = np.intersect1d(ids, s_ids)

            if cs_ids.shape[0] > 0:
                p_1_0 = np.mean(np.ravel(y_pred[cs_ids]))
                p_0_0 = float(1 - p_1_0)
            else:
                p_1_0 = 0.5
                p_0_0 = 0.5

            test_ns = np.equal(sensitive_data.to_numpy(), protected)
            ns_ids = np.argwhere(test_ns)
            cns_ids = np.intersect1d(ids, ns_ids)

            if cns_ids.shape[0] > 0:
                p_1_1 = np.mean(np.ravel(y_pred[cns_ids]))
                p_0_1 = float(1 - p_1_1)
            else:
                p_1_1 = 0.5
                p_0_1 = 0.5


            try:
                OR = (p_1_0 / p_0_0) * (p_0_1 / p_1_1)

                if np.isinf(OR) == False and np.isnan(OR) == False:
                    ROD.extend([OR])
                    weights.extend([ids.shape[0] / contexts.shape[0]])
                else:
                    pass
            except ZeroDivisionError:
                pass

        result = abs(np.dot(np.squeeze(ROD), weights) - 1) * -1
    else:
        test_s = np.not_equal(sensitive_data.to_numpy(), protected)
        s_ids = np.argwhere(test_s)
        if s_ids.shape[0] > 0:
            p_1_0 = np.mean(np.ravel(y_pred[s_ids]))
            p_0_0 = float(1 - p_1_0)
        else:
            p_1_0 = 0.5
            p_0_0 = 0.5

        test_ns = np.equal(sensitive_data.to_numpy(), protected)
        ns_ids = np.argwhere(test_ns)

        if ns_ids.shape[0] > 0:
            p_1_1 = np.mean(np.ravel(y_pred[ns_ids]))
            p_0_1 = float(1 - p_1_1)
        else:
            p_1_1 = 0.5
            p_0_1 = 0.5


        try:
            OR = (p_1_0 / p_0_0) * (p_0_1 / p_1_1)

            if np.isinf(OR) == False and np.isnan(OR) == False:
                result = abs(OR - 1) * -1
            else:
                pass
        except ZeroDivisionError:
            pass

    if type(result) is np.ndarray:
        result = result[0]

    print(result)
    return result


