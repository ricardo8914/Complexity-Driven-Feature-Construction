import pandas as pd
import numpy as np
import itertools
from nimfa import Nmf

def get_contigency_matrices(df, sensitive_attribute=None, admissible_attributes=None, target=None):

    z = []
    for i in admissible_attributes:
        z.append(sorted(df[i].unique()))

    Z = df.groupby(admissible_attributes).size().reset_index()[admissible_attributes].values.tolist()

    I = [sensitive_attribute]
    I.extend([i for i in list(df) if i not in admissible_attributes and i not in [target, sensitive_attribute]])

    y = []
    for i in I:
        if i != 'kx':
            y.append(sorted((df[i].unique()).tolist()))

    if len(y) > 1:
        y_ = list(itertools.product(*y))
    else:
        y_ = [item for sublist in y for item in sublist]

    x = []
    x.extend(sorted((df[target].unique()).tolist()))

    y.append(x)

    m = list(itertools.product(*y))

    column_list = I.copy()
    column_list.extend([target])

    df_i = df.set_index(admissible_attributes)
    df_i = df_i[column_list]
    df_i.sort_index(inplace=True)

    f = []
    real_Z = []
    for j in Z:
        try:

            m_ = df_i.loc[j]
            tuples = [tuple(x) for x in m_.to_numpy()]
            h = []

            for i in m:
                h.extend([tuples.count(i)])

            nz = np.asarray(h)
            v = nz.reshape((len(x), len(y_)))
            f.append(v)
            real_Z.append(j)

        except KeyError:
            pass

    c = []
    c.append(real_Z)
    c.append(m)
    q = list(itertools.product(*c))
    res = [(*a, *b) for a, b in q]

    return f, x, y_, real_Z, I, res

def factorize_matrices(contigency_matrix):

    repaired = []
    for i in contigency_matrix:

        nmf = Nmf(i, max_iter=200, rank=1, update='euclidean', objective='fro', seed="nndsvd")
        b = nmf.factorize()
        repaired.append(np.array(b.fitted()))

    return repaired


def repair_dataset(df, sensitive_attribute, admissible_attributes, target):

    print('get matrices')
    matrices, X, Y, Z, I, res = get_contigency_matrices(df=df, sensitive_attribute=sensitive_attribute, admissible_attributes=admissible_attributes, target=target)
    print('factorize matrices')
    M = factorize_matrices(matrices)
    print('done with factorization')


    columns = []
    columns.extend(admissible_attributes)
    columns.extend(I)
    columns.extend([target])

    rows = 0
    for i in M:
        rows += np.sum(i.flatten())

    w = []
    for i in M:
        s = i.flatten()
        for j in s:
            w.extend([j / rows])

    new_distribution = (np.rint(np.dot(rows, w))).astype(int)

    repaired_tuples = []
    for idx, i in enumerate(res):
        if new_distribution[idx] > 0:
            repaired_tuples.extend([list(i)] * new_distribution[idx])
        else:
            pass

    repaired_df = pd.DataFrame(repaired_tuples, columns=columns)

    return repaired_df

