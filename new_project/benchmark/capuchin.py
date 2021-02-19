import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import NMF
from nimfa import Nmf
from sklearn.preprocessing import LabelEncoder


def get_contigency_matrices(df, sensitive_attribute=None, admissible_attributes=None, target=None):
    encoded_df = df.copy()
    encoders_dict = {}
    inadmissible_features = [i for i in list(df) if
                             i not in admissible_attributes and i not in [target, sensitive_attribute]]
    for i in list(encoded_df):
        if i not in (sensitive_attribute, *inadmissible_features, target):
            le = LabelEncoder()
            df[i] = df[i].astype(str, copy=False)
            df[i] = df[i].astype('category', copy=False)
            encoded_df[i] = le.fit_transform(df[i])
            encoders_dict[i] = le

    Z = encoded_df.groupby(admissible_attributes).size().reset_index()[admissible_attributes].values.tolist()

    Z = [tuple(x) for x in Z]

    I = [sensitive_attribute]
    I.extend([i for i in list(df) if i not in admissible_attributes and i not in [target, sensitive_attribute]])

    x = []
    x.extend(sorted((df[target].unique()).tolist()))

    len_sensitive = df[sensitive_attribute].unique().shape[0]
    sorted_sensitive = sorted((df[sensitive_attribute].unique()).tolist())
    sorted_target = sorted((df[target].unique()).tolist())

    sorted_inadmissible = []
    for i in inadmissible_features:
        sorted_inadmissible.extend(sorted((df[i].unique()).tolist()))

    column_list = I.copy()
    column_list.extend([target])

    df_i = encoded_df.set_index(admissible_attributes)
    df_i = df_i[column_list]
    df_i.sort_index(inplace=True)

    f = []
    real_Z = []
    res = []
    check_count = 0
    for j in Z:
        try:
            m_ = df_i.loc[[j]]
            tuples = [tuple(x) for x in m_.to_numpy()]

            h = []

            if sorted_inadmissible:
                for y in sorted_target:
                    for i in sorted_inadmissible:
                        for s in sorted_sensitive:
                            check_count += tuples.count((s, i, y))
                            h.extend([tuples.count((s, i, y))])
                            res.append(j + (s,) + (i,) + (y,))
            else:
                for y in sorted_target:
                    for s in sorted_sensitive:
                        check_count += tuples.count((s, y))
                        h.extend([tuples.count((s, y))])
                        res.append(j + (s,) + (y,))

            nz = np.asarray(h)

            if sorted_inadmissible:
                v = nz.reshape((len(sorted_target), len_sensitive * len(sorted_inadmissible)))
            else:
                v = nz.reshape((len(sorted_target), len_sensitive))

            f.append(v)
            real_Z.append(j)

        except KeyError:
            pass

    res_df = pd.DataFrame(res, columns=admissible_attributes + [sensitive_attribute] + inadmissible_features + [target])
    contexts_df = pd.DataFrame(real_Z, columns=admissible_attributes)
    for i in list(encoded_df):
        if i not in (sensitive_attribute, *inadmissible_features, target):
            res_df[i] = encoders_dict[i].inverse_transform(res_df[i])
            contexts_df[i] = encoders_dict[i].inverse_transform(contexts_df[i])

    return f, res_df, contexts_df

def factorize_matrices(contigency_matrix):

    repaired = []
    fair_contexts = []
    for i in contigency_matrix:

        #nmf = Nmf(i, max_iter=1000, rank=1, update='euclidean', objective='fro', seed="nndsvd")
        #b = nmf.factorize()
        #repaired.append(np.array(b.fitted()))
        model = NMF(1, max_iter=100)
        W = model.fit_transform(i)
        H = model.components_

        #print('original matrix: \n', i)
        #print('NMF capuchin \n', W * H)
        #print('NMF nimfa \n', b.fitted())
        repaired.append(W*H)

        fair_contexts.append(np.all(i == W*H))

    return repaired, fair_contexts


def repair_dataset(df, sensitive_attribute, admissible_attributes, target):

    matrices, res, contexts = get_contigency_matrices(df=df, sensitive_attribute=sensitive_attribute,
                                            admissible_attributes=admissible_attributes, target=target)

    M, contexts_mask = factorize_matrices(matrices)

    rows = 0
    for i in M:
        rows += np.sum(i.flatten())

    w = []
    for i in M:
        s = i.flatten()
        for j in s:
            w.extend([j / rows])

    new_distribution = (np.ceil(np.dot(rows, w))).astype(int)
    old_distribution = np.asarray(matrices).flatten()
    #
    retained_array = np.where(new_distribution == old_distribution, old_distribution, 0)

    print(retained_array.shape)

    repaired_df = pd.DataFrame(columns=list(res))
    retained_df = pd.DataFrame(columns=list(res) + ['freq'])
    #multi_index = pd.MultiIndex.from_tuples([tuple(x) for x in res.to_numpy()], names=list(res))
    for idt, t in enumerate(new_distribution):
        if t > 0:
            newdf = pd.DataFrame(np.repeat(res.iloc[idt].values.reshape((1, res.shape[1])), t, axis=0), columns=list(res))
            repaired_df = pd.concat([repaired_df, newdf])

            freq = np.append(res.iloc[idt].to_numpy(), retained_array[idt]).reshape((1, res.shape[1] + 1))
            newr_df = pd.DataFrame(freq, columns=list(retained_df))
            retained_df = pd.concat([retained_df, newr_df])


    #retained_df = repaired_df.copy()
    #retained_df.reset_index(inplace=True, drop=False)
    #retained_df.set_index(list(res), inplace=True)
    #contexts['repair'] = contexts_mask
    #valid_contexts = [tuple(x) for x in contexts.loc[contexts['repair'] == True, admissible_attributes].to_numpy()]
    #retained_df = retained_df.loc[valid_contexts]
    #retained_df['freq'] = retained_array
    #retained_df = retained_df.loc[retained_df['freq'] > 0]
    #
    retained_df.set_index(list(res), inplace=True)

    repaired_df[target] = pd.to_numeric(repaired_df[target])

    return repaired_df







