import pandas as pd
from pathlib import Path

home = str(Path.home())


def reweighting(df, target, sensitive, protected):

    protected_f = df.loc[df[sensitive] == protected].shape[0] / df.shape[0]
    nonprotected_f = df.loc[df[sensitive] != protected].shape[0] / df.shape[0]

    pos_f = df.loc[df[target] == 1].shape[0] / df.shape[0]
    neg_f = df.loc[df[target] == 0].shape[0] / df.shape[0]

    protected_positive = df.loc[(df[sensitive] == protected) & (df[target] == 1)].shape[0] / df.shape[0]
    protected_negative = df.loc[(df[sensitive] == protected) & (df[target] == 0)].shape[0] / df.shape[0]
    nonprotected_positive = df.loc[(df[sensitive] != protected) & (df[target] == 1)].shape[0] / df.shape[0]
    nonprotected_negative = df.loc[(df[sensitive] != protected) & (df[target] == 0)].shape[0] / df.shape[0]

    w_protected_pos = (protected_f * pos_f) / protected_positive
    w_protected_neg = (protected_f * neg_f) / protected_negative
    w_nonprotected_pos = (nonprotected_f * pos_f) / nonprotected_positive
    w_nonprotected_neg = (nonprotected_f * neg_f) / nonprotected_negative

    df_ = df.copy()

    def weight(row):
        if row[sensitive] == protected and row[target] == 1:
            return w_protected_pos
        elif row[sensitive] == protected and row[target] == 0:
            return w_protected_neg
        elif row[sensitive] != protected and row[target] == 1:
            return w_nonprotected_pos
        else:
            return w_nonprotected_neg

    df_['weight'] = df_.apply(lambda row: weight(row), axis=1)

    return df_['weight'].to_numpy()
