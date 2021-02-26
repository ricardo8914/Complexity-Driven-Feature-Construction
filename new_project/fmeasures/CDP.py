from fairlearn.metrics import MetricFrame, selection_rate
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_binned_df(df, admissible):
    df_ = df.copy()
    for i in list(df_):
        if i in admissible and df_[i].dtype in (float, int):
            out, bins = pd.qcut(df_[i], q=3, retbins=True, duplicates='drop')
            if bins.shape[0] == 2:
                out, bins = pd.cut(df_[i], bins=3, retbins=True, duplicates='drop')
            df_.loc[:, i] = out.astype(str)
    return df_


def encodeDF(df, sensitive, admissible):
    encoded_df = df.copy()
    for i in list(encoded_df):
        if i in (sensitive, *admissible):
            le = LabelEncoder()
            df[i] = df[i].astype(str, copy=False)
            df[i] = df[i].astype('category', copy=False)
            encoded_df[i] = le.fit_transform(df[i])

    return encoded_df


def CDP(y_test, y_pred, df, sensitive, admissible):

    df_ = df.copy()
    binned_df = generate_binned_df(df_, admissible)
    encoded_df = encodeDF(binned_df, sensitive, admissible)

    sensitive_features = encoded_df.loc[:, sensitive]
    sensitive_features.reset_index(inplace=True, drop=True)
    control = encoded_df.loc[:, admissible]
    control.reset_index(inplace=True, drop=True)

    contexts = encoded_df.loc[:, admissible].to_numpy()
    unique_contexts = np.array(list(set([tuple(x) for x in contexts])))

    cdp = []
    weights = []
    for z in unique_contexts:
        try:
            test_c = np.char.equal(contexts, z)
        except TypeError:
            test_c = np.equal(contexts, z)

        ids = np.argwhere(np.all(test_c, axis=1))

        ids = [i[0] for i in ids]

        sensitive_features_ = sensitive_features.iloc[ids]
        sensitive_features_.reset_index(inplace=True, drop=True)

        if sensitive_features_.unique().shape[0] > 1:

            m = MetricFrame(selection_rate, y_test[ids], y_pred[ids],
                                  sensitive_features=sensitive_features_)

            cdp.append(m.difference())
            weights.append(len(ids) / contexts.shape[0])
        else:
            continue

    result = np.dot(np.squeeze(cdp), weights)

    return result