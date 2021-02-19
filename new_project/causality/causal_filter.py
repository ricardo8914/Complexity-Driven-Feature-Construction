from causality.d_separation import d_separation

def causal_filter(candidate_df, sensitive_features):
    #result = True

    c = ''
    for f in sensitive_features:
        if c == '':
            c = f
        else:
            c = c + '_' + f

    mb = d_separation(candidate_df, sensitive=c, target='outcome')

    if set(sensitive_features).intersection(set(mb)):
        result = False
    else:
        result = True

    return result, mb