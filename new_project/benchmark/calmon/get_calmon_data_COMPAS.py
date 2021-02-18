from tqdm import tqdm
from cvxpy import *
from benchmark.calmon.DTools3 import *
import time
from pathlib import Path
from sklearn.model_selection import KFold
import itertools


home = str(Path.home())

COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'
output = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/calmon_output/COMPAS'

df = pd.read_csv(COMPAS_path + '/compas-scores-two-years.csv')

df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
                    'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
ix = df['days_b_screening_arrest'] <= 30
ix = (df['days_b_screening_arrest'] >= -30) & ix
ix = (df['is_recid'] != -1) & ix
ix = (df['c_charge_degree'] != "O") & ix
ix = (df['score_text'] != 'N/A') & ix
df = df.loc[ix, :]
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

dfcut = df.loc[~df['race'].isin(['Native American', 'Hispanic', 'Asian', 'Other']), :]

dfcutQ = dfcut[['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text', 'priors_count', 'is_recid',
                'two_year_recid', 'length_of_stay']].copy()


# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x):
    if x <= 0:
        return '0'
    elif 1 <= x <= 3:
        return '1 to 3'
    else:
        return 'More than 3'


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return '<week'
    if 8 < x <= 93:
        return '<3months'
    else:
        return '>3 months'


# Quantize length of stay
def adjustAge(x):
    if x == '25 - 45':
        return '25 to 45'
    else:
        return x


# Quantize score_text to MediumHigh
def quantizeScore(x):
    if (x == 'High') | (x == 'Medium'):
        return 'MediumHigh'
    else:
        return x


dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

features = ['race', 'age_cat', 'c_charge_degree', 'priors_count', 'is_recid']

# Pass vallue to df
df = dfcutQ[features]

kf1 = KFold(n_splits=5, random_state=42, shuffle=True)  ### CHANGE SEED FOR DIFFERENT SPLITS!
df_list = []
for train_index,test_index in kf1.split(df):
    df_list.append((df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()))

DT = DTools(df=df, features=features)

D_features = ['race']
Y_features = ['is_recid']
X_features = ['age_cat', 'c_charge_degree', 'priors_count']

DT.setFeatures(D=D_features, X=X_features, Y=Y_features)


class Dclass():
    # adjust education
    def adjustPrior(self, v):
        if v == '0':
            return 0
        elif v == '1 to 3':
            return 1
        else:
            return 2

    def adjustAge(self, a):
        if a == 'Less than 25':
            return 0
        elif a == '25 to 45':
            return 1
        else:
            return 2

    # distortion metric
    def getDistortion(self, vold, vnew):
        '''
        Distortion metric.

        Inputs:
        *vold : dictionary of the form {attr:value} with old values
        *vnew : dictionary of the form {attr:value} with new values

        Output
        *d : distortion value
        '''

        # value that will be returned for events that should not occur
        bad_val = 1e4
        #
        # # Adjust prior
        pOld = self.adjustPrior(vold['priors_count'])
        pNew = self.adjustPrior(vnew['priors_count'])
        #
        # # Priors cannot be increased, or lowered by more than 1 category. A change has a unit penalty
        if (pNew > pOld) | (pNew < pOld - 1):
            return bad_val
        #
        # # adjust age
        aOld = self.adjustAge(vold['age_cat'])
        aNew = self.adjustAge(vnew['age_cat'])

        # Age cannot be increased or decreased in more than one category
        if np.abs(aOld - aNew) > 1.0:
            return bad_val

        # Recidivism should not be increased
        if vold['is_recid'] < vnew['is_recid']:
            return bad_val

        cum_sum = 0.0

        if np.abs(aOld - aNew) > 0:
            #             cum_sum+=1
            #             cum_sum = cum_sum**2
            cum_sum = cum_sum + 1

        # Penalty of 1 if priors is decreased or increased
        if np.abs(pNew - pOld) > 0:
            #             cum_sum+=1
            #             cum_sum = cum_sum**2
            cum_sum = cum_sum + 1

        # cum_sum = cum_sum**2
        if vold['is_recid'] > vnew['is_recid']:
            #             cum_sum+=1
            #             cum_sum = cum_sum**2
            cum_sum = cum_sum + 1

        # final penalty of 2 for changing misdemeanor to felony and vice-verse
        if vold['c_charge_degree'] != vnew['c_charge_degree']:
            #             cum_sum+=2
            #             cum_sum = cum_sum**2
            cum_sum = cum_sum + 4

        return cum_sum


c1 = 1 # value of (delta1,c1): to keep.
c2 = 2 # value of (delta2,c2): value that should no happen
c3 = 3 # penalty for adjusting age
clist = [c1, c2, c3]
Dclass = Dclass()

DT.setDistortion(Dclass)
epsilon=.2
values = list(itertools.product(*DT.D_values))
dlist = [.4, .3, 0]
# for v in values:
#     if 'African-American' in v:
#         mean_value = .4 #original in ICML submission - mean_value = .25
#         #mean_value=.22
#     else:
#         mean_value = .3 #original in ICML submission - mean_value = .25
#         #mean_value=.22
#     dlist.append((v, mean_value))

######### CHANGE SEED HERE ###########
seed = sum([ord(b) for b in 'fairexp'])
np.random.seed(seed=seed)

def randomize(df, dfMap, features=[]):
    df2 = df.copy()
    print('Randomizing...')
    for idx in tqdm(df2.index):
        rowTest = df2.loc[idx, :]
        vals = rowTest[features]
        draw = dfMap.loc[tuple(vals.tolist())]
        # randomly select value

        try:
            mapVal = np.random.choice(range(len(draw)), p=draw.tolist())
        except ValueError:
            mapVal = np.random.choice(range(len(draw)))

        draw.index[mapVal]
        df2.loc[idx, draw.index.names] = draw.index[mapVal]

    return df2


split_num = 0
runtimes = []
# iterate over pairs
for (df_train, df_test) in df_list:
    start_time = time.time()
    file_name = str(split_num)

    print('-----------------')
    print('Current split: ' + file_name)

    # initialize a new DT object
    DT = DTools(df=df_train, features=features)

    # Set features
    DT.setFeatures(D=D_features, X=X_features, Y=Y_features)

    # Set Distortion
    DT.setDistortion(Dclass, clist=clist)

    # solve optimization for previous parameters -- This uses and older implementation, based on the FATML submission.
    DT.optimize(epsilon=epsilon, dlist=dlist, verbose=True)

    DT.computeMarginals()

    # randomized mapping for training
    # this is the dataframe with the randomization for the train set
    dfPtrain = DT.dfP.applymap(lambda x: 0 if x < 1e-8 else x)
    dfPtrain = dfPtrain.divide(dfPtrain.sum(axis=1), axis=0)

    # randomized mapping for testing (Beware of ugly code)
    d1 = DT.dfFull.reset_index().groupby(D_features + X_features).sum()
    d2 = d1.transpose().reset_index().groupby(X_features).sum()
    dTest = d2.transpose()
    dTest = dTest.drop(Y_features, 1)
    dTest = dTest.applymap(lambda x: x if x > 1e-8 else 0)
    dTest = dTest / dTest.sum()

    # this is the dataframe with the randomization for the test set
    dfPtest = dTest.divide(dTest.sum(axis=1), axis=0)

    # Randomize train data
    print('Randomizing training set...')
    df_train_new = randomize(df_train, dfPtrain, features=D_features + X_features + Y_features)

    # Randomize test data
    print('Randomizing test set...')
    df_test_new = randomize(df_test, dfPtest, features=D_features + X_features)

    # Save train files
    df_train.to_csv(output + '/train_' + file_name + '.csv')
    df_train_new.to_csv(output + '/train_new_' + file_name + '.csv')

    print(df_train.groupby('race')['is_recid'].mean())
    print(df_train_new.groupby('race')['is_recid'].mean())

    # Save test files
    df_test.to_csv(output + '/test_' + file_name + '.csv')
    df_test_new.to_csv(output + '/test_new_' + file_name + '.csv')

    end_time = time.time() - start_time

    runtimes.append(['Calmon', split_num, end_time])

    # increment split number
    split_num += 1

runtimes_df = pd.DataFrame(runtimes, columns=['Method', 'Fold', 'Runtimes'])
runtimes_df.to_csv(output + '/runtimes.csv')

