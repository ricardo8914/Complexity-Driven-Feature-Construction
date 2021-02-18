from tqdm import tqdm
from benchmark.calmon.DTools3 import *
import time
from pathlib import Path
from sklearn.model_selection import KFold

home = str(Path.home())


capuchin_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/capuchin'
feldman_folder = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/feldman'

df = pd.read_csv(capuchin_path + '/bin_german_credit.csv', sep=',', header=0)

df.drop(columns=['Unnamed: 0', 'purpose', 'personal_status_sex', 'other_debtors',
                                 'property', 'other_installment_plans', 'housing', 'job', 'telephone',
                                 'foreign_worker', 'credit_history'], inplace=True)

output = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/calmon_output/german'


def quantizeCheckStatus(x):
    if x['account_check_status'] in ('< 0 DM', 'no checking account'):
        return 0
    elif x['account_check_status'] == '0 <= ... < 200 DM':
        return 1
    else:
        return 2


def quantizeDuration(x):
    if x['duration_in_month'] <= 3:
        return 0
    if 3 < x['duration_in_month'] <= 6:
        return 1
    else:
        return 2


def quantizeCreditAmount(x):
    if x['credit_amount'] <= 3:
        return 0
    if 3 < x['credit_amount'] <= 6:
        return 1
    else:
        return 2


def quantizeSavings(x):
    if x['savings'] in ('unknown/ no savings account', '... < 100 DM'):
        return 0
    elif x['savings'] in ('100 <= ... < 500 DM', '500 <= ... < 1000 DM '):
        return 1
    else:
        return 2


def quantizeEmpl(x):
    if x['present_emp_since'] in ('unemployed', '... < 1 year '):
        return 0
    elif x['present_emp_since'] in ('1 <= ... < 4 years', '4 <= ... < 7 years'):
        return 1
    else:
        return 2

transformations = [('account_check_status', quantizeCheckStatus), ('duration_in_month', quantizeDuration),
                   ('credit_amount', quantizeCreditAmount), ('savings', quantizeSavings),
                   ('present_emp_since', quantizeEmpl)]

def prepare_df_calmon(df):
    df_ = df.copy()

    for n, t in transformations:
        df_[n] = df_.apply(lambda row: t(row), axis=1)

    return df_

df_enc = prepare_df_calmon(df)

list(df_enc)

features = ['account_check_status', 'duration_in_month', 'credit_amount', 'savings', 'present_emp_since',
            'people_under_maintenance'] + ['age'] + ['default']
D_features = ['age']
Y_features = ['default']
X_features = [i for i in features if i not in ('age', 'default')]

kf1 = KFold(n_splits=5, random_state=42, shuffle=True)  ### CHANGE SEED FOR DIFFERENT SPLITS!
df_list = []
for train_index, test_index in kf1.split(df_enc):
    df_list.append((df_enc.iloc[train_index, :].copy(), df_enc.iloc[test_index, :].copy()))

print(list(df))

class Dclass():

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
        bad_val = 3.0

        count_one_jumps = 0
        for f in vold.keys():
            if np.abs(vold[f]-vnew[f]) == 1:
                count_one_jumps += 1

        count_two_jumps = 0
        for f in vold.keys():
            if np.abs(vold[f]-vnew[f]) == 2:
                count_two_jumps += 1

        count_three_jumps = 0
        for f in vold.keys():
            if np.abs(vold[f] - vnew[f]) == 3:
                count_three_jumps += 1

        if count_three_jumps >= 2:
            return bad_val
        elif count_two_jumps > 2:
            return 2.0
        elif count_one_jumps > 3:
            return 1.0
        else:
            return 0

c1 = .99 # value of (delta1,c1): to keep.
c2 = 1.99  # value of (delta2,c2): value that should no happen
c3 = 2.99 # penalty for adjusting age
clist = [c1, c2, c3]
Dclass = Dclass()

dlist = [.1, 0.05, 0]
epsilon = .05

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

    print('DT created')

    # Set features
    DT.setFeatures(D=D_features, X=X_features, Y=Y_features)

    print('Features were set')

    # Set Distortion
    DT.setDistortion(Dclass, clist=clist)

    print('Distortion was applied')

    # solve optimization for previous parameters -- This uses and older implementation, based on the FATML submission.
    print('Start optimization')
    DT.optimize(epsilon=epsilon, dlist=dlist, verbose=True)

    print('Start computing marginals')
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

    # Save test files
    df_test.to_csv(output + '/test_' + file_name + '.csv')
    df_test_new.to_csv(output + '/test_new_' + file_name + '.csv')

    end_time = time.time() - start_time

    runtimes.append(['Calmon', split_num, end_time])

    print(df_train.groupby('age')['default'].mean())
    print(df_train_new.groupby('age')['default'].mean())

    # increment split number
    split_num += 1

runtimes_df = pd.DataFrame(runtimes, columns=['Method', 'Fold', 'Runtimes'])
runtimes_df.to_csv(output + '/runtimes.csv')