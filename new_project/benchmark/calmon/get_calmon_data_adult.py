from tqdm import tqdm
from benchmark.calmon.DTools3 import *
import time
from pathlib import Path
from sklearn.model_selection import KFold

home = str(Path.home())


path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data'

output = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/calmon_output/adult'

df = pd.read_csv(path + '/adult.csv', sep=',', header=None, names=["age", "workclass", "fnlwgt",
                        "education", "education_num",
                        "marital_status", "occupation",
                        "relationship", "race", "sex",
                        "capital_gain", "capital_loss",
                        "hours_per_week", "native_country", "income"], na_values='?')


def group_edu(x):
    if x <= 5:
        return '<6'
    elif x >= 13:
        return '>12'
    else:
        return x


def age_cut(x):
    if x >= 70:
        return '>=70'
    else:
        return x


# Limit education range
df['Education Years'] = df['education_num'].apply(lambda x: group_edu(x))

# Limit age range
df['Age (decade)'] = df['age'].apply(lambda x: age_cut(x))

# Transform all that is non-white into 'minority'
df['race'] = df['race'].apply(lambda x: x if x == ' White' else 'Minority')

df['Income Binary'] = df['income'].apply(lambda x : 1 if x == " >50K" else 0)


#features = ['Age (decade)','Education Years','Income','Gender','Race','Income Binary']
features = ['Age (decade)', 'Education Years', 'income', 'sex', 'Income Binary']
#D_features = ['Gender','Race']
D_features = ['sex']
Y_features = ['Income Binary']
X_features = ['Age (decade)', 'Education Years']

# keep only the features we will use
df = df[features]


kf1 = KFold(n_splits=5, random_state=42, shuffle=True)  ### CHANGE SEED FOR DIFFERENT SPLITS!
df_list = []
for train_index,test_index in kf1.split(df):
    df_list.append((df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()))


class Dclass():
    # adjust education
    def adjustEdu(self, v):
        if v == '>12':
            return 13
        elif v == '<6':
            return 5
        else:
            return v

    def adjustAge(self, a):
        if a == '>=70':
            return 70.0
        else:
            return a

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

        # Adjust education years
        eOld = self.adjustEdu(vold['Education Years'])
        eNew = self.adjustEdu(vnew['Education Years'])

        # Education cannot be lowered or increased in more than 1 year
        if (eNew < eOld) | (eNew > eOld + 1):
            return bad_val

        # adjust age
        aOld = self.adjustAge(vold['Age (decade)'])
        aNew = self.adjustAge(vnew['Age (decade)'])

        # Age cannot be increased or decreased in more than a decade
        if np.abs(aOld - aNew) > 10.0:
            return bad_val

        # Penalty of 2 if age is decreased or increased
        if np.abs(aOld - aNew) > 0:
            return 2.0

        # final penalty according to income
        if vold['Income Binary'] > vnew['Income Binary']:
            return 1.0
        else:
            return 0.0

c1 = .99 # value of (delta1,c1): to keep.
c2 = 1.99  # value of (delta2,c2): value that should no happen
c3 = 2.99 # penalty for adjusting age
clist = [c1, c2, c3]
Dclass = Dclass()

#DT = DTools(df=df, features=features)

# Set features
#DT.setFeatures(D=D_features, X=X_features, Y=Y_features)

# Set Distortion
#DT.setDistortion(Dclass, clist=clist)

dlist = [.1, 0.05, 0]
epsilon = .05

######### CHANGE SEED HERE ###########
seed = sum([ord(b) for b in 'fairexp'])
np.random.seed(seed=seed)


####################################

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

    # increment split number
    split_num += 1

runtimes_df = pd.DataFrame(runtimes, columns=['Method', 'Fold', 'Runtimes'])
runtimes_df.to_csv(output + '/runtimes.csv')



