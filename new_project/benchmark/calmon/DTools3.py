"""
Created on Fri Aug  5 11:17:36 2016
@author: fdcalmon
"""

import numpy as np
import pandas as pd
from cvxpy import *
from itertools import product


class DTools():
    '''
    Class that contains the optimization problem for the data discrimination problem.
    The class is initialized with a data frame, and a set of features (columns) of the
    data frame that will be used for computing the joint frequency (estimated distribution)
    across the dataset.
    '''

    def __init__(self, df=[], features=[]):
        self.df = df.copy()

        if len(df) == 0:
            print('Initialize with a dataframe...')
            return

        if not features:
            self.features = list(df)
        else:
            self.features = features

        # build joint distribution
        self.dfJoint = self.df.groupby(self.features).size().reset_index()
        self.dfJoint.rename(columns={0: 'Count'}, inplace=True)
        # print 'Here'
        self.dfJoint['Frequency'] = self.dfJoint['Count'].apply(lambda x: x / float(len(self.df)))

        # initialize the features that will be used for optimization
        self.D_features = []  # discriminatory features
        self.Y_features = []  # binary decision variable
        self.X_features = []  # variables used for decision making

        # values that each feature can assume
        self.D_values = []
        self.Y_values = []

        # place holder for mapping dataframe
        self.dfP = []  # this will hold the conditional mappings

        # place holder for the distortion mapping
        self.dfD = []

        # excess distortion constraint placeholder
        self.clist = []

        # excess distortion matrices
        self.CMlist = []

    # this is a general function that creates a mask based on overlapping,
    # multilevel indices. It is not a beautiful function, but it's what we have.
    def getMask(self, dfRef):
        # generates a mask assuming the multindex column is a subset of the multindex rows
        dfRows = pd.DataFrame(index=dfRef.index).reset_index()
        dfCols = pd.DataFrame(index=dfRef.columns).reset_index()
        target_ix = list(dfRef.columns.names)

        val_1 = dfRows.loc[:, target_ix].to_numpy()
        val_2 = dfCols.loc[:, target_ix].to_numpy()

        dfRef_c = dfRef.copy()
        for idx1, x1 in enumerate(val_1):
            test = np.equal(val_2, x1)
            f = np.argwhere(np.all(test, axis=1))[0]
            dfRef_c.iloc[idx1, f[0]] = 1.0

        #dfRef_c.iloc[idx1, f[0]] = 1.0

        # print('done with matrix')
        #
        #
        # for i in range(dfRef.shape[0]):
        #     val1 = dfRows.loc[i, target_ix].tolist()
        #     for j in range(dfRef.shape[1]):
        #         val2 = dfCols.loc[j, target_ix].tolist()
        #         if all([x == y for (x, y) in zip(val1, val2)]):
        #             dfRef.iloc[[i], [j]] = 1.0
        #
        # print(dfRef_c.equals(dfRef))

        return dfRef_c

    # method for setting the features
    def setFeatures(self, D=[], X=[], Y=[]):
        self.D_features = D
        self.Y_features = Y
        self.X_features = X

        # Get values for Pandas multindex
        self.D_values = [self.dfJoint[feature].unique().tolist() for feature in self.D_features]
        self.Y_values = [self.dfJoint[feature].unique().tolist() for feature in self.Y_features]
        self.X_values = [self.dfJoint[feature].unique().tolist() for feature in self.X_features]

        print('got values for pd MI')

        # Create multindex for mapping dataframe
        self.DXY_features = self.D_features + self.X_features + self.Y_features
        self.DXY_values = self.D_values + self.X_values + self.Y_values
        self.DXY_index = pd.MultiIndex.from_product(self.DXY_values, names=self.DXY_features)

        print('MI for mapping was created')

        # Create multindex for distortion dataframe
        self.XY_features = self.X_features + self.Y_features
        self.XY_values = self.X_values + self.Y_values
        self.XY_index = pd.MultiIndex.from_product(self.XY_values, names=self.XY_features)

        print('MI for distortion was created')

        # Initialize mapping dataframe
        self.dfP = pd.DataFrame(np.zeros((len(self.DXY_index), len(self.XY_index))),
                                index=self.DXY_index, columns=self.XY_index)

        # Initialize distortion dataframe
        self.dfD = pd.DataFrame(np.zeros((len(self.XY_index), len(self.XY_index))),
                                index=self.XY_index.copy(), columns=self.XY_index.copy())

        print('DF were created')

        ###
        # Generate masks for recovering marginals
        ###
        self.dfPxyd = pd.DataFrame(index=self.dfP.index, columns=['Frequency'])
        index_list = [list(x) for x in self.dfPxyd.index.tolist()]

        print('masks were generated')

        # find corresponding frequency value
        i = 0
        for comb in self.dfJoint[self.DXY_features].values.tolist():
            idx = index_list.index(comb)  # get the entry corresponding to the combination
            # add marginal to list
            self.dfPxyd.iloc[idx, 0] = self.dfJoint.loc[i, 'Frequency']
            i += 1

        print('frequencies were found')

        # create mask that reduces Pxyd to Pxy
        # so Pxyd.dot(dfMask1) = Pxy
        self.dfMask_Pxyd_to_Pxy = pd.DataFrame(np.zeros((len(self.dfP), len(self.dfD))), index=self.dfP.index,
                                               columns=self.dfD.index)
        print('getting mask')
        self.dfMask_Pxyd_to_Pxy = self.getMask(self.dfMask_Pxyd_to_Pxy)

        print('created mask that reduces Pxyd to Pxy')

        # compute mask that reduces Pxyd to Pyd
        self.YD_features_index = self.dfJoint.groupby(self.Y_features + self.D_features)['Frequency'].sum().index
        self.dfMask_Pxyd_to_Pyd = pd.DataFrame(np.zeros((len(self.dfP), len(self.YD_features_index))),
                                               index=self.dfP.index, columns=self.YD_features_index)
        self.dfMask_Pxyd_to_Pyd = self.getMask(self.dfMask_Pxyd_to_Pyd)

        # get  matrix for p_yd, with y varying in the columns
        self.dfD_to_Y_address = pd.DataFrame(range(len(list(self.dfMask_Pxyd_to_Pyd))),
                                             index=self.dfMask_Pxyd_to_Pyd.columns)
        self.dfD_to_Y_address = pd.pivot_table(self.dfD_to_Y_address.reset_index(), columns=self.D_features,
                                               index=self.Y_features, values=0)

        # compute mask that reduces Pxyd to Py
        self.y_index = self.dfD_to_Y_address.index
        self.dfMask_Pxyd_to_Py = pd.DataFrame(np.zeros((len(self.dfP), len(self.y_index))),
                                              index=self.dfP.index, columns=self.y_index)
        self.dfMask_Pxyd_to_Py = self.getMask(self.dfMask_Pxyd_to_Py)

        # compute mask that reduces Pxy to Py
        self.dfMask_Pxy_to_Py = pd.DataFrame(np.zeros((len(list(self.dfP)), len(self.y_index))),
                                             index=self.dfP.columns, columns=self.y_index)
        self.dfMask_Pxy_to_Py = self.getMask(self.dfMask_Pxy_to_Py)

        # compute mask that reduces Pxyd to Pd
        self.dfMask_Pxyd_to_Pd = pd.DataFrame(np.zeros((len(self.dfP), self.dfD_to_Y_address.shape[1])),
                                              index=self.dfP.index, columns=self.dfD_to_Y_address.columns)
        self.dfMask_Pxyd_to_Pd = self.getMask(self.dfMask_Pxyd_to_Pd)

        print('done setting the features')

        # method for passing the distortion value

    def setDistortion(self, Dclass, clist=[]):
        '''
            Inputs:
            * 'Dclass' is a class with a method 'getDistortion' that computes the distortion between two dictionaries of features
            * clist: list of excess disotion constraints. Sould be of the form (delta_i,c_i) (see paper)
        '''

        # set constraint list
        self.clist = clist

        # create row dictionay (rows represent old values)
        # this will make it easier to compute distrotion metric
        rows_tuple = self.dfD.index.tolist()
        rows_dict = [{self.XY_features[i]: t[i] for i in range(len(self.XY_features))} for t in rows_tuple]

        # create columns dictionay (columns represent new values)
        cols_tuple = self.dfD.columns.tolist()
        cols_dict = [{self.XY_features[i]: t[i] for i in range(len(self.XY_features))} for t in cols_tuple]

        p = product(rows_dict, cols_dict)

        def map_distortion(p):
            return Dclass.getDistortion(*p)

        l = list((map(map_distortion, p)))
        l = np.asarray(l)
        l = l.reshape((self.dfD.shape[0], self.dfD.shape[1]))

        dfD_c = pd.DataFrame(l, index=self.XY_index.copy(), columns=self.XY_index.copy())
        self.dfD = dfD_c

        # for i in range(self.dfD.shape[0]):
        #     old_values = rows_dict[i]
        #     for j in range(self.dfD.shape[1]):
        #         new_values = cols_dict[j]
        #         self.dfD.iloc[[i], [j]] = Dclass.getDistortion(old_values, new_values)


        Dmatrix = self.dfD.values

        # Create constraint matrix list for excess distortion
        # since old values index the rows, we go through the D matrix line by line, marking as 1 events where
        # the threshold is violated. This will be multiplied by the probability matrix, resulting in the excess
        # distortion metric

        if self.clist:

            self.CMlist = [np.zeros(Dmatrix.shape) for i in range(len(self.clist))]

            # This part of the code is slow and stupid.
            # It can be improved.

            for x in range(len(self.CMlist)):
                c = self.clist[x]
                for i in range(Dmatrix.shape[0]):
                    for j in range(Dmatrix.shape[1]):
                        if Dmatrix[i, j] >= c:
                            self.CMlist[x][i, j] = 1.0

        print('done with distortion')

    def optimize(self, epsilon=1., dlist=[], verbose=True, mean_distortion=1e6):

        self.epsilon = epsilon
        self.dlist = dlist
        self.mean_distortion = mean_distortion

        Pmap = Variable(shape=(self.dfP.shape[0], self.dfP.shape[1]))  # main conditional map
        # PmapXY = Variable(dfP.shape[1],dfP.shape[1])  # main conditional map
        PXhYh = Variable(shape=(self.dfMask_Pxyd_to_Pxy.shape[1]))  # marginal distribution of (Xh Yh)
        PYhgD = Variable(shape=(self.dfD_to_Y_address.shape[1], self.dfD_to_Y_address.shape[0]))  # rows represent p_(y|D)
        PYhD = Variable(shape=(self.dfD_to_Y_address.shape[1], self.dfD_to_Y_address.shape[0]))

        # marginal distribution
        dfMarginal = self.dfJoint.groupby(self.DXY_features)['Frequency'].sum()
        PxydMarginal = pd.concat([self.dfP, dfMarginal], axis=1).fillna(0)['Frequency'].values
        self.PxydMarginal = PxydMarginal
        PyMarginal = PxydMarginal.dot(self.dfMask_Pxyd_to_Py).T
        PdMarginal = PxydMarginal.dot(self.dfMask_Pxyd_to_Pd).T
        PxyMarginal = PxydMarginal.dot(self.dfMask_Pxyd_to_Pxy).T
        # add constraints

        # valid distribution
        constraints = [sum(Pmap, axis=1) == 1]
        constraints.append(Pmap >= 0)

        # definition of marginal PxhYh
        constraints.append(PXhYh == sum(np.diag(PxydMarginal) * Pmap, axis=0).T)

        # Define the joint distribution P_{YH,D}
        # constraints.append(PYhD == (self.dfMask_Pxyd_to_Pd.values.T).dot(np.diag(PxydMarginal))*Pmap*self.dfMask_Pxy_to_Py.values)

        # add the conditional mapping
        # constraints.append(PYhgD == np.diag(PdMarginal**(-1))*PYhD)
        constraints.append(PYhgD == np.diag(np.ravel(PdMarginal) ** (-1)).dot(self.dfMask_Pxyd_to_Pd.values.T).dot(
            np.diag(PxydMarginal)) * Pmap * self.dfMask_Pxy_to_Py.values)

        # add excess distorion
        Pxy_xhyh = np.nan_to_num(np.diag(PxyMarginal ** (-1))).dot(self.dfMask_Pxyd_to_Pxy.values.T).dot(
            np.diag(PxydMarginal)) * Pmap

        for i in range(len(self.CMlist)):
            # Cost = mul_elemwise(self.CMlist[i],Pxy_xhyh)
            constraints.append(sum(multiply(self.CMlist[i], Pxy_xhyh), axis=1) <= self.dlist[i])

        # epsilon definition for discrimination control
        # epsilon = .1
        # disc_upper = Variable()
        # disc_lower = Variable()

        # Discrimination control
        for d in range(self.dfMask_Pxyd_to_Pd.shape[1]):
            # constraints.append(sum(kl_div(PYhgD[d,:].T,PyMarginal))<=self.epsilon)
            # constraints.append(norm(PYhgD[d,:].T-PyMarginal,1)<=disc)
            # constraints.append(PYhgD[d,:].T-PyMarginal<=PyMarginal*(self.epsilon))
            # constraints.append(PYhgD[d,:].T-PyMarginal>=-PyMarginal*(self.epsilon))
            for d2 in range(self.dfMask_Pxyd_to_Pd.shape[1]):
                if d > d2:
                    continue
                constraints.append(PYhgD[d, :].T <= PYhgD[d2, :].T * (1 + self.epsilon))
                constraints.append(PYhgD[d2, :].T <= PYhgD[d, :].T * (1 + self.epsilon))

        # Mean distortion
        # Pxy_xhyhJoint = (self.dfMask_Pxyd_to_Pxy.values.T).dot(np.diag(PxydMarginal))*Pmap
        # constraints.append(sum(mul_elemwise(self.dfD.as_matrix(),Pxy_xhyhJoint),axis=1)<=self.mean_distortion)

        # constraints.append(disc<=self.epsilon)

        # add objective
        # constraints.append(norm(PXhYh-PxyMarginal, 1)<=self.epsilon)

        obj = Minimize(norm(PXhYh - PxyMarginal, 1) / 2)
        # obj = Minimize(sum(kl_div(PXhYh,PxyMarginal)))

        prob = Problem(obj, constraints)
        prob.solve(verbose=verbose)

        self.dfP.loc[:, :] = Pmap.value
        self.optimum = prob.value
        self.const = []
        for i in range(len(self.CMlist)):
            # Cost = mul_elemwise(self.CMlist[i],Pxy_xhyh)
            self.const.append(sum(multiply(self.CMlist[i], Pxy_xhyh), axis=1).value.max())

    def computeMarginals(self):
        # mask = self.dfMask_Pxyd_to_Pxy.T.values
        self.dfFull = pd.DataFrame((np.diag(self.PxydMarginal)).dot(self.dfP.values), index=self.dfP.index,
                                   columns=self.dfP.columns)

        self.dfPyMarginal = pd.DataFrame(self.PxydMarginal.dot(self.dfMask_Pxyd_to_Py).T,
                                         index=self.dfMask_Pxyd_to_Py.columns)
        self.dfPdMarginal = pd.DataFrame(self.PxydMarginal.dot(self.dfMask_Pxyd_to_Pd).T,
                                         index=self.dfMask_Pxyd_to_Pd.columns)
        self.dfPxyMarginal = pd.DataFrame(self.PxydMarginal.dot(self.dfMask_Pxyd_to_Pxy).T,
                                          index=self.dfMask_Pxyd_to_Pxy.columns)

        self.dfPyhgD = pd.DataFrame(
            np.diag(np.ravel(self.dfPdMarginal.values) ** (-1)).dot(self.dfMask_Pxyd_to_Pd.values.T).dot(
                self.dfFull.values).dot(self.dfMask_Pxy_to_Py.values), index=self.dfPdMarginal.index,
            columns=self.dfMask_Pxy_to_Py.columns)

        self.dfPxydMarginal = pd.DataFrame(self.PxydMarginal, index=self.dfMask_Pxyd_to_Pxy.index)

        self.dfPxygdPrior = self.dfPxydMarginal.reset_index().groupby(self.D_features + self.Y_features)[
            0].sum().unstack(self.Y_features)
        self.dfPxygdPrior = self.dfPxygdPrior.div(self.dfPxygdPrior.sum(axis=1), axis=0)

