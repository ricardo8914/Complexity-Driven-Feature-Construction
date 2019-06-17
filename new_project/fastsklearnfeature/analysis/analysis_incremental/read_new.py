import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt

#path = '/home/felix/phd/fastfeatures/results/11_03_incremental_construction'
#path = '/home/felix/phd/fastfeatures/results/12_03_incremental_03_threshold'
#path = '/home/felix/phd/fastfeatures/results/12_03_incremental_02_threshold'
#path = '/tmp'
#path = '/home/felix/phd/fastfeatures/results/15_03_timed_transfusion'
#path = '/home/felix/phd/fastfeatures/results/15_03_timed_transfusion_node1'
#path = '/home/felix/phd/fastfeatures/results/16_03_test_transfusion_me'
#path = '/home/felix/phd/fastfeatures/results/18_03_banknote'
#path = '/home/felix/phd/fastfeatures/results/18_03_iris'
#path = '/home/felix/phd/fastfeatures/results/20_03_transfusion'
path = '/Users/rsd8914/Complexity-Driven-Feature-Construction/new_project/data/fastfeature_logs'


cost_2_raw_features: Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed : Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed: Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination : Dict[int, List[CandidateFeature]]  = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))

#find best candidate per cost


#print(cost_2_unary_transformed)

for k,v in cost_2_raw_features.items():
    for candidate in v:
        c='fair_test_score'
        if c in list(candidate.runtime_properties.keys()):
            print(str(candidate) + ": Test Score = "+ "{0:.2f}".format(candidate.runtime_properties['test_score']) + ' ,Fair Test Score = ' + "{0:.2f}".format(candidate.runtime_properties['fair_test_score']))

for k,v in cost_2_binary_transformed.items():
    for candidate in v:
        c='fair_test_score'
        if c in list(candidate.runtime_properties.keys()):
            print(str(candidate) + ": Test Score = "+ "{0:.2f}".format(candidate.runtime_properties['test_score']) + ' ,Fair Test Score = ' + "{0:.2f}".format(candidate.runtime_properties['fair_test_score']))
