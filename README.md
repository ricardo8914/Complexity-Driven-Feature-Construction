# Automated Feature Engineering for Algorithmic Fairness

One of the fundamental problems of machine ethics is to avoid the perpetuation and amplification of discrimination through machine learning applications. In particular, it is desired to exclude the influence of attributes with sensitive information, such as gender or race, and other causally related attributes on the machine learning task. The state-of-the-art bias reduction algorithm Capuchin breaks the causality chain of such attributes by adding and removing tuples. However, this horizontal approach can be considered invasive because it changes the data distribution. A vertical approach would be to prune sensitive features entirely. While this would ensure fairness without tampering with the data, it could also hurt the machine learning accuracy. Therefore, we propose a novel multi-objective feature selection strategy that leverages feature construction to generate more features that lead to both high accuracy and fairness. On three well-known datasets, our system achieves higher accuracy than other fairness-aware approaches while maintaining similar or higher fairness.

## Setup 
```
cd new_project/
conda create -n fairexp -c conda-forge r-base=3.6.1 python=3.8
conda activate fairexp
python -m pip install .
```
