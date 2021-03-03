# Complexity-Driven Feature Construction

Feature engineering is a critical but time-consuming task in machine learning.
In particular, in cases where raw features can be transformed and combined into new features, the search space is exponentially large.
Existing feature selection methods try to identify the best representations. However, the selected feature representations are often very complex, hard to understand, and might suffer from overfitting.
Therefore, we propose a system that leverages feature set complexity to prune the huge feature search space.
Preliminary experiments show that our system generates representations that are less complex, yield higher classification accuracy, and generalize better to unseen data than current state-of-the-art feature selection and construction methods.

## Using our system
To run the experiments, first, you need to set the paths in a configuration file with the name of your machine. Examples can be found here: ~/new_project/fastsklearnfeature/configuration/resources

We provide a small jupyter notebook as an example: [Example Notebook](../master/new_project/fastsklearnfeature/documentation/Example.ipynb)

## Setup 
```
cd new_project/
conda create -n vldb_revision -c conda-forge r-base=3.6.1 python=3.8
conda activate vldb_revision
python -m pip install .
```

## Experiments
We already applied our system for the datasets [Blood Transfusion Service Center](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center), [Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication), [Ecoli](https://archive.ics.uci.edu/ml/datasets/ecoli), [Statlog (Heart)](http://archive.ics.uci.edu/ml/datasets/statlog+(heart)), [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)), [House Prices](https://www.kaggle.com/surya635/house-price-prediction):

<img src="https://user-images.githubusercontent.com/5217389/54563865-a6131280-49ca-11e9-8fde-eba0feb4f3ee.png" align="left" width="300" >
<img src="https://user-images.githubusercontent.com/5217389/54511804-5db80e00-4952-11e9-98c8-4f76b56c76e0.png" align="left" width="300" >
<img src="https://user-images.githubusercontent.com/5217389/54512643-2a2ab300-4955-11e9-84e9-2ea661bcbcda.png" align="left" width="300" >
<img src="https://user-images.githubusercontent.com/5217389/54512707-5d6d4200-4955-11e9-96ca-07ea912598d4.png" align="left" width="300" >
<img src="https://user-images.githubusercontent.com/5217389/55276710-0c673180-52f7-11e9-903d-cc49cc7294b3.png" align="left" width="300" >
<img src="https://user-images.githubusercontent.com/5217389/55877226-cc773880-5b99-11e9-9d1c-8ec06738b187.png" align="left" width="300" >





