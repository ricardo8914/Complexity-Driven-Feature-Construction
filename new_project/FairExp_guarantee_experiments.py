from FairExp_Adult_guarantee import adult_experiment
from FairExp_COMPAS_guarantee import COMPAS_experiment
from FairExp_German_guarantee import german_experiment
import pandas as pd
import multiprocessing as mp
from pathlib import Path

from fastsklearnfeature.configuration.Config import Config
home = str(Config.get('path_to_project'))
#home = str(Path.home())

results_path = Path(home + '/Complexity-Driven-Feature-Construction/results')
results_path.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    mp.set_start_method('fork')

    adult_results = adult_experiment()
    compas_results = COMPAS_experiment()
    german_results = german_experiment()

    all_results = pd.concat([adult_results, compas_results, german_results], axis=0, ignore_index=True)

    all_results.to_csv(
        path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/FairExp_guarantee_experiments.csv',
        index=False)




