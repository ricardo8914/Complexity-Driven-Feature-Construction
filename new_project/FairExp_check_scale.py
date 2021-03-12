from FairExp_Adult import adult_experiment
from FairExp_COMPAS import COMPAS_experiment
from FairExp_German import german_experiment
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

    german_results = german_experiment(number_model_parallelism=1, number_speculativ_parallelism=1,
                                       number_kfold_parallelism=1)
    german_results.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/mp1,sp1,fp1.csv',
                          index=False)

    german_results = german_experiment(number_model_parallelism=mp.cpu_count(), number_speculativ_parallelism=1, number_kfold_parallelism=1)
    german_results.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/mpA,sp1,fp1.csv', index=False)

    german_results = german_experiment(number_model_parallelism=1, number_speculativ_parallelism=1,
                                       number_kfold_parallelism=5)
    german_results.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/mp1,sp1,fp5.csv',
                          index=False)

    german_results = german_experiment(number_model_parallelism=1, number_speculativ_parallelism=mp.cpu_count(),
                                       number_kfold_parallelism=1)
    german_results.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/mp1,spA,fp1.csv',
                          index=False)

    german_results = german_experiment(number_model_parallelism=1, number_speculativ_parallelism=5,
                                       number_kfold_parallelism=5)
    german_results.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/mp1,sp5,fp5.csv',
                          index=False)

    german_results = german_experiment(number_model_parallelism=mp.cpu_count(), number_speculativ_parallelism=5,
                                       number_kfold_parallelism=5)
    german_results.to_csv(path_or_buf=home + '/Complexity-Driven-Feature-Construction/results/mpA,sp5,fp5.csv',
                          index=False)




