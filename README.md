# Case-based off-policy evaluation using prototype learning

This repository contains code to run the experiments in the [UAI 2022 paper](https://www.auai.org/uai2022/) _Case-based off-policy evaluation using prototype learning_.
The experiments are defined in `src/cased_based_ope/sepsis` and `src/cased_based_ope/sepsis_sim` and utilize code provided by [Komorowski et al.](https://github.com/matthieukomorowski/AI_Clinician) and [Obserst and Sontag](https://github.com/clinicalml/gumbel-max-scm/tree/master), respectively. All settings for the experiments are defined in `sepsis_config.yml` and `sepsis_sim_config.yml`.

## Installation

```bash
$ conda create --name case_based_ope_env --file <Conda lockfile>
$ conda activate case_based_ope_env
$ poetry install
```

## Collect sepsis dataset

The main sepsis experiment is based on data from the [MIMIC-III database](https://physionet.org/content/mimiciii/1.4/). 
The final dataset cannot be publically shared, but it is possible to recreate this dataset if one has access to the MIMIC-III database.
To create the dataset, one should run the notebook `src/case_based_ope/sepsis/ai_clinician/AIClinician_Data_extract_MIMIC3_140219.ipynb` provided by [Komorowski et al.](https://github.com/matthieukomorowski/AI_Clinician) to extract raw data from the database.
The extracted data should be placed in the folder `data/sepsis/raw/`. Then, one can simply type the following commands to create the dataset:
```bash
$ export MATLAB_PATH=<path to Matlab installation>
$ $MATLAB_PATH/matlab -r "addpath('src/case_based_ope/sepsis/'); create_dataset('data/sepsis/')" < /dev/null
```

## Generate paper results

Download the compressed results and extract the files:
```bash
$ wget https://github.com/antmats/case_based_ope/releases/download/v0.1.0/results.zip 
$ tar -xzf results.zip
```

Recreate results for the main sepsis experiment:
```bash
$ python -m case_based_ope.sepsis.recreate_paper_results -c sepsis_config.yml
```

Recreate results for the simulated sepsis experiment:
```bash
$ python -m case_based_ope.sepsis_sim.recreate_paper_results -c sepsis_sim_config.yml
```

## Start from scratch

Build the AI Clinician:
```bash
$ export MATLAB_PATH=<path to Matlab installation>
$ $MATLAB_PATH/matlab -r "addpath('src/case_based_ope/sepsis/'); \
> build_ai_clinician('data/sepsis/interim/ai_clinician_workspace.mat', 'results/sepsis/))" < /dev/null
```

Train models of the behavior policy:
```bash
$ python -m case_based_ope.sepsis.train_models -c sepsis_config.yml
```

Evaluate models of the behavior policy:
```bash
$ python -m case_based_ope.sepsis.evaluate_models -c sepsis_config.yml
```

Perform policy evaluation:
```bash
$ python -m case_based_ope.sepsis.perform_policy_evaluation -c sepsis_config.yml
```

To run the code for the simulated sepsis experiment, one must first collect the "true" MDP parameters by executing the notebook `src/case_based_ope/sepsis_sim/gumbel_max_scm/learn_mdp_parameters.ipynb` provided by [Obserst and Sontag](https://github.com/clinicalml/gumbel-max-scm/tree/master). The notebook outputs the file `data/sepsis_sim/diab_txr_mats-replication.pkl`. To run the experiment, then type:
```bash
$ python -m case_based_ope.sepsis_sim.run_experiment -c sepsis_sim_config.yml
```

## License

`case_based_ope` was created by Anton Matsson. It is licensed under the terms of the MIT license.

## Acknowledgements

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. 

The computations in this work were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2018-05973.