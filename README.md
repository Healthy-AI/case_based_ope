# Case-based off-policy evaluation using prototype learning

This repository contains code to run the experiments in the UAI 2022 paper _Case-based off-policy evaluation using prototype learning_.
The two experiments---sepsis and sepsis_sim---are based on code provided by [Komorowski et al](https://github.com/matthieukomorowski/AI_Clinician) and [Obserst and Sontag](https://github.com/clinicalml/gumbel-max-scm/tree/master), respectively.

## Installation

```bash
$ conda create --name case_based_ope_env --file <Conda lockfile>
$ conda activate case_based_ope_env
$ poetry install
```

## License

`case_based_ope` was created by Anton Matsson. It is licensed under the terms of the MIT license.

## Acknowledgements

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. 
The computations in this work were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2018-05973.