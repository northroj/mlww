[![Testing Suite](https://github.com/northroj/mlww/actions/workflows/run-tests.yml/badge.svg)](https://github.com/northroj/mlww/actions/workflows/run-tests.yml)   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15033855.svg)](https://doi.org/10.5281/zenodo.15033855)

# mlww

Machine learning weight window generation for Monte Carlo neutron transport.

This python package can randomly generate input decks for the Monte Carlo neutron transport code [MC/DC](https://github.com/CEMeNT-PSAAP/MCDC), perform simulations to create a dataset of input parameters and flux tallies for training machine learning models, train convolutional neural networks from the data, and extract flux predictions from the models to create automatic weight window maps for MC/DC simulations.

## Install Instructions

Set up a conda environment (venv and other versions of python should work but the author has only tested conda on 3.11).

Clone the repository.

Pip install it.

```
conda create --name mlww_env python=3.11
conda activate mlww_env
git clone https://github.com/northroj/mlww.git
cd mlww
pip install .
```

## Basic Usage

The examples folder contains prebuilt scripts to exercise the code.

- example_generate.py randomly generates input grids for the simulations, builds the MC/DC input decks for each random case, performs MC/DC simulations to collect flux tallies, and stores the input and output data in hdf5 files for use with the machine learning models.
- example_train.py trains a CNN on the data files, calculates training and validation accuracy and saves the model to the models folder.
- example_use.py uses the default model included in the repository to predict the flux and create weight windows for an example input generation. The problem is simulated in MC/DC with and without the weight windows to compare efficiency gains. *<span style="color:#FF5733"> This is recommended for basic use with the pretrained model included in the repository. Running example_generate will take a long time to produce enough results to train a new model on. </span>*
- example_test.py just runs the testing suite that exists in the tests folder










