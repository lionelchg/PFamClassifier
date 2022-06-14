# PFamClassifier repository

This is a repository for classifying protein domains of the PFam dataset available at on [Kaggle](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split). Mirror of a cleaned private repository done for a coding challenge.

## Overview of the repository

The repository contains several directories:

1. `code/`: this is the main code for running the training and get the results of the neural networks
2. `latex/`: directory containing the LaTeX source files as well as the `main.pdf` report
3. `proteinclass/`: Python library written and used in `code/` written using [PyTorch](https://pytorch.org) for the deep learning part.

## Structure of the `proteinclass/` library

The library is written in an object-oriented fashion and is inspired by the [pytorch-template](https://github.com/victoresque/pytorch-template) made by @victoresque on GitHub and the [PlasmaNet](https://github.com/lionelchg/PlasmaNet) project. The content of each file is described below:

- `dataloader.py`: contains the `ProteinDataset` class which overloads the base `Dataset` class from PyTorch and reads the PFam database
- `log.py`: contains functions for creating logging objects
- `model.py`: contains the neural network architectures. A base class is first written which inherits from the `nn.Module` class of PyTorch. This base class holds a method for computing the number of training parameters and the other three classes inherit from it.
- `multiple_train.py`: script to launch multiple trainings sequentially
- `pproc.py`: routines for post-processing the `metrics.h5` generated during training
- `predict.py`: script for inference of a given set of sequences
- `train.py`: performs the actual training of neural networks
- `trainer.py`: contains the `Trainer` class instantiated in `train.py`
- `util.py`: contains common functions used across the library

The library is called via executables created in `setup.py`:

```python
entry_points={
    'console_scripts': [
        'train_network=proteinclass.train:main',
        'train_networks=proteinclass.multiple_train:main',
        'plot_metrics=proteinclass.pproc:main',
        'predict=proteinclass.predict:main'
    ],
},
```

Each of these executables are run via a configuration file written in `YAML` format (for example `code/results/200labels/simple.yml` for the `train_network` executable). These configuration files are converted into Python dictionnaries which are parsed throughout the code.

To install the library you need Python 3.8 or above version. Run the following command at the root of the repository (preferably in a virtual environment):

```shell
pip install -e .
```

If there are some missing packages, an image of a working python 3.9 version on Mac can be found in `requirements.txt` and so the following command should be run:

```shell
pip install -r requirements.txt
```

To run PyTorch on GPUs special versions of PyTorch should be installed with CUDA. Please refer to this [link](https://pytorch.org/get-started/previous-versions/) for a proper installation. For PyTorch 1.10.1 with CUDA 11.1 on Linux for example run the following command on terminal:

```shell
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Run the code

To run the code the dataset from https://www.kaggle.com/googleai/pfam-seed-random-split should be downloaded and put inside the `code/` repository so that there is an `code/input/random_split` directory.

After that the 200 labels training can be launched by executing `run.sh` inside `results/200labels/`:

```shell
./run.sh
```

The training done using 5000 and 15000 labels can be performed by executing `run.sh` inside the `results/MoreLabels/` directory.

## Dataset analysis

All the figures of the first part of the report can be plotted by running the `main.py` script inside `code/dataset_analysis`:

```shell
cd code/dataset_analysis
python main.py
```
