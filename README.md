# L-CODE: Labelless Concept Drift Detection and Explanation

This repository provides supplementary materials and source code for reproduction of the results in the L-CODE paper. 

## Requirements

This code has been tested with Python 3.8.1 and the following packages:

    scipy==1.6.2
    numpy==1.19.2
    scikit-learn==0.24.1
    matplotlib==3.4.1
    h5py==2.10.0
    xlrd==2.0.1
    pandas==1.2.4
    seaborn==0.11.1
    natsort==7.1.1
    shap==0.39.0
    scikit-multiflow==0.5.3
    autorank==1.1.1
    Orange3==3.28.0
    tables==3.6.1
    statsmodels==0.12.2

    # just for IncrementalKS
    cffi==1.14.5

You also need to install Python C++ Wrapper version of [IncrementalKS](https://github.com/denismr/incremental-ks) to test the Incremental Kolmogorov Smirnov algorithm in our experiments.

## Contents
### notebooks
- Experiment_1
  - Notebooks for the experiment on the UCI datasets.
- Experiment_2
  - Notebooks for the experiment on the USP DS datasets.
- Experiment_3 (supplementary material)
  - Notebooks to see the robustness of L-CODE with respect to the parameter setting.
- Experiment_4 (supplementary material)
  - Notebooks for the comparison of the running time of drift detection methods.

### src
Python source codes used in the notebooks.

## Reference
