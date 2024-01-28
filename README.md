# Gas Turbine Modeling

## Overview

This project focuses on modeling gas turbine behavior using a novel approach that integrates knowledge-guided deep
learning. It addresses the challenges in dynamical systems modeling, where traditional deep learning methods often
struggle due to physical inconsistencies and generalization issues. The project aims to provide a more accurate,
reliable, and physically consistent model for gas turbines.

## Getting Started

To set up the project, install the necessary packages using the provided `env.yml` file:

```bash
conda env create - f env.yml
```

## Data Handling and Modeling

- `src/dataset.py`: Manages data setup and preprocessing for both synthetic and experimental data types.
- `src/model.py`: Manages model setup, training, and evaluation.
- 'src/loss_functions.py': Manages domain-informed loss functions for training. It incorporates multi-state constraints
  informed by domain knowledge, enhancing prediction accuracy.

## Visualization and Analysis

Jupyter notebooks like `Collect_Results.ipynb` and `Create_Plots.ipynb` are provided for data analysis and creating
visualizations that aid in understanding model performance.

## Usage

To use this project, start by setting up the environment and then follow the example provided in `Example.ipynb` to get
a sense of the workflow. `run_experiments.py` is provided as a template for running experiments.
