# Gas Turbine Modeling

## Overview

This project focuses on modeling gas turbine behavior using a novel approach that integrates knowledge-guided deep
learning. It addresses the challenges in dynamical systems modeling, where traditional deep learning methods often
struggle due to physical inconsistencies and generalization issues. The project aims to provide a more accurate,
reliable, and physically consistent model for gas turbines.

## Getting Started

To set up the environment for this project:

```bash
conda env create - f env.yml
```

## Code Structure

- `src/dataset.py`: Data loading and preprocessing functionalities.
- `src/model.py`: LSTM for gas turbine simulation.
- `src/loss_functions.py`: Custom loss function incorporating knowledge-guided constraints.
- `src/utils.py`: Utility functions supporting data handling and model operations.

## Key Features

- **Comprehensive Experiment Suite**: Script (`run_experiments.py`) to facilitate the execution of various modeling
  experiments.
- **Result Analysis Tools**: Jupyter notebooks (`Collect_Results.ipynb` and `Create_Plots.ipynb`) for aggregating and
  visualizing experiment outcomes.
- **Example Demonstrations**: `Example.ipynb` provides a walkthrough of the model's usage and capabilities.