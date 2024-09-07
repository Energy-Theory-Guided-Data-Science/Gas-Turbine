# Gas Turbine Modeling

## Overview

This repository contains the code to reproduce the experiments from the paper:

**Knowledge-Guided Learning of Temporal Dynamics and its Application to Gas Turbines**  
by Pawel Bielski, Aleksandr Eismont, Jakob Bach, Florian Leiser, Dustin Kottonau, and Klemens Böhm.  
Published in the 15th ACM International Conference on Future Energy Systems (e-Energy '24), Singapore.

[Read the paper on ACM Digital Library](https://dl.acm.org/doi/10.1145/3632775.3661967)

[Watch the conference presentation video](https://www.youtube.com/watch?v=kG1HU2kaaMw&list=PLn0nrSd4xjjbw168up-HQ6ckv6AOczEii&index=9&ab_channel=AssociationforComputingMachinery%28ACM%29), which explains our approach using simple, easy-to-understand toy examples.

For the dataset and experimental results, including the code used to create the figures and the figures themselves, please refer to:

[Dataset on the UCI ML Repository](https://archive.ics.uci.edu/dataset/994/micro+gas+turbine+electrical+energy+prediction)

[Dataset and Experimental Results](https://doi.org/10.35097/sLJiahifxvfDKMEc)


### Project Focus

This project aims to model gas turbine behavior using a novel knowledge-guided deep learning approach. It addresses challenges in modeling dynamical systems, where traditional deep learning methods often struggle due to physical inconsistencies and generalization issues. The goal is to provide a more accurate, reliable, and physically consistent model for gas turbines.


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
- `data` : Place the csv files with experimental data there. You can find the data [here](https://doi.org/10.35097/sLJiahifxvfDKMEc). 

## Key Features

- **Run experiments**: Run `run_experiments.py` to reproduce experiments from the paper.
- **Organize results**: Jupyter notebook `Collect_Results.ipynb` to collect experimental results to one pandas dataframe.
- **Visualize Results**: The experimental results, including the code used to create the figures and the figures themselves, can be found [here](https://doi.org/10.35097/sLJiahifxvfDKMEc).
