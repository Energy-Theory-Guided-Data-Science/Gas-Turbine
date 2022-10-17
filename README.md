conda env create -f env.yml# Gas-Turbine
Can domain knowledge help to model a gas turbine?

## Getting Started
When you download this code, ensure, the data available is in a subfolder called "Raw_Data" in a "Data" folder which is on the same level as the code folder goes. Only in that way, the code will work properly. The data used for modelling gas turbines is available at the repo owners.

Within the Code folder is a file called requirements.txt which includes all packages which need to be installed. Install these packages with 

```python
conda env create -f env.yml
```

If no errors occur, all packages are installed correctly. The package `lttb` might be needed to be installed seperately using `pip install lttb`.

___

## How the files work:
* 01_Preprocessing.ipynb: When having real data loaded in the "./Data/Raw_Data/" folder, we execute this file to downsample the data to the corresponding size and cut unnecessary data.
* 02_Visualization.ipynb: When having preprocessed the data, we can plot the different variables for each experiment in this file.
* 03_Synthetic_Data.ipynb: In this file we generate synthethic data. When having a quadratic relation from input to stationary levels and a linear transition phase, the parameters can be changed in cell 3. If other relations need to be changed, adapt the arguments in cell 10 (`create_dataframe`) line 3 (`create_synthetic_data`) accordingly. The synthetically generated data is then stored at "./Data/Synthetic_Data/".
* 10_Theory_Baseline.ipynb: File for the creation of the theoretical baseline. Specified parameters are shown in cell 4. Cell 12 specifies which data is used for training.
* 20_Data_Baseline.ipynb: The same for data baseline. All parameters can be specified in cell 3. The model structure itself can be changed in file Neural_Networks.py
* 30_Design_of_Architecture-Input.ipynb: Same as Data Baseline. The different approach is changed in the preprocessing file "Data_Preprocessing.py"
* 40_Design_of_Architecture-Output.ipynb: Same as Data Baseline. The different approach is changed in the preprocessing file "Data_Preprocessing.py"
* 50_Loss_function.ipynb: Loss function approach. Model specifications in cell 3, hyper parameter optimization of the weights in cell 15. The loss function is specified in cell 13.
* 91_Analysis_Availability.ipynb: Comparison of approaches when varying data availability. Change the folder names to fit your experiments.
* 92_Analysis_Steepness.ipynb: Comparison of approaches when varying steepness. Change the folder names to fit your experiments.
* 97_Comparison different_Approaches.ipynb: Comparison of approaches on one configuration. Change title and folders fitting to your experiments.
* 99_Analysis.ipynb: Check loss function approach which hyper parameter is optimal.
* Data_Processing.py: Preprocessing and scaling of data for NNs. Here is the adaptation for DoA-Input and -Output approaches.
* Global_Functions.py: Auxiliary methods, e.g. color codes and functions
* Neural_Networks.py: Functions specifying model structure and the inclusion.
* requirements.txt: summary of all libraries needed.

## Examplary run through:
When wanting to compare the different approaches on varying steepness levels you have to do the following:
1. Create the corresponding synthetic data by varying the parameter l_1 in cell 3.
2. Run all the different approaches once for each configuration. For each run you have to specify the folder in which the respective data lies.
3. When all models are trained, insert the corresponding folders where the model and images are saved into the respective fields in file 92_
4. Create the plot and discuss the results.

## Useful Materials
* [How to Write a Good Git Commit Message](https://chris.beams.io/posts/git-commit/)
* [Python Styleguide by Google](http://google.github.io/styleguide/pyguide.html)
