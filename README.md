# parent-predictor
 A real use case Deep Learning regression model

# Info
## settings.json
The app configuration. Describes the neural network hyperparameters, training columns and the target columns to predict.
Also defines the normalization and encoding information for some columns.

## learner.py
The main app component. It will create the Keras neural network and initiate the learning process using the sample training dataset in `./dataset/train_dataset.csv`.

It will also create a directory `./generated` and will add the execution logs along with the model configuration.

Teansorboard will be also available at the end of the execution by typing the command `tensorboard --logdir=./generated/tensorboard/<logs-file>`.
The command to run it is shown in your terminal or can be found in `./generated/learner/<latest log>`

## predictor.py
Load the model configuration in `./generated/config/` and makes the predictions using the specified data source.
It will then append the prediction in `./generated/output_data.csv`.

The default target column to predict is "age_parent"

## lib.py
Functions librairy used by the learner & predictor and provides data preparation/transformation routines.

# Installation
Before package installation make sure you are running on Python 3.5+ 64-bit. 

Scripts dependencies & installation :
```
pip install tensorflow
pip install keras
pip install scikit-learn
pip install pandas
pip install numpy
pip install h5py
pip install matplotlib
pip install loguru
```

To use Tensorflow with GPU support (Recommended) :
`pip install tensorflow-gpu`
and follow this [Guide](https://www.tensorflow.org/install/gpu)

# Run 
`python .\app\learner.py .\dataset\train_dataset.csv`

## App Parameters
`data_file` Path to the training data file.

`--benchmark` Compare the mode performance to linear regression.

`--feature-importance` Evaluate data feature importance.

`--kfold-validation` Evaluate model performance with kfold validation.

`--grid-search` Use grid search to find which hyperparameters are the best for the model.

### Example
`python .\app\learner.py .\dataset\train_dataset.csv --benchmark --feature-importance`
