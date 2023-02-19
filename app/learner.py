"""
Learner is the main component of the application.

It serves too:
1. Prepare the data
2. create the neural network
3. train the network
4. evaluate amd benchmark the robustness of the model 

DEPENDENCIES: 
    -tensorflow
    -sklearn
    -numpy
    -h5py
"""

import argparse
import os
# Configure which GPU device to use
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential as K_Sequential
from keras.layers import Dense as K_Dense
from keras.layers import BatchNormalization as K_BatchNormalization
from keras.layers import Dropout as K_Dropout
from keras.constraints import MaxNorm as K_MaxNorm

from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib 

# Scikit learn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Benchmark 
from sklearn.linear_model import LinearRegression

# Other
import pandas
import numpy as np
from matplotlib import pyplot
import json
import pickle
import datetime
import time
from loguru import logger

# Import custom
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import lib
from predictor import Predictor


def gpuCheck():
    # Get device informations
    print(device_lib.list_local_devices())
    if tf.test.is_built_with_cuda(): 
        print("TensorFLow is using Cuda")
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print("Using GPU")
    else:
        print("No GPU found, using CPU")

class TrainTimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []
        logger.info("Training: begins at {}".format(datetime.datetime.now().time()))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

class TestTimeHistory(Callback):
    def on_test_begin(self, logs=None):
        self.start = datetime.datetime.now()
        logger.info("Test: begins at {}".format(self.start.time()))

    def on_test_end(self, logs=None):
        self.end = datetime.datetime.now() 
        self.execTime = (self.end  - self.start).total_seconds()
        logger.info("Test: end at {}".format(self.end.time()))

class KerasModel():
    def __init__(self, n_input_cols, n_output_cols, grid_search=False, with_keras=False): 
        self.n_input_cols = n_input_cols
        self.n_output_cols = n_output_cols
        self.grid_search = grid_search
        self.with_keras = with_keras

    def __call__(self, optimizer="adam", init="normal", activation="relu",
     neurons=20, dropout_rate=0.0, weight_constraint=0, config={}): 
        # Get config variable
        init = config.get("init", init)
        activation = config.get("activation", activation)
        neurons = config.get("neurons", neurons)
        dropout_rate = config.get("dropout_rate", dropout_rate)
        weight_constraint = config.get("weight_constraint", weight_constraint)
        
        if self.with_keras is False:
            # Create model
            model = Sequential()
            # Input layer
            model.add(Dense(neurons, input_dim=self.n_input_cols, kernel_initializer=init, 
            activation=activation , kernel_constraint=MaxNorm(weight_constraint)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

            # Hidden layer
            model.add(Dense(neurons, kernel_initializer=init, activation=activation, kernel_constraint=MaxNorm(weight_constraint)))
            model.add(Dropout(dropout_rate))

            model.add(Dense(neurons, kernel_initializer=init, activation=activation, kernel_constraint=MaxNorm(weight_constraint)))
            model.add(Dropout(dropout_rate))

            model.add(Dense(neurons, kernel_initializer=init, activation=activation, kernel_constraint=MaxNorm(weight_constraint)))
            model.add(Dropout(dropout_rate))

            model.add(Dense(neurons, kernel_initializer=init, activation=activation, kernel_constraint=MaxNorm(weight_constraint)))
            model.add(Dropout(dropout_rate))
            
            # Output layer
            model.add(Dense(self.n_output_cols, kernel_initializer=init))
        else: 
            # Create model
            model = K_Sequential()
            # Input layer
            model.add(K_Dense(neurons, input_dim=self.n_input_cols, kernel_initializer=init, 
            activation=activation, kernel_constraint=K_MaxNorm(weight_constraint)))
            model.add(K_BatchNormalization())
            model.add(K_Dropout(dropout_rate))
            # Hidden layer
            model.add(K_Dense(neurons, kernel_initializer=init, activation=activation, kernel_constraint=K_MaxNorm(weight_constraint)))
            model.add(K_Dropout(dropout_rate))
            model.add(K_Dense(neurons, kernel_initializer=init, activation=activation, kernel_constraint=K_MaxNorm(weight_constraint)))
            model.add(K_Dropout(dropout_rate))
            # Output layer
            model.add(K_Dense(self.n_output_cols, kernel_initializer=init))
            
        # Load custom optimizer
        if not self.grid_search:
            try:
                optimizer = config["optimizer"].get("class_name", optimizer)
                optimizers_param = config["optimizer"].get("param", {})
                optimizer = optimizer + "(**{})".format(optimizers_param)
                optimizer = eval("tf.keras.optimizers." + optimizer)
                if self.with_keras:
                    optimizer = eval("keras.optimizers." + optimizer)
            except Exception as e:
                logger.error("Optimizer class name or it's parameters is not valid. Check your optimizer in setings file : {}", e)
                sys.exit(1)

        # Compile model
        model.compile(loss="mean_squared_error", optimizer=optimizer, 
         metrics=["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "cosine_proximity"]) 
        logger.success("Keras model compiled")
        return model

class Learner():
    def __init__(self, dataframe, settings_file, working_dir):
        """
        File preparation
        """
        self.source_dataframe = dataframe
        self.settings_file = settings_file
        self.working_dir = working_dir

        self.setup = lib.Setup()
        self.setup.loadLearnerSettings(self.settings_file)

        # Create working dir folder if not exists
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.setup.setWorkingDir(working_dir, log_dir="/learner")

        # Create tensorboard folder if not exists
        # Tensorboard --logdir
        self.tensorboard_dir = working_dir + "/tensorboard"
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.tensorboard_dir = self.tensorboard_dir + "/logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def dataPreparation(self):   
        # Check if daframe contains NAN values
        self.setup.hasDataframeNan(self.source_dataframe)
        # Encode data to numerical values
        self.setup.encodeLabel(self.source_dataframe, load_file=False)
        # iterating the columns
        # Normalize data
        self.setup.normalizeDataframe(self.source_dataframe, load_file=False)
        
    def splitData(self, train_data=0.7, validation_data=0.2, test_data=0.1):
        if (train_data + validation_data + test_data) <= 1:
            # Split data into train, validation and test set
            train_size = int(len(self.source_dataframe) * train_data)
            validation_size = int(len(self.source_dataframe) * validation_data)
            # test_size = len(self.source_dataframe) - (train_size + validation_size)
            test_size = int(len(self.source_dataframe) * test_data)
            # Select range of row for each dataframe
            self.train_dataframe = self.source_dataframe[0:train_size]
            self.validation_dataframe = self.source_dataframe[train_size:(train_size + validation_size)]
            self.test_dataframe = self.source_dataframe[validation_size:(validation_size + test_size)]
            logger.info("Dataset length : Train = %s, Validation = %s, Test = %s"
             % (len(self.train_dataframe), len(self.validation_dataframe), len(self.test_dataframe)))

            try:
                train_column = self.setup.loaded_settings["train_column"]
                target_prediction = self.setup.loaded_settings["target_prediction"]
                # Defining network input (X) and output (Y) variables
                self.X_train = self.train_dataframe[train_column].to_numpy()
                self.X_validation = self.validation_dataframe[train_column].to_numpy()
                self.X_test = self.test_dataframe[train_column].to_numpy()
                # Extract target prediction variables for learning output 
                self.Y_train = self.train_dataframe[target_prediction].to_numpy()
                self.Y_validation = self.validation_dataframe[target_prediction].to_numpy()
                self.Y_test = self.test_dataframe[target_prediction].to_numpy()
            except KeyError as e:
                logger.error(("Could not load header file. " + \
                 "Make sure that the header file has all the train and target prediction columns: %s" % (e)))
                sys.exit(1)
    
            """
            for dateTypeColumn in self.loaded_settings["target_date_prediction"]:
                # Date transformation
                # Formating date type string to datetime
                self.train_dataframe[dateTypeColumn] = pandas.to_datetime(self.train_dataframe[dateTypeColumn], format = "%Y-%m-%d", errors = "coerce")
                assert self.train_dataframe[dateTypeColumn].isnull().sum() == 0, "missing ScheduledDay dates"

                # Split date variable
                self.train_dataframe[dateTypeColumn + "_year"] = self.train_dataframe[dateTypeColumn].dt.year
                self.train_dataframe[dateTypeColumn + "_month"] = self.train_dataframe[dateTypeColumn].dt.month
                # self.train_dataframe[dateTypeColumn + "_day"] = self.train_dataframe[dateTypeColumn].dt.day
                self.train_dataframe = self.train_dataframe.drop(columns=[dateTypeColumn])
            """
        else:
            logger.error("SplitData() parameters invalid, " + \
            "be sure that the sum of the split does not exceed 1")
            sys.exit(1)
      
    def trainModel(self, update_model=False):
        logger.info("Learner is working with : Input=%d, Output=%d" % (self.X_train.shape[1], self.Y_train.shape[1]))
        
        # Keras Callback instance
        # Set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(monitor="mean_squared_error", patience=50)
        # Track network learning performance
        tensorboard = TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=1)
        # Get fit execution time
        train_time_callback = TrainTimeHistory()
        test_time_callback = TestTimeHistory()

        # ANN Parameters
        epochs = int(self.setup.hyperparameter["epoch"])
        batch_size = int(self.setup.hyperparameter["batch_size"])

        if update_model:
            # Load model
            logger.info("Learner will train on an existing model")
            compiledModel = self.setup.loadModel()
        else: 
            # Model building/compiling model with training dataset
            model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1])
            compiledModel = model(config=self.setup.hyperparameter["model"])

            history = compiledModel.fit(self.X_train, self.Y_train,
             epochs=epochs, batch_size=batch_size, validation_data=(self.X_validation, self.Y_validation),
             verbose=0, callbacks=[train_time_callback, tensorboard])
            
            logger.info("Model Train execution time = %.2f sec" % round(sum(train_time_callback.times), 2))
            logger.info("Average Epoch execution time = %.2f sec" % round(np.mean(train_time_callback.times), 2))
            logger.info("You can now launch Tensorboard by typing in your command line: " + \
            "tensorboard --logdir=%s" % (self.tensorboard_dir))

            # Save learning curve plot
            pyplot.plot(history.history["loss"])
            pyplot.plot(history.history["val_loss"])
            pyplot.title("Learning curve")
            pyplot.ylabel("loss")
            pyplot.xlabel("epoch")
            pyplot.legend(["train", "validation"], loc="upper right")
            pyplot.savefig(self.working_dir + "/learning_curve.png")

            # Evaluate model with training dataset
            score = compiledModel.evaluate(self.X_test, self.Y_test, verbose=0, callbacks=[test_time_callback])
            logger.info("Model Evaluate execution time = %.2f sec" % round(test_time_callback.execTime, 2))
            for index in range(len(compiledModel.metrics_names)):
                logger.info("%s: %.2f" % (compiledModel.metrics_names[index], score[index]))
            # Saving model
            self.setup.exportModel(compiledModel)

    def evaluateFeatureImportance(self, n_jobs=1):
        # Deactivate logger
        # NOTE: Cannot use any logging module that are runnning live I/O operation while having multiple workers,
        #      because Keras estimator is not pickable 
        logger.remove()
        # Create log file
        ouput_file = self.working_dir + "/logs/evaluateFeatureImportance_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        sys.stdout=open(ouput_file, "w")

        # ANN Parameters
        epochs = int(self.setup.hyperparameter["epoch"])
        batch_size = int(self.setup.hyperparameter["batch_size"])

        model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1], with_keras=True)
        estimator = KerasRegressor(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=0,
         **dict(config=self.setup.hyperparameter["model"]))
        estimator.fit(self.X_train, self.Y_train)
        
        lib.printDebug("Starting feature importance evaluation")
        featureImportance = permutation_importance(estimator, self.X_train, self.Y_train,
         scoring=None, n_repeats=5, n_jobs=n_jobs, random_state=None) 

        # Retrieve input header column name
        X_header = self.setup.loaded_settings["train_column"]
        # Convert to numpy array
        featureImportance = featureImportance.importances_mean.astype(float)
        # Create dataframe from result
        featureImportance = pandas.DataFrame(featureImportance, index=X_header, columns=["importance"])
        # Sort dataframe from highest to lowest
        featureImportance = featureImportance.sort_values(by=["importance"], ascending=False)
        lib.printDebug(featureImportance.to_string())

        # Plot feature importance
        featureImportance = featureImportance.sort_values("importance", ascending=True)
        featureImportance.plot.barh(y="importance", use_index=True, logx=True, title="Feature importance")
        pyplot.show()
        pyplot.savefig(self.working_dir + "/feature_importance.png")
        
        sys.stdout.close()

    def kFoldValidation(self):
        model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1])
        estimator = KerasRegressor(build_fn=model, epochs=50, batch_size=5, verbose=0)

        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, self.X_train, self.Y_train, cv=kfold)
        print("KFold validation results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    
    def testModelParameters(self, n_jobs=1):
        """
        Searching for the best model parameters configuration
        """
        # Deactivate logger
        # NOTE: Cannot use any logging module that are runnning live I/O operation while having multiple workers,
        #      because Keras estimator is not pickable 
        logger.remove()
        # Create log file
        ouput_file = self.working_dir + "/logs/testModelParameters_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        sys.stdout=open(ouput_file, "w")

        # NOTE : This operation may take some time
        model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1], grid_search=True, with_keras=False)
        # Grid search epochs, batch size and optimizer
        estimator = KerasRegressor(build_fn=model, verbose=0)

        # Hyperparameters
        optimizer = ["Adagrad"]
        init = ["uniform"]
        activation = ["relu"]
        epochs = [450]
        batch_size = [50, 150]
        neurons = [100, 150]
        dropout_rate = [0.1]
        weight_constraint = [2]

        param_grid = dict(optimizer=optimizer, activation=activation, epochs=epochs, batch_size=batch_size, init=init, 
         neurons=neurons, dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        lib.printDebug("Start testing model parameters with GridSearch. This will take time...")
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=10, n_jobs=n_jobs)
        grid_result = grid.fit(self.X_train, self.Y_train)
        # Summarize results
        lib.printDebug("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params = grid_result.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            lib.printDebug("%f (%f) with:  %r" % (mean, stdev, param))

        sys.stdout.close()
    
    def debugPrediction(self):
        logger.remove(self.setup.log_handler)

        predict_dataframe = self.test_dataframe[(self.setup.loaded_settings["train_column"] + 
         self.setup.loaded_settings["target_prediction"])]

        predictor = Predictor(predict_dataframe, self.settings_file, self.working_dir, with_debug_target=True)
        predictor.predictData(show_prediction=False)
        predictor.exportPredictionFile(self.working_dir)

    def benchmark(self):
        # Using a Linear Regression model
        logger.info("Benchmark reference with a Linear Regression model")
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.Y_train)
        y_pred= regressor.predict(self.X_test)
        logger.info("Coefficients: \n", regressor.coef_)
        # The mean squared error
        logger.info("Mean squared error: %.2f"
            % mean_squared_error(self.Y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        logger.info("Coefficient of determination: %.2f"
            % r2_score(self.Y_test, y_pred))

        dnz_debug_Y = self.setup.denormalizeDataset(y_pred, 
         self.setup.loaded_settings["target_prediction"]).round(0).astype(int)
        dnz_debug_X = self.setup.denormalizeDataset(self.Y_test, 
         self.setup.loaded_settings["target_prediction"]).round(0).astype(int)
        
        # Debug each entry prediction
        # for i in range(len(dnz_debug_Y)):
        #     logger.debug("Real=%s VS Predicted=%s" % (dnz_debug_X[i],  dnz_debug_Y[i]))

        prediction_deviation = []
        for i in range(len(dnz_debug_Y)): prediction_deviation.append(abs(dnz_debug_X[i] - dnz_debug_Y[i]))

        prediction_deviation = pandas.DataFrame(prediction_deviation,
         columns=self.setup.loaded_settings["target_prediction"])

        logger.info("Max predicted deviation:\n%s" % (prediction_deviation.max()))
        logger.info("Min predicted deviation:\n%s" % (prediction_deviation.min()))
        logger.info("Mean predicted deviation:\n%s" % (prediction_deviation.mean().round(2)))
        exactPrediction = ((prediction_deviation == 0).sum(axis=0) / prediction_deviation.size) * 100
        logger.info("Exact prediction rate (percent):\n%s" % (exactPrediction.round(2)))

if __name__ == "__main__":
    print("-- Librairies loaded --")
    gpuCheck()

    # Add optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help="Path to the training dataset file")
    parser.add_argument("--feature-importance", action="store_true", help="Evaluate data feature importance")
    parser.add_argument("--kfold-validation", action="store_true", help="Evaluate model performance with kfold validation")
    parser.add_argument("--grid-search", action="store_true", help="Use grid search to find which hyperparameters are the best for the model")
    parser.add_argument("--benchmark", action="store_true", help="Compare the mode performance to linear regression")
    # Parse the command-line arguments
    args = parser.parse_args()

    print("\nThe program will now begin...\
        \nDo not interrupt the program. It might take a while to finish depending on your build.")

    # Load dataset
    file_obj = open(args.data_file)
    source_dataframe = pandas.read_csv(file_obj, encoding="latin-1")

    # Init Learner
    learner = Learner(
        dataframe = source_dataframe,
        settings_file = "./settings.json",
        working_dir = "./generated"
    )
        
    learner.dataPreparation()
    learner.splitData()
    
    if args.feature_importance:
        print("Evaluating data feature importance...")
        learner.kFoldValidation()
    if args.kfold_validation:
        print("Evaluating model performance with kfold validation...")
        learner.kFoldValidation()
    if args.grid_search:
        print("Using grid search to find best hyperparameters...")
        learner.testModelParameters(n_jobs = 10)

    learner.trainModel(update_model=False)
    learner.debugPrediction()

    if args.benchmark:
        print("Comparing model performance to linear regression...")
        learner.benchmark()