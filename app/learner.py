"""
Learner is the main component of the application.

It serves too:
1. Prepare the data
2. Create the neural network
3. Train the network
4. Evaluate amd benchmark the robustness of the model 
"""

import argparse
import os
import sys
import datetime
import time

import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard
from tensorflow.python.client import device_lib

from keras.models import Sequential as K_Sequential
from keras.layers import Dense as K_Dense
from keras.layers import BatchNormalization as K_BatchNormalization
from keras.layers import Dropout as K_Dropout
from keras.constraints import MaxNorm as K_MaxNorm
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Configure which GPU device to use
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import custom modules
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import lib
from predictor import Predictor

def gpu_check():
    """Checks for GPU availability and prints device information."""

    print(device_lib.list_local_devices())
    if tf.test.is_built_with_cuda():
        print("TensorFlow is using CUDA")
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print("Using GPU")
    else:
        print("No GPU found, using CPU")


class TrainTimeHistory(Callback):
    """Custom Keras callback to track training time."""

    def on_train_begin(self, logs=None):
        self.times = []
        logger.info("Training begins at {}".format(datetime.datetime.now().time()))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

class TestTimeHistory(Callback):
    """Custom Keras callback to track testing time."""

    def on_test_begin(self, logs=None):
        self.start = datetime.datetime.now()
        logger.info("Test begins at {}".format(self.start.time()))

    def on_test_end(self, logs=None):
        self.end = datetime.datetime.now()
        self.exec_time = (self.end - self.start).total_seconds()
        logger.info("Test ends at {}".format(self.end.time()))

class KerasModel:
    def __init__(self, n_input_cols, n_output_cols, grid_search=False, with_keras=False):
        self.n_input_cols = n_input_cols
        self.n_output_cols = n_output_cols
        self.grid_search = grid_search
        self.with_keras = with_keras

    def __call__(self, optimizer="adam", init="normal", activation="relu",
                 neurons=20, dropout_rate=0.0, weight_constraint=0, config={}):
        """
        Builds a Keras model based on the given configuration.

        Args:
            optimizer (str): The optimizer to use.
            init (str): The weight initialization method.
            activation (str): The activation function.
            neurons (int): The number of neurons in each layer.
            dropout_rate (float): The dropout rate.
            weight_constraint (float): The weight constraint.
            config (dict): Additional configuration parameters.

        Returns:
            model (Sequential/K_Sequential): The compiled Keras model.
        """

        # Get config variables
        init = config.get("init", init)
        activation = config.get("activation", activation)
        neurons = config.get("neurons", neurons)
        dropout_rate = config.get("dropout_rate", dropout_rate)
        weight_constraint = config.get("weight_constraint", weight_constraint)

        if not self.with_keras:
            # Create model using TensorFlow
            model = Sequential()
            # Input layer
            model.add(Dense(neurons, input_dim=self.n_input_cols, kernel_initializer=init,
                            activation=activation, kernel_constraint=MaxNorm(weight_constraint)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

            # Hidden layers
            for _ in range(4):
                model.add(Dense(neurons, kernel_initializer=init, activation=activation,
                                kernel_constraint=MaxNorm(weight_constraint)))
                model.add(Dropout(dropout_rate))

            # Output layer
            model.add(Dense(self.n_output_cols, kernel_initializer=init))
        else:
            # Create model using Keras
            model = K_Sequential()
            # Input layer
            model.add(K_Dense(neurons, input_dim=self.n_input_cols, kernel_initializer=init,
                              activation=activation, kernel_constraint=K_MaxNorm(weight_constraint)))
            model.add(K_BatchNormalization())
            model.add(K_Dropout(dropout_rate))
            # Hidden layers
            for _ in range(2):
                model.add(K_Dense(neurons, kernel_initializer=init, activation=activation,
                                  kernel_constraint=K_MaxNorm(weight_constraint)))
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
                logger.error("Optimizer class name or its parameters are not valid. Check your optimizer in the settings file: {}", e)
                sys.exit(1)

        # Compile model
        model.compile(loss="mean_squared_error", optimizer=optimizer,
                      metrics=["mean_squared_error", "mean_absolute_error",
                               "mean_absolute_percentage_error", "cosine_proximity"])
        logger.success("Keras model compiled")
        return model

class Learner:
    def __init__(self, dataframe, settings_file, working_dir):
        """
        Initializes the Learner class.

        Args:
            dataframe (pd.DataFrame): The source dataframe.
            settings_file (str): The path to the settings file.
            working_dir (str): The working directory.
        """

        self.source_dataframe = dataframe
        self.settings_file = settings_file
        self.working_dir = working_dir

        self.setup = lib.Setup()
        self.setup.load_learner_settings(self.settings_file)

        # Create working dir folder if it doesn't exist
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.setup.set_working_dir(working_dir, log_dir="/learner")

        # Create tensorboard folder if it doesn't exist
        self.tensorboard_dir = os.path.join(self.working_dir, "tensorboard")
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.tensorboard_dir = os.path.join(self.tensorboard_dir, "logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def data_preparation(self):
        """
        Performs data preparation steps.
        """

        # Check if dataframe contains NaN values
        self.setup.has_dataframe_nan(self.source_dataframe)
        # Encode data to numerical values
        self.setup.encode_label(self.source_dataframe, load_file=False)
        # Normalize data
        self.setup.normalize_dataframe(self.source_dataframe, load_file=False)

    def split_data(self, train_data=0.7, validation_data=0.2, test_data=0.1):
        """
        Splits the data into train, validation, and test sets.

        Args:
            train_data (float): The proportion of data for training.
            validation_data (float): The proportion of data for validation.
            test_data (float): The proportion of data for testing.
        """

        if (train_data + validation_data + test_data) <= 1:
            # Split data into train, validation, and test sets
            train_size = int(len(self.source_dataframe) * train_data)
            validation_size = int(len(self.source_dataframe) * validation_data)
            test_size = int(len(self.source_dataframe) * test_data)

            # Select range of rows for each dataframe
            self.train_dataframe = self.source_dataframe[:train_size]
            self.validation_dataframe = self.source_dataframe[train_size:train_size + validation_size]
            self.test_dataframe = self.source_dataframe[train_size + validation_size:train_size + validation_size + test_size]

            logger.info("Dataset length: Train = %s, Validation = %s, Test = %s" % (
                len(self.train_dataframe),
                len(self.validation_dataframe),
                len(self.test_dataframe)
            ))

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
                logger.error("Could not load header file. Make sure that the header file has all the train and target prediction columns: %s" % e)
                sys.exit(1)
        else:
            logger.error("SplitData() parameters are invalid. Make sure that the sum of the split does not exceed 1")
            sys.exit(1)

    def train_model(self, update_model=False):
        """
        Trains the model.

        Args:
            update_model (bool): Whether to update an existing model or build a new one.
        """

        logger.info("Learner is working with: Input = %d, Output = %d" % (self.X_train.shape[1], self.Y_train.shape[1]))

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
            compiled_model = self.setup.load_model()
        else:
            # Model building/compiling model with training dataset
            model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1])
            compiled_model = model(config=self.setup.hyperparameter["model"])

            history = compiled_model.fit(
                self.X_train, self.Y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_validation, self.Y_validation),
                verbose=0,
                callbacks=[train_time_callback, tensorboard]
            )

            logger.info("Model Train execution time = %.2f sec" % round(sum(train_time_callback.times), 2))
            logger.info("Average Epoch execution time = %.2f sec" % round(np.mean(train_time_callback.times), 2))
            logger.info("You can now launch Tensorboard by typing in your command line: tensorboard --logdir=%s" % self.tensorboard_dir)

            # Save learning curve plot
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("Learning curve")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "validation"], loc="upper right")
            plt.savefig(os.path.join(self.working_dir, "learning_curve.png"))

            # Evaluate model with test dataset
            score = compiled_model.evaluate(self.X_test, self.Y_test, verbose=0, callbacks=[test_time_callback])
            logger.info("Model Evaluate execution time = %.2f sec" % round(test_time_callback.exec_time, 2))
            for index, metric_name in enumerate(compiled_model.metrics_names):
                logger.info("%s: %.2f" % (metric_name, score[index]))

            # Saving model
            self.setup.export_model(compiled_model)

    def evaluate_feature_importance(self, n_jobs=1):
        """
        Evaluates the feature importance using permutation importance and generates a plot.

        Args:
            n_jobs (int): The number of parallel jobs to run.

        Returns:
            None
        """

        # Deactivate logger
        logger.remove()

        # Create log file
        output_file = os.path.join(self.working_dir, "logs", "evaluateFeatureImportance_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        sys.stdout = open(output_file, "w")

        # ANN Parameters
        epochs = int(self.setup.hyperparameter["epoch"])
        batch_size = int(self.setup.hyperparameter["batch_size"])

        model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1], with_keras=True)
        estimator = KerasRegressor(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=0,
                                **dict(config=self.setup.hyperparameter["model"]))
        estimator.fit(self.X_train, self.Y_train)

        lib.print_debug("Starting feature importance evaluation")
        feature_importance = permutation_importance(estimator, self.X_train, self.Y_train,
                                                    scoring=None, n_repeats=5, n_jobs=n_jobs, random_state=None)

        # Retrieve input header column name
        X_header = self.setup.loaded_settings["train_column"]
        # Convert to numpy array
        feature_importance = feature_importance.importances_mean.astype(float)
        # Create dataframe from result
        feature_importance = pd.DataFrame(feature_importance, index=X_header, columns=["importance"])
        # Sort dataframe from highest to lowest
        feature_importance = feature_importance.sort_values(by=["importance"], ascending=False)
        lib.print_debug(feature_importance.to_string())

        # Plot feature importance
        feature_importance = feature_importance.sort_values("importance", ascending=True)
        feature_importance.plot.barh(y="importance", use_index=True, logx=True, title="Feature importance")
        pyplot.show()
        plt.savefig(os.path.join(self.working_dir, "feature_importance.png"))

        sys.stdout.close()

    def k_fold_validation(self):
        """Performs k-fold cross-validation."""

        model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1])
        estimator = KerasRegressor(build_fn=model, epochs=50, batch_size=5, verbose=0)

        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, self.X_train, self.Y_train, cv=kfold)
        print("KFold validation results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    def test_model_parameters(self, n_jobs=1):
        """
        Tests different model parameter configurations using grid search.
        
        Args:
            n_jobs (int): The number of parallel jobs to run.
        """

        # Deactivate logger
        logger.remove()
        # Create log file
        output_file = os.path.join(self.working_dir, "logs", "testModelParameters_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        sys.stdout = open(output_file, "w")

        # NOTE: This operation may take some time
        model = KerasModel(self.X_train.shape[1], self.Y_train.shape[1], grid_search=True, with_keras=False)
        # Grid search epochs, batch size, and optimizer
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

        param_grid = dict(
            optimizer=optimizer,
            activation=activation,
            epochs=epochs,
            batch_size=batch_size,
            init=init,
            neurons=neurons,
            dropout_rate=dropout_rate,
            weight_constraint=weight_constraint
        )
        lib.print_debug("Start testing model parameters with GridSearch. This will take time...")
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=10, n_jobs=n_jobs)
        grid_result = grid.fit(self.X_train, self.Y_train)
        # Summarize results
        lib.print_debug("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params = grid_result.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            lib.print_debug("%f (%f) with: %r" % (mean, stdev, param))

        sys.stdout.close()

    def debug_prediction(self):
        """Debugs the prediction by generating predictions and exporting them."""

        logger.remove(self.setup.log_handler)

        predict_dataframe = self.test_dataframe[
            self.setup.loaded_settings["train_column"] + self.setup.loaded_settings["target_prediction"]
        ]

        predictor = Predictor(predict_dataframe, self.settings_file, self.working_dir, with_debug_target=True)
        predictor.predict_data(show_prediction=False)
        predictor.export_prediction_file(self.working_dir)

    def benchmark(self):
        """Performs benchmarking using a Linear Regression model."""

        # Using a Linear Regression model
        logger.info("Benchmark reference with a Linear Regression model")
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.Y_train)
        y_pred = regressor.predict(self.X_test)
        logger.info("Coefficients: \n%s" % regressor.coef_)
        # The mean squared error
        logger.info("Mean squared error: %.2f" % mean_squared_error(self.Y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        logger.info("Coefficient of determination: %.2f" % r2_score(self.Y_test, y_pred))

        dnz_debug_Y = self.setup.denormalize_dataset(y_pred, self.setup.loaded_settings["target_prediction"]).round(0).astype(int)
        dnz_debug_X = self.setup.denormalize_dataset(self.Y_test, self.setup.loaded_settings["target_prediction"]).round(0).astype(int)

        prediction_deviation = []
        for i in range(len(dnz_debug_Y)):
            prediction_deviation.append(abs(dnz_debug_X[i] - dnz_debug_Y[i]))

        prediction_deviation = pd.DataFrame(prediction_deviation, columns=self.setup.loaded_settings["target_prediction"])

        logger.info("Max predicted deviation:\n%s" % (prediction_deviation.max()))
        logger.info("Min predicted deviation:\n%s" % (prediction_deviation.min()))
        logger.info("Mean predicted deviation:\n%s" % (prediction_deviation.mean().round(2)))
        exactPrediction = ((prediction_deviation == 0).sum(axis=0) / prediction_deviation.size) * 100
        logger.info("Exact prediction rate (percent):\n%s" % (exactPrediction.round(2)))


if __name__ == "__main__":
    print("-- Libraries loaded --")
    gpu_check()

    # Add optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help="Path to the training dataset file")
    parser.add_argument("--feature-importance", action="store_true", help="Evaluate data feature importance")
    parser.add_argument("--kfold-validation", action="store_true", help="Evaluate model performance with kfold validation")
    parser.add_argument("--grid-search", action="store_true", help="Use grid search to find the best hyperparameters for the model")
    parser.add_argument("--benchmark", action="store_true", help="Compare the model performance to linear regression")
    # Parse the command-line arguments
    args = parser.parse_args()

    print("\nThe program will now begin..."
          "\nDo not interrupt the program. It might take a while to finish depending on your build.")

    # Load dataset
    with open(args.data_file) as file_obj:
        source_dataframe = pd.read_csv(file_obj, encoding="latin-1")

    # Init Learner
    learner = Learner(
        dataframe=source_dataframe,
        settings_file="./settings.json",
        working_dir="./generated"
    )

    learner.data_preparation()
    learner.split_data()

    if args.feature_importance:
        print("Evaluating data feature importance...")
        learner.evaluate_feature_importance(n_jobs=10)
    if args.kfold_validation:
        print("Evaluating model performance with kfold validation...")
        learner.k_fold_validation()
    if args.grid_search:
        print("Using grid search to find the best hyperparameters...")
        learner.test_model_parameters(n_jobs=10)

    learner.train_model(update_model=False)
    learner.debug_prediction()

    if args.benchmark:
        print("Comparing model performance to linear regression...")
        learner.benchmark()