"""
Predictor is the second component of the application.

It serves too:
1. Load the keras predictor model generated from the learner
2. Load the data encoding & normalizer config 
4. Load the source data and populate with the predictions for each row
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import Callback

# Custom library
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import lib

class PredictTime(Callback):
    """Callback to log the prediction time."""
    
    def on_predict_begin(self, logs=None):
        self.start = datetime.datetime.now()
        logger.info(f"Predict: begins at {self.start.time()}")

    def on_predict_end(self, logs=None):
        self.end = datetime.datetime.now() 
        self.execTime = (self.end  - self.start).total_seconds()
        logger.info(f"Predict: end at {self.end.time()}")

class Predictor:
    """Class to predict the data using a trained model."""
    
    def __init__(self, dataframe, settings_file, working_dir, with_debug_target=False):
        self.source_dataframe = dataframe
        self.transform_dataframe = dataframe
        self.settings_file = settings_file
        self.working_dir = working_dir
        self.with_debug_target = with_debug_target
        self.setup = lib.Setup()
        self.setup.load_learner_settings(self.settings_file)
        self.setup.set_working_dir(working_dir, log_dir="/predictor")

        # File paths
        self.encoder_file = os.path.join(self.setup.config_dir, "encoder.pickle")
        self.normalization_file = os.path.join(self.setup.config_dir, "normalization_minmax.json")
        self.model_file = os.path.join(self.setup.config_dir, "model.h5")

        # Check if required files exist
        self._check_required_files()

        # Check device
        print(device_lib.list_local_devices())
        print("Using GPU") if tf.test.gpu_device_name() else print("No GPU found, using CPU")

    def _check_required_files(self):
        """Check if the required files exist."""
        
        required_files = [self.encoder_file, self.normalization_file, self.model_file]
        for file in required_files:
            if not os.path.exists(file): 
                logger.error(f"Could not found {os.path.basename(file)}")
                sys.exit()

    def transformData(self):
        """Transforms the data for prediction."""
        
        try:
            if self.with_debug_target:
                # Keep target prediction column for dubug prediction
                train_column = self.source_dataframe[self.setup.loaded_settings["train_column"]]
                target_column = self.source_dataframe[self.setup.loaded_settings["target_prediction"]]
                self.transform_dataframe = pd.concat([train_column, target_column], axis=1)
            else:
                self.transform_dataframe = self.transform_dataframe[self.setup.loaded_settings["train_column"]]     
        except KeyError as e:
            logger.error(f"Could not parse header file. Make sure that the header file is the same as the columns used for the train set. {e}")
            sys.exit(1)
        
        # Check if dataframe contains NAN values
        self.setup.has_dataframe_nan(self.transform_dataframe)
        
        # Encode and Normalize data
        self.setup.encode_label(self.transform_dataframe, load_file=True)
        self.setup.normalize_dataframe(self.transform_dataframe, load_file=True)

    def predictData(self, show_prediction=False):
        """Predicts the data using the trained model."""

        new_X = self.transform_dataframe
        target_prediction = self.setup.loaded_settings["target_prediction"]

        if self.with_debug_target:
            # Prepare debug target prediction
            new_X, debug_new_Y, dnz_debug_Y = self._prepare_debug_target(new_X, target_prediction)

        # Get fit execution time
        predict_time_callback = PredictTime()

        # Load model and predict target columns
        loaded_model = self.setup.load_model()

        try:
            predicted_Y = loaded_model.predict(new_X, callbacks=[predict_time_callback])
        except Exception as e:
            logger.error(e)
            sys.exit()

        logger.info(f"Data prediction execution time = {round(predict_time_callback.execTime, 2)} sec")

        # Denormalize predictions
        self.dnz_predicted_Y = self.setup.denormalize_dataset(predicted_Y, target_prediction).round(0).astype(int)

        # Record predictions
        self._record_predictions(new_X, dnz_debug_Y, show_prediction, target_prediction)

        # Debug prediction analysis
        if self.with_debug_target:
            self._analyze_debug_prediction(new_X, dnz_debug_Y, target_prediction)

    def _prepare_debug_target(self, new_X, target_prediction):
        """Prepares the debug target prediction data."""

        # Drop target prediction
        new_X = self.transform_dataframe.drop(columns=target_prediction, axis=1)
        
        # Cache debug target prediction
        debug_new_Y = self.transform_dataframe[target_prediction]

        # Convert to np array
        new_X = new_X.to_numpy()
        debug_new_Y = debug_new_Y.to_numpy()

        # Denormalize debug Y
        dnz_debug_Y = self.setup.denormalize_dataset(debug_new_Y, target_prediction).round(0).astype(int)

        return new_X, debug_new_Y, dnz_debug_Y

    def _record_predictions(self, new_X, dnz_debug_Y, show_prediction, target_prediction):
        """Records the predictions and logs them if required."""

        for i in range(len(new_X)):
            # Get "année de naissance"
            test = self.source_dataframe.loc[self.source_dataframe.index[i], "annedenaissance"] + self.dnz_predicted_Y[i]

            # Debug real data with predicted outputs
            if self.with_debug_target:
                logger.debug(f"Real={dnz_debug_Y[i]} VS Predicted={self.dnz_predicted_Y[i]} TEST={test}")   
            else:    
                logger.debug(f"Predicted={self.dnz_predicted_Y[i]} TEST={test}")

    def _analyze_debug_prediction(self, new_X, dnz_debug_Y, target_prediction):
        """Analyzes the debug predictions and plots the deviation."""

        prediction_deviation = []
        # Get deviation
        for i in range(len(new_X)):
            prediction_deviation.append(abs(dnz_debug_Y[i] - self.dnz_predicted_Y[i]))

        # Convert numpy array to dataframe
        prediction_deviation = pd.DataFrame(prediction_deviation, columns=target_prediction)

        self._log_deviation(prediction_deviation)
        
        # Plot deviation
        prediction_deviation.plot(kind="box", grid=True, title="Debug prediction deviation per column")
        # Save as png meaning prediction plot
        plt.savefig(os.path.join(self.working_dir, "debug_prediction_deviation.png"))

    def _log_deviation(self, prediction_deviation):
        """Logs the deviation metrics."""

        logger.info(f"Max predicted deviation:\n{prediction_deviation.max()}")
        logger.info(f"Min predicted deviation:\n{prediction_deviation.min()}")
        logger.info(f"Mean predicted deviation:\n{prediction_deviation.mean().round(2)}")
        
        exact_prediction = ((prediction_deviation == 0).sum(axis=0) / prediction_deviation.size) * 100
        logger.info(f"Exact prediction rate (percent):\n{exact_prediction.round(2)}")

    def exportPredictionFile(self, output_dir):
        """Exports the prediction data to a file."""

        # Numpy array to Dataframe
        prediction_dataframe = pd.DataFrame(self.dnz_predicted_Y, 
         columns=self.setup.loaded_settings["target_prediction"])
        # Append predicted column into the source dataframe
        # output_dataframe = self.source_dataframe
        # output_dataframe = self.setup.denormalize_dataset(output_dataframe)
        # output_dataframe = self.setup.decode_label(output_dataframe)
        # output_dataframe = output_dataframe.reset_index().drop(columns=["index"])
        output_dataframe = pd.concat([self.source_dataframe, prediction_dataframe], axis=1)
        
        # Create output file
        export_file = os.path.join(output_dir, "output_data.csv")
        output_dataframe.to_csv(export_file, encoding="latin-1")
        
        logger.success(f"Predictions has been exported in {export_file} file")
