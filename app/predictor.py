"""
Predictor is the second component of the application.

It serves too:
1. load the keras predictor model generated from the learner
2. load the data encoding & normalizer config 
4. Load the source data and populate with the predictions for each row

DEPENDENCIES: 
    -tensorflow
    -sklearn
    -numpy
    -h5py
"""

import os
# Configure which GPU device to use
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.python.client import device_lib 
from tensorflow.keras.callbacks import Callback

# Import custom
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import json
import datetime
import numpy as np
import pandas
from loguru import logger
from matplotlib import pyplot

import lib

class PredictTime(Callback):
    def on_predict_begin(self, logs=None):
        self.start = datetime.datetime.now()
        logger.info("Predict: begins at {}".format(self.start.time()))

    def on_predict_end(self, logs=None):
        self.end = datetime.datetime.now() 
        self.execTime = (self.end  - self.start).total_seconds()
        logger.info("Predict: end at {}".format(self.end.time()))

class Predictor():
    def __init__(self, dataframe, settings_file, working_dir, with_debug_target=False):
        self.source_dataframe = dataframe
        self.transform_dataframe = dataframe
        self.settings_file = settings_file
        self.working_dir = working_dir
        self.with_debug_target = with_debug_target

        self.setup = lib.Setup()
        self.setup.loadLearnerSettings(self.settings_file)
        self.setup.setWorkingDir(working_dir, log_dir="/predictor")

        self.encoder_file = self.setup.config_dir + "/encoder.pickle"
        self.noramalization_file = self.setup.config_dir + "/normalization_minmax.json"
        self.model_file = self.setup.config_dir + "/model.h5"

        # Check if required file exist
        if not os.path.exists(self.encoder_file): sys.exit(logger.error("Could not found encoder.pickle file"))
        if not os.path.exists(self.noramalization_file): sys.exit(logger.error("Could not found normalization_minmax.json"))
        if not os.path.exists(self.model_file): sys.exit(logger.error("Could not found Keras model file"))
        
        print(device_lib.list_local_devices())
        if tf.test.gpu_device_name():
            print("Using GPU")
        else:
            print("No GPU found, using CPU")

    def transformData(self):
        try:
            # Set dataframe column      
            if self.with_debug_target:
                # Keep target prediction column for dubug prediction
                train_column = self.source_dataframe[self.setup.loaded_settings["train_column"]]
                target_column  = self.source_dataframe[self.setup.loaded_settings["target_prediction"]]
                self.transform_dataframe = pandas.concat([train_column, target_column], axis=1)
            else:
                self.transform_dataframe = self.transform_dataframe[self.setup.loaded_settings["train_column"]]     
        except KeyError as e:
            logger.error("Could not parse header file. " + \
                "Make sure that the header file is the same as the columns used for the train set. %s" % (e))
            sys.exit(1)
        # Check if dataframe contains NAN values
        self.setup.hasDataframeNan(self.transform_dataframe)
        # Encoding data to numerical values
        self.setup.encodeLabel(self.transform_dataframe, load_file=True)
        # Normalize data
        self.setup.normalizeDataframe(self.transform_dataframe, load_file=True)
        
    def predictData(self, show_prediction=False):
        # Set models inputs
        new_X = self.transform_dataframe
        target_prediction = self.setup.loaded_settings["target_prediction"]
       
        if self.with_debug_target:
            # Drop target prediction
            new_X = self.transform_dataframe.drop(columns=target_prediction, axis=1)
            # Cache debug target prediction
            debug_new_Y = self.transform_dataframe[target_prediction]
            # Convert to np array
            new_X = new_X.to_numpy()
            debug_new_Y = debug_new_Y.to_numpy()
            # Denormalized debug Y
            dnz_debug_Y = self.setup.denormalizeDataset(debug_new_Y, target_prediction).round(0).astype(int)
            
        # Keras Callback instance
        # Get fit execution time
        predict_time_callback = PredictTime()
        # Predict target columns
        # Load model
        loadedModel = self.setup.loadModel()
        try:
            predicted_Y = loadedModel.predict(new_X, callbacks=[predict_time_callback])
        except Exception as e:
            logger.error(e)
            sys.exit()
        logger.info("Data prediction execution time = %.2f sec" % round(predict_time_callback.execTime, 2))
        # Denormalized predictions
        self.dnz_predicted_Y = self.setup.denormalizeDataset(predicted_Y, target_prediction).round(0).astype(int)

        # Record predictions
        if show_prediction:
            for i in range(len(new_X)):
                # Get "année de naissance"
                test = self.source_dataframe.loc[self.source_dataframe.index[i], "annedenaissance"] + self.dnz_predicted_Y[i]
                # Debug real data with predicted outputs
                if self.with_debug_target:
                    logger.debug("Real=%s VS Predicted=%s TEST=%s" % (dnz_debug_Y[i],  self.dnz_predicted_Y[i], test))   
                else:    
                    logger.debug("Predicted=%s TEST=%s" % (self.dnz_predicted_Y[i], test ) )
        
        # Retrieve debug prediction meaning
        if self.with_debug_target:
            prediction_deviation = []
            # Get deviation
            for i in range(len(new_X)): prediction_deviation.append(abs(dnz_debug_Y[i] - self.dnz_predicted_Y[i]))
            # Convert numpy array to dataframe
            prediction_deviation = pandas.DataFrame(prediction_deviation, columns=target_prediction)
            logger.info("Max predicted deviation:\n%s" % (prediction_deviation.max()))
            logger.info("Min predicted deviation:\n%s" % (prediction_deviation.min()))
            logger.info("Mean predicted deviation:\n%s" % (prediction_deviation.mean().round(2)))
            exactPrediction = ((prediction_deviation == 0).sum(axis=0) / prediction_deviation.size) * 100
            logger.info("Exact prediction rate (percent):\n%s" % (exactPrediction.round(2)))
            # Plot deviation
            prediction_deviation.plot(kind="box", grid=True, title="Debug prediction deviation per column")
            # Save as png meaning prediction plot
            pyplot.savefig(self.working_dir + "/debug_prediction_deviation.png")

    def exportPredictionFile(self, output_dir):
        # Numpy array to Dataframe
        prediction_dataframe = pandas.DataFrame(self.dnz_predicted_Y, 
         columns=self.setup.loaded_settings["target_prediction"])
        # Append predicted column into the source dataframe
        # output_dataframe = self.source_dataframe
        # output_dataframe = self.setup.denormalizeDataframe(output_dataframe)
        # output_dataframe = self.setup.decodeLabel(output_dataframe)
        # output_dataframe = output_dataframe.reset_index().drop(columns=["index"]) 
        
        output_dataframe = pandas.concat([self.source_dataframe, prediction_dataframe], axis=1)
        # Create output file
        export_file = output_dir + "/output_data.csv"
        pandas.DataFrame(output_dataframe).to_csv(export_file, encoding="latin-1")
        logger.success("Predictions has been exported in %s file" % (export_file))

# def standalone(): 
#     # Load dataset
#     # file_obj = open("")
#     # output_dataframe = pandas.read_csv(file_obj, encoding="latin-1")
#     # predictor = Predictor(output_dataframe, "settings.json", "Etmp/", with_debug_target=True)
#     # predictor.transformData()
#     # predictor.predictData(show_prediction=True)
#     # predictor.exportPredictionFile("")