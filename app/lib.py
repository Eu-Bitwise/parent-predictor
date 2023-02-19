"""
Function lib used in the learner and predictor
"""

# from tensorflow.keras.models import save_model
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from matplotlib import pyplot

import numpy as np
import pickle
import os
import datetime
import json
import sys
from loguru import logger

from colorama import init
from colorama import Fore, Back, Style

class Setup():
    
    def loadLearnerSettings(self, settings_file):
        # Retrieve learner settings file
        self.settings_file = settings_file
        if os.path.exists(settings_file):
            json_file = open(settings_file, "r")
            self.loaded_settings = json.load(json_file)
            self.hyperparameter = self.loaded_settings["hyperparameter"]
            json_file.close()
            logger.success("Loaded learner settings from disk")
        else: 
            logger.error("Could not found  %s file" % (settings_file))
            sys.exit(1)
        return self.loaded_settings 
    
    def setWorkingDir(self, working_dir, log_dir):
        # Check if working directory exists
        if not os.path.exists(working_dir): 
            logger.error("Could not found  %s directory" % (working_dir))
            sys.exit(1)

        # Create folder for logs if not exist 
        logs_dir = working_dir + "/logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        # Create log directory if not exists
        learner_log_dir = logs_dir + log_dir
        if not os.path.exists(learner_log_dir):
            os.makedirs(learner_log_dir)
        # Create logs handler
        self.log_handler = logger.add(learner_log_dir + "/logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
         format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="DEBUG")

        # Create config folder if not exist
        self.config_dir = working_dir + "/config"
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def hasDataframeNan(self, dataframe):
        try:
            if dataframe.isnull().values.any():
                logger.warning("%d NaN detected in the dataframe:" % (dataframe.isnull().sum().sum()))
                for (columnName, columnData) in dataframe.iteritems():
                    if dataframe[columnName].isnull().sum() > 0: 
                        logger.debug("- [%s] => %d NaN" % (columnName, dataframe[columnName].isnull().sum()))
                        if self.loaded_settings["replace_nan"]:
                            datasetColumn = dataframe[columnName].to_numpy().reshape(-1, 1)
                            dataframe[columnName] = SimpleImputer(strategy="most_frequent").fit_transform(datasetColumn)
                            logger.success("NaN value has been replaced in column [%s]" % (columnName))
                        else: 
                            logger.error("Dataframe contains NaN value. If you wish to replace them automatically, check your settings file")
                            sys.exit(1)
        except Exception as e:
            logger.error("An error occured in hasDataframeNan() function: %s" % (e))
            sys.exit(1)
            
    def encodeLabel(self, dataframe, load_file=False) :
        try: 
            # Encoding data to numerical values
            if not load_file:
                encoder_dict = {}
                categorical_columns = self.loaded_settings["encode_variables"]
                for (columnName, columnData) in dataframe.iteritems():
                    for encode_target in categorical_columns:
                        if columnName == encode_target["name"]:
                            if encode_target["values"]:
                                encoder = LabelEncoder().fit(encode_target["values"])
                            else:
                                # If "values" key empty encode from column data
                                encoder = LabelEncoder().fit(columnData)
                            try:
                                dataframe[columnName] = encoder.transform(columnData)
                            except Exception as e:
                                logger.error("Cannot encode column [%s]. It may contains values that has not been all described in settings: %s"
                                    % (columnName, e))
                                sys.exit(1)
                            encoder_dict[columnName] = encoder
                # Save encoder to a pickle file
                path = os.path.dirname(os.path.realpath(__file__))
                encoder_pickle = self.config_dir + "/encoder.pickle"
                with open(encoder_pickle, "wb") as File:
                    pickle.dump(encoder_dict, File)
                    logger.success("Data encoder transform saved")
            else:
                # Load encoder transform from pickle file
                encoder_pickle = self.config_dir + "/encoder.pickle"
                if os.path.exists(encoder_pickle):
                    with open(encoder_pickle, "rb") as File:
                        encoder_dict = pickle.load(File)
                        logger.success("Data encoder transform loaded")
                        for (columnName, columnData) in dataframe.iteritems():
                            if columnName in encoder_dict:
                                dataframe[columnName] = encoder_dict[columnName].transform(columnData)
            logger.success("Encoding dataframe done")
            return dataframe
        except Exception as e:
            logger.error("An error occured in encodeLabel() function: %s" % (e))
            sys.exit(1)

    def decodeLabel(self, dataframe):
        try: 
            path = os.path.dirname(os.path.realpath(__file__))
            encoder_pickle = self.config_dir + "/encoder.pickle"
            if os.path.exists(encoder_pickle):
                with open(encoder_pickle, "rb") as File:
                    encoder_dict = pickle.load(File)
                # Décoding dataset
                for (columnName, columnData) in dataframe.iteritems():
                    if columnName in encoder_dict:
                        dataframe[columnName] = encoder_dict[columnName].inverse_transform(columnData.astype(int))
                logger.success("Dataframe has been decoded")
                return dataframe
            else:
                logger.error("Could not found %s file" % (encoder_pickle))
                sys.exit(1)
        except Exception as e:
            logger.error("An error occured in decodeLabel() function: %s" % (e))
            sys.exit(1)

    def normalizeDataframe(self, dataframe, load_file=False):
        try:
            normalizer_file = self.config_dir + "/normalization_minmax.json"
            custom_normalization = self.loaded_settings["custom_normalization"]
            if not load_file:
                with open(normalizer_file, "w") as jsonFile:
                    normalizer = {}
                    for (columnName, columnData) in dataframe.iteritems():
                        if dataframe[columnName].dtype == object:
                            logger.error("Cannot normalize column [%s] type<%s>, must be numeric " 
                             % (columnName, dataframe[columnName].dtype))
                            sys.exit(1)
                        min = np.min(dataframe[columnName])
                        max = np.max(dataframe[columnName])
                        if custom_normalization:
                            for normalize_target in custom_normalization:
                                if columnName == normalize_target["name"]:
                                    min = normalize_target["min"]
                                    max = normalize_target["max"]
                                    break
                        if min == max:
                            logger.warning("Cannot normalize column [%s], contains only => %.2f, set whole column to 0"
                             % (columnName, min))
                            normalizer[columnName] = {"min": 0.0, "max": 1.0}
                        else:
                            dataframe[columnName] = (columnData - min) / (max - min)
                            normalizer[columnName] = {"min": float(min), "max": float(max)}
                    # Write min/max column
                    json.dump(normalizer, jsonFile, indent=4)
                    logger.success("Data normalization min max saved")
            else:
                if os.path.exists(normalizer_file):
                    with open(normalizer_file, "r") as jsonFile:
                        normalizer = json.load(jsonFile)
                        logger.success("Data normalization min max loaded")
                        for (columnName, columnData) in dataframe.iteritems():
                            if columnName in normalizer:     
                                min = normalizer[columnName]["min"]
                                max = normalizer[columnName]["max"]
                                dataframe[columnName] = (columnData - min) / (max - min)
                else:
                    logger.error("Could not found  %s file" % (normalizer_file))
                    sys.exit(1)
            logger.success("Normalizing dataframe done")
            return dataframe
        except Exception as e:
            logger.error("An error occured in normalizeDataframe() function: %s" % (e))
            sys.exit(1)

    def denormalizeDataframe(self, dataframe):
        try:
            path = os.path.dirname(os.path.realpath(__file__))
            normalizer_file = self.config_dir + "/normalization_minmax.json"
            if os.path.exists(normalizer_file):
                with open(normalizer_file, "r") as jsonFile:
                    entry = json.load(jsonFile)
                    for (columnName, columnData) in dataframe.iteritems():
                        if columnName in entry:
                            min = entry[columnName]["min"]
                            max = entry[columnName]["max"]
                            dataframe[columnName] = columnData * (max - min) + min
                    logger.success("Dataframe has been denormalized")
                    return dataframe
            else:
                logger.error("Could not found %s file" % (normalizer_file))
                sys.exit(1)
        except Exception as e:
            logger.error("An error occured in denormalizeDataframe() function: %s" % (e))
            sys.exit(1)

    def denormalizeDataset(self, dataset, header_column):
        try:
            normalizer_file = self.config_dir + "/normalization_minmax.json"
            if os.path.exists(normalizer_file):
                with open(normalizer_file, "r") as jsonFile:
                    entry = json.load(jsonFile)
                    for index, columnName in enumerate(header_column):
                        if columnName in entry:
                            min = entry[columnName]["min"]
                            max = entry[columnName]["max"]
                            dataset[:, index] = dataset[:, index] * (max - min) + min
                    logger.success("Dataset has been denormalized")
                    return dataset
            else:
                logger.error("Could not found %s file" % (normalizer_file))
                sys.exit(1)
        except Exception as e:
            logger.error("An error occured in denormalizeDataset() function: %s" % (e))
            sys.exit(1)

    def exportModel(self, model):
        try:
            model_file = self.config_dir + "/model.h5"
            # Serialize model to HDF5
            model.save(model_file)
            logger.success("Saved Keras model to disk")
        except Exception as e:
            logger.error("An error occured in exportModel() function: %s" % (e))
            sys.exit(1)

    def loadModel(self):
        try: 
            model_file = self.config_dir + "/model.h5"
            if os.path.exists(model_file):
                # Load existing model
                loadedModel = load_model(model_file)
                # Summarize model.
                print(loadedModel.summary())
                logger.success("Loaded Keras model from disk")
                return loadedModel
            else:
                logger.error("Could not found %s file" % (model_file))
                sys.exit(1)
        except Exception as e:
            logger.error("An error occured in loadModel() function: %s" % (e))
            sys.exit(1)

def setLearnerParam(dataframe, settings_file, target_prediction=[]): 
    try:
        with open(settings_file, "r") as jsonFile:
            data = json.load(jsonFile)
            trainColumn = dataframe.columns.tolist()
            if target_prediction:
                trainColumn = [column for column in trainColumn if column not in target_prediction] 
                data["target_prediction"] = target_prediction
            data["train_column"] = trainColumn
        with open(settings_file, "w+") as jsonFile:
            json.dump(data, jsonFile, indent=4)
        return dataframe.columns.tolist()
    except Exception as e:
        logger.error("An error occured in setLearnerParam() function: %s" % (e))
        sys.exit(1)
        
def hasDatasetNan(dataset, replaceNan=False):
    if np.isnan(dataset).any():
        logger.warning("NAN value detected in the dataset")
        logger.debug(np.argwhere(np.isnan(dataset)))
        return True
    return False

def getNewBorn(birthday, predicted_age):
    child_birhYear = birthday + predicted_age
    return child_birhYear

def printConsole(message, color=Fore.RESET, back=Back.RESET):
    print(color + back + message + Style.RESET_ALL)

def printDebug(msg):
    now = datetime.datetime.now()
    print("%s | %s" % (now.strftime("%Y-%m-%d %H:%M:%S"), msg) )

'''
def replaceDataframeColumn(dataframe_dst, dataframe_src):
    for (columnName_dst, columnData_dst) in dataframe_dst.iteritems():
        for (columnName_src, columnData_src) in dataframe_src.iteritems():
            if columnName_dst == columnName_src:
                dataframe_dst[columnName_dst] = columnData_src
    return dataframe_dst
'''
'''
def getDataframeColumnIndex(dataframe, header_column):
    columnIndex = []
    for (columnName, columnData) in dataframe.iteritems():
        if columnName in header_column: 
            columnIndex.append(dataframe.columns.get_loc(columnName))
    return columnIndex
'''