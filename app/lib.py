"""
Function lib used in the learner and predictor
"""
import os
import sys
import datetime
import json
import numpy as np
import pickle
from loguru import logger

from colorama import init, Fore, Back, Style
from matplotlib import pyplot
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


class Setup:
    def load_learner_settings(self, settings_file):
        """
        Load learner settings from a JSON file.
        """
        self.settings_file = settings_file
        if os.path.exists(settings_file):
            with open(settings_file, "r") as json_file:
                self.loaded_settings = json.load(json_file)
                self.hyperparameter = self.loaded_settings["hyperparameter"]
                logger.success("Loaded learner settings from disk")
        else:
            logger.error("Could not find file: %s" % settings_file)
            sys.exit(1)
        return self.loaded_settings

    def set_working_dir(self, working_dir, log_dir):
        """
        Set the working directory and create necessary folders.
        """
        if not os.path.exists(working_dir):
            logger.error("Could not find directory: %s" % working_dir)
            sys.exit(1)

        logs_dir = os.path.join(working_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        learner_log_dir = os.path.join(logs_dir, log_dir)
        if not os.path.exists(learner_log_dir):
            os.makedirs(learner_log_dir)

        log_file_path = os.path.join(learner_log_dir, f"logs_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.log_handler = logger.add(log_file_path, format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="DEBUG")

        self.config_dir = os.path.join(working_dir, "config")
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def has_dataframe_nan(self, dataframe):
        """
        Check if the dataframe contains any NaN values.
        Replace NaN values if specified in the loaded settings file.
        """
        try:
            if dataframe.isnull().values.any():
                nan_count = dataframe.isnull().sum().sum()
                logger.warning("%d NaN detected in the dataframe:" % nan_count)
                for column_name, column_data in dataframe.iteritems():
                    if dataframe[column_name].isnull().sum() > 0:
                        logger.debug("- [%s] => %d NaN" % (column_name, dataframe[column_name].isnull().sum()))
                        if self.loaded_settings["replace_nan"]:
                            dataset_column = dataframe[column_name].to_numpy().reshape(-1, 1)
                            dataframe[column_name] = SimpleImputer(strategy="most_frequent").fit_transform(dataset_column)
                            logger.success("NaN value has been replaced in column [%s]" % column_name)
                        else:
                            logger.error("Dataframe contains NaN value. If you wish to replace them automatically, check your settings file")
                            sys.exit(1)
        except Exception as e:
            logger.error("An error occurred in has_dataframe_nan() function: %s" % e)
            sys.exit(1)

    def encode_label(self, dataframe, load_file=False):
        """
        Encode categorical variables in the dataframe using LabelEncoder.
        Save the encoder to a pickle file if load_file is False, else load the encoder from the pickle file.
        """
        try:
            if not load_file:
                encoder_dict = {}
                categorical_columns = self.loaded_settings["encode_variables"]
                for column_name, column_data in dataframe.iteritems():
                    for encode_target in categorical_columns:
                        if column_name == encode_target["name"]:
                            if encode_target["values"]:
                                encoder = LabelEncoder().fit(encode_target["values"])
                            else:
                                encoder = LabelEncoder().fit(column_data)
                            try:
                                dataframe[column_name] = encoder.transform(column_data)
                            except Exception as e:
                                logger.error("Cannot encode column [%s]. It may contain values that have not been described in the settings: %s"
                                             % (column_name, e))
                                sys.exit(1)
                            encoder_dict[column_name] = encoder

                encoder_pickle = os.path.join(self.config_dir, "encoder.pickle")
                with open(encoder_pickle, "wb") as file:
                    pickle.dump(encoder_dict, file)
                    logger.success("Data encoder transform saved")
            else:
                encoder_pickle = os.path.join(self.config_dir, "encoder.pickle")
                if os.path.exists(encoder_pickle):
                    with open(encoder_pickle, "rb") as file:
                        encoder_dict = pickle.load(file)
                        logger.success("Data encoder transform loaded")
                        for column_name, column_data in dataframe.iteritems():
                            if column_name in encoder_dict:
                                dataframe[column_name] = encoder_dict[column_name].transform(column_data)
            logger.success("Encoding dataframe done")
            return dataframe
        except Exception as e:
            logger.error("An error occurred in encode_label() function: %s" % e)
            sys.exit(1)

    def decode_label(self, dataframe):
        """
        Decode encoded labels in the dataframe using the encoder pickle file.
        """
        try:
            encoder_pickle = os.path.join(self.config_dir, "encoder.pickle")
            if os.path.exists(encoder_pickle):
                with open(encoder_pickle, "rb") as file:
                    encoder_dict = pickle.load(file)
                for column_name, column_data in dataframe.iteritems():
                    if column_name in encoder_dict:
                        dataframe[column_name] = encoder_dict[column_name].inverse_transform(column_data.astype(int))
                logger.success("Dataframe has been decoded")
                return dataframe
            else:
                logger.error("Could not find file: %s" % encoder_pickle)
                sys.exit(1)
        except Exception as e:
            logger.error("An error occurred in decode_label() function: %s" % e)
            sys.exit(1)

    def normalize_dataframe(self, dataframe, load_file=False):
        """
        Normalize the numerical columns in the dataframe to the range [0, 1] using min-max normalization.
        Save the normalization parameters to a JSON file if load_file is False, else load the parameters from the JSON file.
        """
        try:
            normalizer_file = os.path.join(self.config_dir, "normalization_minmax.json")
            custom_normalization = self.loaded_settings["custom_normalization"]
            if not load_file:
                normalizer = {}
                with open(normalizer_file, "w") as json_file:
                    for column_name, column_data in dataframe.iteritems():
                        if dataframe[column_name].dtype == object:
                            logger.error("Cannot normalize column [%s], type<%s>, must be numeric" 
                                         % (column_name, dataframe[column_name].dtype))
                            sys.exit(1)
                        min_val = np.min(dataframe[column_name])
                        max_val = np.max(dataframe[column_name])
                        if custom_normalization:
                            for normalize_target in custom_normalization:
                                if column_name == normalize_target["name"]:
                                    min_val = normalize_target["min"]
                                    max_val = normalize_target["max"]
                                    break
                        if min_val == max_val:
                            logger.warning("Cannot normalize column [%s], contains only %.2f. Setting the whole column to 0"
                                            % (column_name, min_val))
                            normalizer[column_name] = {"min": 0.0, "max": 1.0}
                        else:
                            dataframe[column_name] = (column_data - min_val) / (max_val - min_val)
                            normalizer[column_name] = {"min": float(min_val), "max": float(max_val)}
                    json.dump(normalizer, json_file, indent=4)
                    logger.success("Data normalization min-max saved")
            else:
                if os.path.exists(normalizer_file):
                    with open(normalizer_file, "r") as json_file:
                        normalizer = json.load(json_file)
                        logger.success("Data normalization min-max loaded")
                        for column_name, column_data in dataframe.iteritems():
                            if column_name in normalizer:
                                min_val = normalizer[column_name]["min"]
                                max_val = normalizer[column_name]["max"]
                                dataframe[column_name] = (column_data - min_val) / (max_val - min_val)
                else:
                    logger.error("Could not find file: %s" % normalizer_file)
                    sys.exit(1)
            logger.success("Normalizing dataframe done")
            return dataframe
        except Exception as e:
            logger.error("An error occurred in normalize_dataframe() function: %s" % e)
            sys.exit(1)

    def denormalize_dataframe(self, dataframe):
        """
        Denormalize the normalized numerical columns in the dataframe using the normalization parameters from the JSON file.
        """
        try:
            normalizer_file = os.path.join(self.config_dir, "normalization_minmax.json")
            if os.path.exists(normalizer_file):
                with open(normalizer_file, "r") as json_file:
                    entry = json.load(json_file)
                    for column_name, column_data in dataframe.iteritems():
                        if column_name in entry:
                            min_val = entry[column_name]["min"]
                            max_val = entry[column_name]["max"]
                            dataframe[column_name] = column_data * (max_val - min_val) + min_val
                    logger.success("Dataframe has been denormalized")
                    return dataframe
            else:
                logger.error("Could not find file: %s" % normalizer_file)
                sys.exit(1)
        except Exception as e:
            logger.error("An error occurred in denormalize_dataframe() function: %s" % e)
            sys.exit(1)

    def denormalize_dataset(self, dataset, header_column):
        """
        Denormalize the normalized numerical columns in the dataset using the normalization parameters from the JSON file.
        """
        try:
            normalizer_file = os.path.join(self.config_dir, "normalization_minmax.json")
            if os.path.exists(normalizer_file):
                with open(normalizer_file, "r") as json_file:
                    entry = json.load(json_file)
                    for index, column_name in enumerate(header_column):
                        if column_name in entry:
                            min_val = entry[column_name]["min"]
                            max_val = entry[column_name]["max"]
                            dataset[:, index] = dataset[:, index] * (max_val - min_val) + min_val
                    logger.success("Dataset has been denormalized")
                    return dataset
            else:
                logger.error("Could not find file: %s" % normalizer_file)
                sys.exit(1)
        except Exception as e:
            logger.error("An error occurred in denormalize_dataset() function: %s" % e)
            sys.exit(1)

    def export_model(self, model):
        """
        Export the Keras model to a HDF5 file.
        """
        try:
            model_file = os.path.join(self.config_dir, "model.h5")
            model.save(model_file)
            logger.success("Saved Keras model to disk")
        except Exception as e:
            logger.error("An error occurred in export_model() function: %s" % e)
            sys.exit(1)

    def load_model(self):
        """
        Load the Keras model from the saved model file.
        """
        try:
            model_file = os.path.join(self.config_dir, "model.h5")
            if os.path.exists(model_file):
                loaded_model = load_model(model_file)
                print(loaded_model.summary())
                logger.success("Loaded Keras model from disk")
                return loaded_model
            else:
                logger.error("Could not find file: %s" % model_file)
                sys.exit(1)
        except Exception as e:
            logger.error("An error occurred in load_model() function: %s" % e)
            sys.exit(1)

def set_learner_param(dataframe, settings_file, target_prediction=[]):
    """Set the learner parameters in the settings file based on the dataframe and target prediction columns."""

    try:
        with open(settings_file, "r") as json_file:
            data = json.load(json_file)
            train_column = dataframe.columns.tolist()
            if target_prediction:
                train_column = [column for column in train_column if column not in target_prediction]
                data["target_prediction"] = target_prediction
            data["train_column"] = train_column
        with open(settings_file, "w+") as json_file:
            json.dump(data, json_file, indent=4)
        return dataframe.columns.tolist()
    except Exception as e:
        logger.error("An error occurred in set_learner_param() function: %s" % e)
        sys.exit(1)

def has_dataset_nan(dataset, replace_nan=False):
    """Check if the dataset contains any NaN values."""

    if np.isnan(dataset).any():
        logger.warning("NAN value detected in the dataset")
        logger.debug(np.argwhere(np.isnan(dataset)))
        return True
    return False

def get_new_born(birthday, predicted_age):
    """Calculate the birth year of a child based on the birthday and predicted age."""

    child_birth_year = birthday + predicted_age
    return child_birth_year

def print_console(message, color=Fore.RESET, back=Back.RESET):
    """Print a message to the console with specified color and background."""

    print(color + back + message + Style.RESET_ALL)

def print_debug(msg):
    """Print a debug message with timestamp."""
    
    now = datetime.datetime.now()
    print("%s | %s" % (now.strftime("%Y-%m-%d %H:%M:%S"), msg))