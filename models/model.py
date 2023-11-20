"""
This module defines how a general Deep Learning model should look
"""
import abc
import numpy as np
from tensorflow import keras
from utilities.logger import Logger


class Model(abc.ABC):
    """
    This is an abstract class defining a Deep Learning model behavior
    """
    def __init__(self, logger: Logger, model_base: keras.Model, optimizer: keras.optimizers.Optimizer, loss: keras.losses.Loss, print_model_summary: bool,
                 compilation_metrics: list[keras.metrics.Metric], batch_size: int, epochs: int):
        """
        A constructor
        :param logger: A logger to use for the output messages
        :param model_base: A base model object
        :param optimizer: The optimizer to use during the training
        :param loss: The loss to use for training and evaluation
        :param print_model_summary: True if it is required to print the model summary, False otherwise
        :param compilation_metrics: Additional metrics to track during the training process
        :param batch_size: The size of each optimization data batch
        :param epochs: Number of training epochs
        """
        self._logger: Logger = logger
        self._model: keras.Model = model_base
        # Store parameters for the 'build' method
        self._optimizer: keras.optimizers.Optimizer = optimizer
        self._loss: keras.losses.Loss = loss
        self._print_model_summary: bool = print_model_summary
        self._metrics: list[keras.metrics.Metric] = compilation_metrics
        # Store the batch size and the epochs number (to use in the 'train' method)
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._model_exists: bool = False

    @abc.abstractmethod
    def _build_model(self) -> None:
        """
        Abstract method. Serves for building the model (e.g., defining its architecture).
        Each subclass should define its model architecture on its own.
        :return: None (the model is built using the "self._model" attribute)
        """
        pass

    def build(self) -> None:
        """
        Builds and compiles the model
        :return: None
        """
        # Invoke the "_build_model" method. Each subclass should define it on its own (in this class the method is abstract)
        self._logger.info(f'Building the model ({self.__class__.__name__})')
        self._build_model()
        # Compile the model
        self._logger.info('Compiling the model')
        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        # If required, print the model summary
        if self._print_model_summary:
            self._logger.info('Model summary:')
            self._model.summary()
        self._model_exists = True
        self._logger.info('Finished build and compilation of the model')

    def train(self, training_features: np.ndarray, training_labels: np.ndarray, validation_features: np.ndarray, validation_labels: np.ndarray, early_stop_by: str) -> keras.callbacks.History:
        """
        Performs the model training process
        :param training_features: A tensor containing the training data
        :param training_labels: A tensor containing the training labels
        :param validation_features: A tensor containing the validation data
        :param validation_labels: A tensor containing the validation labels
        :param early_stop_by: A metric to base the early stopping mechanism on
        :return: An object describing the training history
        """
        self._logger.info(f'Starting the training process. It will last for {self._epochs} epochs')
        early_stopping: keras.callbacks.EarlyStopping = keras.callbacks.EarlyStopping(monitor=early_stop_by, patience=20, restore_best_weights=True)
        result: keras.callbacks.History = self._model.fit(training_features, training_labels, validation_data=(validation_features, validation_labels),
                                                          batch_size=self._batch_size, epochs=self._epochs, callbacks=[early_stopping])
        self._logger.info('Finished the training process')
        return result

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        """
        Performs the model evaluation
        :param features: The data on which the model should be evaluated
        :param labels: The labels corresponding to the provided data
        :return:
        """
        self._logger.info(f'Starting the evaluation process on {labels.shape[0]} samples')
        loss, accuracy = self._model.evaluate(features, labels)
        return loss, accuracy

    def save(self, path: str) -> None:
        """
        Saves the model to a file
        :param path: A path to a file where the model should be saved (without extension)
        :return: None
        """
        path += '.keras'
        self._logger.info(f'Saving the model to {path}')
        self._model.save(path)

    def load(self, path: str) -> None:
        """
        Loads keras model from the provided path, and stores it as an object model
        :param path: A path to a file where the model is stored (without extension)
        :return: None
        """
        if self._model_exists:
            self._logger.warning(31, 'Model class was requested to load a pre-trained keras model, but there already exists one. The existing model will be overriden')
        self._logger.info(f'Loading a pre-trained model from {path}')
        self._model: keras.Model = keras.models.load_model(f'{path}.keras')
        if self._print_model_summary:
            self._logger.info('Model summary:')
            self._model.summary()
