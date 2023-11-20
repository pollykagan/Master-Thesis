"""
This module defines how a general Deep Learning classification model should look
"""
import abc
import numpy as np
from tensorflow import keras
from utilities.logger import Logger
from .model import Model


class Classifier(Model):
    """
    This is an abstract class defining a classifier behavior
    """
    def __init__(self, logger: Logger, model_base: keras.Model, optimizer: keras.optimizers.Optimizer, loss: keras.losses.Loss, print_model_summary: bool,
                 compilation_metrics: list[keras.metrics.Metric], batch_size: int, epochs: int):
        """
        A constructor
        :param logger: A logger to use for the output messages
        :param model_base: The type of the model to use
        :param optimizer: The optimizer to use during the training
        :param loss: The loss to use for training and evaluation
        :param print_model_summary: True if it is required to print the model summary, False otherwise
        :param compilation_metrics: Additional metrics to track during the training process
        :param batch_size: The size of each optimization data batch
        :param epochs: Number of training epochs
        """
        super().__init__(logger=logger, model_base=model_base, optimizer=optimizer, loss=loss, print_model_summary=print_model_summary, compilation_metrics=compilation_metrics, batch_size=batch_size,
                         epochs=epochs)

    @abc.abstractmethod
    def _build_model(self) -> None:
        """
        Abstract method. Serves for building the model (e.g., defining its architecture).
        Each subclass should define its model architecture on its own.
        :return: None (the model is built using the "self._model" attribute)
        """
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Performs the prediction on the provided data
        :param features: The data to perform the predictions on
        :return: The predicted labels (for each input sample)
        """
        # Calculate the probabilities for each class
        predictions: np.ndarray = self._model.predict(features)
        # Choose the label based on the higher probability
        labels: np.ndarray = np.argmax(predictions, axis=1)
        return labels
