"""
This module implements the emotions classifier
"""
from tensorflow import keras
from utilities.logger import Logger
from .classifier import Classifier


class EmotionClassifier(Classifier):
    """
    This class implements the Classifier (and the Model) interface for emotion classification task
    """
    def __init__(self, logger: Logger, input_shape: tuple[int, int, int, int], print_model_summary: bool, emotions_number: int, batch_size: int, epochs: int, learning_rate: float, kernel: int,
                 channels: list[int], pooling: int, pooling_layers: list[int], activation: str, dropout: float, regularization_coefficient: float, mlp_units: list[int]):
        """
        A constructor
        :param logger: A logger to use for the output messages
        :param input_shape: The shape of each sample for the model (since it is a CNN classifier, all the samples must have the same shape)
        :param print_model_summary: True if it is required to print the model summary, False otherwise
        :param batch_size: The size of each optimization data batch
        :param epochs: Number of training epochs
        :param learning_rate: The learning rate of the optimizer
        :param kernel: The size of the kernel (the model use square kernels, so it's enough to pass a single integer, e.g. kernel_size=3 refers to a kernel of shape (3, 3))
        :param channels: A list containing the number of kernels (channels) for each convolutional layer
        :param pooling: The size of the pooling kernel, also defines the stride of the pooling (e.g., pooling=2 ==> pooling_kernel = (2, 2) and pooling_stride = (2, 2)
        :param pooling_layers: The indices of the CNN layers on which the pooling should be applied
        :param activation: A string representing an activation function to be used for the model layers
        :param dropout: A probability for the Dropout
        :param regularization_coefficient: The coefficient of the L1, L2 regularization
        :param mlp_units: A list containing the number of units for each linear layer
        """
        if emotions_number < 2:
            logger.fatal(32, f'{self.__class__.__name__} received bad number of emotions (labels):  {emotions_number}', ValueError)
        if emotions_number == 2:
            loss = keras.losses.BinaryCrossentropy()
            accuracy_metric = keras.metrics.BinaryAccuracy()
        else:
            loss = keras.losses.SparseCategoricalCrossentropy()
            accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
        super().__init__(logger=logger, model_base=keras.Sequential(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, print_model_summary=print_model_summary,
                         compilation_metrics=[accuracy_metric], batch_size=batch_size, epochs=epochs)
        self._input_shape: tuple[int, int, int, int] = input_shape
        self._labels_number: int = emotions_number
        self._kernel: tuple[int, int] = (kernel, kernel)
        self._channels: list[int] = channels
        self._pooling: tuple[int, int] = (pooling, pooling)
        self._pooling_indices: list[int] = pooling_layers
        self._activation: str = activation
        self._dropout: float = dropout
        self._regularization_coefficient: float = regularization_coefficient
        self._mlp_units: list[int] = mlp_units
        # Check the model parameters
        if not 0 <= self._dropout <= 1:
            self._logger.fatal(33, f'{self.__class__.__name__} received bad probability for dropout: {self._dropout}', ValueError)

    def _build_cnn(self) -> None:
        """
        Builds the convolutional part of the model
        :return: None
        """
        for index, channels_number in enumerate(self._channels):
            self._model.add(keras.layers.Conv2D(name=f'Layer_{index}__Convolution', filters=channels_number, kernel_size=self._kernel, activation=self._activation,
                                                kernel_regularizer=keras.regularizers.L2(self._regularization_coefficient)))
            if index in self._pooling_indices:
                self._model.add(keras.layers.MaxPooling2D(name=f'Layer_{index}__Max_Pooling', pool_size=self._pooling, strides=self._pooling))
            self._model.add(keras.layers.BatchNormalization(name=f'Layer_{index}__Batch_Normalization'))
            self._model.add(keras.layers.Dropout(name=f'Layer_{index}__Dropout', rate=self._dropout))

    def _build_mlp(self) -> None:
        """
        Builds the linear part of the model
        :return: None
        """
        for index, mlp_units in enumerate(self._mlp_units, start=len(self._channels)):
            self._model.add(keras.layers.Dense(name=f'Layer_{index}__Linear', units=mlp_units, activation=self._activation,
                                               kernel_regularizer=keras.regularizers.L2(self._regularization_coefficient)))
            self._model.add(keras.layers.Dropout(name=f'Layer_{index}__Dropout', rate=self._dropout))
        if self._labels_number == 2:
            last_layer_units_number = 1
            last_layer_activation = 'sigmoid'
        else:
            last_layer_units_number = self._labels_number
            last_layer_activation = 'softmax'
        # The last linear layer activation function is not configurable, since we want to get as output the probabilities for each class.
        # This is done using Sigmoid activation for Binary Classification, or using Softmax activation for Multi-Label Classification
        self._model.add(keras.layers.Dense(name=f'{last_layer_activation}_Layer'.capitalize(), units=last_layer_units_number, activation=last_layer_activation))

    def _build_model(self) -> None:
        """
        Builds the entire model
        :return: None
        """
        self._build_cnn()
        # CNN works with 4D tensors (<number of samples>, <height>, <width>, <channels>), but MLP works with 2D tensors (<number of samples>, <number of features>).
        # So, when moving from the CNN to the MLP part, we need to convert each sample from a 3D tensor to 1D vector
        self._model.add(keras.layers.Flatten())
        self._build_mlp()
        self._model.build(self._input_shape)
