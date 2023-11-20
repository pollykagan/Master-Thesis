"""
This module is responsible for performing the emotions classification task pipeline
"""
import os
import time
import configargparse
import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utilities.logger import Logger
from models.emotion_classifier import EmotionClassifier


class EmotionClassificationPipeline:
    """
    This class performs the emotions classification task pipeline
    """
    def __init__(self, args: configargparse.Namespace, logger: Logger):
        """
        A constructor
        :param args: A namespace containing the parsed command line arguments
        :param logger: A logger to use for the output messages
        """
        self._args: configargparse.Namespace = args
        self._logger: Logger = logger
        self._input_shape: tuple = ()
        self._classes_number: int = 0
        self._accuracy_metric: str = ''

    def _get_features_and_labels(self):
        """
        Extracts the features and the labels, according to the specified emotion and dataset/s (if requested "-dataset all")
        :return: A pair (tuple) - the first item is numpy.ndarray of features, and the second is numpy.ndarray of labels
        """
        if self._args.dataset == 'all':
            datasets: list[str] = ['crema_d', 'emo_db', 'emovo', 'ravdess']
        else:
            datasets: list[str] = [self._args.dataset]
        features_tensors_list: list[np.ndarray] = []
        labels_tensors_list: list[np.ndarray] = []
        for dataset in datasets:
            preprocessed_data_directory: str = os.path.join('data', self._args.emotion, 'preprocessed', dataset, 'mfcc')
            # The loaded tensor is 3D (<number of samples>, <height>, <width>), so the 4-th dimension (<channel>) need to be added
            current_features: np.ndarray = np.load(os.path.join(preprocessed_data_directory, 'features.npz'))['arr_0'][..., np.newaxis]
            current_labels: np.ndarray = np.load(os.path.join(preprocessed_data_directory, 'labels.npz'))['arr_0']
            features_tensors_list.append(current_features)
            labels_tensors_list.append(current_labels)
        if len(features_tensors_list) == 1 and len(labels_tensors_list) == 1:
            return features_tensors_list[0], labels_tensors_list[0]
        return np.concatenate(features_tensors_list), np.concatenate(labels_tensors_list)

    @staticmethod
    def _plot_history(history: keras.callbacks.History, accuracy_metric_string: str, output_file: [str, None] = None) -> None:
        """
        Plots accuracy and loss for training and validation sets as a function of the epochs
        :param history: The object containing the training process history (the one returned by tensorflow 'fit' method
        :param accuracy_metric_string: A string representing the name of the accuracy metric (depends on the number of output labels)
        :param output_file: A path where the graph should be saved (use None to display it on the screen without saving)
        :return: None
        """
        fig, axs = plt.subplots(2, figsize=(15, 10))
        # Create accuracy subplot
        axs[0].plot(list(map(lambda accuracy: accuracy * 100, history.history[accuracy_metric_string])), label='Train Accuracy')
        axs[0].plot(list(map(lambda accuracy: accuracy * 100, history.history[f'val_{accuracy_metric_string}'])), label='Validation Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend(loc='lower right')
        axs[0].set_title('Model Accuracy (in %)')
        # Create loss subplot
        axs[1].plot(history.history['loss'], label='Train Loss')
        axs[1].plot(history.history['val_loss'], label='Validation Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Error')
        axs[1].legend(loc='upper right')
        axs[1].set_title('Model Loss (Categorical Cross-Entropy)')
        # Display the graph (or store it to a file)
        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()

    def _load_and_split_data(self, create_validation_set: bool = True) -> tuple[np.ndarray, np.ndarray, [np.ndarray, None], [np.ndarray, None], np.ndarray, np.ndarray]:
        """
        Loads the stored dataset (using "--dataset" value) and splits it to the training, validation and test sets
        :param create_validation_set: If True - will create training, validation and test sets, otherwise - only the training and the test sets (good for K-Fold CV)
        :return: A tuple containing the (training features, training labels, validation features, validation labels, test features, test labels).
                If create_validation_set=False then (training features, training labels, test features, test labels)
        """
        # Load the preprocessed features
        self._logger.info('Loading the data')
        features_and_labels: tuple[np.ndarray, np.ndarray] = self._get_features_and_labels()
        features: np.ndarray = features_and_labels[0]
        labels: np.ndarray = features_and_labels[1]
        self._logger.info(f'Finished loading the data. Features shape: {features.shape}. Labels shape: {labels.shape}')
        # Extract the input shape of the data (it can be extracted from the first sample, because all the samples has the same shape)
        self._input_shape: tuple = features.shape
        self._classes_number: int = len(np.unique(labels)) if self._args.emotion == 'all' else 2
        self._accuracy_metric: str = 'sparse_categorical_accuracy' if self._args.emotion == 'all' else 'binary_accuracy'
        self._logger.info('Splitting the data to train, validation and test sets')
        work_data, test_data, work_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=13)
        if not create_validation_set:
            self._logger.info(f'Finished splitting the data. Train set size: {work_data.shape[0]}. Test set size: {test_data.shape[0]}')
            return work_data, work_labels, None, None, test_data, test_labels
        train_data, validation_data, train_labels, validation_labels = train_test_split(work_data, work_labels, test_size=0.25, random_state=13)
        self._logger.info(f'Finished splitting the data. Train set size: {train_data.shape[0]}. Validation set size: {validation_data.shape[0]}. Test set size: {test_data.shape[0]}')
        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    def perform_single_configuration_training(self) -> None:
        """
        Performs a single training, according to the received configuration
        :return: None
        """
        with tensorflow.device('/device:GPU:0'):
            train_data, train_labels, validation_data, validation_labels, test_data, test_labels = self._load_and_split_data()
            # Instantiate the classifier
            classifier: EmotionClassifier = EmotionClassifier(self._logger, self._input_shape, self._args.model_summary, self._classes_number, self._args.batch_size, self._args.epochs,
                                                              self._args.learning_rate, self._args.kernel_size, self._args.channels, self._args.pooling_size, self._args.pooling_indices,
                                                              self._args.activation, self._args.dropout, self._args.regularization_coefficient, self._args.mlp_units)
            classifier.build()
            history: keras.callbacks.History = classifier.train(train_data, train_labels, validation_data, validation_labels, f'val_{self._accuracy_metric}')
            self._plot_history(history, self._accuracy_metric)
            classifier.save(self._args.save_model_to)
            model_directory: str = os.path.dirname(os.path.realpath(f'{self._args.save_model_to}.keras'))
            self._logger.info(f'Saving evaluation tensors to {model_directory}')
            np.savez_compressed(os.path.join(model_directory, 'train_features.npz'), train_data)
            np.savez_compressed(os.path.join(model_directory, 'train_labels.npz'), train_labels)
            np.savez_compressed(os.path.join(model_directory, 'validation_features.npz'), validation_data)
            np.savez_compressed(os.path.join(model_directory, 'validation_labels.npz'), validation_labels)
            np.savez_compressed(os.path.join(model_directory, 'test_features.npz'), test_data)
            np.savez_compressed(os.path.join(model_directory, 'test_labels.npz'), test_labels)

    def evaluate_trained_model(self) -> None:
        """
        Performs the evaluation of a provided pre-trained model
        :return: None
        """
        # Instantiate the classifier
        classifier: EmotionClassifier = EmotionClassifier(self._logger, (0, 0, 0, 0), self._args.model_summary, 13, 0, 0, 0, 0, [],
                                                          0, [], '', 0, 0, [])
        model_directory: str = os.path.dirname(os.path.realpath(f'{self._args.load_model_from}.keras'))
        classifier.load(self._args.load_model_from)
        self._logger.info(f'Loading train, validation and test sets from {model_directory}')
        train_data: np.ndarray = np.load(os.path.join(model_directory, 'train_features.npz'))['arr_0']
        train_labels: np.ndarray = np.load(os.path.join(model_directory, 'train_labels.npz'))['arr_0']
        validation_data: np.ndarray = np.load(os.path.join(model_directory, 'validation_features.npz'))['arr_0']
        validation_labels: np.ndarray = np.load(os.path.join(model_directory, 'validation_labels.npz'))['arr_0']
        test_data: np.ndarray = np.load(os.path.join(model_directory, 'test_features.npz'))['arr_0']
        test_labels: np.ndarray = np.load(os.path.join(model_directory, 'test_labels.npz'))['arr_0']
        self._logger.info(f'Evaluating the model on the training set')
        train_loss, train_accuracy = classifier.evaluate(train_data, train_labels)
        self._logger.info(f'Finished the evaluation process. Training Loss = {train_loss}. Training Accuracy = {train_accuracy}')
        self._logger.info(f'Evaluating the model on the validation set')
        validation_loss, validation_accuracy = classifier.evaluate(validation_data, validation_labels)
        self._logger.info(f'Finished the evaluation process. Validation Loss = {validation_loss}. Validation Accuracy = {validation_accuracy}')
        self._logger.info(f'Evaluating the model on the test set')
        test_loss, test_accuracy = classifier.evaluate(test_data, test_labels)
        self._logger.info(f'Finished the evaluation process. Test Loss = {test_loss}. Test Accuracy = {test_accuracy}')
        self._logger.info('Removing training, validation and test sets')
        os.remove(os.path.join(model_directory, 'train_features.npz'))
        os.remove(os.path.join(model_directory, 'train_labels.npz'))
        os.remove(os.path.join(model_directory, 'validation_features.npz'))
        os.remove(os.path.join(model_directory, 'validation_labels.npz'))
        os.remove(os.path.join(model_directory, 'test_features.npz'))
        os.remove(os.path.join(model_directory, 'test_labels.npz'))

    def tune(self) -> None:
        """
        Performs the hyper-parameters tuning process
        :return: None
        """
        iteration: int = 0
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = self._load_and_split_data()
        self._logger.info('Starting the tuning process')
        architectures = [([10, 20, 30], [250], [0, 1], 3, 5),
                         ([20, 13, 5], [50, 7], [1, 2], 4, 5),
                         ([20, 40, 30, 10], [400, 100, 13], [0, 1], 3, 4),
                         ([50, 30, 20, 10], [300, 100, 20], [0, 1, 3], 3, 3),
                         ([80, 60, 40, 20], [800, 80], [0, 1, 3], 3, 3)]
        for channels, mlp_units, pooling_indices, pooling_size, kernel_size in architectures:
            for dropout in map(lambda number: number / 100, range(10, 46, 5)):
                for regularization_coefficient in [0, 0.000001, 0.00001]:
                    for batch_size in map(lambda index: 16 * index, range(1, 7)):
                        for learning_rate in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
                            iteration += 1
                            self._logger.info(f'Tuning iteration {iteration}')
                            if iteration < 1:
                                continue
                            self._logger.info(f'{batch_size=}')
                            self._logger.info(f'{learning_rate=}')
                            self._logger.info(f'{dropout=}')
                            self._logger.info(f'{regularization_coefficient=}')
                            self._logger.info(f'{kernel_size=}')
                            self._logger.info(f'{channels=}')
                            self._logger.info(f'{pooling_size=}')
                            self._logger.info(f'{pooling_indices=}')
                            self._logger.info(f'{mlp_units=}')
                            # Instantiate the classifier
                            try:
                                classifier: EmotionClassifier = EmotionClassifier(self._logger, self._input_shape, self._args.model_summary, self._classes_number, batch_size, self._args.epochs,
                                                                                  learning_rate, kernel_size, channels, pooling_size, pooling_indices, self._args.activation, dropout,
                                                                                  regularization_coefficient, mlp_units)
                                classifier.build()
                                history: keras.callbacks.History = classifier.train(train_data, train_labels, validation_data, validation_labels, f'val_{self._accuracy_metric}')
                            except Exception as error:
                                self._logger.warning(41, f'Could not perform training with the provided configuration due to the following error: {error}')
                                continue
                            train_loss, train_accuracy = classifier.evaluate(train_data, train_labels)
                            validation_loss, validation_accuracy = classifier.evaluate(validation_data, validation_labels)
                            # test_loss, test_accuracy = classifier.evaluate(test_data, test_labels)
                            file_name: str = f'val_acc {validation_accuracy:.2f}, tr_acc {train_accuracy:.2f}, val_loss {validation_loss:.4f}, tr_loss {train_loss:.4f}, '\
                                             f'diff {abs(train_accuracy - validation_accuracy):.2f}, bs {batch_size}, lr {learning_rate}, drop {dropout}, reg {regularization_coefficient}, '\
                                             f'ker {kernel_size}, chan {channels}, mlp {mlp_units}, pool {pooling_size}, pool_i {pooling_indices}'\
                                             '.png'
                            graph_path: str = os.path.join('plots', self._args.dataset, file_name)
                            if os.path.exists(graph_path):
                                os.remove(graph_path)
                                time.sleep(2)
                            self._plot_history(history, self._accuracy_metric, graph_path)

    def cross_validation(self):
        """
        Performs the K-Fold Cross Validation
        :return: None
        """
        train_data, train_labels, _, _, test_data, test_labels = self._load_and_split_data(create_validation_set=False)
        folds_number: int = 5
        self._logger.info(f'Starting K-Fold CV with K={folds_number}')
        iteration: int = 0
        architectures = [([10, 20, 30], [250], [0, 1], 3, 5, [0.25, 0.3, 0.35]),
                         ([20, 40, 30, 10], [400, 100, 13], [0, 1], 3, 4, [0.1, 0.15, 0.2]),
                         ([80, 60, 40, 20], [800, 80], [0, 1, 3], 3, 3, [0.15, 0.2, 0.25])]
        for channels, mlp_units, pooling_indices, pooling_size, kernel_size, dropout_options in architectures:
            for dropout in dropout_options:
                for regularization_coefficient in (0.000001,):
                    for batch_size in (16, 32):
                        for learning_rate in (0.001, 0.005):
                            iteration += 1
                            self._logger.info(f'{folds_number}-Fold CV iteration {iteration}')
                            if iteration < 1:
                                continue
                            average_training_accuracy, average_training_loss, average_validation_accuracy, average_validation_loss = 0, 0, 0, 0
                            for train_indices, validation_indices in KFold(n_splits=folds_number, shuffle=True, random_state=13).split(train_data):
                                training_features, validation_features = train_data[train_indices], train_data[validation_indices]
                                training_labels, validation_labels = train_labels[train_indices], train_labels[validation_indices]
                                try:
                                    classifier: EmotionClassifier = EmotionClassifier(self._logger, self._input_shape, self._args.model_summary, self._classes_number, batch_size, self._args.epochs,
                                                                                      learning_rate, kernel_size, channels, pooling_size, pooling_indices, self._args.activation, dropout,
                                                                                      regularization_coefficient, mlp_units)
                                    classifier.build()
                                    classifier.train(training_features, training_labels, validation_features, validation_labels, f'val_{self._accuracy_metric}')
                                except Exception as error:
                                    self._logger.debug(str(error))
                                    continue
                                train_loss, train_accuracy = classifier.evaluate(training_features, training_labels)
                                validation_loss, validation_accuracy = classifier.evaluate(validation_features, validation_labels)
                                average_training_accuracy += train_accuracy
                                average_training_loss += train_loss
                                average_validation_accuracy += validation_accuracy
                                average_validation_loss += validation_loss
                            average_training_accuracy /= folds_number
                            average_training_loss /= folds_number
                            average_validation_accuracy /= folds_number
                            average_validation_loss /= folds_number
                            results: str = f'{average_validation_accuracy=}; {average_training_accuracy=}; {average_validation_loss=}; {average_training_loss=}'
                            params: str = f'{channels=}; {mlp_units=}; {pooling_indices=}; {pooling_size=}; {kernel_size=}; {dropout=}; {regularization_coefficient=}; {batch_size=}; {learning_rate=}'
                            with open(os.path.join('plots', self._args.dataset, f'{results}.txt'), 'w') as results_file:
                                results_file.write(params)
