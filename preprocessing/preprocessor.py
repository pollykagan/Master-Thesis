"""
This module implements the general preprocessing pipeline for audio data
"""
import os
import abc
import shutil
import configargparse
import functools
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utilities.general import get_files_full_paths_list
from utilities.logger import Logger
from utilities.audio_features import AudioFeaturesReturnType
from utilities.audio_features import AudioFeaturesExtractor
from utilities.data_augmentation import AugmentorConfiguration
from utilities.data_augmentation import Augmentor


class Preprocessor(abc.ABC):
    """
    This class is an abstract class defining the interface (and most of the behavior) of all the data preprocessing classes
    """
    _dataset_name: [str, None] = None
    supported_emotions: tuple[str, str, str, str, str, str] = ('angry', 'fearful', 'disgusted', 'happy', 'neutral', 'sad')

    def __init__(self, emotion: str, args: configargparse.Namespace, logger: Logger):
        """
        A constructor
        :param emotion: Emotion for which the related samples should be preprocessed
        :param args: An object containing parsed command line (and ini) arguments
        :param logger: The logger
        """
        logger.info(f'Initializing {self.__class__.__name__} for the dataset {self._dataset_name} with emotion {emotion}')
        # Store the received arguments
        if emotion is None:
            logger.fatal(21, 'Preprocessor received emotion=None. Please provide a valid emotion from the list with --emotion flag', ValueError)
        self._emotion: str = emotion
        self._args: configargparse.Namespace = args
        self._logger: Logger = logger
        self._base_emotion: str = self._args.base_emotion
        # The following attributes store different useful paths
        self._project_root: str = os.path.dirname(os.path.dirname(__file__))
        self._dataset_path: str = os.path.join(self._project_root, 'data', 'datasets', self._dataset_name)
        self._original_data_path: str = os.path.join(self._project_root, 'data', self._emotion, 'original', self._dataset_name)
        self._augmented_data_path: str = os.path.join(self._project_root, 'data', self._emotion, 'augmented', self._dataset_name)
        self._preprocessed_data_path: str = os.path.join(self._project_root, 'data', self._emotion, 'preprocessed', self._dataset_name, self._args.preprocessing_feature)
        # If the dataset path doesn't exist, it's a critical error (the program can't continue without a data)
        if not os.path.exists(self._dataset_path):
            self._logger.fatal(22, f'The dataset path ({self._dataset_path}) does not exist', FileNotFoundError)
        # Create directories (if still don't exist) to store the samples of the provided dataset with the provided emotion
        os.makedirs(self._original_data_path, exist_ok=True)
        os.makedirs(self._augmented_data_path, exist_ok=True)
        os.makedirs(self._preprocessed_data_path, exist_ok=True)
        # Take only the files that represent the requested emotion
        self._filter_by_emotion()
        # If data augmentation is needed, the following method performs it
        if self._args.augment_data:
            self._generate_augmented_data()
        # Build the feature extractor object for future use
        all_audio_paths: list[str] = get_files_full_paths_list(self._original_data_path) + get_files_full_paths_list(self._augmented_data_path)
        self._features_extractor: AudioFeaturesExtractor = AudioFeaturesExtractor(all_audio_paths, self._logger, AudioFeaturesReturnType.NUMPY_ARRAY, self._args.sample_rate, self._args.duration,
                                                                                  self._args.pad)
        # In the future, when we'll extract the audio features, we'd like to do it by a single function call (without multiple if-statements, defining for each feature what to do).
        # So here, we create a dictionary of callables with a single interface (receive nothing as arguments, return a numpy tensor containing the features) that we will use.
        # To do that, we provide each feature extraction method all the arguments it needs to give us exactly what we need (almost all the arguments come from flags)
        self._features_methods: dict = {'waveform': functools.partial(self._features_extractor.waveform),
                                        'spectrogram': functools.partial(self._features_extractor.spectrogram, self._args.frame_size, self._args.hop_length),
                                        'log_spectrogram': functools.partial(self._features_extractor.spectrogram, self._args.frame_size, self._args.hop_length, True),
                                        'mel_spectrogram': functools.partial(self._features_extractor.mel_spectrogram, self._args.frame_size, self._args.hop_length, self._args.mel_bands),
                                        'log_mel_spectrogram': functools.partial(self._features_extractor.mel_spectrogram, self._args.frame_size, self._args.hop_length, self._args.mel_bands, True),
                                        'mfcc': functools.partial(self._features_extractor.mfcc, self._args.frame_size, self._args.hop_length, self._args.mfcc_number)}

    def _generate_augmented_data(self) -> None:
        """
        Performs data augmentation and stores the augmented files under "self._augmented_dataset_path" directory
        :return: None
        """
        self._logger.info(f'{self.__class__.__name__} performs data augmentation')
        configuration: AugmentorConfiguration = AugmentorConfiguration(self._logger, self._args.data_augmentation_json, self._args.apply_gaussian_noise, self._args.apply_pitch_shift,
                                                                       self._args.apply_time_stretch, self._args.apply_gain_transition, self._args.apply_room_simulator)
        augmentor: Augmentor = Augmentor(self._args.augmentation_factor, configuration, self._logger)
        augmentor.perform_augmentation(get_files_full_paths_list(self._original_data_path), self._dataset_name, self._emotion, self._args.sample_rate, duration=self._args.duration)

    @classmethod
    @abc.abstractmethod
    def _extract_emotion_from_full_path(cls, file_full_path: str) -> str:
        """
        Abstract method. Given a full path to an audio file from one of the datasets, returns the emotion it represents.
        Each implementing preprocessor should define a way to extract the emotions from its dataset files
        :param file_full_path: A full path to an audio file for which the emotion should be extracted
        :return: A string representing emotion
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _get_emotion_integer_label(emotion: str) -> int:
        """
        Returns an integer that should serve as a label for the provided emotion in the classifier
        :param emotion: A string representing an emotion
        :return: An integer that will serve as a label for a classifier
        """
        pass

    def _filter_by_emotion(self) -> None:
        """
        Filters the files related to the requested emotion from the "_dataset_path" to the "_original_data_path"
        :return: None
        """
        for path in get_files_full_paths_list(self._dataset_path):
            # The following line determines whether the audio track (sample) represented by "path" will be taken to the dataset.
            # The sample will be taken in 2 cases:
            # 1. Multi-Label Classification on all the emotions
            # 2. Binary Classification case, and the sample belongs to the emotion provided with "--emotion" or with "--base_emotion"
            take_sample: bool = self._emotion == 'all' or self._extract_emotion_from_full_path(path) in [self._base_emotion, self._emotion]
            if take_sample:
                shutil.copyfile(path, os.path.join(self._original_data_path, os.path.basename(path)))

    def _extract_labels(self) -> np.ndarray:
        """
        Extracts the labels of the dataset and returns them as a numpy array
        :return: A 1D numpy array containing the dataset labels
        """
        all_audio_paths: list[str] = get_files_full_paths_list(self._original_data_path) + get_files_full_paths_list(self._augmented_data_path)
        if self._emotion == 'all':
            # Multi-Label Classification case.
            # For each audio file in the original and the augmented datasets, sets the label as the index of the corresponding emotion, defined by a relevant preprocessor
            return np.array(list(map(lambda full_path: self._get_emotion_integer_label(self._extract_emotion_from_full_path(full_path)), all_audio_paths)), dtype=np.int8)
        else:
            # Binary Classification case.
            # For each audio file in the original and the augmented datasets, extracts the component of the file name and converts it to the emotion label, using the dictionary above
            return np.array(list(map(lambda full_path: int(self._extract_emotion_from_full_path(full_path) == self._emotion), all_audio_paths)), dtype=np.int8)

    def __call__(self) -> None:
        """
        This method is invoked when the "Preprocessor" objects are executed as callables.
        It extracts the features and the labels from the dataset, and stores them in ".npz" files
        :return: None
        """
        # Extract the requested features (received by command line/ini arguments)
        self._logger.info(f'{self.__class__.__name__} performs "{self._args.preprocessing_feature}" feature extraction for both original and augmented data')
        features: np.ndarray = self._features_methods[self._args.preprocessing_feature]()
        # Extract the labels
        self._logger.info(f'{self.__class__.__name__} extracts the labels of both original and augmented data')
        labels: np.ndarray = self._extract_labels()
        # Store the features and the labels in separate files
        self._logger.info(f'{self.__class__.__name__} stores the features and the labels')
        features_path: str = os.path.join(self._preprocessed_data_path, 'features.npz')
        labels_path: str = os.path.join(self._preprocessed_data_path, 'labels.npz')
        np.savez_compressed(features_path, features)
        np.savez_compressed(labels_path, labels)
        self._logger.info(f'{self.__class__.__name__} finished preprocessing')
        self._logger.info(f'Original audio files were taken from: {self._original_data_path}')
        self._logger.info(f'Augmented audio files were taken/generated at: {self._augmented_data_path}')
        self._logger.info(f'The features and the labels are saved to: {self._preprocessed_data_path}')

    @classmethod
    def plot_data_histograms(cls) -> None:
        """
        Plots a histogram of the audio tracks in all the datasets, by duration and labels
        :return: None
        """
        sample_rate = 16000
        dataset_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'datasets', cls._dataset_name)
        durations: list[float] = []
        labels: list[int] = []
        for audio_path in get_files_full_paths_list(dataset_path):
            signal: np.ndarray = librosa.load(audio_path, sr=sample_rate)[0]
            durations.append(signal.shape[0] / sample_rate)
            labels.append(cls._get_emotion_integer_label(cls._extract_emotion_from_full_path(audio_path)))
        plt.hist(durations, color='blue', edgecolor='red')
        plt.title(f'Durations of {cls._dataset_name}')
        plt.show()
        plt.hist(labels, color='green', edgecolor='red')
        plt.title(f'Labels distribution of {cls._dataset_name}')
        plt.show()
