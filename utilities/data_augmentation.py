"""
This module implements audio data augmentation
"""
import os
import json
import soundfile
import numpy as np
from audiomentations import Compose
from audiomentations import AddGaussianNoise
from audiomentations import PitchShift
from audiomentations import TimeStretch
from audiomentations import GainTransition
from audiomentations import RoomSimulator
from audiomentations.core.transforms_interface import BaseWaveformTransform
from .logger import Logger
from .audio_features import AudioFeaturesReturnType
from .audio_features import AudioFeaturesExtractor


class AugmentorConfiguration:
    """
    This class provides a robust way to define the data augmentation parameters
    """
    def __init__(self, logger: Logger, arguments_json: str, use_gaussian_noise: bool, use_pitch_shift: bool, use_time_stretch: bool, use_gain_transition: bool, use_room_simulator: bool):
        """
        A constructor
        :param logger: A logger to use to write the class messages
        :param arguments_json: The name of the JSON file which contains parameters for the audio transformations
        :param use_gaussian_noise: If True, AddGaussianNoise transformation will be applied
        :param use_pitch_shift: If True, PitchShift transformation will be applied
        :param use_time_stretch: If True, TimeStretch transformation will be applied
        :param use_gain_transition: If True, GainTransition transformation will be applied
        :param use_room_simulator: If True, RoomSimulator transformation will be applied
        """
        self._logger: Logger = logger
        self._logger.debug(f'Initializing {self.__class__.__name__}')
        self._json: str = os.path.join('config', arguments_json)
        self._use_gaussian_noise: bool = use_gaussian_noise
        self._use_pitch_shift: bool = use_pitch_shift
        self._use_time_stretch: bool = use_time_stretch
        self._use_gain_transition: bool = use_gain_transition
        self._use_room_simulator: bool = use_room_simulator

    def to_dictionary(self) -> dict:
        """
        Returns the representation of the object (AugmentorConfiguration instance) as a dictionary,
        which will allow more comfortable work for the Augmentor class.
        The dictionary will have the following format:
        {'<transformation name>':
            {'apply': <True/False> (whether to apply the transformation),
             'class': <audiomentations class implementing the transformation>,
             'arguments': <dictionary of keyword arguments to the transformation class constructor>}}
        :return: Described above
        """
        result: dict = {'gaussian_noise': {'apply': self._use_gaussian_noise, 'class': AddGaussianNoise},
                        'pitch_shift': {'apply': self._use_pitch_shift, 'class': PitchShift},
                        'time_stretch': {'apply': self._use_time_stretch, 'class': TimeStretch},
                        'gain_transition': {'apply': self._use_gain_transition, 'class': GainTransition},
                        'room_simulator': {'apply': self._use_room_simulator, 'class': RoomSimulator}}
        with open(self._json, 'r') as json_file_object:
            arguments_dictionary: dict = json.load(json_file_object)
        for transformation in result.keys():
            result[transformation]['arguments'] = arguments_dictionary[transformation]
        return result


class Augmentor:
    """
    This class performs audio augmentation, using the requested techniques
    """
    def __init__(self, augmentation_factor: int, configuration: AugmentorConfiguration, logger: Logger):
        """
        A constructor
        :param augmentation_factor: The number of audio files to generate, as a multiplication of original data files number
        :param configuration: An object providing the configuration for the augmentor transformations (instance of AugmentorConfiguration)
        :param logger: A logger to use to write the class messages
        """
        self._logger: Logger = logger
        self._logger.debug(f'Initializing {self.__class__.__name__}')
        if augmentation_factor <= 0:
            self._logger.fatal(1, f'Non-positive augmentation factor: {augmentation_factor}', ValueError)
        self._augmentation_factor: int = augmentation_factor
        configuration_dictionary: dict = configuration.to_dictionary()
        self._logger.debug(f'Augmentation configuration: {configuration_dictionary}')
        # The following list will contain all the transformations that should be composed and applied
        self.transformations: list[BaseWaveformTransform] = []
        for transformation, transformation_setting in configuration_dictionary.items():
            if transformation_setting['apply']:
                transformation_probability: float = transformation_setting['arguments']['p']
                self._logger.info(f'The transformation "{transformation}" will be applied by Augmentor with probability {transformation_probability}')
                # If current transformation should be applied - create the instance of the related class (using the related arguments' dictionary)
                current_transformation: BaseWaveformTransform = transformation_setting['class'](**transformation_setting['arguments'])
                # Add the created transformation object to the transformations list
                self.transformations.append(current_transformation)
            else:
                self._logger.info(f'The transformation "{transformation}" won\'t be applied by Augmentor')
        self._progress_percents: int = -1

    def _report_progress(self, iteration: int, total_number_of_files: int, augmentation_cycle: int, total_cycles: int):
        """
        Reports the progress of the data augmentation process (reports for each 5% of progress)
        :param iteration: The index of the current iteration (represents the number of the files for which the augmentations were already performed)
        :param total_number_of_files: The number of audio files used for the whole augmentation process
        :param augmentation_cycle: The index of the augmentation cycle
        :param total_cycles: Total number of augmentation cycles to perform (equals to augmentation factor)
        :return:
        """
        current_progress_percents: int = int((iteration / total_number_of_files) * 100)
        if int(current_progress_percents / 5) > int(self._progress_percents / 5):
            self._logger.info(f'Augmentation progress:   Cycle {augmentation_cycle}/{total_cycles};   Augmentations: {iteration}/{total_number_of_files}   ({current_progress_percents}%)')
            self._progress_percents: int = current_progress_percents

    def generate_single_audio(self, signal: np.ndarray, sample_rate: int, file_name: str) -> np.ndarray:
        """
        Composes and applies the requested transformations
        :param signal: Waveform to apply the transformations on
        :param sample_rate: The sample rate of the transformations
        :param file_name: The name of the audio file to which the transformations are applied
        :return: New waveforms that were generated from the original ones using the requested transformations
        """
        transformation_names: list[str] = list(map(lambda transformation_object: transformation_object.__class__.__name__, self.transformations))
        self._logger.debug(f'Applying the selected transformations: {transformation_names} to {file_name}')
        transformations_compose_object: Compose = Compose(self.transformations)
        return transformations_compose_object(signal, sample_rate)

    def perform_augmentation(self, audio_files_paths: list[str], dataset_name: str, emotion:str, sample_rate: int, duration: [int, None] = None) -> None:
        """
        Performs the data augmentation, based on the received (original) audio files, and writes the augmented files to the 'data/augmented' directory
        :param audio_files_paths: List of paths of the original audio files
        :param dataset_name: The name of the dataset for which the augmentation is performed
        :param emotion: The emotion for whose classification the augmentation is performed
        :param sample_rate: The sample rate to use when sampling the original data files
        :param duration: The sampling duration for the original data files
        :return: None
        """
        self._logger.info('Performing audio data augmentation')
        self._logger.info(f'Starting extracting original waveforms')
        features_extractor: AudioFeaturesExtractor = AudioFeaturesExtractor(audio_files_paths, self._logger, AudioFeaturesReturnType.DICTIONARY, sample_rate, duration)
        audio_signals_dictionary: dict = features_extractor.waveform()
        number_of_original_files: int = len(audio_signals_dictionary)
        self._logger.info(f'Finished extracting {number_of_original_files} original waveforms')
        self._logger.info(f'For each file {self._augmentation_factor} augmentations will be generated (augmentation cycles)')
        for augmentation_cycle in range(1, self._augmentation_factor + 1):
            self._progress_percents: int = -1
            self._logger.info(f'Augmentation cycle: {augmentation_cycle}')
            for progress_index, (audio_file_path, signal) in enumerate(audio_signals_dictionary.items()):
                self._report_progress(progress_index, number_of_original_files, augmentation_cycle, self._augmentation_factor)
                new_signal: np.ndarray = self.generate_single_audio(signal, sample_rate, audio_file_path)
                # The augmented file name is constructed from the basename of the original file (without '.wav' extension ([:-4])),
                # the augmentation cycle index (starting from 1) and the '.wav' extension
                augmented_file_name: str = f'{os.path.basename(audio_file_path)[:-4]}_{augmentation_cycle}.wav'
                augmented_file_path: str = os.path.join('data', emotion, 'augmented', dataset_name, augmented_file_name)
                if os.path.exists(augmented_file_path):
                    self._logger.warning(2, f'The file {augmented_file_path} generated by Augmentor exists - replacing it')
                self._logger.debug(f'Saving the file {augmented_file_path}')
                soundfile.write(augmented_file_path, new_signal, sample_rate)
            self._report_progress(number_of_original_files, number_of_original_files, augmentation_cycle, self._augmentation_factor)
        self._logger.info('Finished audio data augmentation process')
