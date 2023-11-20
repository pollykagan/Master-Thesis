"""
This module implements the preprocessing pipeline for RAVDESS dataset
"""
import os
import configargparse
from utilities.logger import Logger
from .preprocessor import Preprocessor


class RavdessPreprocessor(Preprocessor):
    """
    This class implements the Preprocessor interface for RAVDESS dataset
    """
    _dataset_name = 'ravdess'
    _emotion_codes_mapping: dict = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised'}

    def __init__(self, emotion: str, args: configargparse.Namespace, logger: Logger):
        """
        A constructor
        :param emotion: Emotion for which the related samples should be preprocessed
        :param args: An object containing parsed command line (and ini) arguments
        :param logger: The logger
        """
        # The labels of RAVDESS dataset are part of the audio files names. Each file name is constructed from 7 two-digits integers, separated by '-'.
        # The emotion is encoded as the third component.
        # For more details, please see: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
        super().__init__(emotion, args, logger)

    @classmethod
    def _extract_emotion_from_full_path(cls, file_full_path: str) -> str:
        """
        Given a full path to an audio file from one of the datasets, returns the emotion it represents
        :param file_full_path: A full path to an audio file for which the emotion should be extracted
        :return: A string representing emotion
        """
        return cls._emotion_codes_mapping[os.path.basename(file_full_path).split('-')[2]]

    @staticmethod
    def _get_emotion_integer_label(emotion: str) -> int:
        """
        Returns an integer that should serve as a label for the provided emotion in the classifier
        :param emotion: A string representing an emotion
        :return: An integer that will serve as a label for a classifier
        """
        ordered_emotions = ('angry', 'fearful', 'disgusted', 'happy', 'neutral', 'sad', 'surprised', 'calm')
        return ordered_emotions.index(emotion)
