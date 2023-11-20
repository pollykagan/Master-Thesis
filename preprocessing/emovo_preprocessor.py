"""
This module implements the preprocessing pipeline for EMOVO dataset
"""
import os
import configargparse
from utilities.logger import Logger
from .preprocessor import Preprocessor


class EmovoPreprocessor(Preprocessor):
    """
    This class implements the Preprocessor interface for EMOVO dataset
    """
    _dataset_name = 'emovo'
    _emotion_codes_mapping: dict = {'dis': 'disgusted', 'pau': 'fearful', 'rab': 'angry', 'gio': 'happy', 'sor': 'surprised', 'tri': 'sad', 'neu': 'neutral'}

    def __init__(self, emotion: str, args: configargparse.Namespace, logger: Logger):
        """
        A constructor
        :param emotion: Emotion for which the related samples should be preprocessed
        :param args: An object containing parsed command line (and ini) arguments
        :param logger: The logger
        """
        # The labels of EMOVO dataset are part of the audio files names. Each file name is constructed from 3 short strings, separated by '-'.
        # The emotion is encoded as the first component.
        # For more details, please see: https://www.kaggle.com/datasets/sourabhy/emovo-italian-ser-dataset
        super().__init__(emotion, args, logger)

    @classmethod
    def _extract_emotion_from_full_path(cls, file_full_path: str) -> str:
        """
        Given a full path to an audio file from one of the datasets, returns the emotion it represents
        :param file_full_path: A full path to an audio file for which the emotion should be extracted
        :return: A string representing emotion
        """
        return cls._emotion_codes_mapping[os.path.basename(file_full_path).split('-')[0]]

    @staticmethod
    def _get_emotion_integer_label(emotion: str) -> int:
        """
        Returns an integer that should serve as a label for the provided emotion in the classifier
        :param emotion: A string representing an emotion
        :return: An integer that will serve as a label for a classifier
        """
        ordered_emotions = ('angry', 'fearful', 'disgusted', 'happy', 'neutral', 'sad', 'surprised')
        return ordered_emotions.index(emotion)
