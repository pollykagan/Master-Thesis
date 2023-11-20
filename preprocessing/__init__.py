"""
Preprocessing package
"""
from typing import Type
from .crema_d_preprocessor import CremaDPreprocessor
from .emo_db_preprocessor import EmoDbPreprocessor
from .emovo_preprocessor import EmovoPreprocessor
from .ravdess_preprocessor import RavdessPreprocessor
from .shemo_preprocessor import ShemoPreprocessor
from .preprocessor import Preprocessor


def get_preprocessor_class(dataset: str) -> Type[Preprocessor]:
    """
    Returns the preprocessor relevant to the provided dataset
    :param dataset: A string representing the dataset for which the preprocessor should be returned
    :return: A preprocessor object related to the provided dataset
    """
    if dataset == 'crema_d':
        return CremaDPreprocessor
    if dataset == 'emo_db':
        return EmoDbPreprocessor
    if dataset == 'emovo':
        return EmovoPreprocessor
    if dataset == 'ravdess':
        return RavdessPreprocessor
    if dataset == 'shemo':
        return ShemoPreprocessor
