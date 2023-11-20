"""
This module provides command line and INI arguments parsing
"""
import configargparse
from configargparse import ArgParser


class ArgumentsParser:
    """
    This class serves for the arguments parsing.
    The arguments can come from both the command line and ".ini" files
    """
    def __init__(self):
        """
        A constructor
        """
        self._parser: ArgParser = ArgParser(ignore_unknown_config_file_keys=False)
        self._define_arguments()

    def _define_arguments(self) -> None:
        """
        Private method.
        Defines the project arguments, and adds them to the parser
        :return: None
        """
        general_group = self._parser.add_argument_group('Arguments parser arguments')
        general_group.add_argument('-c', '--config_file', is_config_file=True, help='A path to a configuration ".ini" file')

        logger_group = self._parser.add_argument_group('Logger arguments')
        logger_group.add_argument('--verbosity', help='The verbosity of the logger')
        logger_group.add_argument('--log_to_console', action='store_true', help='If set, the logger will write messages to a console (stdout)')
        logger_group.add_argument('--log_to_file', action='store_true', help='If set, the logger will write messages to files')
        logger_group.add_argument('--logger_name', help='The name of the logger')

        augmentation_group = self._parser.add_argument_group('Data augmentation arguments')
        augmentation_group.add_argument('--augment_data', action='store_true', help='If set, the data augmentation will be performed')
        augmentation_group.add_argument('--data_augmentation_json', help='A path to the JSON file describing the arguments of the data augmentation transformations')
        augmentation_group.add_argument('--apply_gaussian_noise', action='store_true', help='If set, the gaussian noise transformation will be applied in the data augmentation process')
        augmentation_group.add_argument('--apply_pitch_shift', action='store_true', help='If set, the pitch shift transformation will be applied in the data augmentation process')
        augmentation_group.add_argument('--apply_time_stretch', action='store_true', help='If set, the time stretch transformation will be applied in the data augmentation process')
        augmentation_group.add_argument('--apply_gain_transition', action='store_true', help='If set, the gain transition transformation will be applied in the data augmentation process')
        augmentation_group.add_argument('--apply_room_simulator', action='store_true',  help='If set, the room simulator transformation will be applied in the data augmentation process')
        augmentation_group.add_argument('--augmentation_factor', type=int, help='The number of audio files to generate, as a multiplication of original data files number')

        audio_features_group = self._parser.add_argument_group('Audio features extractor arguments')
        audio_features_group.add_argument('--sample_rate', type=int, help='The sample rate using which the audio files will be sampled')
        audio_features_group.add_argument('--duration', type=float, help='The duration for which the audio files will be sampled')
        audio_features_group.add_argument('--pad', action='store_true', help='If set, files shorter than the required duration (see -d) will be padded with zeros')
        audio_features_group.add_argument('--frame_size', type=int, help='The frame size for spectrogram-like features')
        audio_features_group.add_argument('--hop_length', type=int, help='The hop length for spectrogram-like features')
        audio_features_group.add_argument('--mel_bands', type=int, help='The number of mel bands (relevant for mel-spectrograms only)')
        audio_features_group.add_argument('--mfcc_number', type=int, help='The number of MFCCs (relevant for MFCC feature extraction only)')

        preprocessing_group = self._parser.add_argument_group('Preprocessor arguments')
        preprocessing_group.add_argument('--preprocess_data', action='store_true',
                                         help='If set, the data will be preprocessed. It will be stored at data/preprocessed/<dataset>/<feature type>')
        preprocessing_group.add_argument('--preprocessing_feature', choices=['waveform', 'spectrogram', 'log_spectrogram', 'mel_spectrogram', 'log_mel_spectrogram', 'mfcc'],
                                         help='A feature type to use for preprocessing')

        dataset_arguments = self._parser.add_argument_group('Dataset related arguments')
        dataset_arguments.add_argument('--dataset', choices=['crema_d', 'emo_db', 'emovo', 'ravdess', 'shemo', 'all'],
                                       help='The name of the dataset on which the preprocessing/training should be applied')
        dataset_arguments.add_argument('--emotion', choices=('angry', 'fearful', 'disgusted', 'happy', 'neutral', 'sad', 'all'),
                                       help='The emotion on which the model should operate (will get the label 1)')
        dataset_arguments.add_argument('--base_emotion', choices=('angry', 'fearful', 'disgusted', 'happy', 'neutral', 'sad'), default='neutral',
                                       help='The base emotion for the model (will get the label 0)')

        model_group = self._parser.add_argument_group('Model arguments')
        model_group.add_argument('--model_summary', action='store_true', help='If set, prints the model summary')
        model_group.add_argument('--epochs', type=int, help='Number of epochs of the training process')
        model_group.add_argument('--batch_size', type=int, help='The size of the batch')
        model_group.add_argument('--learning_rate', type=float, help='The learning rate of the optimizer')
        model_group.add_argument('--activation', help='The name of the activation function to use for the model layers')
        model_group.add_argument('--dropout', type=float, help='The probability of the dropout (0 means no dropout)')
        model_group.add_argument('--regularization_coefficient', type=float, help='The L1/L2 regularization coefficient')

        convolution_group = self._parser.add_argument_group('CNN models arguments')
        convolution_group.add_argument('--kernel_size', type=int, help='The size of the (square) kernel to use for CNN models')
        convolution_group.add_argument('--channels', nargs='*', help='A list representing amount of kernels (channels) at each layer for CNN models')
        convolution_group.add_argument('--pooling_size', type=int, help='The size of the pooling (square) kernel and stride')
        convolution_group.add_argument('--pooling_indices', nargs='*', help='CNN layer indices on which the pooling should be applied')
        convolution_group.add_argument('--mlp_units', nargs='*', help='A list representing amount of units of the MLP part of CNN models')

        pipeline_group = self._parser.add_argument_group('General task pipeline arguments')
        pipeline_group.add_argument('--mode', choices=['training', 'evaluation', 'tuning', 'cross_validation'], help='The model pipeline mode')
        pipeline_group.add_argument('--save_model_to', help='A path to save the trained model. Relevant only in "training" mode')
        pipeline_group.add_argument('--load_model_from', help='A path to a pre-trained model for the evaluation. Relevant only in "evaluation" mode')


    def parse_arguments(self) -> configargparse.Namespace:
        """
        Parses the arguments (command line and ".ini"), using the "parse_args" method of ArgParser class
        :return: A namespace containing all the parsed arguments
        """
        result: configargparse.Namespace = self._parser.parse_args()
        # The list arguments (e.g., channels) are parsed as lists of strings - need to convert the list items to the desired type manually
        if result.channels:
            result.channels = list(map(int, result.channels))
        if result.pooling_indices:
            result.pooling_indices = list(map(int, result.pooling_indices))
        if result.mlp_units:
            result.mlp_units = list(map(int, result.mlp_units))
        return result
