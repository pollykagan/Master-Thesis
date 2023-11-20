"""
This module implements extraction of audio features
"""
import os
import enum
import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
from .logger import Logger


class AudioFeaturesReturnType(enum.Enum):
    """
    This class defines the enum values for the return type of multiple sound files representation (e.g., multiple waveforms or spectrograms).
    Supports list (of waveforms/spectrograms), dictionary (keys are paths of the audio files, values are relevant representations) or ndarray of numpy
    (e.g., if AudioFeaturesExtractor loads 10 audio files where every file has 44100 samples, it will return ndarray of shape (10, 44100))
    """
    LIST = enum.auto()
    DICTIONARY = enum.auto()
    NUMPY_ARRAY = enum.auto()


class AudioFeaturesExtractor:
    """
    This class serves for extracting audio features
    """
    def __init__(self, audio_files: list[str], logger: Logger, return_type: AudioFeaturesReturnType, sample_rate: int, duration: int = None, pad_short_files: bool = False):
        """
        A constructor
        :param audio_files: List of audio files for which the audio features should be extracted
        :param logger: A logger to use to write the class messages
        :param return_type: The type of the audio features database (defined by the "AudioFeaturesReturnType" enum)
        :param sample_rate: The sample rate that should be used for sampling the received audio files
        :param duration: The maximal duration to sample the audio files
        :param pad_short_files: If True (and if duration is provided) - files whose length shorter than the provided duration will be padded with zeros
        """
        self._audio_files: list[str] = audio_files
        self._logger: Logger = logger
        self._logger.debug(f'Initializing {self.__class__.__name__}')
        self._return_type: AudioFeaturesReturnType = return_type
        self._sample_rate: int = sample_rate
        self._frame_size: [int, None] = None
        self._hop_length: [int, None] = None
        self._mel_bands_number: [int, None] = None
        self._mfcc_number: [int, None] = None
        self._used_db_scale: bool = False
        # Build the features extractor database - a dictionary of waveforms for the received audio files
        self._waveforms_dictionary: dict[str, np.ndarray] = {}
        self._logger.info(f'{self.__class__.__name__} starts to process {len(audio_files)} audio files')
        for audio_file_path in audio_files:
            if not os.path.exists(audio_file_path):
                self._logger.error(11, f'{self.__class__.__name__} received a file {audio_file_path} that does not exist. Skipping it')
                continue
            if not audio_file_path.endswith('.wav'):
                self._logger.error(12, f'{self.__class__.__name__} received non-audio (.wav) file {audio_file_path}. Skipping it')
                continue
            # The sample rate returned by librosa.load is not needed, so taking only the first element
            current_signal: np.ndarray = librosa.load(audio_file_path, sr=sample_rate, duration=duration)[0]
            if pad_short_files:
                if duration is None:
                    self._logger.error(13, f'AudioFeaturesExtractor was asked to perform padding, but duration is not provided')
                else:
                    required_number_of_samples: int = int(sample_rate * duration)
                    if current_signal.shape[0] != required_number_of_samples:
                        current_signal: np.ndarray = librosa.util.fix_length(current_signal, size=required_number_of_samples)
            self._waveforms_dictionary[audio_file_path] = current_signal
        self._logger.info(f'AudioFeaturesExtractor loaded {len(self._waveforms_dictionary)} audio files')

    def _convert_result_to_return_type(self, result_dictionary: dict[str, np.ndarray]) -> [list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        """
        Converts the received dictionary of audio features to the expected return type
        :param result_dictionary: A dictionary containing the extracted audio features
        :return: A list/ndarray/dictionary containing the extracted audio features
        """
        if self._return_type == AudioFeaturesReturnType.LIST:
            return list(result_dictionary.values())
        elif self._return_type == AudioFeaturesReturnType.NUMPY_ARRAY:
            return np.stack(list(result_dictionary.values()))
        else:
            # self._return_type == AudioFeaturesReturnType.DICTIONARY
            return result_dictionary

    def update_return_type(self, new_type: AudioFeaturesReturnType) -> None:
        """
        Updates the return type of the audio features database (stored in the internal '_return_type') attribute
        :param new_type: The new return type
        :return: None
        """
        self._return_type: AudioFeaturesReturnType = new_type

    def durations(self) -> list[float]:
        """
        Returns a list containing a duration for each audio file provided to the constructor
        :return: A list of float values - each value represents a duration of a single audio file provided to the constructor
        """
        return list(map(lambda waveform: waveform.shape[0], self._waveforms_dictionary.values()))

    def waveform(self) -> [list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        """
        Returns the waveforms for the audio files provided to the constructor
        :return: A list/ndarray/dictionary containing the waveforms for the audio files provided to the constructor
        """
        self._logger.info(f'Extracting waveforms of {len(self._waveforms_dictionary)} provided audio files')
        return self._convert_result_to_return_type(self._waveforms_dictionary)

    def spectrogram(self, frame_size: int, hop_length: int, log_amplitude: bool = False) -> [list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        """
        Returns the spectrograms for the audio files provided to the constructor
        :param frame_size: The frame size to use in STFT
        :param hop_length: The hop length to use in STFT
        :param log_amplitude: If True - returns the log-amplitude spectrogram (in dB), otherwise - power spectrogram
        :return: A list/ndarray/dictionary containing the spectrograms for the audio files provided to the constructor
        """
        # Store the spectrogram parameters
        self._frame_size: int = frame_size
        self._hop_length: int = hop_length
        self._used_db_scale: bool = log_amplitude
        self._logger.info(f'Extracting spectrograms of {len(self._waveforms_dictionary)} provided audio files')
        self._logger.info(f'Spectrograms parameters: {frame_size=}, {hop_length=}, {log_amplitude=}')
        # This dictionary will map each audio file to its spectrogram
        spectrogram_dictionary: dict[str, np.ndarray] = {}
        for audio_file_path, signal in self._waveforms_dictionary.items():
            stft: np.ndarray = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)
            # Power spectrogram is the squared absolute value of the calculated STFT
            spectrogram: np.ndarray = np.abs(stft) ** 2
            if log_amplitude:
                # If log-amplitude spectrogram was requested, convert the power to dB
                spectrogram: np.ndarray = librosa.power_to_db(spectrogram)
            spectrogram_dictionary[audio_file_path] = spectrogram
        self._logger.info('Finished extracting spectrograms')
        return self._convert_result_to_return_type(spectrogram_dictionary)

    def mel_spectrogram(self, frame_size: int, hop_length: int, mel_bands_number: int, log_amplitude: bool = False) -> [list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        """
        Returns the mel-spectrograms for the audio files provided to the constructor
        :param frame_size: The frame size to use in STFT
        :param hop_length: The hop length to use in STFT
        :param mel_bands_number: The number of mel bands to use when building mel-spectrogram
        :param log_amplitude: If True - returns the log-amplitude spectrogram (in dB), otherwise - power spectrogram
        :return: A list/ndarray/dictionary containing the mel-spectrograms for the audio files provided to the constructor
        """
        # Store the spectrogram parameters
        self._frame_size: int = frame_size
        self._hop_length: int = hop_length
        self._mel_bands_number: int = mel_bands_number
        self._used_db_scale: bool = log_amplitude
        self._logger.info(f'Extracting mel-spectrograms of {len(self._waveforms_dictionary)} provided audio files')
        self._logger.info(f'Mel-spectrograms parameters: {frame_size=}, {hop_length=}, {mel_bands_number=}, {log_amplitude=}')
        # This dictionary will map each audio file to its mel-spectrogram
        spectrogram_dictionary: dict[str, np.ndarray] = {}
        for audio_file_path, signal in self._waveforms_dictionary.items():
            mel_spectrogram: np.ndarray = librosa.feature.melspectrogram(y=signal, sr=self._sample_rate, n_fft=frame_size, hop_length=hop_length, n_mels=mel_bands_number)
            if log_amplitude:
                # If log-amplitude spectrogram was requested, convert the power to dB
                mel_spectrogram: np.ndarray = librosa.power_to_db(mel_spectrogram)
            spectrogram_dictionary[audio_file_path] = mel_spectrogram
        self._logger.info('Finished extracting mel-spectrograms')
        return self._convert_result_to_return_type(spectrogram_dictionary)

    def mfcc(self, frame_size: int, hop_length: int, mfcc_number: int) -> [list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        """
        Returns the MFCCs for the audio files provided to the constructor
        :param frame_size: The frame size to use in the MFCC extraction
        :param hop_length: The hop length to use in the MFCC extraction
        :param mfcc_number: The number of MFCCs to extract
        :return: A list/ndarray/dictionary containing the MFCCs for the audio files provided to the constructor
        """
        # Store the MFCCs parameters
        self._logger.info(f'Extracting MFCCs of {len(self._waveforms_dictionary)} provided audio files')
        self._logger.info(f'MFCCs parameters: {frame_size=}, {hop_length=}, {mfcc_number=}')
        self._frame_size: int = frame_size
        self._hop_length: int = hop_length
        self._mfcc_number: int = mfcc_number
        # This dictionary will map each audio file to its MFCCs
        mfcc_dictionary: dict[str, np.ndarray] = {}
        for audio_file_path, signal in self._waveforms_dictionary.items():
            mfcc: np.ndarray = librosa.feature.mfcc(y=signal, sr=self._sample_rate, n_fft=frame_size, hop_length=hop_length, n_mfcc=mfcc_number)
            mfcc_dictionary[audio_file_path] = mfcc.T
        self._logger.info('Finished extracting MFCCs')
        return self._convert_result_to_return_type(mfcc_dictionary)

    @staticmethod
    def _set_axes_labels(x_label: str, y_label: str) -> None:
        """
        Sets the names of the plot axes
        :param x_label: The name of the x-axis
        :param y_label: The name of the y-axis
        :return: None
        """
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    @staticmethod
    def _set_title(plot_type: str, plotted_file_name: str) -> None:
        """
        Sets the title of the plot
        :param plot_type: The type of plot (e.g., 'Waveform', 'Spectrogram')
        :param plotted_file_name: The name of the audio file for which the plot is generated
        :return: None
        """
        title: str = plot_type
        if plotted_file_name:
            title += f' of "{plotted_file_name}"'
        plt.title(title)

    def _output_plot(self, png_file_name: str) -> None:
        """
        Outputs the plot. If the file name is provided - adds '.png' extension and saves it, otherwise - plots it without saving
        :param png_file_name: The name of a file where the plot should be stored
        :return: None
        """
        if png_file_name:
            png_file_name += '.png'
            if os.path.exists(png_file_name):
                self._logger.warning(13, f'The plot file {png_file_name} already exists - replacing it')
            plt.savefig(png_file_name)
        else:
            plt.show()
        plt.close()

    def plot_waveform(self, signal: np.ndarray, audio_file_name: str = '', output_file_name: str = '') -> None:
        """
        Plots the received waveforms using the provided sample rate
        :param signal: A waveform to plot
        :param audio_file_name: The name of the audio file for which the waveform is plotted
        :param output_file_name: The name of the file to save the figure (it won't be plotted)
        :return: None
        """
        plt.figure(figsize=(25, 10))
        librosa.display.waveshow(signal, sr=self._sample_rate, max_points=signal.size)
        self._set_axes_labels('Time (s)', 'Amplitude')
        self._set_title('Waveform', audio_file_name)
        self._output_plot(output_file_name)

    def _plot_general_spectrogram(self, spectrogram: np.ndarray, y_axis: str, audio_file_name: str, output_file_name: str) -> None:
        """
        Used to plot spectrograms of any type
        :param spectrogram: ndarray representing a spectrogram to plot
        :param y_axis: Defines the type of the spectrogram that should be plotted. Valid values: 'linear', 'log', 'mel'
        :param audio_file_name: The name of the audio file for which the spectrogram is plotted
        :param output_file_name: The name of the file to save the figure (it won't be plotted)
        :return: None
        """
        allowed_spectrogram_types: list = ['linear', 'log', 'mel']
        if y_axis not in allowed_spectrogram_types:
            self._logger.error(14, f'AudioFeaturesExtractor._plot_general_spectrogram received incorrect spectrogram type: {y_axis}')
        # y_axis can be linear, log or mel
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(spectrogram, sr=self._sample_rate, hop_length=self._hop_length, x_axis='time', y_axis=y_axis)
        color_bar_format: str = '%+2.f'
        if self._used_db_scale:
            color_bar_format += ' dB'
        plt.colorbar(format=color_bar_format)
        self._set_title(f'Spectrogram ({y_axis})', audio_file_name)
        self._output_plot(output_file_name)

    def plot_spectrogram(self, spectrogram: np.ndarray, audio_file_name: str = '', output_file_name: str = '') -> None:
        """
        Plots a spectrogram
        :param spectrogram: ndarray representing a spectrogram to plot
        :param audio_file_name: The name of the audio file for which the spectrogram is plotted
        :param output_file_name: The name of the file to save the figure (it won't be plotted)
        :return: None
        """
        self._plot_general_spectrogram(spectrogram, 'linear', audio_file_name, output_file_name)

    def plot_log_frequency_spectrogram(self, spectrogram: np.ndarray, audio_file_name: str = '', output_file_name: str = '') -> None:
        """
        Plots a log-frequency spectrogram
        :param spectrogram: ndarray representing a spectrogram to plot
        :param audio_file_name: The name of the audio file for which the spectrogram is plotted
        :param output_file_name: The name of the file to save the figure (it won't be plotted)
        :return: None
        """
        self._plot_general_spectrogram(spectrogram, 'log', audio_file_name, output_file_name)

    def plot_mel_spectrogram(self, spectrogram: np.ndarray, audio_file_name: str = '', output_file_name: str = '') -> None:
        """
        Plots a mel-spectrogram
        :param spectrogram: ndarray representing a spectrogram to plot
        :param audio_file_name: The name of the audio file for which the spectrogram is plotted
        :param output_file_name: The name of the file to save the figure (it won't be plotted)
        :return: None
        """
        self._plot_general_spectrogram(spectrogram, 'mel', audio_file_name, output_file_name)

    def plot_mfcc(self, mfcc: np.ndarray, audio_file_name: str = '', output_file_name: str = '') -> None:
        """
        Plots MFCCs
        :param mfcc: ndarray representing MFCCs to plot
        :param audio_file_name: The name of the audio file for which the spectrogram is plotted
        :param output_file_name: The name of the file to save the figure (it won't be plotted)
        :return: None
        """
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(mfcc, sr=self._sample_rate, hop_length=self._hop_length)
        self._set_axes_labels('Time (s)', 'MFCC coefficients')
        plt.colorbar()
        self._set_title('MFCCs', audio_file_name)
        self._output_plot(output_file_name)
