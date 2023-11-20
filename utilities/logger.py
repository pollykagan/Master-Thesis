"""
This module provides logging feature
"""
import os
import sys
import re
import time
import logging
import simple_colors


class Logger:
    """
    This class serves for the program messages logging
    """
    def __init__(self, name: str, verbosity: str, use_console: bool = True, use_file: bool = True):
        """
        A constructor
        :param name: The name to give to the logger
        :param verbosity: The verbosity level of the logger
        :param use_console: True if the logger messages should be printed to the console, False otherwise
        :param use_file: True if the logger messages should be written to the log file, False otherwise
        """
        # Initialize internal attributes
        self._use_file: bool = use_file
        # The log file name is derived from the logger name
        if self._use_file:
            self._log_file_path: str = f'{name}.log'
        else:
            self._log_file_path: str = ''
        self._warning_counter: int = 0
        self._error_counter: int = 0
        # Create the logging.Logger object to be used for generating the log messages, and set the log verbosity
        self._logger: logging.Logger = logging.getLogger(name)
        self._logger.setLevel(verbosity.upper())
        # Set up the format of the log messages
        self._formatter: logging.Formatter = logging.Formatter('%(asctime)s   %(levelname)-8s ||| %(name)-15s ||| %(message)s', datefmt='%d/%m/%y %H:%M:%S')
        # If console logging is required - set up the console handler
        if use_console:
            self._console_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
            self._console_handler.setFormatter(self._formatter)
            self._logger.addHandler(self._console_handler)
        # If file logging is required - set up the console handler
        if use_file:
            self._file_handler: logging.FileHandler = logging.FileHandler(self._log_file_path, mode='w')
            self._file_handler.setFormatter(self._formatter)
            self._logger.addHandler(self._file_handler)
        # The following dictionary maps each logger verbosity to a related coloring function from "simple_colors" module
        self._colors: dict = {'debug': simple_colors.cyan, 'info': simple_colors.green, 'warning': simple_colors.yellow, 'error': simple_colors.red}

    @staticmethod
    def _remove_ansi_characters(string: str) -> str:
        """
        Private and static method. Receives a string, removes all the ANSI characters from it, and returns the result
        :param string: A string to remove the ANSI characters from
        :return: The processed string
        """
        return re.sub(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]', '', string)

    def _get_log_message(self, error_code: int, message: str, message_type: str) -> str:
        """
        Private method. Processes the received log message, by adding an error code and coloring the message
        :param error_code: The error code of the message (valid error code is a positive integer - any other integer will point that no error code is required)
        :param message: A log message text
        :param message_type: The type of the message (verbosity)
        :return: The processed log message
        """
        # If error code is a positive integer - put it in the beginning of the message
        if error_code > 0:
            code_string: str = f'{error_code} - '
        else:
            code_string: str = ''
        # Construct the result message by concatenating the error code and the received string, and coloring the message
        return self._colors[message_type](code_string + message)

    def _summary(self) -> None:
        """
        Logs the summary of warnings and errors (using the "info" verbosity)
        :return: None
        """
        self.info(f'Summary: {self._warning_counter} warning, {self._error_counter} errors')

    def debug(self, message: str) -> None:
        """
        Logs the received message with a "debug" verbosity
        :param message: A message to log
        :return: None
        """
        self._logger.debug(self._get_log_message(0, message, 'debug'))

    def info(self, message: str) -> None:
        """
        Logs the received message with an "info" verbosity
        :param message: A message to log
        :return: None
        """
        self._logger.info(self._get_log_message(0, message, 'info'))

    def warning(self, warning_code: int, message: str) -> None:
        """
        Logs the received message with a "warning" verbosity
        :param warning_code: An integer code of the warning
        :param message: A message to log
        :return: None
        """
        self._warning_counter += 1
        self._logger.warning(self._get_log_message(warning_code, message, 'warning'))

    def error(self, error_code: int, message: str) -> None:
        """
        Logs the received message with an "error" verbosity
        :param error_code: An integer code of the error
        :param message: A message to log
        :return: None
        """
        self._error_counter += 1
        self._logger.error(self._get_log_message(error_code, message, 'error'))

    def fatal(self, error_code: int, message: str, exception: Exception.__class__) -> None:
        """
        Logs the received message with an "error" verbosity, and raises the provided exception
        :param error_code: An integer code of the error
        :param message: A message to log
        :param exception: An exception to raise
        :return: None
        """
        self.error(error_code, message)
        raise exception(message)

    def close_logger(self) -> None:
        """
        Print the summary of the logger (for warnings and errors), and remove all the ANSI characters from the log file -
        these characters define the coloring of the messages in the console, but they appear in the log file and affect the readability
        :return: None
        """
        self._summary()
        # Sleep for a second, so the summary can be successfully written to the log file
        time.sleep(3)
        # Since in this project there won't be big log files, the following trick is applied:
        # read the whole log file to a string, remove all the ANSI characters, and write the result to the log file
        if self._use_file and self._log_file_path and os.path.exists(self._log_file_path):
            with open(self._log_file_path, 'r') as logger_file_object:
                logged_messages: str = logger_file_object.read()
            with open(self._log_file_path, 'w') as logger_file_object:
                logger_file_object.write(self._remove_ansi_characters(logged_messages))
