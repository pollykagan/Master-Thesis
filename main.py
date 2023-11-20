"""
The main module
"""
import os
import re
from utilities.general import get_files_full_paths_list
from utilities.parser import ArgumentsParser
from utilities.logger import Logger
import preprocessing
from pipelines.emotions_classification import EmotionClassificationPipeline


def plot_results():
    paths = get_files_full_paths_list(os.path.join(os.getcwd(), 'plots', 'emo_db'))[:-1]
    results = {'[10, 20, 30]': {'0.0001': 0, '0.001': 0, '0.005': 0, '0.01': 0, '0.05': 0},
               '[20, 40, 30, 10]': {'0.0001': 0, '0.001': 0, '0.005': 0, '0.01': 0, '0.05': 0},
               '[80, 60, 40, 20]': {'0.0001': 0, '0.001': 0, '0.005': 0, '0.01': 0, '0.05': 0}}
    for file_path in paths:
        with open(file_path, 'r') as file:
            file_content = file.read().strip()
            match = re.search(r'channels=([^;]+); .* learning_rate=([^;]+)$', file_content)
        results[match.group(1)][match.group(2)] += 1
    print(results)


def plot_histograms():
    datasets_path: str = os.path.realpath(os.path.join('data', 'datasets'))
    for dataset in os.listdir(datasets_path):
        preprocessing.get_preprocessor_class(dataset).plot_data_histograms()


def main():
    """
    The main function
    :return: None
    """

    # plot_histograms()
    parser = ArgumentsParser()
    args = parser.parse_arguments()
    logger = Logger(args.logger_name, args.verbosity, args.log_to_console, args.log_to_file)
    # plot_results()
    if args.preprocess_data:
        preprocessing.get_preprocessor_class(args.dataset)(args.emotion, args, logger)()
    else:
        pipeline = EmotionClassificationPipeline(args, logger)
        if args.mode == 'training':
            pipeline.perform_single_configuration_training()
        elif args.mode == 'evaluation':
            pipeline.evaluate_trained_model()
        elif args.mode == 'tuning':
            pipeline.tune()
        elif args.mode == 'cross_validation':
            pipeline.cross_validation()
    logger.close_logger()


if __name__ == '__main__':
    main()
