"""
General utilities module
"""
import os


def get_files_full_paths_list(directory: str) -> list[str]:
    """
    Returns a list of full paths for all the files contained in the provided directory
    :param directory: A directory to return the files from
    :return: A list of strings - each one represent a full path for a file in the provided directory
    """
    directory_real_path: str = os.path.realpath(directory)
    return list(map(lambda file_name: os.path.join(directory_real_path, file_name), os.listdir(directory_real_path)))
