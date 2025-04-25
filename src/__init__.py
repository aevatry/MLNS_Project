import os
import pathlib

def get_data_dir() -> str:
    """
    Get the data directory path
    """
    return os.path.abspath(pathlib.Path(__file__).parent.parent / "data")