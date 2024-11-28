import yaml
from vmpred.exception import vmException
import os
import sys
import numpy as np
import pandas as pd
from vmpred.constant import *
import joblib

def read_yaml_file(file_path:str) -> dict:
    """
    Reads a YAML file and returns the content as a dictionary
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise vmException(e,sys) from e
    
def read_parquet(file_path:str) -> pd.DataFrame:
    """
    Reads a parquet file and returns the content as a DataFrame

    Args:
        file_path : str

    Returns:
        pd.DataFrame: Pandas DataFrame
    """

    try:
        return pd.read_parquet(file_path, engine="pyarrow")
    except Exception as e:
        raise vmException(e,sys) from e
    
def save_object(file_path:str, obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise vmException(e,sys) from e
    
def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path,'rb') as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise vmException(e,sys) from e