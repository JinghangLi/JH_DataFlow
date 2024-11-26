import json
import pickle
from typing import Dict, List

import yaml
import numpy as np
import cv2
from pathlib import Path

def read_json(path: str) -> Dict:
    '''Reads a json file and returns the content as a dictionary.'''
    with open(path, 'r') as f:
        return json.load(f)
    
def write_json(dict: Dict, path: str) -> None:
    '''Writes a dictionary to a json file.'''
    json_data = json.dumps(dict, indent=4, ensure_ascii=False)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    return

def read_txt(path: str) -> List[str]:
    '''Reads a text file and returns the content as a list of strings.'''
    with open(path, 'r') as f:
        l = f.readlines()
    return [itm.rstrip() for itm in l]

def write_txt(obj: List, path: str):
    s = '\n'.join(obj)
    with open(path, 'w+') as f:
        f.write(s)


def read_image(path: str) -> np.ndarray:
    '''Reads an image file and returns it as a numpy array.'''
    return cv2.imread(path)

def write_image(path: str, img: np.ndarray) -> None:
    '''Writes an image to a file.'''
    cv2.imwrite(path, img)
    return

def read_yaml(path: str) -> Dict:
    '''Reads a yaml file and returns the content as a dictionary.'''
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def read_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def write_pkl(path: str, obj: dict) -> None:
    '''Writes an object to a pickle file.'''
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return
    
# lidar related 
def save_points_bin(path: str, points: np.ndarray) -> bool:
    '''Saves a numpy array of points to a binary file.'''
    try:
        with open(path, 'wb') as f:
            f.write(points.tobytes())
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def list_files_by_extension(dir_path: str, file_extension: str) -> List[str]:
    """
    列出指定目录下所有指定后缀的文件的绝对路径，并按文件名排序。
    Args:
        dir_path (str): 目标文件夹路径。
        file_extension (str): 文件后缀（例如 '.json'）。

    Returns:
        List[str]: 排序后的文件绝对路径列表。
    """
    folder_path = Path(dir_path)
    if not folder_path.is_dir():
        raise ValueError(f"The specified path '{dir_path}' is not a valid directory.")
    
    files = sorted(folder_path.glob(f"*{file_extension}"))
    return [str(file.resolve()) for file in files]
