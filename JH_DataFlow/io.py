import json
import pickle
from typing import Dict, List

import yaml
import numpy as np
import cv2

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



