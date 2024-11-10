import os
import json
import pickle
import importlib.util
import importlib.metadata
from typing import List, Dict, Any


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def save_json(paths: str, datas: List[Dict[Any, Any]]) -> None:
    """
    保存json文件
    """
    with open(paths, "w", encoding="utf-8") as json_file:
        json.dump(datas, json_file, indent=2, ensure_ascii=False)


def read_json(paths: str) -> List[Dict[Any, Any]]:
    """
    读取json文件
    """
    with open(paths, "r", encoding="UTF-8") as f:
        load_dict = json.load(f)
    return load_dict


def load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def save_pkl(pkl_data, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(pkl_data, f)


def check_dir_exist(dir_path: str):
    """检查文件夹是否存在，若不存在则新建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
