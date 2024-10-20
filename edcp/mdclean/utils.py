import os
from io import StringIO
from typing import List, Dict, Tuple

import re
from loguru import logger
from datasets import Dataset


def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


def split_text(s: str, split_char="\n") -> List[str]:
    """将字符串按split_char分割，存放到List中"""
    lines = [line.rstrip() for line in s.split(split_char) if line.strip()]
    return lines


def remove_and_replace(
    text: str,
    remove_words: List[str] = None,
    replacements: List[Tuple[str, str]] = None,
):
    """
    对字符串进行字符删除或使用正则表达式进行替换
    Args:
        text: 需要处理的字符串
        remove_words: 需要删除的字符串，使用replace方法
        replacements: 正则表达式

    Returns:

    """
    if remove_words:
        for word in remove_words:
            text = text.replace(word, "")
    if replacements:
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text


def filter_info(op_name: str, og_num: int, new_num: int, file_name: str = None):
    """
    计算过滤前后样本数量及比例
    Args:
        op_name: 操作名称
        og_num: 过滤前样本数量
        new_num: 过滤后样本数量
        file_name: 当前处理文件的名称
    Returns:

    """
    percentage = new_num / og_num * 100
    if file_name is not None:
        logger.info(f"File name: {file_name}")
    logger.info(f"Start {op_name}")
    logger.info(f"The number of samples before {op_name} is {og_num}")
    logger.info(f"The number of samples after {op_name} is {new_num}")
    logger.info(f"The proportion of data retained is {percentage:.2f}%")


def chunk_list(lst: List[str], batch_size: int) -> List[List[str]]:
    """
    将列表lst按照batch_size个元素进行分组
    """
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def select_strings(str_list: List[str], bool_list: List[str]) -> List[str]:
    """
    根据bool_list中的值从str_list中选择字符串。
    Args:
        str_list: 字符串列表
        bool_list: 布尔值列表，用于指示是否选择str_list中的对应项

    Returns:包含由bool_list选择的str_list中字符串的新列表

    """
    selected_strings = []
    for i in range(len(str_list)):
        if bool_list[i] == "True":  # 如果对应的布尔值为True
            selected_strings.append(str_list[i])
        elif bool_list[i] == "False":
            continue
        else:
            logger.warning(f"Context : {str_list} \n LLM output not legal")
    return selected_strings


def search_file_suffix(directory: str, suffix: str):
    """
    搜索文件夹下所有后缀为suffix的文件
    Args:
        directory:需要搜索的文件夹
        suffix:文件后缀

    Returns:

    """
    suffix_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                suffix_files.append(os.path.join(root, file))
    return suffix_files


def list2dataset(lst: List[Dict[str, str]]) -> Dataset:
    """将List变为datasets格式"""
    return Dataset.from_list(lst)


def save_txt(string_list: List[str], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in string_list:
            file.write(item + '\n')


def filename_add_suffix(file_path, suffix):
    """
    在给定的文件路径的文件名部分添加指定的后缀。

    参数:
    file_path (str): 原始文件路径。
    suffix (str): 要添加到文件名的后缀。

    返回:
    str: 添加后缀后的新的文件路径。
    """
    # 分离目录和文件名
    dir_name, file_name = os.path.split(file_path)
    # 分离文件名和扩展名
    name_only, extension = os.path.splitext(file_name)
    # 创建新的文件名
    new_file_name = f"{name_only}_{suffix}{extension}"
    # 合并目录和新的文件名
    return os.path.join(dir_name, new_file_name)
