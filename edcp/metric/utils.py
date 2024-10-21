import re
from typing import List, Dict, Any, Union, Tuple


import jieba
import jionlp as jio
import nltk
from nltk import FreqDist
from nltk.util import ngrams
from .mcdict import McDict


def cat_dict(*args: Union[McDict[str, Any], Dict[str, Any]]) -> McDict[str, Any]:
    temp_dict: McDict[str, Any] = McDict()
    for a in args:
        temp_dict = McDict(**temp_dict, **a)
    return temp_dict


def cat_dict_with_pool(
    og_ld: List[Dict[str, Any]],
    add_ld: List[Dict[str, Any]],
) -> List[McDict[str, Any]]:
    for idx, og, add in enumerate(zip(og_ld, add_ld)):
        og_ld[idx] = McDict(**og, **add)
    return og_ld


def remove_puncs(text: str) -> str:
    all_puncs = re.compile(
        r"[，\_《。》、？；：‘’＂“”【「】」·！@￥…（）—\,\<\.\>\/\?\;\:\'\"\[\]\{\}\~\`\!\@\#\$\%\^\&\*\(\)\-\=\+]"
    )
    text = re.sub(all_puncs, "", text)
    return text


def split_word(
    sentence: str, enhance_med: bool = False
) -> Tuple[str, List[str], List[str]]:
    """分词操作"""
    if enhance_med:
        jieba.load_userdict('medical_words.txt')
    # 去除所有标点符号
    no_pinc_text = remove_puncs(sentence)
    # 分词操作
    cut_s = [word for word in jieba.lcut(no_pinc_text) if word.strip()]
    # 去除停用词
    cut_s_stop = jio.remove_stopwords(cut_s)
    # 返回去除标点符号后的文本、分词列表、去除停用词后的词表
    return no_pinc_text, cut_s, cut_s_stop


def generate_ngrams(
    token_list: List[str], ngram_size: int
) -> tuple[list[Any], FreqDist]:
    """产生ngrams模型"""
    bigrams = list(ngrams(token_list, ngram_size))
    # 计算grams出现频率
    freq = nltk.FreqDist(bigrams)
    return bigrams, freq
