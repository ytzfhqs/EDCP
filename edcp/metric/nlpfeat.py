import re
import math
from collections import Counter
from typing import List, Dict, Any


import numpy as np
import jionlp as jio
from mpire import WorkerPool
from . import utils
from .mcdict import McDict
from loguru import logger


class SimpleInfo:
    def __init__(
        self, sentence: str, no_pinc_text: str, cut_s: List[str], cut_s_stop: List[str]
    ):
        self.sentence = sentence
        self.no_pinc_text = no_pinc_text
        self.cut_s = cut_s
        self.cut_s_stop = cut_s_stop

    def stop_radio(self) -> float:
        """计算停用词占分词列表比例"""
        return round(len(self.cut_s_stop) / len(self.cut_s), 4)

    def puncs_radio(self) -> float:
        """计算标点符号占分词列表比例"""
        return round((len(self.sentence) - len(self.no_pinc_text)) / len(self.cut_s), 4)

    def word_unique_radio(self) -> float:
        """计算唯一词占分词列表比例"""
        return round(
            (len(self.cut_s) - len(list(set(self.cut_s)))) / len(self.cut_s), 4
        )

    def num_sentences(self) -> int:
        """计算句子个数"""
        return len(jio.split_sentence(self.sentence))

    def word_entropy(self) -> float:
        """计算词的信息熵"""
        total_words = len(self.cut_s)
        word_counts = Counter(self.cut_s)
        entropy = sum(
            -count / total_words * math.log2(count / total_words)
            for count in word_counts.values()
        )
        return round(entropy, 4)

    def is_ending_with_terminal_punctution(self) -> bool:
        """文本是否以结束标点符号终止"""
        terminal_puncs = '。！；？.!;?"”'
        if self.sentence[-1] in terminal_puncs:
            return True
        else:
            return False

    def mean_word_length(self) -> float:
        """词平均长度"""
        words_len = [len(c) for c in self.cut_s]
        return round(np.mean(words_len), 4)

    def curly_bracket(self) -> float:
        """括号出现次数与文本字符数比例"""
        bracket_num = len(re.findall(r"\(|\)|（|）|\{|\}", self.sentence))
        return round(bracket_num, len(self.no_pinc_text))

    def word_count(self) -> int:
        """文本字数"""
        return len(self.no_pinc_text)

    def num_words(self) -> int:
        """文本单词数"""
        return len(self.cut_s)


class NlpFeat:
    def __init__(self, text_column: str = "text"):
        logger.info("Starts initialising the NlpFeat pipeline.")
        self.text_column = text_column

    @staticmethod
    def simple_info(
        sentence: str, no_pinc_text: str, cut_s: List[str], cut_s_stop: List[str]
    ) -> Dict[str, Any]:
        """
        计算简单的NLP特征指标
        Args:
            sentence: 原始文本
            no_pinc_text: 去除标点符号后的文本
            cut_s: 分词列表
            cut_s_stop: 去除停用词后的分类列表

        Returns:
        各NLP特征指标分数
        """
        simple_dict = {}
        si = SimpleInfo(sentence, no_pinc_text, cut_s, cut_s_stop)
        simple_dict["stop_radio"] = si.stop_radio()
        simple_dict["puncs_radio"] = si.puncs_radio()
        simple_dict["word_unique_radio"] = si.word_unique_radio()
        simple_dict["num_sentences"] = si.num_sentences()
        simple_dict["word_entropy"] = si.word_entropy()
        simple_dict["is_ending_with_terminal_punctution"] = (
            si.is_ending_with_terminal_punctution()
        )
        simple_dict["mean_word_length"] = si.mean_word_length()
        simple_dict["curly_bracket"] = si.curly_bracket()
        return simple_dict

    @staticmethod
    def chars_dupe_ngrams(token_list: List[str], ngram_size: int) -> float:
        """计算ngrams中重复词的字符占比"""
        word_lengths = np.array([len(token) for token in token_list])
        bigrams, freq = utils.generate_ngrams(token_list, ngram_size)
        duplicate_ngrams = [key for key in freq if freq[key] > 1]
        duplicate_mask = np.zeros(len(token_list), dtype=int)
        for i in range(len(token_list) - ngram_size + 1):
            ngram = tuple(token_list[i : i + ngram_size])
            if ngram in duplicate_ngrams:
                duplicate_mask[i : i + ngram_size] = 1
        repeated_chars_count = np.sum(word_lengths * duplicate_mask)
        total_chars_count = np.sum(word_lengths)

        if total_chars_count == 0:
            return 0.0

        return round(repeated_chars_count / total_chars_count, 4)

    @staticmethod
    def chars_top_ngrams(token_list: List[str], ngram_size: int) -> float:
        """统计ngrams中频率最高的grams占比"""
        bigrams, freq = utils.generate_ngrams(token_list, ngram_size)
        max_freq_keys = max(freq, key=lambda k: freq[k])
        if freq[max_freq_keys] == 1:
            return 0.0
        total_chars = sum(len(w) for w in token_list)
        top_ngram_chars = sum(len(w) for w in max_freq_keys)
        score = top_ngram_chars * freq[max_freq_keys] / total_chars
        return score

    @classmethod
    def key_ngrams(cls, token_list: List[str]) -> McDict[str, float]:
        """循环计算ngrams指标"""
        ng_dict: McDict[str, float] = McDict()
        # 检查token长度
        max_n = min(len(token_list), 10)
        # 2~10 grams模型
        for i in range(2, max_n + 1):
            if i <= 4:
                ng_dict[f"chars_top_{str(i)}grams"] = cls.chars_top_ngrams(
                    token_list, i
                )
            ng_dict[f"chars_dupe_{str(i)}grams"] = cls.chars_dupe_ngrams(token_list, i)
        return ng_dict

    def do_process(
        self, single_sample: Dict[str, Any], only_mid_res: bool = False
    ) -> McDict[str, Any]:
        """
        单样本处理函数
        Args:
            single_sample: 待处理的样本
            only_mid_res: 是否仅返回指标结果（不包含原样本）

        Returns:

        """
        sentence: str = single_sample[self.text_column]
        # 去除标点符号后的文本、词表、去除停用词后的词表
        no_pinc_text, cut_s, cut_s_stop = utils.split_word(sentence)
        if only_mid_res:
            res_mid = utils.cat_dict(
                # ngrams指标
                self.key_ngrams(cut_s_stop),
                # 简单的nlp指标
                self.simple_info(sentence, no_pinc_text, cut_s, cut_s_stop),
            )
            return res_mid
        else:
            # 计算评估指标，并与原字典进行合并
            res = utils.cat_dict(
                single_sample,
                # ngrams指标
                self.key_ngrams(cut_s_stop),
                # 简单的nlp指标
                self.simple_info(sentence, no_pinc_text, cut_s, cut_s_stop),
            )
        return res

    def forward(self, data: List[Dict[str, Any]]) -> list[dict[str, Any]]:
        for idx in range(len(data)):
            data[idx] = self.do_process(data[idx])
        return data

    def forward_pool(
        self, data: List[Dict[str, Any]], num_proc: int
    ) -> List[McDict[str, Any]]:
        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            nlp_feat_res: List[Dict[str, float]] = pool.map(
                self.do_process, data, progress_bar=True
            )
        data = utils.cat_dict_with_pool(data, nlp_feat_res)
        return data
