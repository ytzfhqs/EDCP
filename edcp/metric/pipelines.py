from typing import List, Dict, Any, Union, Optional


from . import utils
from .nlpfeat import NlpFeat
from .calppl import CPPl
from .minhash import CalMinHash
from ..tool import read_json, save_json
from .mcdict import McDict
from .language import IdentLanguage
from .importance import ImportFeat
from .check_type import check_path_data


class MetricProcess:
    def __init__(
        self,
        data_or_filepath: Union[str, List[Dict[str, Any]]],
        book_data_or_path: Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]],
        wordgram_model_path: List[Dict[str, str]],
        save_wordgram_model_dir: Optional[str],
        llm_model_path: str,
        fasttext_model_path: str,
        text_column: str,
        idx_column: str,
        num_perm: int,
        res_save_path: str
    ):
        """
        MetricProcess初始化方法
        Args:
            data_or_filepath: 进入评分管道的变量或json文件路径
            book_data_or_path: 用于重要性采样的书籍数据变量或json文件路径
            wordgram_model_path: 重要性采样书籍数据训练的wordgram模型路径，若此参数不为None，则book_data_or_path将被忽略
            save_wordgram_model_dir: 仅在传入book_data_or_path时有效。保存书籍数据训练的wordgram模型文件夹路径，若为None，则不保存
            llm_model_path: 用于计算PPL指标的LLM路径
            fasttext_model_path:
            text_column: Dict中文本的Key
            idx_column: Dict中样本ID的Key
            num_perm: MinHash中构造哈希值所用哈希函数个数
        """

        if check_path_data(data_or_filepath) == "str":
            self.data = read_json(data_or_filepath)
        elif check_path_data(data_or_filepath) == "data":
            self.data = data_or_filepath

        self.nf = NlpFeat(text_column)
        self.cppl = CPPl(llm_model_path, text_column)
        self.cmh = CalMinHash(self.data, text_column, idx_column, num_perm)
        self.idl = IdentLanguage(fasttext_model_path, text_column)
        self.imf = ImportFeat(
            book_data_or_path,
            wordgram_model_path,
            text_column,
            save_wordgram_model_dir,
        )

        self.res_save_path = res_save_path

    def merger_cal(self, single_sample: Dict[str, Any]) -> McDict[str, Any]:
        return utils.cat_dict(
            # 原数据
            single_sample,
            # NLP特征指标
            self.nf.do_process(single_sample, only_mid_res=True),
            # LLM的PPL指标
            self.cppl.do_process(single_sample, only_mid_res=True),
            # Minhash指标
            self.cmh.do_process(single_sample, only_mid_res=True),
            # 中英语言识别指标
            self.idl.do_process(single_sample, only_mid_res=True),
            # 重要性采样指标
            self.imf.do_process(single_sample, only_mid_res=True),
        )

    def forward(self):
        for idx in range(len(self.data)):
            self.data[idx] = self.merger_cal(self.data[idx])
        save_json(self.res_save_path, self.data)
        return self.data
