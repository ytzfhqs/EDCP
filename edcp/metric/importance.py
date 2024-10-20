import os
import math
from typing import List, Dict, Any, Optional, Union

from nltk import FreqDist
from loguru import logger
from .mcdict import McDict
from .. import tool
from . import utils
from pydantic import BaseModel, ValidationError


class BookData(BaseModel):
    book_data: Dict[str, List[Dict[str, str]]]


class BookPath(BaseModel):
    book_path: List[Dict[str, str]]


class ImportFeat:
    def __init__(
        self,
        book_data_or_path: Union[
            Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]
        ] = None,
        wordgram_model_path: List[Dict[str, str]] = None,
        text_column: str = "text",
        save_wordgram_model_dir: Optional[str] = "wordgram_model",
    ):
        """

        Args:
            book_data_or_path: 若为变量，则最外层字典键为采样对象名称。若为文件路径，必须包含'name','file_path'键。
            wordgram_model_path: 重要性采样书籍数据训练的wordgram模型路径，若此参数不为None，则book_data_or_path将被忽略
            text_column: Dict中文本的Key
            save_wordgram_model_dir: 仅在传入book_data_or_path时有效。保存书籍数据训练的wordgram模型文件夹路径，若为None，则不保存
        """
        self.text_column: str = text_column
        self.wordgram_model: Dict[str, FreqDist] = dict()
        book_datas: Dict[str, List[Dict[str, Any]]] = dict()

        logger.info(f"Starts initialising the ImportFeat pipeline.")

        # 若传入wordgram_model_path参数则忽略book_data_or_path参数
        if wordgram_model_path:
            logger.info(f"Loading of wordgram models.")
            # [{'name':, 'model_path':}]
            for wmp in wordgram_model_path:
                self.wordgram_model[wmp["name"]] = tool.load_pkl(wmp["model_path"])
            logger.info(f"Wordgram models loading complete.")

        # 若未传入wordgram_model_path参数，则根据book_data_or_path训练wordgram模型
        elif book_data_or_path:
            try:
                BookData(book_data=book_data_or_path)
                logger.info(f"Read book data from variables.")
                # 数据格式为{'wiki':[{'text':},{'text':}], 'med':[{'text':},{'text':}]}
                book_datas = book_data_or_path
            except ValidationError as de:
                try:
                    BookPath(book_path=book_data_or_path)
                    logger.info(f"Read book data from json file.")
                    for bd in book_data_or_path:
                        # 文件路径[{'name':'','file_path':''},]
                        book_datas[bd["name"]] = tool.read_json(bd["file_path"])
                except ValidationError as pe:
                    raise

            for name, v in book_datas.items():
                tokens = self.get_tokens(v, self.text_column)
                for i in range(2, 4):
                    suffix_name = name + f"({i}-gram)"
                    logger.info(f"Training {suffix_name} models.")
                    self.wordgram_model[suffix_name] = self.build_wordgram(
                        suffix_name, tokens, i, save_wordgram_model_dir
                    )

        if book_data_or_path is None and wordgram_model_path is None:
            raise Exception(
                "Either book_data_or_path or wordgram_model_path must be set."
            )

    @staticmethod
    def get_tokens(book_data: List[Dict[str, Any]], text_column: str):
        total_text = "".join(item[text_column] for item in book_data)
        _, cut_s, _ = utils.split_word(total_text)
        return cut_s

    @staticmethod
    def build_wordgram(
        book_name: str,
        tokens: List[str],
        ngram_size: int = 2,
        model_save_dir: Optional[str] = None,
    ) -> FreqDist:
        _, freq = utils.generate_ngrams(tokens, ngram_size)
        if model_save_dir:
            tool.check_dir_exist(model_save_dir)
            model_save_path = os.path.join(model_save_dir, f"{book_name}.pkl")
            tool.save_pkl(freq, model_save_path)
            logger.info(
                f"The {book_name} wordgram model training is complete, and the model save path is {model_save_path}"
            )
        else:
            logger.info(f"The {book_name} wordgram model training is complete")
        return freq

    @staticmethod
    def ngram_prob(ngram_freq: FreqDist, word: str) -> float:
        total_ngrams = sum(ngram_freq.values())
        return (ngram_freq[word] + 1) / (total_ngrams + len(ngram_freq))

    def cls_prob(
        self, wordgram_model: FreqDist, single_sample: Dict[str, Any]
    ) -> float:
        _, tokens, _ = utils.split_word(single_sample[self.text_column])
        prob = 1.0
        for token in tokens:
            prob *= self.ngram_prob(wordgram_model, token)
        return round(math.log(prob), 4)

    def do_process(
        self, single_sample: Dict[str, Any], only_mid_res: bool = False
    ) -> Dict[str, Any] | McDict[str, Any]:
        res_mid = dict()
        for name, model in self.wordgram_model.items():
            res_mid[f"Importance_sample_with_{name}"] = self.cls_prob(
                model, single_sample
            )
        if only_mid_res:
            return res_mid
        else:
            # 计算重要性采样，并与原字典进行合并
            res = utils.cat_dict(single_sample, res_mid)
        return res

    def forward(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx in range(len(data)):
            data[idx] = self.do_process(data[idx])
        return data
