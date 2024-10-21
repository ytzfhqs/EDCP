import re
from typing import List, Dict, Any

import fasttext
import numpy as np
from mpire import WorkerPool
from . import utils
from .mcdict import McDict
from loguru import logger


class IdentLanguage:
    def __init__(self, fasttext_model_path: str, text_column: str = "text"):
        logger.info(f"Starts initialising the IdentLanguage pipeline.")
        self.model = fasttext.load_model(fasttext_model_path)
        logger.info(f"Successful Loading FastText model from {fasttext_model_path}.")
        self.text_column = text_column

    @staticmethod
    def text_trans(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def split_res(res: tuple[tuple, np.ndarray[Any, np.dtype]]) -> tuple[str, float]:
        label = re.findall(r"__label__(\w+)", res[0][0])[0]
        prop = round(np.clip(res[1], 0, 1).item(), 4)
        return label, prop

    def do_process(
        self, single_sample: Dict[str, Any], only_mid_res: bool = False
    ) -> Dict[str, Any] | McDict[str, Any]:
        res_mid = dict()
        tran_t = self.text_trans(single_sample[self.text_column])
        label, prop = self.split_res(self.model.predict(tran_t))
        res_mid["language"] = label
        res_mid["prop"] = prop
        if only_mid_res:
            return res_mid
        else:
            return utils.cat_dict(single_sample, res_mid)

    def forward(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx in range(len(data)):
            data[idx] = self.do_process(data[idx])
        return data

    def forward_pool(
        self, data: List[Dict[str, Any]], num_proc: int
    ) -> List[McDict[str, Any]]:
        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            cmh_res: List[Dict[str, float]] = pool.map(
                self.do_process, data, progress_bar=True
            )
        data = utils.cat_dict_with_pool(data, cmh_res)
        return data
