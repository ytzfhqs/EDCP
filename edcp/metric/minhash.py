from multiprocessing import cpu_count
from typing import List, Dict, Any


from mpire import WorkerPool
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from . import utils
from .mcdict import McDict
from loguru import logger


class CalMinHash:
    def __init__(
        self,
        data: List[Dict[str, Any]],
        text_column: str = "text",
        idx_column: str = "int_id",
        num_perm: int = 256,
    ):
        logger.info("Starts initialising the CalMinHash pipeline.")
        self.data = data
        self.text_column = text_column
        self.idx_column = idx_column
        self.num_perm = num_perm
        logger.info(
            "Initialise MinHashLSH with thresholds of 0.7, 0.8 and 0.9 respectively."
        )
        self.ml_lsh0_7: MinHashLSH
        self.ml_lsh0_8: MinHashLSH
        self.ml_lsh0_9: MinHashLSH
        self.mh_dict: Dict[Any, Any]

        # 尝试多进程
        try:
            logger.info(f"Update MinHash key-value pairs with parallel computing using {cpu_count()} cpu cores")
            self._update_mh_pool()
        # 单进程
        except:
            logger.info("Update MinHash key-value pairs")
            self._update_mh()
        logger.info("Updates MinHash key-value pairs completed")

    def _init_mhlsh(self):
        """初始化MinHashLSH类以及键值对"""
        self.ml_lsh0_7 = MinHashLSH(threshold=0.7, num_perm=self.num_perm)
        self.ml_lsh0_8 = MinHashLSH(threshold=0.8, num_perm=self.num_perm)
        self.ml_lsh0_9 = MinHashLSH(threshold=0.9, num_perm=self.num_perm)
        self.mh_dict: Dict[Any, Any] = {}

    def _update_mh(self):
        # 初始化MinHashLSH类
        self._init_mhlsh()
        for d in tqdm(self.data):
            # 初始化MinHash类（为每个样本单独初始化MinHash类）
            mh = MinHash(num_perm=self.num_perm)
            # 取分词后的词表
            _, words, _ = utils.split_word(d[self.text_column])
            for word in set(words):
                mh.update(word.encode("utf-8"))
            self.ml_lsh0_7.insert(d[self.idx_column], mh)
            self.ml_lsh0_8.insert(d[self.idx_column], mh)
            self.ml_lsh0_9.insert(d[self.idx_column], mh)
            self.mh_dict[d[self.idx_column]] = mh

    @staticmethod
    def _word_encode(d, text_column):
        encode_list = []
        # 取分词后的词表
        _, words, _ = utils.split_word(d[text_column])
        for word in set(words):
            encode_list.append(word.encode("utf-8"))
        return encode_list

    def _update_mh_pool(self):
        # 初始化MinHashLSH类
        self._init_mhlsh()
        n_jobs = min(cpu_count(), len(self.data))
        # 利用多进程快速编码
        with WorkerPool(n_jobs=n_jobs, start_method="spawn") as pool:
            we_res: List[List[str]] = pool.map(
                self._word_encode,
                [{"d": d, "text_column": self.text_column} for d in self.data],
            )

        # 整合结果
        for we, d in zip(we_res, self.data):
            # 初始化MinHash类（为每个样本单独初始化MinHash类）
            mh = MinHash(num_perm=self.num_perm)
            for e in we:
                mh.update(e)
            self.ml_lsh0_7.insert(d[self.idx_column], mh)
            self.ml_lsh0_8.insert(d[self.idx_column], mh)
            self.ml_lsh0_9.insert(d[self.idx_column], mh)
            self.mh_dict[d[self.idx_column]] = mh

    def do_find(self, ml_lsh: MinHashLSH, single_sample: Dict[str, Any]):
        sim_idx = ml_lsh.query(self.mh_dict[single_sample[self.idx_column]])
        return [s_idx for s_idx in sim_idx if s_idx != single_sample[self.idx_column]]

    def do_process(self, single_sample: Dict[str, Any], only_mid_res: bool = False):
        res_mid = dict()
        res_mid["signature_sim0.7"] = self.do_find(self.ml_lsh0_7, single_sample)
        res_mid["signature_sim0.8"] = self.do_find(self.ml_lsh0_8, single_sample)
        res_mid["signature_sim0.9"] = self.do_find(self.ml_lsh0_9, single_sample)
        if only_mid_res:
            return res_mid
        else:
            return utils.cat_dict(single_sample, res_mid)

    def forward(self) -> List[Dict[str, Any]]:
        for idx in range(len(self.data)):
            self.data[idx] = self.do_process(self.data[idx])
        return self.data

    def forward_pool(self, num_proc: int) -> List[McDict[str, Any]]:
        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            self.data: List[McDict[str, Any]] = pool.map_unordered(
                self.do_process,
                [{"single_sample": d} for d in self.data],
                progress_bar=True
            )
        return self.data
