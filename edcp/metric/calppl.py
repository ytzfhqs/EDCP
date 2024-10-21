from typing import List, Dict, Any


import torch
from mpire import WorkerPool
from . import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from .mcdict import McDict
from loguru import logger


class CPPl:
    def __init__(self, llm_model_path: str, text_column: str = "text"):
        """
        初始化PPL计算管道
        Args:
            llm_model_path: 用于计算PPL的LLM路径
            text_column: 字典中语料的key
        """
        logger.info('Starts initialising the CPPl pipeline.')
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        logger.info('LLM Model pre-training weight and tokenizer loading is complete.')
        self.text_column = text_column

    def calculate_ppl(self, single_sample: Dict[str, Any]) -> float:
        text = single_sample[self.text_column]
        encoders = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encoders.input_ids.to(self.model.device)
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        return torch.exp(neg_log_likelihood).cpu().item()

    def do_process(self, single_sample: Dict[str, Any], only_mid_res: bool = False):
        res_mid = {"llm_ppl": self.calculate_ppl(single_sample)}
        if only_mid_res:
            return res_mid
        else:
            return utils.cat_dict(single_sample, res_mid)

    def forward(self, data: List[Dict[str, Any]]):
        for idx in range(len(data)):
            data[idx] = self.do_process(data[idx])
        return data

    def forward_pool(self, data: List[Dict[str, Any]], num_proc: int) -> List[McDict[str, Any]]:
        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            ppl_res: List[Dict[str, float]] = pool.map(
                self.do_process, data, progress_bar=True
            )
        data = utils.cat_dict_with_pool(data, ppl_res)
        return data
