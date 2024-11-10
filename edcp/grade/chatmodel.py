import re
from typing import Literal, Optional, Dict, Any, List, Tuple


from mpire import WorkerPool
from .template import LLMGradePrompt
from ..metric import utils, McDict


class BaseChat:
    def __init__(self, model_name: str, text_column: str = "text"):
        self.model_name = model_name
        self.text_column = text_column

    @staticmethod
    def make_prompt(
        context: str,
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        **kwargs
    ) -> str:
        assert prompt_type in [
            "domain",
            "general",
        ], "The prompt_type parameter must be 'domain' or 'general'."
        if prompt_type == "domain":
            assert (
                domain
            ), "The domain parameter must be passed to the domain type prompt."
            prompt = LLMGradePrompt.DOMAIN_PROMPT.format(domain=domain, context=context)
        else:
            prompt = LLMGradePrompt.GENERAL_PROMPT.format(context=context)
        return prompt

    @staticmethod
    def ex_score_and_reason(
        model_res: str, prompt_type: str, **kwargs
    ) -> Tuple[str, int]:
        if prompt_type == "domain":
            reason: str
            score_str: str
            score: int
            reason, score_str = model_res.split("得分：")
            reason = reason.split("。")[0] + "。"
            score = re.findall(r"\d+", score_str)[0]
            return reason, score
        else:
            reason: str
            score_str: str
            score: int
            reason, score_str = model_res.split("\n\nQuality score: ")
            score = int(score_str)
            return reason, score

    def do_process(
        self,
        single_sample: Dict[str, Any],
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        only_mid_res: bool = False,
        **kwargs
    ) -> McDict[str, Any]:
        """
        单样本处理函数
        Args:
            single_sample: 待处理的样本
            prompt_type: 提示词类型，可选领域类（domain）和通用类（general）
            domain: 当提示词为领域类（domain）时，必须传入该参数
            only_mid_res: 是否仅返回指标结果（不包含原样本）

        Returns:

        """
        pass

    def forward(
        self,
        data: List[Dict[str, Any]],
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        for idx in range(len(data)):
            data[idx] = self.do_process(data[idx], prompt_type, domain)
        return data

    def forward_pool(
        self,
        data: List[Dict[str, Any]],
        num_proc: int,
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        **kwargs
    ) -> List[McDict[str, Any]]:
        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            glm4_score_res: List[Dict[str, float]] = pool.map(
                self.do_process,
                [
                    {"single_sample": d, "prompt_type": prompt_type, "domain": domain}
                    for d in data
                ],
                progress_bar=True,
            )
        data = utils.cat_dict_with_pool(data, glm4_score_res)
        return data


class ChatGLM4(BaseChat):
    def __init__(self, model_name: str, api_key: str, text_column: str = "text"):
        from zhipuai import ZhipuAI

        self.client = ZhipuAI(api_key=api_key)
        super().__init__(model_name, text_column)

    def do_process(
        self,
        single_sample: Dict[str, Any],
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        only_mid_res: bool = False,
        **kwargs
    ):
        context: str = single_sample[self.text_column]
        prompt: str = self.make_prompt(context, prompt_type, domain)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        reason, score = self.ex_score_and_reason(
            response.choices[0].message.content, prompt_type
        )
        res_mid = {"glm4_score": score, "glm4_reason": reason}
        if only_mid_res:
            return res_mid
        else:
            res = utils.cat_dict(single_sample, res_mid)
            return res


class ChatQwen(BaseChat):
    def __init__(
        self, model_name: str, api_key: str, base_url: str, text_column: str = "text"
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        super().__init__(model_name, text_column)

    def do_process(
        self,
        single_sample: Dict[str, Any],
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        only_mid_res: bool = False,
        **kwargs
    ):
        context: str = single_sample[self.text_column]
        prompt: str = self.make_prompt(context, prompt_type, domain)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        reason, score = self.ex_score_and_reason(
            response.choices[0].message.content, prompt_type
        )
        res_mid = {"qwen_score": score, "qwen_reason": reason}
        if only_mid_res:
            return res_mid
        else:
            res = utils.cat_dict(single_sample, res_mid)
            return res


class ChatGPT4(BaseChat):
    def __init__(self, model_name: str, api_key: str, text_column: str = "text"):
        from openai import OpenAI

        self.client = OpenAI()
        self.api_key = api_key
        super().__init__(model_name, text_column)

    def do_process(
        self,
        single_sample: Dict[str, Any],
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        only_mid_res: bool = False,
        **kwargs
    ) -> McDict[str, Any]:
        context: str = single_sample[self.text_column]
        prompt: str = self.make_prompt(context, prompt_type, domain)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        reason, score = self.ex_score_and_reason(
            response.choices[0].message.content, prompt_type
        )
        res_mid = {"gpt4_score": score, "gpt4_reason": reason}
        if only_mid_res:
            return res_mid
        else:
            res = utils.cat_dict(single_sample, res_mid)
            return res


class ChatQwen2_5(BaseChat):
    def __init__(self, model_name: str, text_column: str = "text"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_column = text_column

    def do_process(
        self,
        single_sample: Dict[str, Any],
        prompt_type: Literal["domain", "general"],
        domain: Optional[str] = None,
        only_mid_res: bool = False,
        **kwargs
    ):
        context: str = single_sample[self.text_column]
        prompt: str = self.make_prompt(context, prompt_type, domain)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        reason, score = self.ex_score_and_reason(
            response.choices[0].message.content, prompt_type
        )
        res_mid = {"Qwen2_5_score": score, "Qwen2_5_reason": reason}
        if only_mid_res:
            return res_mid
        else:
            res = utils.cat_dict(single_sample, res_mid)
            return res
