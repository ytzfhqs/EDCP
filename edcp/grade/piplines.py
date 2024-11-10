from typing import Dict, Any, Literal, Optional, List, Union


from ..tool import read_json, save_json
from ..metric.check_type import check_path_data
from ..metric import utils, McDict


class GradeProcess:
    def __init__(
        self,
        data_or_filepath: Union[str, List[Dict[str, Any]]],
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        prompt_type: Literal["domain", "general"] = "general",
        domain: Optional[str] = None,
        res_save_path: str = "data_grade.json",
        text_column: str = "text",
    ):
        if "qwen" in model_name.lower():
            from .chatmodel import ChatQwen

            self.model = ChatQwen(model_name, api_key, base_url, text_column)
        elif "glm" in model_name.lower():
            from .chatmodel import ChatGLM4

            self.model = ChatGLM4(model_name, api_key, text_column)
        elif "gpt" in model_name.lower():
            from .chatmodel import ChatGPT4

            self.model = ChatGPT4(model_name, api_key, text_column)
        else:
            raise ValueError(
                "Only api calls for Qwen, ChatGPT and ChatGLM series models are supported."
            )

        if check_path_data(data_or_filepath) == "str":
            self.data = read_json(data_or_filepath)
        elif check_path_data(data_or_filepath) == "data":
            self.data = data_or_filepath

        self.prompt_type = prompt_type
        self.domain = domain
        self.res_save_path = res_save_path

    def do_grade(self, single_sample: Dict[str, Any]) -> McDict[str, Any]:
        return utils.cat_dict(
            single_sample,
            self.model.do_process(
                single_sample, self.prompt_type, self.domain, only_mid_res=True
            ),
        )

    def forward(self):
        for idx in range(len(self.data)):
            self.data[idx] = self.do_grade(self.data[idx])
        save_json(self.res_save_path, self.data)
        return self.data
