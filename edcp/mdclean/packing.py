from typing import List
from transformers import AutoTokenizer


class PackText:
    def __init__(self, model_path: str, max_tokens: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_tokens = max_tokens

    def tokens_cont(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def to_max_tokens(self, content: List[str]) -> List[str]:
        max_token_list = []
        temp_str = ""
        i = 0
        while i < len(content):
            temp_str = temp_str + content[i]
            if (
                len(temp_str) >= self.max_tokens
                and self.tokens_cont(temp_str) >= self.max_tokens
            ):
                max_token_list.append(temp_str)
                temp_str = ""
                i = i + 1
            else:
                i = i + 1
            if i == len(content) and temp_str != "":
                max_token_list.append(temp_str)
        return max_token_list

    def forward(self, text_list: List[str]):
        return self.to_max_tokens(text_list)
