import torch
from typing import List
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

ID2LABEL = {0: "正文", 1: "非正文"}
ID2BOOL = {0: "True", 1: "False"}


class QwenCLS:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # book_name变量是为了与其他方法调用形式一致，不可删减
    def forward(self, book_name: str, text: List[str]) -> List[str]:
        encoding = self.tokenizer(text, return_tensors="pt", padding=True)
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        outputs = self.model(**encoding)
        logits = outputs.logits
        ids = torch.argmax(logits, dim=-1).tolist()
        response = [ID2BOOL[id] for id in ids]
        return response
