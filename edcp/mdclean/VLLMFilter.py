from typing import List

from .LLMFilter import ChatModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VLLMChatModel(ChatModel):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
        )
        self.llm = LLM(model=model_path)

    def forward(self, book_name: str, context: List[str]) -> List[str]:
        response: List[str] = []
        message_batch = self.collate_prompt(book_name, context)
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.llm.generate(text_batch, self.sampling_params, use_tqdm=False)
        for output in outputs:
            generated_text = output.outputs[0].text
            response.append(generated_text)
        return response