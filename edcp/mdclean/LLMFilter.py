from typing import List, Dict

from .template import LLMFilterPrompt as lfp
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatModel:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    @staticmethod
    def collate_prompt(
        book_name: str, context: List[str]
    ) -> List[List[Dict[str, str]]]:
        system: str = lfp.SYSTEM.format(book=book_name)
        prompt: List[str] = [
            lfp.PROMPT.format(book=book_name, context=c) for c in context
        ]
        message_batch = [
            [
                {
                    "role": "system",
                    "content": system,
                },
                {"role": "user", "content": p},
            ]
            for p in prompt
        ]
        return message_batch

    def text_encoder(
        self, message_batch: List[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs_batch = self.tokenizer(
            text_batch, return_tensors="pt", padding=True
        ).to(self.model.device)
        generated_ids_batch = self.model.generate(
            **model_inputs_batch,
            max_new_tokens=512,
        )
        generated_ids_batch = generated_ids_batch[
            :, model_inputs_batch.input_ids.shape[1] :
        ]
        return generated_ids_batch

    def forward(self, book_name: str, context: List[str]) -> List[str]:
        messages_batch = self.collate_prompt(book_name, context)
        generated_ids_batch = self.text_encoder(messages_batch)
        response = self.tokenizer.batch_decode(
            generated_ids_batch, skip_special_tokens=True
        )
        return response


if __name__ == "__main__":

    print(res)
