from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MdPipeArgs:
    md_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The home directory where the markdown files need to be processed."
        },
    )
    llm_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "LLM paths for filtering non-text samples."},
    )
    token_cont_model: Optional[str] = field(
        default=None,
        metadata={"help": "LLM path for counting the number of tokens."},
    )
    near_tokens: int = field(
        default=1024, metadata={"help": "Approximate range of token counts."}
    )
    process_method: Literal["chat", "cls"] = field(
        default="cls",
        metadata={
            "help": "Methods used to filter non-text samples, chat for generic causal models and cls for specialised classification models"
        },
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Enter batch size during filtering operation."},
    )
    save_path: str = field(
        default="data.json",
        metadata={"help": "Result file save directory."},
    )
    save_middle: bool = field(
        default=False,
        metadata={"help": "Whether to save intermediate result files."},
    )
