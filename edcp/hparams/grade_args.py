from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional, Union


@dataclass
class GradeArgs:
    data_or_filepath: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the text json file that needs to be scored by llm."},
    )
    model_name: Optional[str] = field(default=None, metadata={"help": "Name of llm."})
    api_key: Optional[str] = field(default=None, metadata={"help": "api key of llm."})
    base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Calling the base_ur that is required to be provided for the Qwen family of models."
        },
    )
    prompt_type: Literal["domain", "general"] = field(
        default=None,
        metadata={
            "help": "Prompt type, options are 'domain' (domain class, domain is required), 'general' (general class, domain is None)."
        },
    )
    domain: Optional[str] = field(default=None, metadata={"help": "Domain Nouns."})
    res_save_path: str = field(
        default=None, metadata={"help": "Path to save the result file."}
    )
    text_column: str = field(default=None, metadata={"help": "Key of text in Dict."})
