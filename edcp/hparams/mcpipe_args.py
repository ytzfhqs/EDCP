from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional, Union


@dataclass
class McPipeArgs:
    llm_model_path: Optional[str] = field(
        default=None, metadata={"help": "LLM paths used to calculate PPL"}
    )
    book_data_or_path: Dict[str, str] = field(
        default=None,
        metadata={
            "help": "Book data variable or json file path for importance sampling."
        },
    )
    wordgram_model_path: List[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "Importance sampled book data trained wordgram model path, if this parameter is not None, book_data_or_path will be ignored."
        },
    )
    save_wordgram_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Only valid when book_data_or_path is passed in. Save the path to the wordgram model folder where the book data was trained, or none if None."
        },
    )
    fasttext_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Fasttext model paths for language recognition."},
    )
    text_column: str = field(
        default="text", metadata={"help": "The key to the corpus in the dictionary"}
    )
    idx_column: str = field(
        default="id_int",
        metadata={"help": "The key to the sample id in the dictionary"},
    )
    num_perm: int = field(
        default=128,
        metadata={
            "help": "Number of hash functions used to construct hashes in MinHash"
        },
    )
