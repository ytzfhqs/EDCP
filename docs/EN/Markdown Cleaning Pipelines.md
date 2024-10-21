# Markdown Cleaning Pipelines

## Table of Contents

- [Preparing Markdown Files](#Preparing Markdown Files)
- [Downloading Model Weights](#Downloading Model Weights)
- [Configuration File](Configuration File)
- [Code File Explanation](#Code File Explanation)
- [Example Usage](#Example Usage)
- [Results Showcase](#Results Showcase)

## Preparing Markdown Files

- After converting each PDF file into a folder format using [MinerU](https://github.com/opendatalab/MinerU), we need to consolidate all the `markdown` files into a single folder. Here's a Python function to achieve this:

```python
import os
import shutil

def copy_md_files(search_folder, result_folder):
    # Ensure the result folder exists; create it if it doesn’t
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Traverse all files in the search folder
    for root, dirs, files in os.walk(search_folder):
        for file in files:
            if file.endswith('.md'):
                # Construct full file paths
                source_file = os.path.join(root, file)
                destination_file = os.path.join(result_folder, file)
                
                # Copy the file to the result folder
                shutil.copy2(source_file, destination_file)

# Example usage
copy_md_files('mineru_output', 'markdown_files')
```

- Suggested file structure:

```
└── edcp
  ├── mdclean
  └── metric
  └── ...
└── markdown_files
  └── test.md
└── ...
```

## Downloading Model Weights

- There are two methods for filtering out non-body text. Choose one of the following methods:

  1. To filter non-body text using a general LLM (e.g., `Qwen2.5-7B-Instruct`), based on a prompt template, you need to download the corresponding model weights from `HuggingFace` or `Modelscope`. We recommend using `git lfs` for the download:

     `HuggingFace`:

     ```bash
     git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
     ```

     `Modelscope`:

     ```bash
     git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
     ```

     > [!NOTE]
     >
     > - According to experiments, a general LLM requires at least `7B` parameters to ensure filtering effectiveness.
     > - The prompt template can be modified in [edcp/mdclean/template.py](https://github.com/ytzfhqs/EDCP/blob/main/edcp/mdclean/template.py), but ensure that `book` (book title) and `context` (context) are included.
     > - If you wish to use the `VLLM` framework to accelerate inference (Linux systems only), configure the environment by following the [official documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html). The code will automatically detect and enable the `VLLM` framework if installed.

  2. Alternatively, use our trained **medical-specific** classification model [[Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification) for filtering non-body text. Download using `git lfs`:

     `HuggingFace`:

     ```bash
     git lfs clone https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification
     ```

     > [!NOTE]
     >
     > - The `Qwen2.5-med-book-main-classification` model is only applicable in the medical field. It may not perform well in other domains.
     > - You need to download the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) from the general LLM model for later token count calculations when packing the text.

- Suggested file structure:

```
└── edcp
  ├── mdclean
  └── metric
  └── ...
└── markdown_files
  └── test.md
  └── ...
└── llm_model
  └── Qwen2.5-med-book-main-classification
    └── ...
  └── Qwen2.5-7B-Instruct
      └── tokenizer.json
      └── tokenizer_config.json
      └── vocab.json
      └── merges.txt
└── ...
```

## Configuration File

- This project uses a `yaml` configuration file for parameter passing. Refer to the [edcp/example/yaml/md_pipe.yaml](https://github.com/ytzfhqs/EDCP/blob/main/example/yaml/md_pipe.yaml) file for details.

```yaml
# MdProcess
## Directory containing markdown files to be processed
md_path: markdown_files
## Path to the LLM model for filtering non-body samples
llm_model_path: llm_model/Qwen2.5-med-book-main-classification
## Path to the LLM model used for token count statistics
token_cont_model: llm_model/Qwen2.5-7B-Instruct
## Approximate range of token count per sample
near_tokens: 1024
## Method for filtering non-body samples. "chat" for a general causal model, "cls" for a specialized classification model
process_method: cls
## Batch size
batch_size: 8
## Directory to save result files
save_path: output/data.json
## Whether to save intermediate result files
save_middle: true
```

> [!NOTE]
>
> - The format of the result file is `List[Dict[str, str]]`, e.g., `[{'text': 'Text1'},{'text': 'Text2'}]`, retaining only the body text, and packed according to the specified token count range.
> - The format of the intermediate result file is `List[Dict[str, str]]`, e.g., `[{"text": 'Text1', "res": 'True'},{"text": 'Text2', "res": 'False'}]`, retaining both body and non-body inference results (via the `res` key), without packing the text.

Suggested file structure:

```
└── edcp
  ├── mdclean
  └── metric
  └── ...
└── markdown_files
  └── test.md
  └── ...
└── llm_model
  └── Qwen2.5-med-book-main-classification
    └── ...
  └── Qwen2.5-7B-Instruct
      └── tokenizer.json
      └── tokenizer_config.json
      └── vocab.json
      └── merges.txt
└── yaml
  └── md_pipe.yaml
└── ...
```

## Code File Explanation

- The core code for cleaning `markdown` files can be found in the [edcp/mdclean](https://github.com/ytzfhqs/EDCP/tree/main/edcp/mdclean) folder:

  - `charreplace.py`: Contains the regular expressions and character sets used for text replacement operations.
  - `CLSFilter.py`: Inference process for the medical field body vs non-body classification model.
  - `LLMFilter.py`: Batch inference process using the `Transformers` framework.
  - `packing.py`: Packs text according to a specified number of tokens.
  - `VLLMFilter.py`: Batch inference process using the `VLLM` framework.
  - `pipelines.py`: Entry point for the processing pipeline.
  - `template.py`: Templates for the LLM filtering prompt.
  - `utils.py`: Contains common utility functions.

## Example Usage

- For an example usage, refer to [edcp/example/md_pipe_demo.py](https://github.com/ytzfhqs/EDCP/blob/main/example/md_pipe_demo.py). Below is a simple implementation:

```python
from typing import Literal
from loguru import logger
from transformers import HfArgumentParser
from edcp.mdclean.pipelines import MdProcess
from edcp.hparams.mdpipe_args import MdPipeArgs

def test_pipelines(
    md_path: str,
    llm_model_path: str,
    batch_size: int,
    save_path: str,
    token_cont_model: str,
    near_tokens: int,
    process_method: Literal["chat", "cls"] = "cls",
    save_middle: bool = True,
):
    mp = MdProcess(
        md_path, llm_model_path, token_cont_model, near_tokens, process_method
    )
    mp.forward(batch_size, save_path, save_middle)

if __name__ == "__main__":
    # Log the run
    logger.add("mdclean_run.log")
    parser = HfArgumentParser(MdPipeArgs)
    mdpipe_arg = parser.parse_yaml_file(
        "yaml/md_pipe.yaml", allow_extra_keys=True
    )[0]
    # Test the full pipeline
    test_pipelines(
        mdpipe_arg.md_path,
        mdpipe_arg.llm_model_path,
        mdpipe_arg.batch_size,
        mdpipe_arg.save_path,
        mdpipe_arg.token_cont_model,
        mdpipe_arg.near_tokens,
        mdpipe_arg.process_method,
        mdpipe_arg.save_middle,
    )
```

## Results Showcase

- The saved result file `data.json` looks like:

```json
[
    {
        "text": "儿童是社会中的弱势群体之一，儿童的健康对家庭乃至社会影响重大。儿童自出生至青少年阶段的生长发育过程中，来自家庭、社会、环境的不利因素时刻影响其健康。...,
        "id_int": 0
    },
    {
        "text": "与西方医学比较而言，我国的中医儿科起源要早得多，自扁鹊“为小儿医”以来已有2400余年，我国最早的儿科专著《颅区经》成书于唐末，首次阐述小儿为“纯阳之体”的理论，也是世界现存最早的儿科专著。...,
        "id_int": 1
    }
]
```

> [!TIP]
> If you want to continue using the pipeline in subsequent steps, try to keep the dictionary keys consistent with the result after this step.

The saved intermediate result file `data_middle.json` looks like:

```json
[
  {
    "text": "儿科学",
    "res": "False"
  },
  {
    "text": "Pediatrics",
    "res": "False"
  },
  {
    "text": "第10版",
    "res": "False"
  }
]
```