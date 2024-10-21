# Text Metric Pipeline

## Table of Contents

- [Download Model Weights](#download-model-weights)
- [Configuration File](#configuration-file)
- [Code File Explanation](#code-file-explanation)
- [Usage Example](#usage-example)
- [Results Display](#results-display)

## Download Model Weights

- In this text data evaluation pipeline, you need to download two model weights: a `Fasttext`-based model for identifying Chinese and English texts, and a general-purpose `LLM` model for computing `PPL` (Perplexity).

  - The [fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification) model, trained on general and medical datasets, improves its accuracy in the medical domain (results in other domains are not guaranteed). You can download the model weights using `git lfs`:

  ```bash
  git lfs clone https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification
  ```

  - For calculating `PPL`, you should choose a `Base` model, not an `Instruct` or `Chat` (instruction-tuned) model. For example, to use `Qwen2.5-7B`, download the model weights with `git lfs`:

  From `HuggingFace`:

  ```bash
  git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B
  ```

  From `modelscope`:

  ```bash
  git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-7B.git
  ```

- Suggested file structure:

```
└── edcp
  ├── mdclean
  └── metric
  └── ...
└── llm_model
  └── fasttext-med-en-zh-identification
  └── Qwen2.5-7B
└── ...
```

## Configuration File

- The project uses a `yaml` configuration file for parameter passing. Refer to the [edcp/example/yaml/mc_pipe.yaml](https://github.com/ytzfhqs/EDCP/blob/main/example/yaml/md_pipe.yaml) file for details.

```yaml
## Path to the JSON file with text data for scoring
data_or_filepath: None

## Path to the JSON file for importance sampling
book_data_or_path:
  - name: msdmanuals
    file_path: spider_med/msdmanuals.json

## Path to the wordgram model for importance sampling (when provided, book_data_or_path will be ignored)
wordgram_model_path:
  - name: msdmanuals(2-gram)
    model_path: wordgram_model/msdmanuals(2-gram).pkl
  - name: msdmanuals(3-gram)
    model_path: wordgram_model/msdmanuals(3-gram).pkl

## Only valid if book_data_or_path is provided. Path to save the wordgram model; if None, the model won't be saved
save_wordgram_model_path: example/wordgram_model
## Path to the general-purpose model for calculating PPL
llm_model_path: edcp/Qwen2.5-7B-Instruct
## Path to the Fasttext model for language identification
fasttext_model_path: edcp/fasttext-med-en-zh-identification/model.bin
## Key for the text in the JSON dict
text_column: text
## Key for the sample ID in the JSON dict
idx_column: id_int
## Number of hash functions used to construct MinHash signatures
num_perm: 256
## Path to save the result file
res_save_path: 'data_metric.json'
```

### Parameter Explanation:

  - `data_or_filepath`: Path to the JSON file containing the text data for scoring. The JSON file should look like:

  ```json
  [
      {"text":"Text 1"},
      {"text":"Text 2"},
      ...
  ]
  ```

  - `book_data_or_path`: Path to the JSON file containing the text data for importance sampling. In the `yaml` file, it is organized as `List[Dict[str, str]]`. The `name` key refers to the name of the data, and the `file_path` key refers to the path of the JSON file containing the data, which must include a `text` key, like this:

  ```json
  [
      {"text":"Text 1"},
      {"text":"Text 2"},
      ...
  ]
  ```

  - `wordgram_model_path`: Path to the wordgram model for importance sampling. When this parameter is provided, `book_data_or_path` will be ignored. In the `yaml` file, it is organized as `List[Dict[str, str]]`. The `name` key refers to the model name, and `model_path` refers to the path of the wordgram model built from that dataset, typically with a `.pkl` extension.
  - `save_wordgram_model_path`: Valid only if `book_data_or_path` is provided. Path to save the wordgram model; if set to `None`, the model will not be saved.
  - `llm_model_path`: Path to the general-purpose model used to calculate `PPL`.
  - `fasttext_model_path`: Path to the Fasttext model used to identify Chinese and English.
  - `text_column`: Key for the text data to be scored.
  - `idx_column`: Key for the text data sample ID.
  - `num_perm`: Number of hash functions used to construct MinHash signatures. A larger value results in more accuracy but also increases computation.
  - `res_save_path`: Path to save the result file.

## Code File Explanation

- The core code for text data scoring is in the [edcp/metric](https://github.com/ytzfhqs/EDCP/tree/main/edcp/metric) folder:
  - `calppl.py`: Pipeline for calculating the `LLM` perplexity (PPL) of text.
  - `check_type.py`: Data class for type checking using the `pydantic` library.
  - `importance.py`: Pipeline for calculating the importance sampling metrics for the text.
  - `language.py`: Pipeline for identifying the language of text using the `Fasttext` model.
  - `mcdict.py`: Redefines the `Dict` class to ensure that keys are sorted according to certain rules.
  - `minhash.py`: Pipeline for calculating MinHash similarity between texts.
  - `nlpfeat.py`: Pipeline for calculating simple NLP features of the text, such as unique word count.
  - `pipelines.py`: Integrates the various metrics calculation classes and sets up the overall processing pipeline.
  - `utils.py`: Common utility functions.

## Text Quality Metrics Table

| Metric Name                                 | Description                                                  | Implementation File |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------- |
| `chars_dupe_2grams`                         | Percentage of duplicated 2-grams in the text                 | `nlpfeat.py`        |
| `chars_dupe_3grams`                         | Percentage of duplicated 3-grams in the text                 | `nlpfeat.py`        |
| `chars_dupe_4grams`                         | Percentage of duplicated 4-grams in the text                 | `nlpfeat.py`        |
| `chars_dupe_5grams`                         | Percentage of duplicated 5-grams in the text                 | `nlpfeat.py`        |
| `chars_dupe_6grams`                         | Percentage of duplicated 6-grams in the text                 | `nlpfeat.py`        |
| `chars_dupe_7grams`                         | Percentage of duplicated 7-grams in the text                 | `nlpfeat.py`        |
| `chars_top_2grams`                          | Percentage of the most frequent 2-grams in the text          | `nlpfeat.py`        |
| `chars_top_3grams`                          | Percentage of the most frequent 3-grams in the text          | `nlpfeat.py`        |
| `chars_top_4grams`                          | Percentage of the most frequent 4-grams in the text          | `nlpfeat.py`        |
| `stop_radio`                                | Ratio of stop words in the word list                         | `nlpfeat.py`        |
| `puncs_radio`                               | Ratio of punctuation marks in the word list                  | `nlpfeat.py`        |
| `word_unique_radio`                         | Ratio of unique words in the word list                       | `nlpfeat.py`        |
| `num_sentences`                             | Number of sentences in the text                              | `nlpfeat.py`        |
| `word_entropy`                              | Word entropy of the text                                     | `nlpfeat.py`        |
| `is_ending_with_terminal_punctution`        | Whether the text ends with a terminal punctuation            | `nlpfeat.py`        |
| `mean_word_length`                          | Average word length in the text                              | `nlpfeat.py`        |
| `curly_bracket`                             | Ratio of curly brackets to total characters                  | `nlpfeat.py`        |
| `word_count`                                | Total number of characters in the text                       | `nlpfeat.py`        |
| `num_words`                                 | Total number of words in the text                            | `nlpfeat.py`        |
| `llm_ppl`                                   | Perplexity (PPL) calculated by the `LLM`                     | `calppl.py`         |
| `signature_sim0.7`                          | Sample `ids` with `MinHash` similarity greater than `0.7` to this sample | `minhash.py`        |
| `signature_sim0.8`                          | Sample `ids` with `MinHash` similarity greater than `0.8` to this sample | `minhash.py`        |
| `signature_sim0.9`                          | Sample `ids` with `MinHash` similarity greater than `0.9` to this sample | `minhash.py`        |
| `language`                                  | language                                                     | `language.py`       |
| `prop`                                      | language confidence                                          | `language.py`       |
| `Importance_sample_with_msdmanuals(2-gram)` | Importance sampling of `word 2gram` models based on `msdmanuals` training | `Importance.py`     |
| `Importance_sample_with_msdmanuals(3-gram)` | Importance sampling of `word 3gram` models based on `msdmanuals` training | `Importance.py`     |

## Usage Example

You can refer to the [edcp/example/mc_pipe_demo.py](https://github.com/ytzfhqs/EDCP/blob/main/example/mc_pipe_demo.py) file for an example. Below is a simple implementation:

```python
from typing import List, Dict, Any


from transformers import HfArgumentParser
from edcp.hparams.mcpipe_args import McPipeArgs
from edcp.metric.calppl import CPPl
from edcp.metric.nlpfeat import NlpFeat
from edcp.metric.pipelines import MetricProcess
from edcp.metric.importance import ImportFeat
from loguru import logger


def test_pipelines(
    book_data_or_path,
    wordgram_model_path,
    save_wordgram_model_path,
    llm_model_path,
    fasttext_model_path,
    text_column,
    idx_column,
    num_perm,
    res_save_path
):
    data = [
        {"text": "你好啊，我叫小松鼠。你好啊，我叫小雪球。", "id_int": 0},
        {"text": "你好啊，我叫松鼠。你好啊，我叫雪球。", "id_int": 1},
        {"text": "你好啊，我叫小松鼠。", "id_int": 2},
    ]
    mcp = MetricProcess(
        data,
        book_data_or_path,
        wordgram_model_path,
        save_wordgram_model_path,
        llm_model_path,
        fasttext_model_path,
        text_column,
        idx_column,
        num_perm,
        res_save_path
    )
    print(mcp.forward())


if __name__ == "__main__":
    # Log the process
    logger.add("metric_run.log")
    parser = HfArgumentParser(McPipeArgs)
    mcpipe_arg = parser.parse_yaml_file(
        "example/yaml/mc_pipe.yaml", allow_extra_keys=True
    )[0]
    test_pipelines(
        mcpipe_arg.book_data_or_path,
        mcpipe_arg.wordgram_model_path,
        mcpipe_arg.save_wordgram_model_dir,
        mcpipe_arg.llm_model_path,
        mcpipe_arg.fasttext_model_path,
        mcpipe_arg.text_column,
        mcpipe_arg.idx_column,
        mcpipe_arg.num_perm,
        mcpipe_arg.res_save_path
    )
```

## Results Display

- The saved result file `data_metric.json` will look like this:

```json
[
    {
        "text": "你好啊，我叫小松鼠。你好啊，我叫小雪球。", 
        "id_int": 0, 
        "chars_dupe_2grams": 0.5, 
        "chars_dupe_3grams": 0.0, 
        "chars_dupe_4grams": 0.0, 
        "chars_dupe_5grams": 0.0, 
        "chars_dupe_6grams": 0.0, 
        "chars_dupe_7grams": 0.0, 
        "chars_dupe_8grams": 0.0, 
        "chars_top_2grams": 0.5, 
        "chars_top_3grams": 0.0, 
        "chars_top_4grams": 0.0, 
        "stop_radio": 0.6667, 
        "puncs_radio": 0.3333, 
        "word_unique_radio": 0.3333, 
        "num_sentences": 2, 
        "word_entropy": 2.9183, 
        "is_ending_with_terminal_punctution": True, 
        "mean_word_length": 1.3333, 
        "curly_bracket": 0, 
        "word_count": 16,
        "num_words": 6,
        "llm_ppl": 151.05621337890625, 
        "signature_sim0.7": [1, 2], 
        "signature_sim0.8": [1, 2], 
        "signature_sim0.9": [1, 2], 
        "language": "zh", 
        "prop": 1.0, 
        "Importance_sample_with_msdmanuals(2-gram)": -119.4148, 
        "Importance_sample_with_msdmanuals(3-gram)": -120.452
    }
    ...
]
```

> [!TIP]
> If you want to continue using further processing pipelines, try to keep the keys consistent with the results dictionary after this step.