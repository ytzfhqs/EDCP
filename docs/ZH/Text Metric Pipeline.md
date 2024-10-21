# Text Metric Pipeline

## 目录

- [下载模型权重文件](#下载模型权重文件)
- [配置文件](#配置文件)
- [代码文件说明](#代码文件说明)
- [使用示例](#使用示例)
- [结果展示](#结果展示)

## 下载模型权重文件

- 在文本数据测评管道中，需要下载两个模型权重，分别是用于识别中文、英文的基于`Fasttext`的模型[fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification)和用于计算`PPL`（困惑度）的通用`LLM`模型。

  - [fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification)模型使用通用数据集与医疗领域数据集共同训练，提高了其在医疗领域的准确率（其他垂域不保证效果），可以使用`git lfs`下载模型权重：

  ```bash
  git lfs clone https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification
  ```

  - 用于计算`PPL`的通用`LLM`模型要选择`Base`（基座模型），不要使用`Instruct`、`Chat`（指令微调模型），这里以`Qwen2.5-7B`为例，使用`git lfs`下载模型权重：

  `HuggingFace`：

  ```bash
  git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B
  ```

  `modelscope`：

  ```bash
  git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-7B.git
  ```


- 建议的文件树组织形式：

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

## 配置文件

- 本项目采用`yaml`配置文件传参，具体可参照[edcp/example/yaml/mc_pipe.yaml](https://github.com/ytzfhqs/EDCP/blob/main/example/yaml/md_pipe.yaml)文件。

```yaml
## 需要进行评分的文本json文件路径
data_or_filepath: None

## 用于重要性采样的json文件路径
book_data_or_path:
  - name: msdmanuals
    file_path: spider_med/msdmanuals.json

## 重要性采样书籍数据训练的wordgram模型路径，传入此参数时，book_data_or_path将被忽略
wordgram_model_path:
  - name: msdmanuals(2-gram)
    model_path: wordgram_model/msdmanuals(2-gram).pkl
  - name: msdmanuals(3-gram)
    model_path: wordgram_model/msdmanuals(3-gram).pkl

## 仅在传入book_data_or_path时有效。保存书籍数据训练的wordgram模型路径，若为None，则不保存
save_wordgram_model_path: example/wordgram_model
## 用于计算PPL的通用领域模型路径
llm_model_path: edcp/Qwen2.5-7B-Instruct
## 用于识别语言的Fasttext模型路径
fasttext_model_path: edcp/fasttext-med-en-zh-identification/model.bin
## Dict中文本的Key
text_column: text
## Dict中样本ID的Key
idx_column: id_int
## MinHash中构造哈希值所用哈希函数个数
num_perm: 256
## 结果文件保存路径
res_save_path: 'data_metric.json'
```

- 参数解释：

  - `data_or_filepath`：需要进行评分的文本数据（`json`文件）路径。`json`文件形如：

  ```json
  [
      {"text":"文本1"},
      {"text":"文本2"},
      ...
  ]
  ```

  - `book_data_or_path`：用于重要性采样的文本数据（`json`文件）路径。在`yaml`中是以`List[Dict[str, str]]`格式组织的。字典中的`name`代表数据名称为`name`，用于构造评分指标的`key`。`file_path`代表`name`的文本数据（`json`文件）路径，`json`文件中必须带有键`text`，形如：

  ```json
  [
      {"text":"文本1"},
      {"text":"文本2"},
      ...
  ]
  ```

  - `wordgram_model_path`：重要性采样书籍数据训练的`word gram`模型路径，传入此参数时，`book_data_or_path`将被忽略。在`yaml`中是以`List[Dict[str, str]]`格式组织的。字典中的`name`代表模型名称为`name`，用于构造评分指标的`key`。`model_path`代表使用`name`数据构建的`word gram`模型路径，文件后缀一般为`.pkl`。
  - `save_wordgram_model_path`：仅在传入`book_data_or_path`时有效。保存书籍数据训练的`word gram`模型路径，若为`None`，则不保存。
  - `llm_model_path`：用于计算`PPL`的通用领域模型权重路径。
  - `fasttext_model_path`：用于识别中英文的`Fasttext`模型路径。
  - `text_column`：进行评分的文本数据，文本语料所在的`key`。
  - `idx_column`：进行评分的文本数据，文本`ID`所在的`key`。
  - `num_perm`：`MinHash`中构造哈希值所用哈希函数个数，该值越大，结果越精确，但也会显著增加计算量。
  - `res_save_path`：结果文件保存路径。

## 代码文件说明

- 对文本数据进行评分的核心代码主要在[edcp/metric](https://github.com/ytzfhqs/EDCP/tree/main/edcp/metric)文件夹下：
  - `calppl.py`：计算`LLM`对文本`PPL`的管道。
  - `check_type.py`：利用`pydantic`库进行类型检查的数据类。
  - `importance.py`：计算文本在不同语料上的重要性采样指标的管道。
  - `language.py`：利用`Fasttext`模型识别文本语言的管道。
  - `mcdict.py`：重写`Dict`类，保证`key`按一定规则进行排序
  - `minhash.py`：计算文本间`MinHash`相似度的管道
  - `nlpfeat.py`：计算文本的简单NLP特征，如唯一词个数
  - `pipelines.py`：整合指标计算类，搭建整体计算管道。
  - `utils.py`：常用工具函数。

## 文本质量量化表

| 指标名                                      | 含义                                             | 实现文件        |
| ------------------------------------------- | ------------------------------------------------ | --------------- |
| `chars_dupe_2grams`                         | 文本`2grams`中重复词的字符占比                   | `nlpfeat.py`    |
| `chars_dupe_3grams`                         | 文本`3grams`中重复词的字符占比                   | `nlpfeat.py`    |
| `chars_dupe_4grams`                         | 文本`4grams`中重复词的字符占比                   | `nlpfeat.py`    |
| `chars_dupe_5grams`                         | 文本`5grams`中重复词的字符占比                   | `nlpfeat.py`    |
| `chars_dupe_6grams`                         | 文本`6grams`中重复词的字符占比                   | `nlpfeat.py`    |
| `chars_dupe_7grams`                         | 文本`7grams`中重复词的字符占比                   | `nlpfeat.py`    |
| `chars_top_2grams`                          | 文本`2grams`中频率最高的grams占比                | `nlpfeat.py`    |
| `chars_top_3grams`                          | 文本`3grams`中频率最高的grams占比                | `nlpfeat.py`    |
| `chars_top_4grams`                          | 文本`4grams`中频率最高的grams占比                | `nlpfeat.py`    |
| `stop_radio`                                | 文本停用词占分词列表比例                         | `nlpfeat.py`    |
| `puncs_radio`                               | 文本标点符号占分词列表比例                       | `nlpfeat.py`    |
| `word_unique_radio`                         | 文本唯一词占分词列表比例                         | `nlpfeat.py`    |
| `num_sentences`                             | 文本句子个数                                     | `nlpfeat.py`    |
| `word_entropy`                              | 文本词的信息熵                                   | `nlpfeat.py`    |
| `is_ending_with_terminal_punctution`        | 文本是否以结束标点符号终止                       | `nlpfeat.py`    |
| `mean_word_length`                          | 文本词平均长度                                   | `nlpfeat.py`    |
| `curly_bracket`                             | 文本括号出现次数与文本字符数比例                 | `nlpfeat.py`    |
| `word_count`                                | 文本字数                                         | `nlpfeat.py`    |
| `num_words`                                 | 文本词数                                         | `nlpfeat.py`    |
| `llm_ppl`                                   | `LLM`困惑度                                      | `calppl.py`     |
| `signature_sim0.7`                          | 与该样本`MinHash`相似度大于`0.7`的样本`id`       | `minhash.py`    |
| `signature_sim0.8`                          | 与该样本`MinHash`相似度大于`0.8`的样本`id`       | `minhash.py`    |
| `signature_sim0.9`                          | 与该样本`MinHash`相似度大于`0.9`的样本`id`       | `minhash.py`    |
| `language`                                  | 文本语言                                         | `language.py`   |
| `prop`                                      | 文本语言置信度                                   | `language.py`   |
| `Importance_sample_with_msdmanuals(2-gram)` | 基于`msdmanuals`训练的`word 2gram`模型重要性采样 | `Importance.py` |
| `Importance_sample_with_msdmanuals(3-gram)` | 基于`msdmanuals`训练的`word 3gram`模型重要性采样 | `Importance.py` |

## 使用示例

使用示例可参照[edcp/example/mc_pipe_demo.py](https://github.com/ytzfhqs/EDCP/blob/main/example/mc_pipe_demo.py)文件，以下是一个简单实现：

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
    # 记录日志文件
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

## 结果展示

- 保存的结果文件`data_metric.json`形如：

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
> 如果想继续使用后面的管道请尽可能与该步骤结束后结果字典中`key`保持一致
