# Markdown Clean Pipelines

## 目录

- [Markdown文件准备](#Markdown文件准备)
- [下载模型权重](#下载模型权重)
- [配置文件](#配置文件)
- [代码文件说明](#代码文件说明)
- [使用示例](#使用示例)
- [结果展示](#结果展示)

## Markdown文件准备

- [MinerU](https://github.com/opendatalab/MinerU)转换后的每个PDF文件以文件夹形式存在，我们要先将`markdown`文件全部整合到一个文件夹中。这里提供一个`python`函数实现该功能：

```python
import os
import shutil

def copy_md_files(search_folder, result_folder):
    # 确保结果文件夹存在，不存在则创建
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 遍历搜索文件夹中的所有文件
    for root, dirs, files in os.walk(search_folder):
        for file in files:
            if file.endswith('.md'):
                # 构造完整文件路径
                source_file = os.path.join(root, file)
                destination_file = os.path.join(result_folder, file)
                
                # 复制文件到结果文件夹
                shutil.copy2(source_file, destination_file)

# 示例调用
copy_md_files('mineru_output', 'markdown_files')
```

- 建议的文件树组织形式：

```
└── edcp
  ├── mdclean
  └── metric
  └── ...
└── markdown_files
  └── test.md
└── ...
```

## 下载模型权重文件

- 过滤非正文文本有两种方式，选择其中一种方法即可：

  1. 若使用通用LLM（以`Qwen2.5-7B-Instruct`为例）根据提示词模版进行非正文文本过滤，需要下载对应模型权重，`HuggingFace`[模型主页](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)，`modelscope`[模型主页](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)，建议使用`git lfs`进行下载：

     `HuggingFace`：

     ```bash
     git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
     ```

     `modelscope`：

     ```bash
     git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
     ```

     > [!NOTE]
     >
     > - 根据实验，通用`LLM`需要至少`7B`以上参数量才能保证过滤效果。
     > - 提示词模版可以在[edcp/mdclean/template.py](https://github.com/ytzfhqs/EDCP/blob/main/edcp/mdclean/template.py)中进行更改，但请保证`book`（书名）与`context`（上下文）存在。
     > - 若想使用`VLLM`框架进行推理加速（仅`Linux`系统），请先按照[官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)配置好环境，代码会自动检测`VLLM`框架的安装情况，启用`VLLM`。

  2. 若使用我们训练的**医疗类**分类模型[[Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)进行非正文过滤，使用`git lfs`进行下载：

     `HuggingFace`：

     ```bash
     git lfs clone https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification
     ```

     > [!NOTE]
     >
     > - `Qwen2.5-med-book-main-classification`模型仅适用医疗领域，其他垂域不保证效果。
     > - 由于后期需要对`token`数进行统计以打包文本，所以需要下载通用`LLM`模型中的`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`文件。

- 建议的文件树组织形式：

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

## 配置文件

- 本项目采用`yaml`配置文件传参，具体可参照[edcp/example/yaml/md_pipe.yaml](https://github.com/ytzfhqs/EDCP/blob/main/example/yaml/md_pipe.yaml)文件。

```yaml
# MdProcess
## 包含需要处理markdown文件的主目录
md_path: markdown_files
## 用于过滤非正文样本的LLM路径
llm_model_path: llm_model/Qwen2.5-med-book-main-classification
## 用于统计token数量的LLM路径
token_cont_model: llm_model/Qwen2.5-7B-Instruct
## 单样本token数量的大致范围
near_tokens: 1024
## 用于过滤非正文样本的方法，chat为通用因果模型，cls为专用分类模型
process_method: cls
## 批处理大小
batch_size: 8
## 结果文件保存目录
save_path: output/data.json
## 是否保存中间结果文件
save_middle: true
```

> [!NOTE]
>
> - 结果文件的格式为`List[Dict[str, str]]`，如：`[{'text': '文本1'},{'text': '文本2'}]`，仅保留了正文，并且进行范围`token`数打包。
> - 中间结果的格式为`List[Dict[str, str]]`，如：`[{"text": '文本1', "res": 'True'},{"text": '文本2', "res": 'False'}]`，保留了正文与非正文推理结果（`res`键），并且未进行文本打包。

建议的文件树组织形式：

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

## 代码文件说明

- 对`markdown`文件清洗的核心代码主要在[edcp/mdclean](https://github.com/ytzfhqs/EDCP/tree/main/edcp/mdclean)文件夹下：

  - `charreplace.py`：需要对文本进行替换操作的正则表达式与字符库。
  - `CLSFilter.py`：医疗领域正文与非正文分类模型推理流程。

   - `LLMFilter.py`：`Transformers`框架批推理流程。

   - `packing.py`：对文本按尽可能接近的`token`数打包。

   - `VLLMFilter.py`：`VLLM`框架批推理流程。

   - `pipelines.py`：处理流程入口。

   - `template.py`：`LLM`过滤提示词模板。

   - `utils.py`：一些常用的工具函数。

## 使用示例

- 使用示例可参照[edcp/example/md_pipe_demo.py](https://github.com/ytzfhqs/EDCP/blob/main/example/md_pipe_demo.py)文件，以下是一个简单实现：

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
    # 记录日志文件
    logger.add("mdclean_run.log")
    parser = HfArgumentParser(MdPipeArgs)
    mdpipe_arg = parser.parse_yaml_file(
        "yaml/md_pipe.yaml", allow_extra_keys=True
    )[0]
    # 测试管道全流程
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

## 结果展示

- 保存的结果文件`data.json`形如：

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
> 如果想继续使用后面的管道请尽可能与该步骤结束后结果字典中`key`保持一致

- 保存的中间结果文件`data_middle.json`形如：

```python
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

