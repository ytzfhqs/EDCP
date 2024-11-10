# Text Grade Pipelines

## 目录

- [配置文件](#配置文件)
- [代码文件说明](#代码文件说明)
- [结果展示](#结果展示)

## 配置文件

本管道采用`yaml`配置文件传参，具体可参照[example/yaml/grade_pipe.yaml](https://github.com/ytzfhqs/EDCP/blob/main/example/yaml/grade_pipe.yaml)文件。

```yaml
## 需要进行llm评分的文本json文件路径
data_or_filepath: None

## llm的名称
model_name: qwen-plus
## llm的api key
api_key: None
## 调用Qwen系列模型所需提供的base_url
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
## 提示词种类，可选项为'domain'（领域类，需要提供domain）, 'general'（通用类，domain为None）
prompt_type: domain
## 领域名词
domain: 医学
## 保存结果文件的路径
res_save_path: data_grade.json
## Dict中文本的Key
text_column: text
```

- 参数解释：

  - `data_or_filepath`：需要进行评分的文本数据（`json`文件）路径。`json`文件形如：

  ```python
  [
      {"text":"文本1"},
      {"text":"文本2"},
      ...
  ]
  ```

  - `model_name`：调用模型的名称或路径。若调用的为闭源`LLM`的`api`，如`qwen-plus`，则为模型的名称。若调用的为开源`LLM`，如`Qwen2.5-7B-Instruct`，则为模型所在路径。
  - `api_key`：`LLM`的`api key`，若为开源`LLM`，可将该参数置为`None`。
  - `base_url`：调用`Qwen`系列闭源模型所需提供的`base_url`，其他系列模型不需要，可将该参数置为`None`。
  - `prompt_type`：提示词种类。可选项为`domain`（领域类，需要提供`domain`参数）, `general`（通用类，`domain`为`None`）。提示词可参考[edcp/grade/template.py](https://github.com/ytzfhqs/EDCP/blob/main/edcp/grade/template.py)文件。
  - `domain`：领域名词。仅在`prompt_type`参数为`domain`时有效，其他情况可置为`None`。
  - `res_save_path`：保存结果文件的路径。
  - `text_column`：进行评分的文本数据，文本语料所在的`key`。

## 代码文件说明

- 利用`LLM`和`prompt`对文本进行评分的核心代码主要在[edcp/grade](https://github.com/ytzfhqs/EDCP/tree/main/edcp/metric)文件夹下：
  - `chatmodel.py`：开源与闭源`LLM`调用过程、提示词构建、结果提取。
  - `piplines.py`：整合`LLM`调用，搭建评分管道。
  - `templaye.py`：`LLM`评分提示词模板。

## 使用示例

- 使用示例可参照[edcp/example/grade_pipe_demo.py](https://github.com/ytzfhqs/EDCP/blob/main/example/grade_pipe_demo.py)文件，以下是一个简单实现：

```python
from typing import Optional, Literal


from loguru import logger
from transformers import HfArgumentParser
from edcp.hparams import GradeArgs
from edcp.grade import GradeProcess


def test_pipelines(
    model_name: str,
    api_key: str,
    base_url: Optional[str],
    prompt_type: Literal["domain", "general"],
    domain: Optional[str],
    res_save_path: str = "data_grade.json",
):
    data = [
        {"text": "你好啊，我叫小松鼠。你好啊，我叫小雪球。"},
        {
            "text": "临床药理学（Clinical Pharmacology）作为药理学科的分支，是研究药物在人体内作用规律和入体与药物间相互作用过程的交叉学科。它以药理学与临床医学为基础，阐述药物代谢动力学（药动学）、药物效应动力学（药效学）、毒副反应的性质及药物相互作用的规律等；其目的是促进医学与药学的结合、基础与临床的结合，以及指导临床合理用药，推动医学与药学的共同发展。目前，临床药理学的主要任务是通过对千血药浓度的监测，不断调整给药方案，为患者能够安全有效地使用药物提供保障，同时对新药的有效性与安全性做出科学评价；对上市后药品的不良反应进行监测，以保障患者用药安全；临床合理使用药物，改善治疗。因此，临床药理学被认为是现代医学及教学与科研中不可或缺的一门学科。随着循证和转化医学概念的提出，临床药理学的内涵得到更进一步的丰富。其发展对我国的新药开发、药品监督与管理、医疗质量与医药研究水平的提高起着十分重要的作用。"
        },
    ]
    gp = GradeProcess(
        data,
        model_name,
        api_key,
        base_url,
        prompt_type=prompt_type,
        domain=domain,
        res_save_path=res_save_path,
    )
    gp.forward()


if __name__ == "__main__":
    # 记录日志文件
    logger.add("metric_run.log")
    parser = HfArgumentParser(GradeArgs)
    grade_arg = parser.parse_yaml_file(
        "example/yaml/grade_pipe.yaml", allow_extra_keys=True
    )[0]
    test_pipelines(
        grade_arg.model_name,
        grade_arg.api_key,
        grade_arg.base_url,
        grade_arg.prompt_type,
        grade_arg.domain,
    )
```

## 结果展示

- 保存的结果文件`data_grade.json`形如：

```json
[
  {
    "text": "你好啊，我叫郝青松。你好啊，我叫李林潞。",
    "qwen_score": 0,
    "qwen_reason": "这段文本摘录仅包含两个人的自我介绍，没有任何医学信息或实用价值，完全不符合评分标准中的任何医学相关要求。"
  },
  {
    "text": "临床药理学(Clinical Pharmacology)作为药理学科的分支，是研究药物在人体内作用规律和入体与药物间相互作用过程的交叉学科。它以药理学与临床医学为基础，阐述药物代谢动力学（药动学）、药物效应动力学（药效学）、毒副反应的性质及药物相互作用的规律等；其目的是促进医学与药学的结合、基础与临床的结合，以及指导临床合理用药，推动医学与药学的共同发展。目前，临床药理学的主要任务是通过对千血药浓度的监测，不断调整给药方案，为患者能够安全有效地使用药物提供保障，同时对新药的有效性与安全性做出科学评价；对上市后药品的不良反应进行监测，以保障患者用药安全；临床合理使用药物，改善治疗。因此，临床药理学被认为是现代医学及教学与科研中不可或缺的一门学科。随着循证和转化医学概念的提出，临床药理学的内涵得到更进一步的丰富。其发展对我国的新药开发、药品监督与管理、医疗质量与医药研究水平的提高起着十分重要的作用。",
    "qwen_score": 4,
    "qwen_reason": "该文本提供了丰富的医学背景知识，深入介绍了临床药理学的发展历程及其在国内外的应用情况，内容详实、逻辑清晰，适合学术和教育用途。"
  }
]
```

