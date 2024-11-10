# Text Grade Pipelines

## Contents

- [Configuration Files](#configuration-files)
- [Code Files Overview](#code-files-overview)
- [Results Display](#results-display)

## Configuration Files

This pipeline uses a `yaml` configuration file for parameter input. See the file [example/yaml/grade_pipe.yaml](https://github.com/ytzfhqs/EDCP/blob/main/example/yaml/grade_pipe.yaml) for details.

```yaml
## Path to the JSON file containing the text for LLM evaluation
data_or_filepath: None

## Name of the LLM
model_name: qwen-plus
## API key for the LLM
api_key: None
## Base URL required for calling Qwen series models
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
## Type of prompt, options are 'domain' (domain-specific, requires domain) and 'general' (general-purpose, domain is None)
prompt_type: domain
## Domain keyword
domain: medicine
## Path to save the result file
res_save_path: data_grade.json
## Key of the text in the dictionary
text_column: text
```

- Parameter explanations:

  - `data_or_filepath`: Path to the text data (in `json` format) to be evaluated. The JSON file should be structured like this:

  ```python
  [
      {"text":"Text 1"},
      {"text":"Text 2"},
      ...
  ]
  ```

  - `model_name`: Name or path of the model to call. For closed-source LLMs with an API (e.g., `qwen-plus`), this should be the model's name. For open-source LLMs (e.g., `Qwen2.5-7B-Instruct`), this should be the model's path.
  - `api_key`: API key for the LLM; for open-source LLMs, set this parameter to `None`.
  - `base_url`: Base URL required for closed-source Qwen series models; for other models, set this parameter to `None`.
  - `prompt_type`: Type of prompt. Options are `domain` (domain-specific, requires the `domain` parameter) and `general` (general-purpose, set `domain` to `None`). For prompt references, see the file [edcp/grade/template.py](https://github.com/ytzfhqs/EDCP/blob/main/edcp/grade/template.py).
  - `domain`: Domain keyword, applicable only when `prompt_type` is set to `domain`; otherwise, set to `None`.
  - `res_save_path`: Path to save the result file.
  - `text_column`: Key of the text data in the dataset for scoring.

## Code Files Overview

- The core code for scoring text using the LLM and prompt is in the [edcp/grade](https://github.com/ytzfhqs/EDCP/tree/main/edcp/metric) folder:
  - `chatmodel.py`: Calls for both open-source and closed-source LLMs, prompt construction, and result extraction.
  - `piplines.py`: Integrates LLM calls and sets up the scoring pipeline.
  - `template.py`: LLM scoring prompt templates.

## Example Usage

- For an example usage, see [edcp/example/grade_pipe_demo.py](https://github.com/ytzfhqs/EDCP/blob/main/example/grade_pipe_demo.py). Below is a simple implementation:

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

## Results Display

- The saved result file `data_grade.json` is structured as follows:

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

