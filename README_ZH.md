# Easy-Data-Clean-Pipeline

[[English](README.md)|中文]

**轻量级中文语料清洗工具包**。

## 目录

- [项目简介](#项目简介)
- [技术路线](#技术路线)
- [更新日志](#更新日志)
- [如何使用](#如何使用)

## 项目简介

本项目旨在开发一个以**中文内容**为基础的**高质量**垂直领域书籍数据集，主要针对使用[MinerU](https://github.com/opendatalab/MinerU)将`PDF`转为`markdown`后的文本清洗过程。根据大语言模型预训练相关论文（`C4`，`Dolma`，`RedPajama`），书籍数据是质量较高的数据集，并且对下游任务的影响较大。而大多数书籍都以`PDF`扫描件为主，借助[MinerU](https://github.com/opendatalab/MinerU)工具可以将`PDF`转换成`markdown`文件，但比较缺乏对书籍类`markdown`文档进行清洗的管道。

同时本项目还借鉴了`CCnet`、`RedPajama`的数据质量评分表，衍生出适用于中文文本的质量评估指标。我们期望能够为`LLM`提供更加**丰富**且**针对性强**的知识来源，从而提升其在**特定领域**的**理解**和**生成能力**。

## 技术路线

- 使用[MinerU](https://github.com/opendatalab/MinerU)对`PDF`进行`OCR`转换为`markdown`格式，保留`PDF`中的公式，过滤无关内容（如出版信息、目录、修订说明等），并使用布隆过滤器进行精确的文档级内容去重。
- 基于数据质量量化表，从3个方面衡量数据（与领域无关）：
  1. LLM评价（困惑度等）
  2. NLP特征（文本词的平均长度、文本唯一词出现的比例、停用词占比等）
  3. 机器学习（训练`n-gram`模型进行重要性采样、`minhash`去重等）
- 调用`GPT-4 API`结合数据质量量化表、相关论文提到的高质量数据集特征（如多样性、无偏性、准确性、逻辑性）和专业领域的相关要求，整合提示词，将数据按0-5分成共6个等级。（考虑到经济性，这个步骤只会使用少量的数据集进行标定，并同时使用人工校准。）
- 根据`GPT-4`与人工校对得到的高质量标定数据集训练`only-decoder`（如`Qwen2`）分类模型（使用`only-decoder`模型主要为了避免分词器最大长度以及模型架构带来的计算限制），以实现对后期大量的新样本进行标定。
- 根据`Qwen2`分类模型的得分和数据质量量化表选取训练数据，从不同角度测评筛选出的高质量文本能否高效的提升模型在专业领域的适应性（消融实验）。
  - 不同的模型的表现；
  - 不同的质量分数阈值对领域适应性的影响；
  - 权衡模型专业领域的适应性与通用能力。

**请注意**：上述技术路线并非最终版本，随着项目推进可能会在细节上有调整。

## 更新日志

[24/10/21] 完成文本数据评分管道搭建，优化`markdown`文件清洗管道

距离上次更新已经过去了1个月，在这1个月中我们完成了文本数据的评分管道搭建，优化了对`markdown`文件的清洗管道。同时，我们还在`HuggingFace`上上传了两个分类模型：[Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)和[fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification)。其中，[Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)用于将[MinerU](https://github.com/opendatalab/MinerU)转换后的`markdown`本文进行正文与非正文的分类，相较于通用`LLM`模型，速度加快10倍，并且对医疗类文本正文与非正文的分类精度更高。[fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification)用于判断文本属于英文还是中文，并给出置信度。具体的数据集构建流程和模型训练细节可以在对应模型`HuggingFace`的`Model card`中查看。

本次更新是一次较为重要的更新，除了显性工作（数据的评分管道搭建），我们还对之前的代码进行了很多优化，包括但不仅限于尽可能遵循`HuggingFace`的代码风格、保证代码一致性、生产环境测试等一系列隐性工作。并且几乎所有功能模块都支持**多进程加速**，为了`edcp`库的易用性，我们编写了[用户文档](https://github.com/ytzfhqs/EDCP/tree/main/docs)，来帮助其他研究者了解和使用`edcp`库。

[24/09/20] 我们更新了对`markdown`文件的粗清洗管道
<details><summary>展开日志</summary>

非常高兴迎来`edcp`库的第一次功能更新，本次更新的是对`markdown`文件的粗清洗管道。

具体的，当我们使用[MinerU](https://github.com/opendatalab/MinerU)将`PDF`转换为`markdown`文件时，会产生很多非正文内容，包括书本简介、出版社信息编写规范等，所以我们尝试使用`LLM`来进行清洗。

因为内容主要以中文为主，所以我们使用了不久前刚发布的[Qwen2.5](https://github.com/QwenLM/Qwen2.5)系列。根据实验，必须使用7B参数量以上的模型才能保证过滤质量，推荐使用14B以上参数量的模型。

我们同时提供了`VLLM`框架和`Transformers`框架批推理流程，实测使用`VLLM`框架比`Transformers`框架推理速度快2倍。如果没有`VLLM`框架也不用太过沮丧，因为我们还提供了并行推理，在显存足够的情况下也能进行一定加速。

文件树：

```
└── edcp
  ├── mdclean
      ├── __init__.py
      └── LLMFilter.py
      └── VLLMFilter.py
      └── charreplace.py
      └── pipelines.py
      └── template.py
      └── utils.py
  └── md_pipe_demo.py
```

各个文件的主要作用如下：

- `LLMFilter.py`：Transformers框架批推理流程。
 - `VLLMFilter.py`：VLLM框架批推理流程。
 - `charreplace.py`：需要对文本进行替换操作的正则表达式与字符库。
 - `pipelines.py`：处理流程入口。
 - `template.py`：LLM过滤提示词模板。
 - `utils.py`：一些常用的工具函数。

## 如何使用

### 安装依赖

> [!IMPORTANT]
> 此步骤为必需。

```bash
git clone https://github.com/ytzfhqs/EDCP
cd EDCP
pip install requirements.txt -r
```

> [!TIP]
> 具体用法请参阅[用户文档](https://github.com/ytzfhqs/EDCP/tree/main/docs)。

