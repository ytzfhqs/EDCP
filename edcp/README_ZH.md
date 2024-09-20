##更新日志
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
`LLMFilter.py`：Transformers框架批推理流程。
`VLLMFilter.py`：VLLM框架批推理流程。
`charreplace.py`：需要对文本进行替换操作的正则表达式与字符库。
`pipelines.py`：处理流程入口。
`template.py`：LLM过滤提示词模板。
`utils.py`：一些常用的工具函数。
