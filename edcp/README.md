# EDCP
## Changelog
[24/09/20] We have updated the coarse cleaning pipeline for markdown files
<details><summary>Full Changelog</summary>

We are thrilled to announce the first feature update of the `edcp` library, which introduces a pipeline for preliminary cleaning of `markdown` files.

Specifically, when we use [MinerU](https://github.com/opendatalab/MinerU) to convert `PDF` files into `markdown` format, a lot of non-content elements, such as book introductions, publisher information, and formatting guidelines, are also included. Therefore, we attempt to leverage an `LLM` to clean up these extraneous parts.

Since most of the content is in Chinese, we use the recently released [Qwen2.5](https://github.com/QwenLM/Qwen2.5) series. Based on our experiments, models with at least 7B parameters are required to ensure the filtering quality, with 14B or larger models recommended.

We offer batch inference processes through both the `VLLM` and `Transformers` frameworks. Our tests show that using the `VLLM` framework is twice as fast as using the `Transformers` framework. However, if you don't have the `VLLM` framework, there's no need to worry, as we also provide parallel inference to accelerate processing, provided there is enough VRAM.

The project file tree is:
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
The main roles of each file are as follows:
 - `LLMFilter.py`: Batch inference process using the Transformers framework.
 - `charreplace.py`: Regular expressions and character library for text replacement operations.
 - `pipelines.py`: Entry point for the processing pipeline.
 - `template.py`: LLM filtering prompt templates.
 - `utils.py`: Common utility functions.
 - `md_pipe_demo.py`: Tests for the key functions above, along with usage examples.
