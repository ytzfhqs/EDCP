![# Easy-Data-Clean-Pipeline](assets/logo.png)

[English | [中文](README_ZH.md)]

**A lightweight toolkit for cleaning Chinese corpora.**

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Approach](#technical-approach)
- [Changelog](#changelog)
- [How to Use](#how-to-use)

## Project Overview

This project aims to develop a **high-quality** book dataset in **Chinese**, focusing on cleaning texts after converting `PDF` files to `markdown` using [MinerU](https://github.com/opendatalab/MinerU). According to various papers on pretraining large language models (e.g., `C4`, `Dolma`, `RedPajama`), book data is considered high-quality and has a significant impact on downstream tasks. Since most books are in `PDF` format, which often comes as scanned documents, the [MinerU](https://github.com/opendatalab/MinerU) tool can convert these `PDF` files into `markdown` format. However, a dedicated pipeline for cleaning book-type `markdown` documents is lacking.

This project also draws from the quality scoring systems of datasets like `CCnet` and `RedPajama`, deriving quality assessment metrics that suit Chinese texts. Our goal is to provide a **rich** and **targeted** source of knowledge for `LLMs`, thereby improving their **understanding** and **generation capabilities** in **specific domains**.

## Technical Approach

- Use [MinerU](https://github.com/opendatalab/MinerU) to perform `OCR` on `PDF` files, converting them into `markdown` format. This process retains formulas from the `PDF` while filtering out irrelevant content (e.g., publication info, table of contents, revision notes, etc.). A Bloom filter is used for precise document-level deduplication.
- Based on a data quality quantification table, evaluate data from three perspectives (domain-independent):
  1. LLM evaluation (e.g., perplexity)
  2. NLP features (e.g., average word length, ratio of unique words, proportion of stop words)
  3. Machine learning techniques (e.g., train `n-gram` models for importance sampling, `minhash` deduplication)
- Use the `GPT-4 API` combined with the data quality quantification table and characteristics of high-quality datasets mentioned in relevant papers (such as diversity, fairness, accuracy, and logical consistency) and professional domain requirements. This integration generates prompt-based scoring, categorizing the data into six levels (from 0 to 5). (Considering cost, this step will only use a small portion of the dataset for calibration, with human oversight for quality control.)
- Use the high-quality labeled dataset, obtained from `GPT-4` and human review, to train an `only-decoder` model (such as `Qwen2`). This classification model is used to label large amounts of new samples in the future. The choice of `only-decoder` models helps avoid computational limitations caused by token length and model architecture.
- Using the scores from the `Qwen2` classification model and the data quality quantification table, select training data and assess whether the selected high-quality texts efficiently enhance model adaptability in specialized fields (ablation study).
  - Performance of different models
  - The impact of varying quality score thresholds on domain adaptability
  - Balancing a model’s domain-specific adaptability with general capabilities.

**Please note**: The above technical approach is not final and may be adjusted as the project progresses.

## Changelog

[24/10/21] We completed the construction of the text data scoring pipeline and optimized the cleaning pipeline for `markdown` files.

In the past month, we completed the construction of a text data scoring pipeline and optimized the cleaning pipeline for `markdown` files. Additionally, we uploaded two classification models on `HuggingFace`: [Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification) and [fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification). The [Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification) model is used to classify the main text and non-main text from the `markdown` content transformed by [MinerU](https://github.com/opendatalab/MinerU). Compared to general `LLM` models, it runs 10 times faster and has higher accuracy for classifying medical text into main and non-main content. The [fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification) model is used to determine whether a text is in English or Chinese and provides a confidence score. The detailed dataset construction process and model training specifics can be found in the respective model cards on `HuggingFace`.

This update is quite significant. In addition to the visible work (the construction of the data scoring pipeline), we also carried out many optimizations on the previous code, including but not limited to adhering as much as possible to `HuggingFace`'s coding style, ensuring code consistency, and testing in production environments. And almost all functional modules support **multi-process acceleration**. To improve the usability of the `edcp` library, we also wrote [user documentation](https://github.com/ytzfhqs/EDCP/tree/main/docs) to help other researchers understand and use the `edcp` library.

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

## How to Use

### Install Dependencies

> [!IMPORTANT]
> This step is required.

```bash
git clone https://github.com/ytzfhqs/EDCP
cd EDCP
pip install requirements.txt -r
```

> [!TIP]
> For detailed usage, refer to the [User Documentation](https://github.com/ytzfhqs/EDCP/tree/main/docs).
