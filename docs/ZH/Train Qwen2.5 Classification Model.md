# Train Qwen2.5 Classification Model

## 数据准备

- 训练数据以`json`格式存储，形如：

```python
[
  {
    "text": "人体寄生虫学",
    "label": 1
  },
  {
    "text": "Human Paras it ology",
    "label": 1
  },
  {
    "text": "第10版",
    "label": 1
  },
  ...
]
```

- 字典中必须包含`text`和`label`键，其中`text`是文本，`lable`表示标签（0为正文，1为非正文）。
- 因为数据可能涉及敏感信息，暂时没有开源数据集的计划。
- 数据整合参照[train_qwen_cls/clsdataset.py](https://github.com/ytzfhqs/EDCP/blob/main/train_qwen_cls/clsdataset.py)文件。

## 贝叶斯优化

- 为了获得更好的精度，我们使用了贝叶斯优化来精细调整学习率和优化器的`weight_decay`。学习率在对数空间进行搜索，搜索范围为`[1.0e-5, 1.0e-4]`，`weight_decay`在浮点数空间以`0.01`的步长进行搜索，搜索范围为`[0.01, 0.1]`。
- 剪枝器使用`HyperbandPruner`，该剪枝器能让我们提前放弃希望不大的实验，节约训练资源，提高参数搜索效率。
- 具体的调参细节可以参照[train_qwen_cls/opt_cls.py](https://github.com/ytzfhqs/EDCP/blob/main/train_qwen_cls/opt_cls.py)文件。

> [!Note]
>
> - 使用贝叶斯优化需要`optuna`库的支持，可以使用`pip install optuna`安装。
> - 剪枝器和优化器细节可以在`optuna`库的[官方文档](https://optuna.readthedocs.io/en/stable/index.html)中查看。

## 模型训练

- 使用贝叶斯优化得到的参数训练模型，参照[train_qwen_cls/run_cls.py](https://github.com/ytzfhqs/EDCP/blob/main/train_qwen_cls/run_cls.py)文件。

## 模型卡片

- `HuggingFace`模型主页[Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)