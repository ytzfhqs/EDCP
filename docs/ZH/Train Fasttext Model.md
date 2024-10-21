# Train Fasttext Model

## 数据准备

- 在[train_fasttext/train_model.py](https://github.com/ytzfhqs/EDCP/blob/main/train_fasttext/train_model.py)文件中，提供两个数据类，分别为`DataProcess`和`OffDataProcess`。如果当前网络环境支持访问`HuggingFace`推荐使用`DataProcess`流式加载数据。否则推荐使用`OffDataProcess`类，使用本地数据加载。
- 数据格式形如：

```json
[
    {
        "text": "这是一条中文文本",
        "word_count": 4, 
        "language": "zh"
    },
    {
        "text": "This is an English text.",
        "word_count": 5, 
        "language": "en"
    }
]
```

- 数据类会将数据转换为适合`fasttext`训练格式的`txt`文件，`fasttext`训练数据格式可以参照[官方文档](https://fasttext.cc/docs/en/supervised-tutorial.html)。
- 由于数据收集细节已经存在与代码中，该数据集也不计划开源。

## 训练细节

- 模型训练参数设定如下：

```python
classifier = fasttext.train_supervised(
        input="data_train.txt",
        autotuneValidationFile="data_valid.txt",
        autotuneModelSize="1024M",
        thread=20,
        autotuneDuration=int(3.5 * 60 * 60),
    )
```

- 关键参数说明：
  - `autotuneValidationFile`：自动参数优化时所用验证集。
  - `autotuneModelSize`：自动参数优化时限制模型大小。
  - `thread`：训练调用线程数。
  - `autotuneDuration`：自动参数优化搜索时间，按秒计算。

## 模型卡片

- `Huggingface`模型主页[fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification)