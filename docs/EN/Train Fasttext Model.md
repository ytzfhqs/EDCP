# Train Fasttext Model

## Data Preparation

- In the file [train_fasttext/train_model.py](https://github.com/ytzfhqs/EDCP/blob/main/train_fasttext/train_model.py), two data classes are provided: `DataProcess` and `OffDataProcess`. If the current network environment supports access to `HuggingFace`, it is recommended to use the `DataProcess` class for streaming data loading. Otherwise, use the `OffDataProcess` class for local data loading.
- The data format is as follows:

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

- The data class will convert the data into a `txt` file format suitable for `fasttext` training. The `fasttext` training data format can be found in the [official documentation](https://fasttext.cc/docs/en/supervised-tutorial.html).
- As the data collection details are already embedded in the code, this dataset will not be open-sourced.

## Training Details

- The model training parameters are set as follows:

```python
classifier = fasttext.train_supervised(
        input="data_train.txt",
        autotuneValidationFile="data_valid.txt",
        autotuneModelSize="1024M",
        thread=20,
        autotuneDuration=int(3.5 * 60 * 60),
    )
```

- Key parameter explanations:
  - `autotuneValidationFile`: Validation set used for automatic parameter tuning.
  - `autotuneModelSize`: Limits the model size during automatic parameter tuning.
  - `thread`: Number of threads used during training.
  - `autotuneDuration`: Time for automatic parameter optimization, in seconds.

## Model Card

- `Huggingface` model homepage: [fasttext-med-en-zh-identification](https://huggingface.co/ytzfhqs/fasttext-med-en-zh-identification)