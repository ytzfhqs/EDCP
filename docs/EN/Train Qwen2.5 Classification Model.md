# Train Qwen2.5 Classification Model

## Data Preparation

- The training data is stored in `json` format, as follows:

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

- The dictionary must contain `text` and `label` keys, where `text` represents the text, and `label` indicates the classification (0 for body text, 1 for non-body text).
- Since the data may involve sensitive information, there are currently no plans to open-source the dataset.
- Data integration can refer to the file [train_qwen_cls/clsdataset.py](https://github.com/ytzfhqs/EDCP/blob/main/train_qwen_cls/clsdataset.py).

## Bayesian Optimization

- To achieve better accuracy, Bayesian optimization is used to fine-tune the learning rate and `weight_decay` of the optimizer. The learning rate is searched in the logarithmic space within the range `[1.0e-5, 1.0e-4]`, and `weight_decay` is searched in the floating-point space with a step size of `0.01`, within the range `[0.01, 0.1]`.
- The pruner used is the `HyperbandPruner`, which helps us abandon less promising experiments early, saving training resources and improving parameter search efficiency.
- Detailed tuning information can refer to the file [train_qwen_cls/opt_cls.py](https://github.com/ytzfhqs/EDCP/blob/main/train_qwen_cls/opt_cls.py).

> [!Note]
>
> - Bayesian optimization requires the `optuna` library, which can be installed via `pip install optuna`.
> - Details on the pruner and optimizer can be found in the official documentation of the `optuna` library [here](https://optuna.readthedocs.io/en/stable/index.html).

## Model Training

- Use the parameters obtained from Bayesian optimization to train the model, referring to the file [train_qwen_cls/run_cls.py](https://github.com/ytzfhqs/EDCP/blob/main/train_qwen_cls/run_cls.py).

## Model Card

- `HuggingFace` model homepage: [Qwen2.5-med-book-main-classification](https://huggingface.co/ytzfhqs/Qwen2.5-med-book-main-classification)