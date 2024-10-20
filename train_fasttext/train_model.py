import re
import nltk
import jieba
import fasttext
import jionlp as jio
from datasets import Dataset
from nltk.corpus import stopwords
from datasets import concatenate_datasets, load_dataset

# Download: https://github.com/nltk/nltk_data
# nltk.download("stopwords")


class DataProcess:
    def __init__(self, max_en_sample, max_zh_sample, num_proc):
        self.max_zh_sample = max_zh_sample
        self.max_en_sample = max_en_sample
        self.num_proc = num_proc
        self.ds_zh = self.zh_data(max_zh_sample)
        self.ds_en = self.en_data(max_en_sample)

    @staticmethod
    def zh_data(max_zh_sample):
        sp = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
        sp = Dataset.from_list(list(sp.take(int(max_zh_sample))))

        smp = load_dataset(
            "ticoAg/shibing624-medical-pretrain",
            data_files="pretrain/medical_book_zh.json",
            split="train",
        )
        return concatenate_datasets([sp, smp], axis=0)

    @staticmethod
    def en_data(max_en_sample):
        rdv2 = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            snapshots=["2023-14"],
            languages=["en"],
            name="default",
            split="train",
            streaming=True,
            trust_remote_code=True,
        ).select_columns(["raw_content"])
        rdv2 = rdv2.rename_column("raw_content", "text")
        rdv2 = Dataset.from_list(list(rdv2.take(int(max_en_sample))))
        mmw = load_dataset(
            "medalpaca/medical_meadow_wikidoc", split="train"
        ).select_columns(["output"])
        mmw = mmw.rename_column("output", "text")

        md = load_dataset("nlp-guild/medical-data", split="train").select_columns(
            ["cause"]
        )
        md = md.rename_column("cause", "text")

        sepc = load_dataset(
            "JWBickel/StrongsChunked_English_Phrase_Counts", split="train[1:]"
        )
        sepc = sepc.rename_column("RowID ^ StrongsChunkedPhrase ^ Count", "text")

        def sepc_text(example):
            match = re.search(r"\^ (.*?) \^", example["text"])
            if match:
                return {"text": match.group(1)}
            else:
                return {"text": ""}

        sepc = sepc.map(sepc_text, num_proc=4)
        return concatenate_datasets([rdv2, mmw, md, sepc], axis=0)

    def save_parq(self):
        self.ds_zh.to_parquet("zh_data.parq")
        self.ds_en.to_parquet("en_data.parq")

    @staticmethod
    def en_seg_opt(example, split_char="\n"):
        lines = [
            line.strip().lower()
            for line in example["text"].split(split_char)
            if line.strip()
        ]
        return {"text": lines}

    @staticmethod
    def zh_seg_opt(example, split_char="\n"):
        lines = [
            line.strip() for line in example["text"].split(split_char) if line.strip()
        ]
        return {"text": lines}

    @staticmethod
    def pack_dict(data):
        temp_ldict = []
        for d in data["text"]:
            for s in d:
                temp_ldict.append({"text": s})
        return Dataset.from_list(temp_ldict)

    @staticmethod
    def zh_word_count(example):
        segs = jieba.lcut(example["text"])
        # 删除非中文字符词汇
        segs = jio.remove_stopwords(segs, remove_non_chinese=True)
        return {"word_count": len(segs), "language": "zh"}

    @staticmethod
    def en_word_count(example):
        stop_words = set(stopwords.words("english"))
        segs = nltk.word_tokenize(example["text"])
        filtered_segs = [word for word in segs if not word.lower() in stop_words]
        return {"word_count": len(filtered_segs), "language": "en"}

    @staticmethod
    def to_train_data(ds, train_radio=0.9):
        temp_list = []
        for d in ds:
            temp_text = f"__label__{d['language']} {d['text']}"
            temp_list.append(temp_text)
        train_len = int(len(temp_list) * train_radio)
        print(f"Train sample: {train_len}")
        print(f"Valid sample: {len(temp_list) - train_len}")
        return temp_list[:train_len], temp_list[train_len:]

    @staticmethod
    def save_txt(string_list, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            for item in string_list:
                file.write(item + "\n")

    def forward(self, min_zh_words=1, min_en_words=1):
        # Datsets -> Dict[str,List[List[str]]]
        self.ds_zh = self.ds_zh.map(self.zh_seg_opt, num_proc=self.num_proc).to_dict()
        self.ds_en = self.ds_en.map(self.en_seg_opt, num_proc=self.num_proc).to_dict()
        # Dict[str,List[List[str]]] -> Datsets
        self.ds_zh = self.pack_dict(self.ds_zh)
        self.ds_en = self.pack_dict(self.ds_en)
        self.ds_zh = self.ds_zh.map(self.zh_word_count, num_proc=self.num_proc).filter(
            lambda example: example["word_count"] >= min_zh_words
        )
        self.ds_en = self.ds_en.map(self.en_word_count, num_proc=self.num_proc).filter(
            lambda example: example["word_count"] >= min_en_words
        )
        print(f"English data sample : {self.ds_en.num_rows}")
        print(f"Chinese data sample : {self.ds_zh.num_rows}")
        # 拼接中文和英文数据集
        ds_mix = concatenate_datasets([self.ds_zh, self.ds_en], axis=0)
        ds_mix = ds_mix.shuffle(seed=42)
        mix_train, mix_valid = self.to_train_data(ds_mix)
        self.save_txt(mix_train, "data_train.txt")
        self.save_txt(mix_valid, "data_valid.txt")


class OffDataProcess(DataProcess):
    def __init__(self, num_proc=12):
        self.num_proc = num_proc
        self.ds_zh = load_dataset("parquet", data_files="zh_data.parq", split="train")
        self.ds_en = load_dataset("parquet", data_files="en_data.parq", split="train")


def main(num_proc):
    # dp = DataProcess(2.2e+5, 3.5e+5, num_proc=num_proc)
    dp = OffDataProcess(num_proc=num_proc)
    dp.forward()
    classifier = fasttext.train_supervised(
        input="data_train.txt",
        autotuneValidationFile="data_valid.txt",
        autotuneModelSize="1024M",
        thread=20,
        autotuneDuration=int(3.5 * 60 * 60),
    )
    # 训练集表现
    train_result = classifier.test("data_train.txt")
    print("train_precision:", train_result[1])
    print("train_recall:", train_result[2])

    # 测试集表现
    train_result = classifier.test("data_valid.txt")
    print("train_precision:", train_result[1])
    print("train_recall:", train_result[2])

    def text_trans(text):
        return text.strip().lower()

    labels = classifier.predict(text_trans("Hello, world!"))
    print(labels)
    classifier.save_model("model.bin")


if __name__ == "__main__":
    main(12)
