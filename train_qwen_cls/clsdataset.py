import re
import os
from typing import Dict, List, Any

from tqdm import tqdm
from rbloom import Bloom
from edcp.tool import read_json, save_json
from edcp.mdclean.utils import search_file_suffix
from edcp.mdclean.utils import remove_and_replace
from edcp.mdclean.charreplace import BookClean

ID2LABEL: Dict[int, str] = {0: "正文", 1: "非正文"}
LABEL2ID: Dict[str, int] = {"正文": 0, "非正文": 1}


class PrepareData:

    def __init__(self, json_dir: str):
        json_path_list = search_file_suffix(json_dir, "json")
        self.book_data = self.aux_read(json_path_list)
        self.bf = Bloom(int(5.0e5), 0.01)

    @classmethod
    def aux_read(cls, json_path_list: str):
        json_data = []
        for path in json_path_list:
            temp_data = read_json(path)
            if cls.check_legally(path, temp_data):
                json_data = json_data + temp_data
            else:
                raise ValueError("Data legitimacy check failed!")
        return json_data

    @staticmethod
    def check_legally(path, datas: List[Dict[str, Any]]):
        file_name = os.path.basename(path)
        for data in datas:
            idx = data["id"]
            if "label" not in list(data.keys()):
                print(f"File {file_name} id {idx} missing 'label' key！")
                return False
            if data["label"] == "":
                print(f"File {file_name} id {idx} missing 'label' value!")
                return False
            if data["text"][0] == " ":
                print(f"File {file_name} id {idx} is a space at the beginning of text!")
                return False
        return True

    @staticmethod
    def batch_amend(datas: List[Dict[str, Any]]):
        for data in datas:
            data["text"] = remove_and_replace(
                data["text"], BookClean.REMOVE_WORDS, BookClean.REPLACEMENTS
            )
            if re.match(r"^【", data["text"]):
                data["label"] = "正文"
            # 以表、图开头的全部是非正文
            if re.match(r"^[图|表]", data["text"]) or data["text"] == "$$":
                data["label"] = "非正文"
        return datas

    def bloom_filter(self, datas: List[Dict[str, Any]]):
        filter_list = []
        for data in tqdm(datas, desc="Bloom Filter"):
            if data["text"] in self.bf:
                continue
            else:
                self.bf.add(data["text"])
                filter_list.append(
                    {"text": data["text"], "label": LABEL2ID[data["label"]]}
                )
        return filter_list

    @staticmethod
    def print_info(datas: List[Dict[str, Any]]):
        total_sample = len(datas)
        print(f'The total sample size is {total_sample}')
        pos_sample = 0
        for data in datas:
            if data['label'] == 0:
                pos_sample += 1
        pos_ratio = pos_sample / total_sample * 100.0
        neg_ratio = 100 - pos_ratio
        print(f'The total positive sample is {pos_sample}, {pos_ratio:.2f}% of total sample size')
        print(f'The total negative sample is {total_sample - pos_sample}, {neg_ratio:.2f}% of total sample size')

    def forward(self):
        book_data = self.batch_amend(self.book_data)
        book_data = self.bloom_filter(book_data)
        self.print_info(book_data)
        save_json("book_data.json", book_data)


if __name__ == "__main__":
    pd = PrepareData("./data")
    pd.forward()
