import re
import json
from typing import Dict, List, Any, Tuple


import requests
from mpire import WorkerPool
from lxml import etree
from tqdm import tqdm


class SpiderMsd:
    def __init__(self, url: str, headers: Dict[str, str]):
        self.url = url
        self.headers = headers
        self.cat_dict = {}
        # 获取各类主题链接
        self.init_url()
        self.spider_res: List[Dict[str, str]] = []

    @staticmethod
    def get_request(url: str, headers: Dict[str, str]) -> Tuple[str, Any]:
        # 发起请求
        resp = requests.get(url=url, headers=headers)
        # 获取网页源代码
        text_res = resp.text
        tree_res = etree.HTML(text_res)
        # 关闭爬虫
        resp.close()
        return text_res, tree_res

    def init_url(self):
        """获取各类主题名称与链接"""
        _, tree_res = self.get_request(self.url, self.headers)
        # 提取主题名称
        cat_name = tree_res.xpath(
            "//div[contains(@class,'SectionList_sectionListItem__NNP4c')]/a/text()"
        )
        # 提取主题链接
        cat_url = tree_res.xpath(
            "//div[contains(@class,'SectionList_sectionListItem__NNP4c')]/a/@href"
        )
        # 检查主题名称数量是否与主题链接一致
        assert len(cat_name) == len(cat_url)
        for name, url in zip(cat_name, cat_url):
            self.cat_dict[name] = "https://www.msdmanuals.cn" + url

    def get_cat_info(self, cat_name: str, cat_url: str) -> List[Dict[str, str]]:
        """获取各类主题中子分支的名称与链接"""
        text_res, _ = self.get_request(cat_url, self.headers)
        res_list: List[Dict[str, str]] = []
        obj = re.compile(
            r'"TopicName":\{"value":"(?P<sub_name>.*?)"\},"TopicUrl":\{"path":"(?P<sub_url>.*?)"\}',
            re.S,
        )
        res = obj.finditer(text_res)
        for it in res:
            # res_list.append(it.groupdict())
            # 主题子分支的命名规则：主题名-子分支名
            sub_name = cat_name + "-" + self.clear_cat_name(it.group("sub_name"))
            sub_url = "https://www.msdmanuals.cn" + it.group("sub_url")
            # 子类层级
            if sub_url.count("/") == 6:
                res_list.append({"cat_name": sub_name, "cat_url": sub_url})
        return res_list

    @staticmethod
    def pack_text(text_list: List[str]) -> str:
        """将xpath提取的元素进行打包合并"""
        end_punc = """()（），,’‘，“”。.！!?？：:\n"""
        res_text = ""
        for idx, t in enumerate(text_list):
            t = t.strip()
            # 若当前字符串以结束符号结尾或为英文字母或为第一条元素
            if (t[-1] in end_punc) or (t[-1].encode("utf-8").isalpha()) or (idx == 0):
                res_text = res_text + t
            else:
                shift_idx = min(idx + 1, len(text_list) - 1)
                if text_list[shift_idx][-1] in end_punc:
                    res_text = res_text + t
                else:
                    res_text = res_text + t + "，"
        return res_text

    def single_process(self, cat_name: str, cat_url: str) -> List[Dict[str, str]]:
        """单主题内容提取函数"""
        # 遍历主题下所有小分支
        res_list = self.get_cat_info(cat_name, cat_url)
        for idx, res in enumerate(res_list):
            _, tree_res = self.get_request(res["cat_url"], self.headers)
            # xpath提取语料
            text = tree_res.xpath(
                "//span[contains(@data-testid,'topicText') or (contains(@id,'v') and contains(@class,'TopicGHead')) or (contains(@class,'genericDrug') and contains(@data-showrollover,'false'))]/text()"
            )
            # 添加text键，用于存放规整后的语料
            res_list[idx]["text"] = self.pack_text(text)
        return res_list

    @staticmethod
    def clear_cat_name(text: str) -> str:
        """清洗主题名称中的特殊字符"""
        pattern = (r"\\u003ci\s*", r"\\u003e\s*", r"\\u003c\s*", r"/i\s*")
        for p in pattern:
            text = re.sub(p, "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def save_txt(save_data, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(save_data)

    @staticmethod
    def save_json(paths: str, datas: List[Dict[Any, Any]]):
        """保存json文件"""
        with open(paths, "w", encoding="utf-8") as json_file:
            json.dump(datas, json_file, indent=2, ensure_ascii=False)

    def forward(self, save_path: str):
        # 遍历主题名称与主题链接
        for n, u in tqdm(self.cat_dict.items()):
            self.spider_res = self.spider_res + self.single_process(n, u)
            self.save_json(save_path, self.spider_res)

    def forward_pool(self, save_path: str):
        with WorkerPool(n_jobs=12, start_method="spawn") as pool:
            self.spider_res = pool.map(
                self.single_process, self.cat_dict, progress_bar=True
            )
        self.save_json(save_path, self.spider_res)


if __name__ == "__main__":
    u = "https://www.msdmanuals.cn/professional/health-topics"
    h = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    }
    smd = SpiderMsd(u, h)
    smd.forward("spidermsd.txt")
