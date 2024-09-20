import os
import importlib
from typing import List, Dict

from . import utils
from tqdm import tqdm
from mpire import WorkerPool
from markdown import Markdown
from .LLMFilter import ChatModel
from .charreplace import BookClean


class MdProcess:
    def __init__(self, md_path: str, llm_model_path: str):
        """
        MdProcess初始化方法
        Args:
            md_path: 包含需要处理markdown文件的主目录
            llm_model_path: 用于过滤非正文样本的LLM路径
        """
        self.md_path_list = utils.search_file_suffix(md_path, "md")
        # 检查VLLM环境是否可用
        try:
            importlib.import_module("vllm")
            from .VLLMFilter import VLLMChatModel

            # 若VLLM可用，使用VLLM框架推理
            self.cm = VLLMChatModel(llm_model_path)
        except ImportError:
            # 若VLLM不可用，使用Transformers框架进行推理
            self.cm = ChatModel(llm_model_path)
        # 暂存处理后结果
        self.res_text: List[Dict[str, str]] = []

    @staticmethod
    def unmark(text: str) -> str:
        # 更改markdown文件的渲染方式，剔除与markdown相关的语法
        Markdown.output_formats["plain"] = utils.unmark_element
        __md = Markdown(output_format="plain")
        __md.stripTopLevelTags = False
        return __md.convert(text)

    @classmethod
    def read_md(cls, md_path: str):
        """
        读取md文件，并过滤文本中存在的markdown语法

        Args:
            md_path: markdown文件路径

        Returns:
            过滤markdown后的字符串
        """
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        return cls.unmark(md_text)

    @staticmethod
    def replace_op(text_list: List[str]) -> List[str]:
        """
        对列表中的字符串逐一进行替换操作
        Args:
            text_list: 字符串列表

        Returns:处理好后的字符串列表
        """
        rep_text: List[str] = []
        for t in text_list:
            c_t = utils.remove_and_replace(
                t, BookClean.REMOVE_WORDS, BookClean.REPLACEMENTS
            )
            rep_text.append(c_t)
        return rep_text

    @staticmethod
    def trans_dict(key_name: str, text_list: List[str]) -> List[Dict[str, str]]:
        return [{key_name: t} for t in text_list]

    def llm_filter(self, book_name, text_list: List[str], batch_size: int) -> List[str]:
        """
        调用LLM对字符串列表进行过滤
        Args:
            book_name: 当前处理的书名
            text_list: 字符串列表
            batch_size: 批处理大小

        Returns:过滤后的字符串列表

        """
        filter_text: List[str] = []
        # 构造Batch输入
        chunk_text_list = utils.chunk_list(text_list, batch_size)
        for chunk_text in tqdm(chunk_text_list, desc="LLM Filtering Process"):
            res: List[str] = self.cm.forward(book_name, chunk_text)
            filter_text = filter_text + utils.select_strings(chunk_text, res)

        utils.filter_info("LLM Filtering Process", len(text_list), len(filter_text))
        return filter_text

    def single_file(self, single_md_path, batch_size=4) -> List[str]:
        """
        单文件处理流程
        Args:
            single_md_path: markdown文件路径
            batch_size: 批处理大小

        Returns: LLM过滤后的字符串列表

        """
        book_name = os.path.splitext(os.path.basename(single_md_path))[0]
        # 读取文件
        text = self.read_md(single_md_path)
        # 分割文件
        text = utils.split_text(text)
        # 替换操作
        text = self.replace_op(text)
        # LLM过滤
        text = self.llm_filter(book_name, text, batch_size=batch_size)
        return text

    def forward(self, batch_size, save_path):
        """
        单核处理流程
        Args:
            batch_size: 批处理大小
            save_path: 结果文件保存路径
        """
        text_list: List[str] = []
        for md_path in self.md_path_list:
            text_list = text_list + self.single_file(md_path, batch_size)
        self.res_text = self.trans_dict("text", text_list)
        utils.save_json(save_path, self.res_text)

    def forward_with_pool(self, num_proc, save_path):
        """
        并行处理流程

        Args:
            num_proc: 进程数
            save_path: 结果文件保存路径
        """
        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            text_list = pool.map(self.single_file, self.md_path_list, progress_bar=True)
        self.res_text = self.trans_dict("text", text_list)
        utils.save_json(save_path, self.res_text)
