import os
import importlib
from typing import List, Dict, Any, Literal

from tqdm import tqdm
from rbloom import Bloom
from mpire import WorkerPool
from markdown import Markdown
from . import utils
from .LLMFilter import ChatModel
from .CLSFilter import QwenCLS
from .charreplace import BookClean
from .packing import PackText
from ..tool import save_json


class BaseProcess:
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
    def trans_dict(key_name: str, text_list: List[str]) -> List[Dict[str, Any]]:
        return [
            {key_name: t, "id_int": idx}
            for t, idx in zip(text_list, range(len(text_list)))
        ]

    @staticmethod
    def bloom_filter(text_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unique_text: List[Dict[str, str]] = []
        bf = Bloom(len(text_list), 0.01)
        for t in text_list:
            if t["text"] in bf:
                continue
            else:
                bf.add(t["text"])
                unique_text.append(t)
        utils.filter_info("Bloom Filter", len(text_list), len(unique_text))
        return unique_text


class MdProcess(BaseProcess):
    def __init__(
        self,
        md_path: str,
        llm_model_path: str,
        token_cont_model: str,
        near_tokens: int,
        process_method: Literal["chat", "cls"],
    ):
        """
        MdProcess初始化方法
        Args:
            md_path: 包含需要处理markdown文件的主目录
            llm_model_path: 用于过滤非正文样本的LLM路径
            token_cont_model: 用于统计token数量的LLM路径
            near_tokens: token数量的大致范围
            process_method: 用于过滤非正文样本的方法，chat为通用因果模型，cls为专用分类模型
        """
        # 查找markdown文件
        self.md_path_list = utils.search_file_suffix(md_path, "md")
        if process_method != "chat" and process_method != "cls":
            raise ValueError("process_method must be 'chat' or 'cls'")
        if process_method == "chat":
            # 检查VLLM环境是否可用
            try:
                importlib.import_module("vllm")
                from .VLLMFilter import VLLMChatModel

                # 若VLLM可用，使用VLLM框架推理
                self.cm = VLLMChatModel(llm_model_path)
            except ImportError:
                # 若VLLM不可用，使用Transformers框架进行推理
                self.cm = ChatModel(llm_model_path)
        else:
            # 加载Qwen2.5分类模型
            self.cm = QwenCLS(llm_model_path)
        # 加载PackText工具
        self.pt = PackText(token_cont_model, near_tokens)
        # 暂存处理后结果
        self.res_text: List[Dict[str, str]] = []
        # 暂存中间结果
        self.middle_res: List[Dict[str, str]] = []

    def llm_filter(
        self,
        book_name: str,
        text_list: List[str],
        batch_size: int,
        save_middle: bool = False,
    ) -> List[str]:
        """
        调用LLM对字符串列表进行过滤
        Args:
            book_name: 当前处理的书名
            text_list: 字符串列表
            batch_size: 批处理大小
            save_middle: 是否保存中间结果

        Returns:过滤后的字符串列表

        """
        # 仅正文样本
        filter_text: List[str] = []
        # 构造Batch输入
        chunk_text_list: List[List[str]] = utils.chunk_list(text_list, batch_size)
        for chunk_text in tqdm(chunk_text_list, desc="LLM Filtering Process"):
            res: List[str] = self.cm.forward(book_name, chunk_text)
            if save_middle:
                self.middle_res = self.middle_res + [
                    {"text": c, "res": r} for c, r in zip(chunk_text, res)
                ]

            filter_text = filter_text + utils.select_strings(chunk_text, res)

        utils.filter_info(
            "LLM Filtering Process", len(text_list), len(filter_text), book_name + ".md"
        )
        return filter_text

    def single_file(
        self, single_md_path, batch_size: int = 4, save_middle: bool = False
    ) -> List[str]:
        """
        单文件处理流程
        Args:
            single_md_path: markdown文件路径
            batch_size: 批处理大小
            save_middle: 是否保存中间结果
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
        text = self.llm_filter(book_name, text, batch_size, save_middle)
        # 按near_tokens打包样本
        text = self.pt.forward(text)
        return text

    def forward(self, batch_size, save_path, save_middle: bool = False):
        """
        单核处理流程
        Args:
            batch_size: 批处理大小
            save_path: 结果文件保存路径
            save_middle: 是否保存中间结果
        """
        text_list: List[str] = []
        for md_path in self.md_path_list:
            text_list = text_list + self.single_file(md_path, batch_size, save_middle)
        self.res_text = self.trans_dict("text", text_list)
        self.res_text = self.bloom_filter(self.res_text)
        save_json(save_path, self.res_text)
        if save_middle:
            save_json(utils.filename_add_suffix(save_path, "_middle"), self.middle_res)

    def forward_with_pool(
        self,
        num_proc: int,
        save_path: str,
        batch_size: int = 4,
        save_middle: bool = False,
    ):
        """
        并行处理流程

        Args:
            num_proc: 进程数
            save_path: 结果文件保存路径
            batch_size: 批处理大小
            save_middle: 是否保存中间结果
        """

        def aux_iter(ls: List[str], **kwargs):
            for l in ls:
                yield {**{"single_md_path": l}, **kwargs}

        with WorkerPool(n_jobs=num_proc, start_method="spawn") as pool:
            text_list = pool.map(
                self.single_file,
                aux_iter(
                    self.md_path_list, batch_size=batch_size, save_middle=save_middle
                ),
                progress_bar=True,
            )
        self.res_text = self.trans_dict("text", text_list)
        self.res_text = self.bloom_filter(self.res_text)
        save_json(save_path, self.res_text)
        if save_middle:
            save_json(utils.filename_add_suffix(save_path, "_middle"), self.middle_res)
