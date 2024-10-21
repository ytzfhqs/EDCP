from typing import Literal


from loguru import logger
from transformers import HfArgumentParser
from edcp.mdclean.pipelines import MdProcess
from edcp.mdclean.LLMFilter import ChatModel
from edcp.hparams.mdpipe_args import MdPipeArgs


def test_llm_filter(llm_model_path: str):
    book_name = "儿科学"
    context = [
        "儿科学／黄国英，孙锟，罗小平主编．--10版北京：人民卫生出版社，2024.6．-",
        "急性支气管炎一般不发热或仅有低热，全身状况好，以咳嗽为主要症状，肺部可闻及十湿啰音，多不固定，随咳嗽而改变。胸部X线检查示肺纹理增多、排列素乱。",
    ]
    cm = ChatModel(llm_model_path)
    res = cm.forward(book_name, context)
    print(res)


def test_pipelines(
    md_path: str,
    llm_model_path: str,
    batch_size: int,
    save_path: str,
    token_cont_model: str,
    near_tokens: int,
    process_method: Literal["chat", "cls"] = "cls",
    save_middle: bool = True,
):
    mp = MdProcess(
        md_path, llm_model_path, token_cont_model, near_tokens, process_method
    )
    mp.forward(batch_size, save_path, save_middle)


if __name__ == "__main__":
    # 记录日志文件
    logger.add("mdclean_run.log")
    parser = HfArgumentParser(MdPipeArgs)
    mdpipe_arg = parser.parse_yaml_file(
        "example/yaml/md_pipe.yaml", allow_extra_keys=True
    )[0]
    # 测试chat-llm过滤管道
    # test_llm_filter(mdpipe_arg.llm_model_path)
    # 测试管道全流程
    test_pipelines(
        mdpipe_arg.md_path,
        mdpipe_arg.llm_model_path,
        mdpipe_arg.batch_size,
        mdpipe_arg.save_path,
        mdpipe_arg.token_cont_model,
        mdpipe_arg.near_tokens,
        mdpipe_arg.process_method,
        mdpipe_arg.save_middle,
    )

