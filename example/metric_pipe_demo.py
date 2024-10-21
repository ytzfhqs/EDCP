from typing import List, Dict, Any


from transformers import HfArgumentParser
from edcp.hparams.mcpipe_args import McPipeArgs
from edcp.metric.calppl import CPPl
from edcp.metric.nlpfeat import NlpFeat
from edcp.metric.pipelines import MetricProcess
from edcp.metric.importance import ImportFeat
from loguru import logger


def test_cal_ppl(llm_model_path):
    data: List[Dict[str, Any]] = [{"text": "测试文本。"}]
    cl = CPPl(llm_model_path)
    print(cl.forward(data))


def test_cal_nlpfeat(text_column):
    data = [{"text": "你好啊，我叫小松鼠。你好啊，我叫小雪球。"}]
    nf = NlpFeat(text_column)
    res = nf.forward(data)
    print(res)


def test_cal_import(book_data_or_path):
    data = [{"text": "你好啊，我叫小松鼠。你好啊，我叫小雪球。"}]
    imf = ImportFeat(book_data_or_path)
    res = imf.forward(data)
    print(res)


def test_pipelines(
    book_data_or_path,
    wordgram_model_path,
    save_wordgram_model_path,
    llm_model_path,
    fasttext_model_path,
    text_column,
    idx_column,
    num_perm,
    res_save_path
):
    data = [
        {"text": "你好啊，我叫小松鼠。你好啊，我叫小雪球。", "id_int": 0},
        {"text": "你好啊，我叫松鼠。你好啊，我叫雪球。", "id_int": 1},
        {"text": "你好啊，我叫小松鼠。", "id_int": 2},
    ]
    mcp = MetricProcess(
        data,
        book_data_or_path,
        wordgram_model_path,
        save_wordgram_model_path,
        llm_model_path,
        fasttext_model_path,
        text_column,
        idx_column,
        num_perm,
        res_save_path
    )
    print(mcp.forward())


if __name__ == "__main__":
    # 记录日志文件
    logger.add("metric_run.log")
    parser = HfArgumentParser(McPipeArgs)
    mcpipe_arg = parser.parse_yaml_file(
        "example/yaml/mc_pipe.yaml", allow_extra_keys=True
    )[0]
    # 测试ppl评估管道
    # test_cal_ppl(mcpipe_arg.llm_model_path)
    # 测试NLP特征评估管道
    # test_cal_nlpfeat(mcpipe_arg.text_column)
    # 测试importance评估管道
    test_cal_import(mcpipe_arg.book_data_or_path)
    test_pipelines(
        mcpipe_arg.book_data_or_path,
        mcpipe_arg.wordgram_model_path,
        mcpipe_arg.save_wordgram_model_dir,
        mcpipe_arg.llm_model_path,
        mcpipe_arg.fasttext_model_path,
        mcpipe_arg.text_column,
        mcpipe_arg.idx_column,
        mcpipe_arg.num_perm,
        mcpipe_arg.res_save_path
    )
