from mdclean.pipelines import MdProcess
from mdclean.LLMFilter import ChatModel


def test_llmfilter(llm_model_path: str):
    book_name = "儿科学"
    context = [
        "儿科学／黄国英，孙锟，罗小平主编．--10版北京：人民卫生出版社，2024.6．-",
        "急性支气管炎一般不发热或仅有低热，全身状况好，以咳嗽为主要症状，肺部可闻及十湿啰音，多不固定，随咳嗽而改变。胸部X线检查示肺纹理增多、排列素乱。",
    ]
    cm = ChatModel(llm_model_path)
    res = cm.forward(book_name, context)
    return res


def test_pipelines(md_path: str, llm_model_path: str, batch_size: int, save_path: str):
    mp = MdProcess(md_path, llm_model_path)
    mp.forward(batch_size, save_path)


if __name__ == "__main__":
    test_pipelines("./", "Qwen2.5-7B-Instruct", 8, "data.json")
