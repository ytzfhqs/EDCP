from typing import Optional, Literal


from loguru import logger
from transformers import HfArgumentParser
from edcp.hparams import GradeArgs
from edcp.grade import GradeProcess


def test_pipelines(
    model_name: str,
    api_key: str,
    base_url: Optional[str],
    prompt_type: Literal["domain", "general"],
    domain: Optional[str],
    res_save_path: str = "data_grade.json",
):
    data = [
        {"text": "你好啊，我叫郝青松。你好啊，我叫李林潞。"},
        {
            "text": "临床药理学（Clinical Pharmacology）作为药理学科的分支，是研究药物在人体内作用规律和入体与药物间相互作用过程的交叉学科。它以药理学与临床医学为基础，阐述药物代谢动力学（药动学）、药物效应动力学（药效学）、毒副反应的性质及药物相互作用的规律等；其目的是促进医学与药学的结合、基础与临床的结合，以及指导临床合理用药，推动医学与药学的共同发展。目前，临床药理学的主要任务是通过对千血药浓度的监测，不断调整给药方案，为患者能够安全有效地使用药物提供保障，同时对新药的有效性与安全性做出科学评价；对上市后药品的不良反应进行监测，以保障患者用药安全；临床合理使用药物，改善治疗。因此，临床药理学被认为是现代医学及教学与科研中不可或缺的一门学科。随着循证和转化医学概念的提出，临床药理学的内涵得到更进一步的丰富。其发展对我国的新药开发、药品监督与管理、医疗质量与医药研究水平的提高起着十分重要的作用。\n第一节临床药理学发展概况\n一、国外药理学发展概况\n临床药理学在近30多年来发展迅速，逐渐从药理学中分割出来并形成了一门独立的学科。近年来，由千循证医学和转化医学的概念的提出也极大地促进了临床药理学的发展。\n早在20世纪30年代，来自康奈尔大学的Hai1-y Gold教授首次提出“临床药理学”的概念，进行了卓有成效的临床药理学研究。从青霉素的发现应用于临床，到磺胺类的合成，成功地用于治疗感染患者，人们对临床药理学有了启蒙的认识。1954年，美国John Hopkins大学建立了第一个临床药理室，开始讲授临床药理学课程。随后，瑞典、日本以及众多欧美国家纷纷成立了临床药理学机构，开设了临床药理学课程。其中以1972年在瑞典卡罗林斯卡(Karolinska)医学院附属霍定(Huddings)医院建立的临床药理室和英国皇家研究生医学院临床药理系规模较大，设备优良，接纳各国学者进修，被分别誉为“国际临床药理室和“国际药理培训中心”。\n然而，在20世纪60年代初期，震惊世界的沙利度胺(thalidomide)事件，给世界人民敲响警钟，促使人们重视新药的毒理学研究，同时加强对临床药理专业人员的培训工作。第十七届世界卫生大会(WHA)决议要求各国制订评价药物安全有效性指导原则；同年，＂赫尔辛基宣言”问世。此后，WHO根据WHA决议颁布了一系列对药物安全性评价、致畸、致突变、致癌、成瘾性等特殊毒性试验的技术要求；1966年Dollery CT在lancet杂志发表“Clinical Pharmacology“文章；1970年WHO对临床药理学的定义、活动范刚、组织、培训等方面作了详细阐明；1975年WHO发表了《人用药物评价指导原则》。\n20世纪70年代至80年代，临床药理学科在国际范围内迅速发展成长。如意大利于1967年在欧洲第一个成立了全国临床药理学会，美国在1971年也成立了临床药理学会。国际药理联合会(International Union of Pharmacology, IUPHAR)建立了临床药理专业组以促进临床药理学的发展。与此同时，世界各地的临床药理学期刊和专著犹如雨后春笋般问世。随着1980年在英国伦敦第一届国际临床药理学与治疗学会议(World Congress on Clinical Pharmacology&Therapeutics)的召开，近几十年来的国际临床药理学会议及学术交流变得越来越频繁。1983年和1986年分别在美国华盛顿和瑞典斯德哥尔摩召开了第二届和第三届国际临床药理学与治疗学会议。以后大约3年召开一次国际临床药理学与治疗学会议。其会议的宗旨是将基础药理与临床药理更密切地结合起来，为临床患者服务。会议内容涉及多个领域，如系统疾病的药物治疗、临床药理学研究设计及合理用药、不良反应监测等。\n目前国际上临床药理学发展较快的有美国、瑞典、英国、德国和日本等国家。\n二、国内药理学发展概况\n我国药理学工作者早在20世纪60年代初就注意到发展我国临床药理学问题，并在1961年千上海围绕“寻找新药的理论基础和临床实际”展开学术讨论会，强烈呼吁在国内建立临床药理学科。随后，1979年7月在北京召开了第一届“全国临床药理专题讨论会＂。由千社会各界人士的高度重视，以及临床药理专业人员及临床工作者的介入，我国临床药理学在很多方面得到了迅速的发展。此后，随着越来越多的专业人员及科研人员加入到大部队中来，形成了一支相当规模的临床药理专业队伍，促使我国临床药理学逐步走向成熟。"
        },
    ]
    gp = GradeProcess(
        data,
        model_name,
        api_key,
        base_url,
        prompt_type=prompt_type,
        domain=domain,
        res_save_path=res_save_path,
    )
    gp.forward()


if __name__ == "__main__":
    # 记录日志文件
    logger.add("metric_run.log")
    parser = HfArgumentParser(GradeArgs)
    grade_arg = parser.parse_yaml_file(
        "example/yaml/grade_pipe.yaml", allow_extra_keys=True
    )[0]
    test_pipelines(
        grade_arg.model_name,
        grade_arg.api_key,
        grade_arg.base_url,
        grade_arg.prompt_type,
        grade_arg.domain,
    )
