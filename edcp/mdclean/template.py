class LLMFilterPrompt:
    SYSTEM = "你是一位专业的{book}医学专家"
    # 用于LLM区分正文与无关内容的提示词
    PROMPT = """
    ## 非正文特征
     - 没有提供任何与{book}相关的内容，完全由无关信息(如书本简介、出版社信息、作者介绍、序言、前言、修订说明、中英文对照表、编写规范等)组成。
     - 以章节标题开头，如：第一节、第一章、一、二，字数较少（<20），且以数字结尾。
     - 仅包含表或图编号与标题，如：表11-4儿童重症肺炎诊断标准、表11-2不同年龄阶段儿童的肺功能检测技术及内容、图1-2细胞破裂图、图3-1藻类结构图。
     - 内容较短，无完整句意。
     - 仅由英文单词或数字与极少量的中文组成。
     - 无标点符号或标点符号使用不恰当。
     
    ## 正文特征
     - 提供大量{book}相关的内容，极少包含无关信息。
     - 表现出清晰一致的写作风格。
    
    ## 主要任务
    根据上述<正文特征>与<非正文特征>判断下面这段话是否属于教科书《{book}》的正文内容，若属于，输出True，若不属于，输出False。禁止输出除True、False外的任何其他无关内容。
    
    {context}
    """
