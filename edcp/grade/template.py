class LLMGradePrompt:
    DOMAIN_PROMPT = """以下是一段文本内容摘录。请使用以下5分制评分系统来评估该文本的写作水平、{domain}价值和实用性:
0分：如果文本没有提供任何{domain}价值,完全由无关信息（如广告、宣传材料、少儿不宜内容）组成。
1分：如果文本提供了一些可能有{domain}价值的基本信息，但包含较多的无关或非学术内容（如广告和宣传材料）。
2分：如果文本涉及某些与{domain}相关的元素，但与{domain}标准不太吻合。它可能将{domain}内容与非{domain}材料混杂，对潜在的有用的主题进行浅显概述,或以不连贯的写作风格呈现信息。
3分：如果文本适合{domain}使用，并介绍了与某些学校课程中可能学到的关键概念，或对个人发展有用的实用信息。它的内容连贯但可能不全面，或包含一些无关信息。它可能类似于教科书的一小段节选，可以学习但有明显局限，如涉及过于复杂的概念、过于具体的不重要事件。
4分：如果文本与{domain}高度相关，对个人学习发展有益，表现出清晰一致的写作风格。它可能类似于教科书的一个章节或教程，提供大量{domain}内容，极少包含无关信息，且概念对学生来说不会过于深奥。内容连贯、重点突出,对结构化学习有价值。
5分：如果文本内容摘录在{domain}价值上表现极好，完全适合小学、中学或大学教学或专业人士学习。它遵循详细的推理过程，写作风格易于理解，对主题提供深刻而全面的见解，不包含任何非{domain}性或无实用意义内容。
文本内容摘录:
{context}

在审查这段文本摘录后：请简要地为您的评分进行合理的解释，最多不超过100字，最后以“{domain}得分：<分数>”的格式结束。请根据所列出的标准系统地赋予分数。
"""

    GENERAL_PROMPT = """Below is an extract from a web page. Evaluate whether the page has a high natural language value and could be useful in an naturanl language task to train a good language model using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Zero score if the content contains only some meaningless content or private content, such as some random code, http url or copyright information, personally identifiable information, binary encoding of images.
- Add 1 point if the extract provides some basic information, even if it includes some useless contents like advertisements and promotional material.
- Add another point if the extract is written in good style, semantically fluent, and free of repetitive content and grammatical errors.
- Award a third point tf the extract has relatively complete semantic content, and is written in a good and fluent style, the entire content expresses something related to the same topic, rather than a patchwork of several unrelated items.
- A fourth point is awarded if the extract has obvious educational or literary value, or provides a meaningful point or content, contributes to the learning of the topic, and is written in a clear and consistent style. It may be similar to a chapter in a textbook or tutorial, providing a lot of educational content, including exercises and solutions, with little to no superfluous information. The content is coherent and focused, which is valuable for structured learning.
- A fifth point is awarded if the extract has outstanding educational value or is of very high information density, provides very high value and meaningful content, does not contain useless information, and is well suited for teaching or knowledge transfer. It contains detailed reasoning, has an easy-to-follow writing style, and can provide deep and thorough insights.


The extract:
<{context}>.

After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Quality score:  <total points>"
"""
