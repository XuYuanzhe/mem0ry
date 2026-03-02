from enum import Enum


class MemoryType(Enum):
    """
    SEMANTIC (语义记忆)
    定义：关于事实、概念、知识和通用信息的记忆。它不依赖于特定的时间或地点。
    人类例子：
        “巴黎是法国的首都。”
        “猫是一种哺乳动物。”
        “用户喜欢喝咖啡，不喜欢喝茶。”（这是从多次对话中总结出的用户画像/偏好）

    EPISODIC (情景记忆)
    定义：关于特定事件、经历、时间点和地点的记忆。它像是一部“自传体电影”。
    人类例子：
        “上周二我和张三在星巴克开会。”
        “昨天我心情很不好，因为下雨了。”
        “刚才用户让我帮他写了一封邮件。”

    PROCEDURAL (程序性记忆)
    定义：关于“怎么做”的记忆，即技能、习惯、操作流程和规则。通常是隐式的，难以用语言完全描述，但体现在行动中。
    人类例子：
        骑自行车、游泳的动作肌肉记忆。
        “遇到愤怒的客户时，先安抚再解决问题”（这是一种处理流程）。
        “每次写代码前都要先写测试用例”（这是一种工作流习惯）。

    可以简单概括：
        Semantic = 知识库 (Wiki)
        Episodic = 日记本 (Diary)
        Procedural = 操作手册 (Manual)
    """
    SEMANTIC = "semantic_memory"
    EPISODIC = "episodic_memory"
    PROCEDURAL = "procedural_memory"
