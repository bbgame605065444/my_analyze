"""
提示工程师 - Chain-of-Thought 实现
为标准提示和CoT提示构建few-shot格式的提示词。

CoT核心概念：这是整个系统的关键组件！
- Few-shot学习：通过几个示例教会模型如何回答问题
- CoT vs 标准提示的关键区别：是否包含逐步推理过程
- 提示格式：Q: 问题 A: 答案(CoT时包含推理过程)
"""

def create_few_shot_prompt(exemplars, new_question, use_cot):
    """
    为语言模型构建few-shot提示词。
    
    参数说明：
        exemplars (list): 示例字典列表，包含 'question', 'answer', 'chain_of_thought'
        new_question (str): 需要回答的新问题
        use_cot (bool): 是否包含思维链推理过程
        
    返回值：
        str: 格式化的提示字符串
        
    CoT学习要点：
    - 这个函数决定了模型看到的提示格式
    - use_cot=True时，模型会看到完整的推理过程，学会逐步思考
    - use_cot=False时，模型只看到问题和直接答案，容易养成直接回答的习惯
    """
    
    prompt_parts = []
    
    # 添加few-shot示例
    # 这些示例教会模型如何回答问题的格式和方法
    for exemplar in exemplars:
        question = exemplar['question']
        answer = exemplar['answer']
        chain_of_thought = exemplar['chain_of_thought']
        
        # 所有示例都以"Q: 问题"的格式开始
        prompt_parts.append(f"Q: {question}")
        
        if use_cot:
            # CoT模式：包含思维链推理过程 + 最终答案
            # 这是CoT的核心！模型学会先推理，再给答案
            prompt_parts.append(f"A: {chain_of_thought} {answer}")
        else:
            # 标准模式：只有直接答案，不包含推理过程
            # 模型学会直接给出答案，跳过中间推理步骤
            prompt_parts.append(f"A: {answer}")
        
        # 示例之间用空行分隔，让格式更清晰
        prompt_parts.append("")
    
    # 添加新问题，这是我们真正想让模型回答的问题
    prompt_parts.append(f"Q: {new_question}")
    prompt_parts.append("A:")  # 让模型知道该开始回答了
    
    # 将所有部分用换行符连接成完整的提示词
    return "\n".join(prompt_parts)