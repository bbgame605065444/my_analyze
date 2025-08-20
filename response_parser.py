"""
响应解析器 - Chain-of-Thought 实现
解析模型响应并评估答案的正确性。

CoT核心概念：从模型的复杂响应中提取最终答案
- CoT响应通常很长，包含推理过程 + 最终答案
- 标准响应通常较短，直接给出答案
- 需要智能解析算法来找到真正的答案
"""

import re
import string

def parse_response(response):
    """
    从模型的响应中提取最终答案。
    
    参数说明：
        response (str): 模型返回的原始文本响应
        
    返回值：
        str: 提取出的答案，如果找不到答案则返回空字符串
        
    CoT学习要点：
    - CoT模式下，模型会产生很长的推理过程，最后才给出答案
    - 我们需要从这段长文本中准确提取出最终答案
    - 这个函数使用多种策略来确保能找到正确答案
    """
    
    if not response:
        return ""
    
    # 策略1：寻找 "The answer is X" 或 "answer is X" 模式
    # 这是最可靠的方法，因为模型经常用这种格式给出最终答案
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+(.+?)(?:\.|$)',
        r'(?:the\s+)?final\s+answer\s+is\s+(.+?)(?:\.|$)',
        r'(?:the\s+)?result\s+is\s+(.+?)(?:\.|$)',
        r'(?:the\s+)?solution\s+is\s+(.+?)(?:\.|$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response.lower())
        if match:
            answer = match.group(1).strip()
            # 清理答案，移除不必要的字符
            answer = re.sub(r'[^\w\s.-]', '', answer)
            return answer.strip()
    
    # 策略2：寻找响应中的最后一个数字
    # 对于数学问题，最后的数字通常就是答案
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    if numbers:
        return numbers[-1]
    
    # 策略3：寻找最后一个可能是答案的词
    # 用于非数字类型的答案（如yes/no, 单词等）
    sentences = response.split('.')
    if sentences:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            words = last_sentence.split()
            if words:
                # 返回最后一个有意义的词
                last_word = words[-1].strip()
                # 移除标点符号
                last_word = last_word.translate(str.maketrans('', '', string.punctuation))
                if last_word:
                    return last_word
    
    # 如果找不到答案，返回空字符串
    return ""

def evaluate_answer(predicted_answer, true_answer):
    """
    检查预测答案是否与真实答案匹配。
    
    参数说明：
        predicted_answer (str): 从模型响应中提取的答案
        true_answer (str): 真实答案（ground truth）
        
    返回值：
        bool: 如果答案匹配返回True，否则返回False
        
    CoT学习要点：
    - 这个函数决定了我们如何评判CoT的效果
    - 需要处理各种格式差异（大小写、标点、空格等）
    - 公平的评估对比较CoT和标准提示的效果至关重要
    """
    
    if not predicted_answer or not true_answer:
        return False
    
    # 标准化答案的函数
    def normalize(text):
        if not text:
            return ""
        # 转换为小写，消除大小写差异
        text = text.lower()
        # 移除多余空格，标准化空白字符
        text = ' '.join(text.split())
        # 移除标点符号，专注于内容本身
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 去除首尾空格
        return text.strip()
    
    # 对预测答案和真实答案都进行标准化
    normalized_predicted = normalize(predicted_answer)
    normalized_true = normalize(true_answer)
    
    # 比较标准化后的答案
    return normalized_predicted == normalized_true