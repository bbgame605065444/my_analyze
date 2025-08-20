"""
数据集处理器 - Chain-of-Thought 实现
为各种推理任务加载和管理数据集。

CoT核心概念：为了测试思维链提示的效果，我们需要不同类型的推理任务数据集。
这个模块提供三种经典的推理任务：数学推理、常识推理、符号推理。
"""

def load_dataset(task_name):
    """
    为给定的推理任务加载特定数据集。
    
    参数说明：
        task_name (str): 任务名称 ('gsm8k', 'csqa', 'last_letter_concatenation')
        
    返回值：
        list: 包含字典的列表，每个字典都有 'question', 'answer', 'chain_of_thought' 键
        
    异常：
        ValueError: 如果任务名称不被支持
        
    CoT学习要点：
    - 每个数据样本都包含问题、答案和思维链推理过程
    - 思维链是解决问题的逐步推理步骤，这是CoT的核心
    - 我们用前面的样本作为few-shot示例，教会模型如何进行逐步推理
    """
    
    if task_name == 'gsm8k':
        # GSM8K：小学数学推理任务
        # 这类问题需要多步骤的算术推理，是CoT最擅长的任务类型
        return [
            {
                'question': 'Janet has 24 apples. She gives 8 to her neighbor and then buys 12 more. How many apples does she have now?',
                'answer': '28',
                # 思维链：展示逐步推理过程，而不是直接给答案
                'chain_of_thought': 'Janet starts with 24 apples. She gives away 8 apples, so she has 24 - 8 = 16 apples left. Then she buys 12 more apples, so she has 16 + 12 = 28 apples.'
            },
            {
                'question': 'A store sells pencils for $0.25 each. If Tom buys 16 pencils, how much does he pay?',
                'answer': '4.00',
                # 乘法推理：需要理解单价×数量=总价的概念
                'chain_of_thought': 'Each pencil costs $0.25. Tom buys 16 pencils. The total cost is 16 × $0.25 = $4.00.'
            },
            {
                'question': 'Maria bakes 36 cookies and puts them in boxes. Each box holds 9 cookies. How many boxes does she need?',
                'answer': '4',
                # 除法推理：需要理解总数÷每份数量=份数的概念
                'chain_of_thought': 'Maria has 36 cookies total. Each box holds 9 cookies. To find the number of boxes, I divide 36 ÷ 9 = 4 boxes.'
            },
            {
                'question': 'A car travels 240 miles in 4 hours. What is its average speed in miles per hour?',
                'answer': '60',
                # 速度计算：需要理解速度=距离÷时间的公式
                'chain_of_thought': 'To find average speed, I divide total distance by total time. The car travels 240 miles in 4 hours, so the average speed is 240 ÷ 4 = 60 miles per hour.'
            }
        ]
        
    elif task_name == 'csqa':
        # CommonsenseQA：常识推理任务
        # 需要运用日常生活常识进行推理，不仅仅是数学计算
        return [
            {
                'question': 'Where would you find a fox that is not real?',
                'answer': 'storybook',
                # 常识推理：需要理解"不真实"意味着虚构，虚构的动物出现在故事中
                'chain_of_thought': 'A fox that is not real would be fictional. Fictional foxes appear in stories, books, movies, and other forms of media. The most common place to find a fictional fox would be in a storybook.'
            },
            {
                'question': 'What do people typically do when they are bored?',
                'answer': 'look for entertainment',
                # 心理状态推理：需要理解无聊的感受和人们的应对行为
                'chain_of_thought': 'When people are bored, they feel understimulated and seek activities to engage their mind. This typically leads them to look for entertainment such as watching TV, reading, playing games, or socializing.'
            },
            {
                'question': 'If you want to learn about the past, what should you do?',
                'answer': 'read history',
                # 学习方法推理：需要理解获取历史知识的最佳途径
                'chain_of_thought': 'To learn about the past, you need to access information about historical events and periods. The most systematic way to do this is to read history books, which contain researched and documented information about past events.'
            },
            {
                'question': 'What happens to food when you put it in the freezer?',
                'answer': 'gets cold',
                # 物理现象推理：需要理解温度对物质状态的影响
                'chain_of_thought': 'A freezer is designed to maintain very low temperatures, typically below 0°F (-18°C). When you put food in the freezer, it gets cold and freezes, which preserves it by slowing down bacterial growth and chemical reactions.'
            }
        ]
        
    elif task_name == 'last_letter_concatenation':
        # Last Letter Concatenation：符号推理任务
        # 这是纯符号操作任务，需要精确的步骤执行能力
        return [
            {
                'question': 'Take the last letter of each word and concatenate them: "apple" "banana" "cherry"',
                'answer': 'eay',
                # 符号操作：需要逐个处理每个单词，提取最后一个字母
                'chain_of_thought': 'I need to find the last letter of each word. The last letter of "apple" is "e". The last letter of "banana" is "a". The last letter of "cherry" is "y". Concatenating them gives "eay".'
            },
            {
                'question': 'Take the last letter of each word and concatenate them: "dog" "cat" "bird"',
                'answer': 'gtd',
                # 重复的符号操作：展示了相同的处理模式
                'chain_of_thought': 'Looking at each word: "dog" ends with "g", "cat" ends with "t", "bird" ends with "d". Concatenating these letters gives "gtd".'
            },
            {
                'question': 'Take the last letter of each word and concatenate them: "sun" "moon" "star"',
                'answer': 'nnr',
                # 使用箭头符号使步骤更清晰
                'chain_of_thought': 'For each word, I take the last letter: "sun" → "n", "moon" → "n", "star" → "r". Concatenating gives "nnr".'
            },
            {
                'question': 'Take the last letter of each word and concatenate them: "red" "blue" "green"',
                'answer': 'den',
                # 另一种表达方式，但逻辑相同
                'chain_of_thought': 'The last letters are: "red" → "d", "blue" → "e", "green" → "n". Concatenating them gives "den".'
            }
        ]
        
    else:
        # 如果任务名称不被支持，抛出错误
        raise ValueError(f"Unsupported task name: {task_name}. Supported tasks: 'gsm8k', 'csqa', 'last_letter_concatenation'")