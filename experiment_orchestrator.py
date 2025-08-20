"""
实验协调器 - Chain-of-Thought 实现
协调实验运行并计算准确率指标。

CoT核心概念：这是整个实验的指挥中心
- 组织实验流程：加载数据 → 构建提示 → 调用模型 → 解析响应 → 评估结果
- 对比实验：同一任务下比较标准提示 vs CoT提示的效果
- 准确率计算：统计正确答案的比例，衡量推理能力
"""

from dataset_handler import load_dataset
from prompt_engineer import create_few_shot_prompt
from model_interface import get_model_response, ModelConfig
from model_config import ConfigManager
from response_parser import parse_response, evaluate_answer

def run_experiment(task_name, use_cot, config=None):
    """
    为给定任务和提示方法运行单个实验。
    
    参数说明：
        task_name (str): 任务名称
        use_cot (bool): 是否使用思维链提示
        config (ModelConfig, optional): 模型配置，默认从环境变量加载
        
    返回值：
        float: 模型在该任务上的准确率（0.0到1.0）
        
    CoT学习要点：
    - 这个函数是科学实验的核心，严格控制变量
    - 唯一变量：是否使用CoT（use_cot参数）
    - 其他条件相同：相同的数据集、相同的模型、相同的评估标准
    - 通过对比准确率，我们能看出CoT的真实效果
    - 支持多种模型：Gemini API、Qwen3本地推理、Qwen3 API
    """
    
    # 获取模型配置
    if config is None:
        config_manager = ConfigManager()
        config = config_manager.get_config()
    
    print(f"\n运行实验: {task_name} ({'CoT' if use_cot else '标准'}) - 模型: {config.model_type}")
    
    # 加载数据集
    dataset = load_dataset(task_name)
    
    if len(dataset) == 0:
        print("警告: 数据集为空")
        return 0.0
    
    # 将前2个样本作为few-shot示例，其余作为测试问题
    # 这是CoT实验的标准做法：用少量示例教会模型，用剩余数据测试效果
    num_exemplars = min(2, len(dataset) - 1)  # 至少留1个用于测试
    exemplars = dataset[:num_exemplars]        # few-shot示例
    test_questions = dataset[num_exemplars:]   # 测试问题
    
    if len(test_questions) == 0:
        print("警告: 没有可用的测试问题")
        return 0.0
    
    print(f"使用 {len(exemplars)} 个示例和 {len(test_questions)} 个测试问题")
    
    correct_answers = 0
    total_questions = len(test_questions)
    
    # 处理每个测试问题
    # 这是实验的核心循环，每次测试都严格按照相同步骤进行
    for i, test_item in enumerate(test_questions):
        print(f"处理问题 {i + 1}/{total_questions}...")
        
        question = test_item['question']
        true_answer = test_item['answer']
        
        # 创建提示词（标准 vs CoT 的关键差异在这里体现）
        prompt = create_few_shot_prompt(exemplars, question, use_cot)
        
        try:
            # 获取模型响应
            # CoT模式：模型会产生长推理过程 + 答案
            # 标准模式：模型倾向于直接给出答案
            # 使用配置的模型进行推理
            response = get_model_response(prompt, config)
            
            # 解析响应以提取答案
            predicted_answer = parse_response(response)
            
            # 评估答案正确性
            is_correct = evaluate_answer(predicted_answer, true_answer)
            
            if is_correct:
                correct_answers += 1
                print(f"  ✓ 正确: {predicted_answer}")
            else:
                print(f"  ✗ 错误: {predicted_answer} (期望: {true_answer})")
                
        except Exception as e:
            print(f"  ✗ 处理问题时出错: {e}")
            # 继续处理下一个问题，确保实验的完整性
    
    # 计算准确率
    # 这是衡量CoT效果的关键指标
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    print(f"准确率: {correct_answers}/{total_questions} = {accuracy:.2%}")
    
    return accuracy