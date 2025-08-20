#!/usr/bin/env python3
"""
Chain-of-Thought 提示实验主脚本
复现论文《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》的核心发现

CoT核心理念：通过在提示中展示逐步推理过程，可以显著提升大语言模型的复杂推理能力。
这个实验将证明：CoT > 标准提示 在各种推理任务上的优越性。
"""

from experiment_orchestrator import run_experiment
from model_config import ConfigManager

def main():
    """
    主执行函数，运行所有实验并报告结果。
    
    CoT学习要点：
    - 这是整个研究的总控制台
    - 我们将在3种不同的推理任务上测试CoT的效果
    - 每个任务都会运行2次：标准提示 vs CoT提示
    - 通过对比结果，验证CoT的有效性
    - 支持多种模型配置：Gemini API、Qwen3本地推理、Qwen3 API
    """
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print("=" * 80)
    print("思维链提示实验 (CHAIN-OF-THOUGHT PROMPTING EXPERIMENTS)")
    print("复现论文: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models'")
    print("=" * 80)
    
    # 显示当前配置
    config_manager.print_config_summary(config)
    
    # 定义要评估的任务
    # 这三个任务代表了不同类型的推理能力
    tasks = [
        'gsm8k',                      # 小学数学 (算术推理)
        'csqa',                       # 常识问答 (常识推理)  
        'last_letter_concatenation'   # 尾字母连接 (符号推理)
    ]
    
    # 存储所有实验结果
    results = {}
    
    # 为每个任务运行实验
    # 这是科学实验的核心：对照实验设计
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"任务: {task.upper()}")
        print(f"{'='*60}")
        
        try:
            # 运行标准提示实验（对照组）
            # 模型只看到问题和直接答案，没有推理过程
            standard_accuracy = run_experiment(task, use_cot=False, config=config)
            
            # 运行思维链提示实验（实验组）  
            # 模型看到问题、推理过程、然后答案
            cot_accuracy = run_experiment(task, use_cot=True, config=config)
            
            # 存储结果
            # improvement > 0 意味着CoT有效
            results[task] = {
                'standard': standard_accuracy,
                'cot': cot_accuracy,
                'improvement': cot_accuracy - standard_accuracy
            }
            
        except Exception as e:
            print(f"运行 {task} 实验时出错: {e}")
            results[task] = {
                'standard': 0.0,
                'cot': 0.0,
                'improvement': 0.0
            }
    
    # 打印总结果
    # 这是整个研究的高潮：揭示CoT的效果
    print(f"\n{'='*80}")
    print("实验结果总结 (EXPERIMENT RESULTS SUMMARY)")
    print(f"{'='*80}")
    
    print(f"{'任务':<25} {'标准提示':<12} {'CoT提示':<12} {'改进幅度':<12}")
    print("-" * 65)
    
    total_standard = 0.0
    total_cot = 0.0
    
    # 逐个显示每个任务的结果
    for task, result in results.items():
        standard_acc = result['standard']
        cot_acc = result['cot']
        improvement = result['improvement']
        
        total_standard += standard_acc
        total_cot += cot_acc
        
        print(f"{task:<25} {standard_acc:<12.2%} {cot_acc:<12.2%} {improvement:+.2%}")
    
    # 计算平均值
    # 平均改进幅度是衡量CoT整体效果的关键指标
    num_tasks = len(tasks)
    avg_standard = total_standard / num_tasks
    avg_cot = total_cot / num_tasks
    avg_improvement = avg_cot - avg_standard
    
    print("-" * 65)
    print(f"{'平均':<25} {avg_standard:<12.2%} {avg_cot:<12.2%} {avg_improvement:+.2%}")
    
    # 打印关键发现
    # 这些发现将验证或反驳CoT的有效性
    print(f"\n{'='*80}")
    print("关键发现 (KEY FINDINGS)")
    print(f"{'='*80}")
    
    print(f"• 标准提示平均准确率: {avg_standard:.2%}")
    print(f"• 思维链提示平均准确率: {avg_cot:.2%}")
    print(f"• CoT平均改进幅度: {avg_improvement:+.2%}")
    
    # 找出改进最大的任务
    best_improvement = max(results.values(), key=lambda x: x['improvement'])
    best_task = [task for task, result in results.items() if result == best_improvement][0]
    
    print(f"• CoT最大改进: {best_improvement['improvement']:+.2%} (任务: {best_task})")
    
    # 检查CoT是否在所有任务上都有帮助
    # 这是验证CoT普适性的重要指标
    all_improved = all(result['improvement'] > 0 for result in results.values())
    if all_improved:
        print("• 思维链提示在所有任务上都提升了性能 ✓")
    else:
        print("• 思维链提示并非在所有任务上都有提升")
    
    print(f"\n{'='*80}")
    print("实验完成 (EXPERIMENT COMPLETED)")
    print(f"\n** CoT核心原理总结 **")
    print("思维链提示通过展示逐步推理过程，教会模型'如何思考'，")
    print("从而在复杂推理任务上获得显著的性能提升。")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()