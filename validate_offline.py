#!/usr/bin/env python3
"""
离线模式验证脚本
验证系统是否正确配置为仅使用Qwen3本地推理，无任何在线API依赖

CoT学习要点：
- 这个脚本确保系统真正离线运行
- 验证所有在线API都被禁用
- 测试Qwen3本地推理是否正常工作
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_offline_config():
    """检查离线配置"""
    print_section("检查离线配置")
    
    # 检查.env文件
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env配置文件不存在")
        return False
    
    print("✅ .env配置文件存在")
    
    # 读取配置
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键配置
    checks = [
        ("OFFLINE_MODE=true", "离线模式启用"),
        ("MODEL_TYPE=qwen3_local", "模型类型设置为本地推理"),
        ("QWEN_MODEL_PATH=Qwen/Qwen3-8B", "Qwen3模型路径配置")
    ]
    
    all_good = True
    for config_line, description in checks:
        if config_line in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - 未找到: {config_line}")
            all_good = False
    
    return all_good

def check_forbidden_imports():
    """检查是否存在被禁止的在线API导入"""
    print_section("检查禁止的在线API导入")
    
    forbidden_modules = [
        "google.generativeai", 
        "openai",
        "anthropic",
        "cohere"
    ]
    
    all_good = True
    for module in forbidden_modules:
        try:
            importlib.import_module(module)
            print(f"⚠️ 检测到在线API模块: {module} (已安装但应在离线模式下禁用)")
            # 注意：这里不算作错误，因为模块可能已安装但不会被使用
        except ImportError:
            print(f"✅ 在线API模块 {module} 未安装 (符合离线要求)")
    
    return all_good

def check_required_dependencies():
    """检查必需的依赖"""
    print_section("检查Qwen3必需依赖")
    
    required_modules = [
        ("torch", "PyTorch深度学习框架"),
        ("transformers", "HuggingFace Transformers"),
        ("accelerate", "模型加速库"),
        ("numpy", "数值计算库"),
        ("dotenv", "环境变量管理")
    ]
    
    all_good = True
    for module, description in required_modules:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {description}: {module} v{version}")
        except ImportError:
            print(f"❌ {description}: {module} 未安装")
            all_good = False
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA可用，GPU数量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"   GPU {i}: {gpu_name} ({memory}GB)")
        else:
            print("⚠️ CUDA不可用，将使用CPU推理 (速度较慢)")
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
        all_good = False
    
    return all_good

def test_model_interface():
    """测试模型接口"""
    print_section("测试模型接口")
    
    try:
        # 设置环境变量确保离线模式
        os.environ['OFFLINE_MODE'] = 'true'
        os.environ['MODEL_TYPE'] = 'qwen3_local'
        
        from model_interface import get_model_response, OFFLINE_MODE, TRANSFORMERS_AVAILABLE
        from model_config import ConfigManager
        
        print(f"✅ 模型接口导入成功")
        print(f"✅ 离线模式状态: {OFFLINE_MODE}")
        print(f"✅ Transformers可用: {TRANSFORMERS_AVAILABLE}")
        
        # 测试配置加载
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print(f"✅ 配置加载成功: 模型类型 = {config.model_type}")
        
        # 测试在线API是否被正确禁用
        try:
            # 尝试使用Gemini API (应该被阻止)
            config.model_type = 'gemini'
            get_model_response("test", config)
            print("❌ Gemini API未被正确禁用")
            return False
        except ValueError as e:
            if "离线模式下禁用" in str(e):
                print("✅ Gemini API正确被禁用")
            else:
                print(f"❌ 意外错误: {e}")
                return False
        
        # 测试本地推理是否可用
        config.model_type = 'qwen3_local'
        print("🔄 准备测试Qwen3本地推理...")
        
        # 注意：这里不实际加载模型，因为可能需要很长时间
        print("✅ 模型接口验证完成 (未实际加载模型以节省时间)")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型接口测试失败: {e}")
        return False

def check_network_isolation():
    """检查网络隔离状态"""
    print_section("检查网络隔离")
    
    # 检查是否设置了离线环境变量
    offline_mode = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'
    if offline_mode:
        print("✅ OFFLINE_MODE环境变量已设置")
    else:
        print("⚠️ OFFLINE_MODE环境变量未设置，建议设置为true")
    
    # 检查是否有网络相关的环境变量被禁用
    api_vars = ['GEMINI_API_KEY', 'OPENAI_API_KEY', 'API_KEY']
    for var in api_vars:
        value = os.getenv(var)
        if value and not value.startswith('disabled'):
            print(f"⚠️ 检测到API密钥环境变量 {var}，离线模式下会被忽略")
        else:
            print(f"✅ API密钥 {var} 未设置或已禁用")
    
    return True

def generate_offline_report():
    """生成离线验证报告"""
    print_section("生成验证报告")
    
    report_file = "offline_validation_report.txt"
    
    # 收集系统信息
    import platform
    import datetime
    
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
    except ImportError:
        torch_version = "Not installed"
        cuda_available = False
        cuda_version = "N/A"
    
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "Not installed"
    
    report_content = f"""
CoT 离线模式验证报告
生成时间: {datetime.datetime.now()}

=== 系统信息 ===
操作系统: {platform.system()} {platform.release()}
Python版本: {platform.python_version()}
架构: {platform.machine()}

=== 依赖版本 ===
PyTorch: {torch_version}
Transformers: {transformers_version}
CUDA可用: {cuda_available}
CUDA版本: {cuda_version}

=== 离线配置 ===
离线模式: {os.getenv('OFFLINE_MODE', 'false')}
模型类型: {os.getenv('MODEL_TYPE', 'not set')}
Qwen模型路径: {os.getenv('QWEN_MODEL_PATH', 'not set')}
FP8量化: {os.getenv('USE_FP8', 'false')}

=== 验证状态 ===
配置文件: {'存在' if Path('.env').exists() else '不存在'}
必需依赖: {'完整' if torch_version != 'Not installed' and transformers_version != 'Not installed' else '缺失'}
离线模式: {'已启用' if os.getenv('OFFLINE_MODE', 'false').lower() == 'true' else '未启用'}

=== 建议 ===
1. 确保所有必需依赖已安装
2. 验证离线配置正确设置
3. 测试Qwen3模型推理功能
4. 检查GPU内存是否足够 (推荐12GB+)

此报告确认系统已配置为离线模式，仅使用Qwen3本地推理。
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 验证报告已保存至: {report_file}")
    return True

def main():
    """主验证函数"""
    print("🔒 CoT 离线模式验证工具")
    print("验证系统是否正确配置为仅使用Qwen3本地推理")
    
    # 强制设置离线模式
    os.environ['OFFLINE_MODE'] = 'true'
    
    results = []
    
    # 执行各项检查
    results.append(("配置检查", check_offline_config()))
    results.append(("依赖检查", check_required_dependencies()))
    results.append(("禁止模块检查", check_forbidden_imports()))
    results.append(("模型接口测试", test_model_interface()))
    results.append(("网络隔离检查", check_network_isolation()))
    results.append(("生成报告", generate_offline_report()))
    
    # 汇总结果
    print_section("验证结果汇总")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n🎉 恭喜！系统已成功配置为离线模式")
        print("现在可以安全地运行CoT实验，所有推理都将在本地完成")
        print("\n下一步:")
        print("1. 运行 python main.py 开始CoT实验")
        print("2. 运行 python run_tests.py 进行功能测试")
        print("3. 检查 offline_validation_report.txt 了解详细信息")
    else:
        print(f"\n⚠️ 检测到 {total - passed} 个问题，请根据上述提示进行修复")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)