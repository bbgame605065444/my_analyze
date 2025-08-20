#!/usr/bin/env python3
"""
Qwen3 8B 本地推理设置脚本
自动配置Qwen3 8B模型用于CoT实验

CoT学习要点：
- 这个脚本帮助快速设置Qwen3本地推理环境
- 支持FP8量化以提升推理速度
- 提供配置验证和测试功能
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """检查系统要求"""
    print("检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本需要 >= 3.8")
        return False
    print("✅ Python版本满足要求")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️ CUDA不可用，将使用CPU推理（速度较慢）")
    except ImportError:
        print("⚠️ PyTorch未安装，稍后会自动安装")
    
    return True

def install_dependencies(use_fp8=False):
    """安装依赖"""
    print("\n安装依赖包...")
    
    # 基础依赖
    packages = [
        "torch>=2.0.0",
        "transformers>=4.51.0", 
        "accelerate>=0.20.0",
        "python-dotenv>=0.19.0"
    ]
    
    # FP8量化支持
    if use_fp8:
        packages.append("bitsandbytes>=0.41.0")
    
    for package in packages:
        print(f"安装 {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
            return False
    
    return True

def download_model(model_path="Qwen/Qwen3-8B"):
    """下载模型"""
    print(f"\n检查模型: {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 检查是否已下载
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("✅ 模型已存在，跳过下载")
            return True
        except:
            pass
        
        print("开始下载模型（这可能需要几分钟）...")
        
        # 下载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer下载完成")
        
        # 下载模型（仅配置，不实际加载权重）
        config = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=None  # 不实际加载到设备
        ).config
        print("✅ 模型配置下载完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        return False

def create_config_file(model_type="qwen3_local", use_fp8=False):
    """创建配置文件"""
    print(f"\n创建配置文件...")
    
    config_content = f"""# CoT 实验配置文件 - Qwen3 8B 本地推理
# 由 setup_qwen3.py 自动生成

# =============================================================================
# 模型配置
# =============================================================================

# 模型类型
MODEL_TYPE={model_type}

# Qwen3模型路径
QWEN_MODEL_PATH=Qwen/Qwen3-8B

# 是否启用FP8量化
USE_FP8={str(use_fp8).lower()}

# 设备映射
DEVICE_MAP=auto

# 数据类型
TORCH_DTYPE=auto

# =============================================================================
# 生成参数 (非思考模式)
# =============================================================================

# 温度参数
TEMPERATURE=0.7

# Top-p参数
TOP_P=0.8

# 最大生成token数
MAX_TOKENS=2048

# =============================================================================
# 实验配置
# =============================================================================

# 最大重试次数
MAX_RETRIES=3

# 基础延迟时间
BASE_DELAY=1

# 详细日志
VERBOSE_LOGGING=true
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ 配置文件 .env 创建完成")

def test_setup():
    """测试设置"""
    print("\n测试Qwen3设置...")
    
    try:
        # 导入配置
        from model_config import ConfigManager
        from model_interface import get_model_response
        
        # 加载配置
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"模型类型: {config.model_type}")
        print(f"模型路径: {config.qwen_model_path}")
        print(f"FP8量化: {'启用' if config.use_fp8 else '禁用'}")
        
        # 测试简单推理
        print("\n进行测试推理...")
        test_prompt = "Q: What is 2 + 3?\nA:"
        
        response = get_model_response(test_prompt, config)
        print(f"测试响应: {response}")
        
        print("✅ Qwen3设置测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Qwen3 8B 本地推理设置向导")
    print("=" * 60)
    
    # 检查系统要求
    if not check_requirements():
        print("❌ 系统要求检查失败")
        return False
    
    # 询问用户配置
    print("\n配置选项:")
    print("1. 是否启用FP8量化？(y/N)")
    use_fp8_input = input().strip().lower()
    use_fp8 = use_fp8_input in ['y', 'yes', '1', 'true']
    
    if use_fp8:
        print("✅ 将启用FP8量化（需要支持的GPU）")
    else:
        print("✅ 将使用标准精度")
    
    # 安装依赖
    if not install_dependencies(use_fp8):
        print("❌ 依赖安装失败")
        return False
    
    # 下载模型
    if not download_model():
        print("❌ 模型下载失败")
        return False
    
    # 创建配置文件
    create_config_file("qwen3_local", use_fp8)
    
    # 测试设置
    if test_setup():
        print("\n" + "=" * 60)
        print("🎉 Qwen3 8B 设置完成！")
        print("=" * 60)
        print("现在可以运行 CoT 实验:")
        print("  python main.py")
        print("\n或者运行测试:")
        print("  python run_tests.py")
        print("=" * 60)
        return True
    else:
        print("❌ 设置完成但测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)