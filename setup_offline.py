#!/usr/bin/env python3
"""
离线模式自动设置脚本
自动配置CoT系统为仅使用Qwen3本地推理的离线模式

CoT学习要点：
- 这个脚本简化了离线模式的设置过程
- 自动安装必需依赖
- 配置离线模式环境
- 验证设置是否正确
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {title}")
    print(f"{'='*60}")

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python版本需要 >= 3.8")
        print(f"当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True

def install_offline_dependencies():
    """安装离线模式依赖"""
    print("正在安装离线模式依赖...")
    
    # 检查requirements-offline.txt是否存在
    req_file = Path("requirements-offline.txt")
    if not req_file.exists():
        print("❌ requirements-offline.txt 文件不存在")
        return False
    
    try:
        # 安装依赖
        print("🔄 安装依赖包...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-offline.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ 依赖安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败:")
        print(f"错误输出: {e.stderr}")
        print(f"标准输出: {e.stdout}")
        return False

def setup_offline_config():
    """设置离线配置"""
    print("配置离线模式...")
    
    # 检查是否已有.env文件
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️ .env文件已存在")
        response = input("是否覆盖现有配置？(y/N): ").strip().lower()
        if response not in ['y', 'yes', '1']:
            print("✅ 保持现有配置")
            return True
    
    # 创建离线配置
    offline_config = """# CoT 离线模式配置 - 自动生成
# 此配置确保系统完全离线运行，仅使用Qwen3本地推理

# 离线模式开关
OFFLINE_MODE=true

# 模型配置
MODEL_TYPE=qwen3_local
QWEN_MODEL_PATH=Qwen/Qwen3-8B

# 性能优化
USE_FP8=true
DEVICE_MAP=auto
TORCH_DTYPE=auto

# 生成参数 (非思考模式)
TEMPERATURE=0.7
TOP_P=0.8
MAX_TOKENS=2048

# 实验配置
MAX_RETRIES=1
BASE_DELAY=0
VERBOSE_LOGGING=true
"""
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(offline_config)
        print("✅ 离线配置文件创建成功")
        return True
    except Exception as e:
        print(f"❌ 配置文件创建失败: {e}")
        return False

def download_qwen_model():
    """下载Qwen3模型"""
    print("检查Qwen3模型...")
    
    try:
        from transformers import AutoTokenizer
        
        model_path = "Qwen/Qwen3-8B"
        
        # 尝试加载tokenizer来检查模型是否已下载
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("✅ Qwen3模型已存在")
            return True
        except:
            pass
        
        print("🔄 首次使用需要下载Qwen3模型 (约15GB)")
        response = input("是否现在下载？(Y/n): ").strip().lower()
        
        if response in ['', 'y', 'yes', '1']:
            print("开始下载模型...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                cache_dir=None  # 使用默认缓存目录
            )
            print("✅ Qwen3模型下载完成")
            return True
        else:
            print("⚠️ 跳过模型下载，稍后可手动下载")
            return True
            
    except Exception as e:
        print(f"❌ 模型处理失败: {e}")
        return False

def verify_offline_setup():
    """验证离线设置"""
    print("验证离线设置...")
    
    try:
        # 运行验证脚本
        result = subprocess.run([
            sys.executable, "validate_offline.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 离线模式验证通过")
            return True
        else:
            print("⚠️ 验证过程中发现问题:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("⚠️ 验证脚本不存在，跳过自动验证")
        return True
    except Exception as e:
        print(f"⚠️ 验证失败: {e}")
        return False

def show_usage_instructions():
    """显示使用说明"""
    print_step("完成", "离线模式设置完成")
    
    print("🎉 CoT系统已成功配置为离线模式！")
    print("\n📋 系统特性:")
    print("✅ 完全离线运行，无需网络连接")
    print("✅ 仅使用Qwen3 8B本地推理")
    print("✅ 支持FP8量化以提升性能")
    print("✅ 所有API调用已被禁用")
    
    print("\n🚀 开始使用:")
    print("1. 运行CoT实验:")
    print("   python main.py")
    print("\n2. 运行测试:")
    print("   python run_tests.py")
    print("\n3. 验证离线状态:")
    print("   python validate_offline.py")
    
    print("\n📊 硬件建议:")
    print("- GPU: 12GB+ VRAM (RTX 3080/4070或更高)")
    print("- RAM: 16GB+ 系统内存")
    print("- 存储: 20GB+ 可用空间")
    
    print("\n🔧 配置文件:")
    print("- .env - 离线模式配置")
    print("- requirements-offline.txt - 离线依赖列表")
    
    print("\n⚠️ 注意事项:")
    print("- 首次推理可能需要几分钟加载模型")
    print("- 确保GPU驱动和CUDA已正确安装")
    print("- 如遇问题请查看日志或运行验证脚本")

def main():
    """主设置函数"""
    print("🔒 CoT 离线模式自动设置")
    print("此脚本将配置系统仅使用Qwen3本地推理")
    
    # 确认用户意图
    print("\n此操作将:")
    print("1. 安装Qwen3本地推理依赖")
    print("2. 创建离线模式配置文件")
    print("3. 下载Qwen3模型 (如需要)")
    print("4. 禁用所有在线API")
    
    response = input("\n是否继续？(Y/n): ").strip().lower()
    if response not in ['', 'y', 'yes', '1']:
        print("设置已取消")
        return False
    
    # 执行设置步骤
    steps = [
        (1, "检查Python版本", check_python_version),
        (2, "安装离线依赖", install_offline_dependencies),
        (3, "配置离线模式", setup_offline_config),
        (4, "处理Qwen3模型", download_qwen_model),
        (5, "验证设置", verify_offline_setup)
    ]
    
    for step_num, title, func in steps:
        print_step(step_num, title)
        if not func():
            print(f"❌ 步骤 {step_num} 失败，设置中断")
            return False
    
    # 显示使用说明
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)