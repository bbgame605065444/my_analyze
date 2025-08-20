"""
模型配置管理器 - Chain-of-Thought 实现
提供便捷的配置管理和模型切换功能。

CoT核心概念：不同的模型可能需要不同的配置
- 本地模型：需要考虑硬件资源、量化等
- API模型：需要考虑费用、速率限制等
- 配置管理：提供统一的配置接口
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class ModelConfig:
    """模型配置数据类"""
    # 基础配置
    model_type: str = 'gemini'  # 'gemini', 'qwen3_local', 'qwen3_api'
    model_name: str = 'gemini-2.0-flash-exp'
    api_key: Optional[str] = None
    api_base: str = 'https://api.openai.com/v1'
    
    # Qwen3特定配置
    qwen_model_path: str = 'Qwen/Qwen3-8B'
    use_fp8: bool = False
    device_map: str = 'auto'
    torch_dtype: str = 'auto'
    
    # 生成参数 (非思考模式推荐值)
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048
    
    # 实验配置
    max_retries: int = 3
    base_delay: int = 1

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认查找 .env 文件
        """
        self.config_file = config_file or self._find_config_file()
        self._load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """查找配置文件"""
        possible_files = ['.env', 'config.env', '.config']
        
        for file in possible_files:
            if os.path.exists(file):
                return file
        
        return None
    
    def _load_config(self):
        """加载配置文件"""
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
    
    def get_config(self, model_type: Optional[str] = None) -> ModelConfig:
        """
        获取模型配置
        
        Args:
            model_type: 指定模型类型，覆盖环境变量设置
            
        Returns:
            ModelConfig: 配置对象
        """
        # 从环境变量读取配置
        config = ModelConfig()
        
        # 基础配置
        config.model_type = model_type or os.getenv('MODEL_TYPE', 'gemini')
        config.model_name = os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp')
        config.api_key = os.getenv('API_KEY') or os.getenv('GEMINI_API_KEY')
        config.api_base = os.getenv('API_BASE', 'https://api.openai.com/v1')
        
        # Qwen3配置
        config.qwen_model_path = os.getenv('QWEN_MODEL_PATH', 'Qwen/Qwen3-8B')
        config.use_fp8 = os.getenv('USE_FP8', 'false').lower() == 'true'
        config.device_map = os.getenv('DEVICE_MAP', 'auto')
        config.torch_dtype = os.getenv('TORCH_DTYPE', 'auto')
        
        # 生成参数
        config.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        config.top_p = float(os.getenv('TOP_P', '0.8'))
        config.max_tokens = int(os.getenv('MAX_TOKENS', '2048'))
        
        # 实验配置
        config.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        config.base_delay = int(os.getenv('BASE_DELAY', '1'))
        
        return config
    
    def save_config(self, config: ModelConfig, file_path: str = 'config.env'):
        """
        保存配置到文件
        
        Args:
            config: 配置对象
            file_path: 保存路径
        """
        config_lines = [
            "# CoT 实验配置文件",
            "# 由 ConfigManager 自动生成",
            "",
            "# 模型配置",
            f"MODEL_TYPE={config.model_type}",
            f"MODEL_NAME={config.model_name}",
            f"API_KEY={config.api_key or ''}",
            f"API_BASE={config.api_base}",
            "",
            "# Qwen3配置",
            f"QWEN_MODEL_PATH={config.qwen_model_path}",
            f"USE_FP8={str(config.use_fp8).lower()}",
            f"DEVICE_MAP={config.device_map}",
            f"TORCH_DTYPE={config.torch_dtype}",
            "",
            "# 生成参数",
            f"TEMPERATURE={config.temperature}",
            f"TOP_P={config.top_p}",
            f"MAX_TOKENS={config.max_tokens}",
            "",
            "# 实验配置",
            f"MAX_RETRIES={config.max_retries}",
            f"BASE_DELAY={config.base_delay}",
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(config_lines))
    
    def get_preset_configs(self) -> Dict[str, ModelConfig]:
        """
        获取预设配置
        
        Returns:
            Dict[str, ModelConfig]: 预设配置字典
        """
        return {
            'gemini': ModelConfig(
                model_type='gemini',
                model_name='gemini-2.0-flash-exp',
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048
            ),
            'qwen3_local': ModelConfig(
                model_type='qwen3_local',
                qwen_model_path='Qwen/Qwen3-8B',
                use_fp8=False,
                device_map='auto',
                torch_dtype='auto',
                temperature=0.7,  # 非思考模式
                top_p=0.8,
                max_tokens=2048
            ),
            'qwen3_local_fp8': ModelConfig(
                model_type='qwen3_local',
                qwen_model_path='Qwen/Qwen3-8B',
                use_fp8=True,  # 启用FP8量化
                device_map='auto',
                torch_dtype='auto',
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048
            ),
            'qwen3_api': ModelConfig(
                model_type='qwen3_api',
                model_name='qwen3-8b',
                api_base='https://api.your-provider.com/v1',
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048
            ),
        }
    
    def print_config_summary(self, config: ModelConfig):
        """打印配置摘要"""
        print("=" * 50)
        print("当前模型配置")
        print("=" * 50)
        print(f"模型类型: {config.model_type}")
        print(f"模型名称: {config.model_name}")
        
        if config.model_type == 'qwen3_local':
            print(f"模型路径: {config.qwen_model_path}")
            print(f"FP8量化: {'启用' if config.use_fp8 else '禁用'}")
            print(f"设备映射: {config.device_map}")
            print(f"数据类型: {config.torch_dtype}")
        elif config.model_type in ['gemini', 'qwen3_api']:
            print(f"API密钥: {'已设置' if config.api_key else '未设置'}")
            print(f"API地址: {config.api_base}")
        
        print(f"温度参数: {config.temperature}")
        print(f"Top-p参数: {config.top_p}")
        print(f"最大Token: {config.max_tokens}")
        print("=" * 50)

def create_quick_config(model_type: str, **kwargs) -> ModelConfig:
    """
    快速创建配置
    
    Args:
        model_type: 模型类型
        **kwargs: 其他配置参数
        
    Returns:
        ModelConfig: 配置对象
    """
    config_manager = ConfigManager()
    presets = config_manager.get_preset_configs()
    
    if model_type in presets:
        config = presets[model_type]
        # 应用额外参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 便捷函数
def get_gemini_config(api_key: Optional[str] = None) -> ModelConfig:
    """获取Gemini配置"""
    return create_quick_config('gemini', api_key=api_key)

def get_qwen3_local_config(model_path: str = 'Qwen/Qwen3-8B', use_fp8: bool = False) -> ModelConfig:
    """获取Qwen3本地配置"""
    return create_quick_config('qwen3_local', qwen_model_path=model_path, use_fp8=use_fp8)

def get_qwen3_api_config(api_key: str, api_base: str) -> ModelConfig:
    """获取Qwen3 API配置"""
    return create_quick_config('qwen3_api', api_key=api_key, api_base=api_base)