"""
模型接口 - Chain-of-Thought 实现
支持多种大语言模型：Gemini API、Qwen3 8B (本地/API)

CoT核心概念：这个模块负责与大语言模型通信
- 模型选择：支持 Gemini、Qwen3 8B 本地推理、Qwen3 8B API
- API调用：将构建好的提示词发送给模型
- 错误处理：网络问题时的重试机制
- FP8推理：Qwen3支持FP8量化以提升推理速度
"""

import os
import time
import logging
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查是否启用离线模式
OFFLINE_MODE = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'

if OFFLINE_MODE:
    logger.info("🔒 离线模式已启用 - 仅使用Qwen3本地推理")
    # 在离线模式下，禁用所有在线API导入
    GEMINI_AVAILABLE = False
    OPENAI_AVAILABLE = False
else:
    # 正常模式：根据配置导入不同的依赖
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        # 配置Gemini API密钥
        if os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    except ImportError:
        GEMINI_AVAILABLE = False
        logger.warning("Gemini API 不可用，请安装 google-generativeai")

    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        logger.warning("OpenAI 不可用，请安装 openai")

# Qwen3本地推理依赖 (离线模式必需)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
    if OFFLINE_MODE:
        logger.info("✅ Qwen3本地推理依赖加载成功")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    error_msg = "Transformers 不可用，请安装 transformers>=4.51.0 torch"
    if OFFLINE_MODE:
        error_msg += " (离线模式必需)"
    logger.error(error_msg)

# 全局变量存储加载的模型
_local_model = None
_local_tokenizer = None

class ModelConfig:
    """模型配置类"""
    def __init__(self):
        # 从环境变量读取配置，默认使用Gemini
        self.model_type = os.getenv('MODEL_TYPE', 'gemini')  # 'gemini', 'qwen3_local', 'qwen3_api'
        self.model_name = os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp')
        self.api_key = os.getenv('API_KEY', os.getenv('GEMINI_API_KEY'))
        self.api_base = os.getenv('API_BASE', 'https://api.openai.com/v1')
        
        # Qwen3 特定配置
        self.qwen_model_path = os.getenv('QWEN_MODEL_PATH', 'Qwen/Qwen3-8B')
        self.use_fp8 = os.getenv('USE_FP8', 'false').lower() == 'true'
        self.device_map = os.getenv('DEVICE_MAP', 'auto')
        self.torch_dtype = os.getenv('TORCH_DTYPE', 'auto')
        
        # 生成参数 (非思考模式)
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.top_p = float(os.getenv('TOP_P', '0.8'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2048'))

def load_qwen3_local(config: ModelConfig):
    """加载本地Qwen3模型"""
    global _local_model, _local_tokenizer
    
    if _local_model is None:
        logger.info(f"正在加载Qwen3模型: {config.qwen_model_path}")
        
        # 加载tokenizer
        _local_tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model_path,
            trust_remote_code=True
        )
        
        # 配置模型加载参数
        model_kwargs = {
            'trust_remote_code': True,
            'device_map': config.device_map
        }
        
        # 设置数据类型
        if config.torch_dtype == 'auto':
            model_kwargs['torch_dtype'] = 'auto'
        elif config.use_fp8:
            # FP8量化配置
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model_kwargs['quantization_config'] = quantization_config
                logger.info("启用FP8量化")
            except ImportError:
                logger.warning("BitsAndBytesConfig不可用，使用默认精度")
                model_kwargs['torch_dtype'] = torch.float16
        else:
            model_kwargs['torch_dtype'] = torch.float16
        
        # 加载模型
        _local_model = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_path,
            **model_kwargs
        )
        
        logger.info("Qwen3模型加载完成")
    
    return _local_model, _local_tokenizer

def get_model_response(prompt: str, config: Optional[ModelConfig] = None, max_retries: int = 3, base_delay: int = 1) -> str:
    """
    向语言模型发送提示并获取响应。
    
    参数说明：
        prompt (str): 要发送给模型的提示词
        config (ModelConfig): 模型配置，默认使用环境变量配置
        max_retries (int): 最大重试次数
        base_delay (int): 指数退避的基础延迟时间（秒）
        
    返回值：
        str: 模型的响应文本
        
    异常：
        Exception: 如果所有重试尝试都失败
        
    CoT学习要点：
    - 这个函数接收完整的提示词（包含few-shot示例和新问题）
    - 模型根据提示词中的示例学习回答格式
    - 如果是CoT提示，模型会模仿示例中的逐步推理风格
    - 离线模式下仅支持Qwen3本地推理
    """
    
    if config is None:
        config = ModelConfig()
    
    # 离线模式验证
    if OFFLINE_MODE:
        if config.model_type != 'qwen3_local':
            logger.warning(f"🔒 离线模式下强制使用Qwen3本地推理，忽略配置的模型类型: {config.model_type}")
            config.model_type = 'qwen3_local'
        
        # 确保离线模式下不使用在线API
        if config.model_type in ['gemini', 'qwen3_api']:
            raise ValueError(f"🚫 离线模式下不允许使用在线API: {config.model_type}")
    
    # 根据模型类型选择不同的推理方式
    if config.model_type == 'gemini':
        if OFFLINE_MODE:
            raise ValueError("🚫 离线模式下禁用Gemini API")
        return _get_gemini_response(prompt, config, max_retries, base_delay)
    elif config.model_type == 'qwen3_local':
        return _get_qwen3_local_response(prompt, config, max_retries, base_delay)
    elif config.model_type == 'qwen3_api':
        if OFFLINE_MODE:
            raise ValueError("🚫 离线模式下禁用Qwen3 API")
        return _get_qwen3_api_response(prompt, config, max_retries, base_delay)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")

def _get_gemini_response(prompt: str, config: ModelConfig, max_retries: int, base_delay: int) -> str:
    """Gemini API响应"""
    if OFFLINE_MODE:
        raise ValueError("🚫 离线模式下禁用Gemini API")
    
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini API 不可用，请安装 google-generativeai")
    
    # 创建Gemini模型实例
    model = genai.GenerativeModel(config.model_name)
    
    # 重试循环：处理网络错误和API限制
    for attempt in range(max_retries + 1):
        try:
            # 向模型发送提示词并获取响应
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Gemini API请求失败 (尝试 {attempt + 1}/{max_retries + 1}). {delay}秒后重试...")
            time.sleep(delay)
    
    raise Exception("超过最大重试次数")

def _get_qwen3_local_response(prompt: str, config: ModelConfig, max_retries: int, base_delay: int) -> str:
    """Qwen3本地推理响应"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers 不可用，请安装 transformers>=4.51.0 torch")
    
    # 加载模型
    model, tokenizer = load_qwen3_local(config)
    
    for attempt in range(max_retries + 1):
        try:
            # 构建消息格式
            messages = [{"role": "user", "content": prompt}]
            
            # 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # 生成响应 (非思考模式参数)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码响应
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Qwen3本地推理失败 (尝试 {attempt + 1}/{max_retries + 1}). {delay}秒后重试...")
            time.sleep(delay)
    
    raise Exception("超过最大重试次数")

def _get_qwen3_api_response(prompt: str, config: ModelConfig, max_retries: int, base_delay: int) -> str:
    """Qwen3 API响应"""
    if OFFLINE_MODE:
        raise ValueError("🚫 离线模式下禁用Qwen3 API")
    
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI库不可用，请安装 openai")
    
    # 配置API客户端
    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.api_base
    )
    
    for attempt in range(max_retries + 1):
        try:
            # 调用API
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Qwen3 API请求失败 (尝试 {attempt + 1}/{max_retries + 1}). {delay}秒后重试...")
            time.sleep(delay)
    
    raise Exception("超过最大重试次数")