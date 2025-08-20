# Chain-of-Thought Prompting Implementation

This project replicates the core findings from the paper "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" by Wei et al. (2022).

## Overview

Chain-of-Thought (CoT) prompting is a technique that improves large language model reasoning by providing step-by-step intermediate reasoning steps in few-shot examples. This implementation demonstrates the effectiveness of CoT across three types of reasoning tasks:

- **Arithmetic Reasoning** (GSM8K): Grade school math word problems
- **Commonsense Reasoning** (CommonsenseQA): Questions requiring common sense knowledge
- **Symbolic Reasoning** (Last Letter Concatenation): Extracting and concatenating last letters

## Supported Models

### 🌐 Cloud API Models
- **Gemini 2.0 Flash** - Google's latest high-performance model
- **Qwen3 8B API** - Through OpenAI-compatible API endpoints

### 🏠 Local Inference Models  
- **Qwen3 8B Local** - Local inference with optional FP8 quantization
  - Standard precision inference
  - FP8 quantization for faster inference
  - Optimized for non-thinking mode reasoning

## System Architecture

The implementation consists of 6 core components:

1. **Dataset Handler** (`dataset_handler.py`) - Loads sample datasets for each task
2. **Prompt Engineer** (`prompt_engineer.py`) - Constructs few-shot prompts (standard vs CoT)
3. **Model Interface** (`model_interface.py`) - Multi-model support (Gemini/Qwen3 local/API)
4. **Model Config** (`model_config.py`) - Configuration management system
5. **Response Parser** (`response_parser.py`) - Extracts and evaluates answers
6. **Experiment Orchestrator** (`experiment_orchestrator.py`) - Runs experiments and calculates accuracy

## Quick Start

### Option 1: 🔒 Offline Mode - Qwen3 Only (Most Secure)
```bash
# 1. Auto-setup offline mode (no internet required after setup)
python setup_offline.py

# 2. Run experiments (100% offline)
python main.py

# 3. Verify offline status
python validate_offline.py
```

### Option 2: Qwen3 Local Inference (Recommended)
```bash
# 1. Auto-setup Qwen3 (interactive)
python setup_qwen3.py

# 2. Run experiments
python main.py
```

### Option 3: Gemini API (Easiest)
```bash
# 1. Install minimal dependencies
pip install -r requirements-minimal.txt

# 2. Set API key
export GEMINI_API_KEY="your_gemini_api_key"

# 3. Run experiments
python main.py
```

### Option 4: Manual Setup

#### For Offline Mode (Manual):
```bash
# 1. Install offline dependencies only
pip install -r requirements-offline.txt

# 2. Create offline configuration
cp config.env .env
# Edit .env: set OFFLINE_MODE=true, MODEL_TYPE=qwen3_local

# 3. Run experiments (offline)
python main.py
```

#### For Qwen3 Local Inference:
```bash
# 1. Install dependencies
pip install -r requirements-qwen3.txt

# 2. Create configuration
cp config.env .env
# Edit .env file with your preferences

# 3. Set model type
export MODEL_TYPE=qwen3_local

# 4. Run experiments
python main.py
```

#### For Qwen3 API:
```bash
# 1. Install API dependencies
pip install openai python-dotenv

# 2. Configure API
export MODEL_TYPE=qwen3_api
export API_KEY="your_api_key"
export API_BASE="https://your-api-provider.com/v1"

# 3. Run experiments
python main.py
```

## Configuration Options

### Environment Variables
```bash
# Offline Mode (Recommended for Privacy)
OFFLINE_MODE=true|false          # Enable offline-only mode

# Model Selection
MODEL_TYPE=gemini|qwen3_local|qwen3_api
MODEL_NAME=gemini-2.0-flash-exp

# API Configuration (Ignored in offline mode)
GEMINI_API_KEY=your_key_here
API_KEY=your_key_here
API_BASE=https://api.openai.com/v1

# Qwen3 Local Configuration
QWEN_MODEL_PATH=Qwen/Qwen3-8B
USE_FP8=true|false               # Enable FP8 quantization
DEVICE_MAP=auto
TORCH_DTYPE=auto

# Generation Parameters (Non-thinking mode)
TEMPERATURE=0.7
TOP_P=0.8
MAX_TOKENS=2048
```

### Configuration File
Create a `.env` file:
```bash
cp config.env .env
# Edit .env with your settings
```

## Model-Specific Features

### Qwen3 8B Local
- **FP8 Quantization**: Faster inference with reduced memory usage
- **Non-thinking Mode**: Optimized parameters (temp=0.7, top_p=0.8)
- **Auto Device Mapping**: Automatically distributes model across available GPUs
- **Context Length**: Supports up to 32K tokens natively

### Performance Comparison
| Model | Speed | Memory | Setup | Cost | Privacy | Offline |
|-------|-------|--------|-------|------|---------|---------|
| 🔒 Offline Mode | Medium | High | Auto | Free | Maximum | ✅ |
| Qwen3 Local | Medium | High | Medium | Free | High | ✅ |
| Qwen3 API | Fast | Low | Easy | Pay-per-use | Medium | ❌ |
| Gemini API | Fast | Low | Easy | Pay-per-use | Medium | ❌ |

## Usage Examples

### Basic Usage
```python
from model_config import get_qwen3_local_config, get_gemini_config
from experiment_orchestrator import run_experiment

# Offline Mode (Recommended)
import os
os.environ['OFFLINE_MODE'] = 'true'
config = get_qwen3_local_config(use_fp8=True)
accuracy = run_experiment('gsm8k', use_cot=True, config=config)

# Use Qwen3 local with FP8
config = get_qwen3_local_config(use_fp8=True)
accuracy = run_experiment('gsm8k', use_cot=True, config=config)

# Use Gemini API (requires internet)
config = get_gemini_config(api_key="your_key")
accuracy = run_experiment('gsm8k', use_cot=True, config=config)
```

### Offline Mode Benefits
- 🔒 **Complete Privacy**: No data leaves your device
- 🌐 **No Internet Required**: Works completely offline after setup
- 💰 **Zero Cost**: No API fees or usage limits
- 🛡️ **Security**: All processing happens locally
- ⚡ **Consistent Performance**: No network latency or rate limits

### Run Tests
```bash
# Run all tests
python run_tests.py

# Run specific component tests
python run_tests.py dataset_handler
python run_tests.py prompt_engineer
python run_tests.py model_interface
python run_tests.py response_parser
python run_tests.py experiment_orchestrator
```

## Testing Strategy

Each component has comprehensive unit tests:

- **`test_dataset_handler.py`** - Tests dataset loading and structure validation
- **`test_prompt_engineer.py`** - Tests prompt construction for standard vs CoT formats
- **`test_model_interface.py`** - Tests API interaction with retry logic and error handling
- **`test_response_parser.py`** - Tests answer extraction and evaluation logic
- **`test_experiment_orchestrator.py`** - Tests experiment orchestration and accuracy calculation
- **`test_main.py`** - Tests main execution flow and result formatting

## Expected Results

The implementation should demonstrate that Chain-of-Thought prompting improves performance across all three reasoning tasks, consistent with the original paper's findings. The system compares standard prompting (direct answers) with CoT prompting (step-by-step reasoning) and reports accuracy improvements.

## Key Features

- **Robust Error Handling**: Exponential backoff retry logic for API calls
- **Comprehensive Testing**: Unit tests for all components with 90%+ coverage
- **Modular Design**: Clean separation of concerns across components
- **Detailed Logging**: Progress tracking and result reporting
- **Extensible**: Easy to add new reasoning tasks and datasets

## Files Structure

```
cot/
├── dataset_handler.py           # Dataset loading functionality
├── prompt_engineer.py           # Prompt construction logic
├── model_interface.py           # Multi-model interface (Gemini/Qwen3)
├── model_config.py              # Configuration management
├── response_parser.py           # Response parsing and evaluation
├── experiment_orchestrator.py   # Experiment execution logic
├── main.py                      # Main execution script
├── setup_offline.py             # 🔒 Offline mode auto-setup
├── setup_qwen3.py               # Qwen3 setup wizard
├── validate_offline.py          # Offline mode validator
├── run_tests.py                 # Test runner script
├── requirements.txt             # All dependencies
├── requirements-offline.txt     # 🔒 Offline-only dependencies
├── requirements-minimal.txt     # Minimal Gemini dependencies
├── requirements-qwen3.txt       # Qwen3 dependencies
├── config.env                   # Configuration template
├── .env                         # Your configuration (created by setup)
├── COT学习指南.md               # Chinese learning guide
├── test_*.py                    # Unit tests for each component
└── README.md                    # Project documentation
```

## Contributing

To extend this implementation:

1. Add new datasets in `dataset_handler.py`
2. Create corresponding test cases
3. Update the task list in `main.py`
4. Run tests to ensure functionality

This implementation serves as both a replication study and a foundation for further Chain-of-Thought prompting research.