# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chain-of-Thought (CoT) prompting implementation that replicates the core findings from Wei et al. (2022). The system demonstrates how step-by-step reasoning in prompts improves LLM performance across three reasoning task types:
- **Arithmetic Reasoning** (GSM8K): Grade school math problems
- **Commonsense Reasoning** (CommonsenseQA): Questions requiring common sense
- **Symbolic Reasoning** (Last Letter Concatenation): Symbol manipulation tasks

## Common Commands

### Setup and Configuration
```bash
# Offline mode setup (most secure, no internet required after setup)
python setup_offline.py

# Qwen3 local setup (interactive wizard)
python setup_qwen3.py

# Manual configuration
cp config.env .env  # Then edit .env with your settings
```

### Running Experiments
```bash
# Run all CoT experiments
python main.py

# Run with specific model type
MODEL_TYPE=qwen3_local python main.py
MODEL_TYPE=gemini python main.py
```

### Testing
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

### Validation
```bash
# Verify offline mode setup
python validate_offline.py
```

## System Architecture

The codebase follows a modular 6-component design:

1. **`dataset_handler.py`** - Loads hardcoded sample datasets for each reasoning task (GSM8K, CommonsenseQA, Last Letter)
2. **`prompt_engineer.py`** - Constructs few-shot prompts with/without chain-of-thought reasoning steps
3. **`model_interface.py`** - Multi-model support layer (Gemini API, Qwen3 local/API) with retry logic
4. **`model_config.py`** - Configuration management system with environment variable support
5. **`response_parser.py`** - Extracts answers from model responses and evaluates correctness
6. **`experiment_orchestrator.py`** - Orchestrates experiments and calculates accuracy metrics

**Main execution flow:** `main.py` → `experiment_orchestrator.py` → other components

## Configuration System

The system uses environment variables loaded from `.env` file:

### Key Variables
- `MODEL_TYPE`: `gemini|qwen3_local|qwen3_api`
- `OFFLINE_MODE`: `true|false` (disables all API calls when true)
- `QWEN_MODEL_PATH`: Hugging Face model path (default: `Qwen/Qwen3-8B`)
- `USE_FP8`: `true|false` (enables FP8 quantization for Qwen3)
- `GEMINI_API_KEY`: API key for Gemini models
- `API_KEY`/`API_BASE`: For custom OpenAI-compatible APIs

### Model-Specific Features
- **Qwen3 Local**: Supports FP8 quantization, auto device mapping, optimized for non-thinking mode (temp=0.7, top_p=0.8)
- **Offline Mode**: Complete privacy, no network calls, Qwen3-only inference
- **API Models**: Rate limiting with exponential backoff retry logic

## Core Concepts

**Chain-of-Thought Implementation**: The system compares standard prompting (direct answers) vs CoT prompting (step-by-step reasoning) by controlling the `use_cot` parameter in prompts.

**Experimental Design**: For each task, runs two experiments:
1. Standard prompting (control group)
2. CoT prompting (experimental group)

**Few-Shot Learning**: Uses hardcoded exemplars from each dataset as teaching examples, with the key difference being inclusion/exclusion of reasoning chains.

## Development Notes

- All datasets are hardcoded in `dataset_handler.py` - no external data loading
- Tests cover all components with mock model responses
- The system prioritizes educational clarity over production optimization
- Chinese comments throughout codebase explain CoT concepts
- Supports both local inference (privacy-focused) and API-based inference (convenience-focused)