# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dual-system implementation combining:

1. **CoT Prompting Replication** - The original implementation from Wei et al. (2022) for basic reasoning tasks
2. **CoT-RAG Medical System** - An advanced medical diagnostic framework using Chain-of-Thought with Retrieval-Augmented Generation

### Original CoT Implementation
Demonstrates step-by-step reasoning improvements across three reasoning task types:
- **Arithmetic Reasoning** (GSM8K): Grade school math problems
- **Commonsense Reasoning** (CommonsenseQA): Questions requiring common sense
- **Symbolic Reasoning** (Last Letter Concatenation): Symbol manipulation tasks

### CoT-RAG Medical System
Advanced 4-stage diagnostic reasoning framework for ECG/cardiac analysis:
- **Stage 1**: Knowledge Graph Generation from expert decision trees
- **Stage 2**: RAG Population with patient-specific clinical data
- **Stage 3**: Reasoning Execution through populated knowledge graphs
- **Stage 4**: Integration testing with real medical datasets (ECG-QA, PTB-XL, MIMIC-IV)

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
# Run all tests (original CoT components)
python run_tests.py

# Run specific component tests
python run_tests.py dataset_handler
python run_tests.py prompt_engineer
python run_tests.py model_interface
python run_tests.py response_parser
python run_tests.py experiment_orchestrator

# Run CoT-RAG stage-specific tests
python test_stage1_simple.py          # Stage 1: Knowledge graph generation
python test_stage1_quick.py           # Quick Stage 1 tests
python test_stage1_kg_generation.py   # Detailed KG generation tests
python test_stage1_to_stage2.py       # Stage 1→2 integration
python test_stage2_simple.py          # Stage 2: RAG population
python test_stage2_quick.py           # Quick Stage 2 tests
python test_stage3_simple.py          # Stage 3: Reasoning execution
python test_stage4_simple.py          # Stage 4: Basic integration
python test_stage4_integration.py     # Full system integration

# Complete pipeline test
python test_complete_pipeline.py
```

### Validation
```bash
# Verify offline mode setup
python validate_offline.py
```

## System Architecture

This codebase implements two complementary systems:

### Original CoT System (6-component modular design)
1. **`dataset_handler.py`** - Loads hardcoded sample datasets for each reasoning task (GSM8K, CommonsenseQA, Last Letter)
2. **`prompt_engineer.py`** - Constructs few-shot prompts with/without chain-of-thought reasoning steps
3. **`model_interface.py`** - Multi-model support layer (Gemini API, Qwen3 local/API) with retry logic
4. **`model_config.py`** - Configuration management system with environment variable support
5. **`response_parser.py`** - Extracts answers from model responses and evaluates correctness
6. **`experiment_orchestrator.py`** - Orchestrates experiments and calculates accuracy metrics

**Main execution flow:** `main.py` → `experiment_orchestrator.py` → other components

### CoT-RAG Medical System (4-stage pipeline)
1. **Stage 1: Knowledge Graph Generation** (`core/stage1_generator.py`)
   - Converts expert decision trees (YAML) into fine-grained knowledge graphs
   - Uses `ExpertDecisionTree` class to load/validate expert knowledge from `expert_knowledge/`
   - Employs LLM decomposition to create detailed diagnostic reasoning chains

2. **Stage 2: RAG Population** (`core/stage2_rag.py`) 
   - Populates knowledge graphs with patient-specific clinical data
   - `PatientData` container for ECG data, clinical notes, demographics
   - Medical ontology mapping for standardized terminology (ICD-10, SNOMED)
   - **Fairseq-signals integration** for real PTB-XL and ECG-QA data loading

3. **Stage 3: Reasoning Execution** (`core/stage3_executor.py`)
   - Executes diagnostic reasoning through populated knowledge graphs
   - Multiple decision methods: rule-based, LLM-based, classifier-based, hybrid
   - **Enhanced with fairseq-signals trained classifiers** for hierarchical ECG diagnosis
   - Generates interpretable diagnostic reports with confidence scores

4. **Stage 4: Integration & Evaluation**
   - **Direct integration with fairseq-signals framework**
   - Real-time PTB-XL and ECG-QA dataset processing
   - **Fairseq-signals trained model inference** with hierarchical classification
   - Performance evaluation against clinical ground truth
   - Model interpretability and validation metrics

### Shared Infrastructure
- **`core/knowledge_graph.py`** - Core KG data structures (`KGNode`, `NodeType`, `KnowledgeGraph`)
- **`utils/llm_interface.py`** - Enhanced wrapper around `model_interface.py` for CoT-RAG
- **`utils/prompt_templates.py`** - Medical-specific prompt engineering
- **`utils/medical_ontology.py`** - Medical terminology standardization
- **`models/`** - ECG-specific classifiers and ensemble managers
  - **`models/fairseq_classifier.py`** - Direct fairseq-signals model integration
  - **`models/se_resnet_classifier.py`** - SE-ResNet with hierarchical classification
  - **`models/han_classifier.py`** - Hierarchical Attention Network for ECG
- **`datasets/`** - Medical dataset loaders (Chapman, MIMIC-ECG, PTB-XL)
- **`data_processing/`** - Clinical data preprocessing and hierarchy building
  - **`data_processing/fairseq_loader.py`** - Direct fairseq-signals integration layer
  - **`data_processing/ecg_loader.py`** - Enhanced with fairseq-signals support
  - **`data_processing/clinical_loader.py`** - Enhanced with ECG-QA integration

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

## Key Data Files and Expert Knowledge

### Expert Decision Trees (`expert_knowledge/`)
- **`cardiology_decision_tree.yaml`** - Main cardiac diagnostic decision tree
- **`arrhythmia_decision_tree.yaml`** - Specialized arrhythmia classification tree
- **`test_simple.yaml`** - Simplified tree for testing/development

### Medical Datasets with Fairseq-Signals Integration
- **PTB-XL Dataset** - Direct integration via `/home/ll/Desktop/codes/fairseq-signals/datasets/physionet.org`
  - SCP-ECG statement hierarchy support
  - Hierarchical classification (superclass → class → subclass)
  - Real-time model inference with fairseq-signals trained models
- **ECG-QA Dataset** - Direct integration via `/home/ll/Desktop/codes/fairseq-signals/datasets/ecg_step2_ptb_ver2`
  - Question-answering dataset with ECG interpretations
  - MIMIC-IV-ECG and PTB-XL subsets with template/paraphrased questions
  - Fairseq-signals processed format for efficient loading
- **Fairseq-Signals Models** - Pre-trained hierarchical ECG classification models
  - ECG Transformer classifiers with attention mechanisms
  - Multi-task models (classification + question answering)
  - Ensemble models with multiple checkpoints
- **Medical Dataset Loaders** (`datasets/`) - Chapman, MIMIC-ECG, PTB-XL data interfaces

### Output and Results
- **Knowledge Graphs** (`output/`) - Generated KG structures (JSON format)
- **Stage Test Results** - JSON files with stage-wise evaluation metrics
- **Clinical Validation** (`validation/`) - Medical accuracy and interpretability metrics
- **Fairseq Integration Logs** - Real-time inference and data loading statistics

## Development Notes

### Original CoT System
- All datasets are hardcoded in `dataset_handler.py` - no external data loading
- Tests cover all components with mock model responses
- The system prioritizes educational clarity over production optimization
- Chinese comments throughout codebase explain CoT concepts

### CoT-RAG Medical System
- Expert knowledge defined in YAML format for easy medical professional input
- Supports integration with real clinical datasets (ECG-QA, MIMIC-IV, PTB-XL)
- Modular architecture allows swapping classifiers, LLMs, and decision methods
- Comprehensive validation against clinical ground truth and expert assessment
- Knowledge graph structures persist as JSON for reproducibility and analysis

### Shared Infrastructure
- Supports both local inference (privacy-focused) and API-based inference (convenience-focused)
- Multi-model support through unified interface
- Extensive testing with stage-specific and integration test suites