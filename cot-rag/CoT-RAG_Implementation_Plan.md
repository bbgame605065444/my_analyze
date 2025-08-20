# CoT-RAG Implementation Plan: From Concept to Code

## Executive Summary

This document provides a comprehensive implementation blueprint for building the CoT-RAG (Chain-of-Thought Retrieval-Augmented Generation) framework, evolving from the existing CoT implementation. The plan integrates expert knowledge with LLM reasoning capabilities to create a robust, interpretable, and clinically-validated diagnostic system.

## 1. Architecture Overview

### 1.1 System Components

```
CoT-RAG System Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    CoT-RAG Framework                        │
├─────────────────────┬─────────────────────┬─────────────────┤
│   Expert Knowledge  │   Reasoning Engine  │ Prediction Engine│
│   - Decision Trees  │   - Knowledge Graph │ - ECG Classifiers│
│   - Medical Rules   │   - RAG Components  │ - Deep Learning  │
│   - Clinical Logic  │   - Execution Flow  │ - Time-series ML │
└─────────────────────┴─────────────────────┴─────────────────┘
```

### 1.2 Data Flow Pipeline

1. **Stage 1**: Expert Decision Tree → Knowledge Graph Generation
2. **Stage 2**: Patient Data + Knowledge Graph → RAG Population
3. **Stage 3**: Populated Knowledge Graph → Reasoning Execution → Diagnostic Report

## 2. Project Structure

```
cot-rag/
├── main.py                     # Main execution orchestrator
├── config.py                   # Configuration management
├── requirements-cotrag.txt     # Additional dependencies
│
├── core/                       # Core CoT-RAG implementation
│   ├── __init__.py
│   ├── knowledge_graph.py      # KG data structures
│   ├── stage1_generator.py     # Expert DT → KG conversion
│   ├── stage2_rag.py          # RAG population logic
│   ├── stage3_executor.py     # Reasoning execution
│   └── evaluation.py          # Hierarchical evaluation metrics
│
├── data_processing/            # Data handling and preprocessing
│   ├── __init__.py
│   ├── ecg_loader.py          # ECG data loading and preprocessing
│   ├── clinical_loader.py     # Clinical text data processing
│   └── hierarchy_builder.py   # Medical taxonomy integration
│
├── models/                     # Prediction engines
│   ├── __init__.py
│   ├── base_classifier.py     # Abstract base for all models
│   ├── ecg_classifiers.py     # ECG-specific deep learning models
│   └── ensemble_manager.py    # Multi-model orchestration
│
├── expert_knowledge/           # Expert-defined decision trees
│   ├── cardiology_dt.yaml     # Cardiology decision tree
│   ├── arrhythmia_dt.yaml     # Arrhythmia-specific rules
│   └── ischemia_dt.yaml       # Ischemia detection rules
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── llm_interface.py       # LLM API wrapper (extends existing)
│   ├── prompt_templates.py    # Structured prompts for CoT-RAG
│   ├── medical_ontology.py    # Medical standard mappings
│   └── visualization.py       # Decision tree and report visualization
│
├── evaluation/                 # Comprehensive evaluation framework
│   ├── __init__.py
│   ├── hierarchical_metrics.py # Medical hierarchy-aware metrics
│   ├── clinical_validation.py  # Clinical expert validation
│   └── benchmark_datasets.py   # Standard evaluation datasets
│
├── experiments/                # Experiment configurations and results
│   ├── ptb_xl_experiments/     # PTB-XL dataset experiments
│   ├── chapman_experiments/    # Chapman-Shaoxing experiments
│   └── physionet_experiments/  # PhysioNet challenge experiments
│
└── docs/                       # Documentation and examples
    ├── api_reference.md
    ├── clinical_workflow.md
    └── examples/
        ├── basic_usage.py
        ├── custom_hierarchy.py
        └── medical_validation.py
```

## 3. Core Implementation Components

### 3.1 Knowledge Graph Framework (`core/knowledge_graph.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from enum import Enum

class NodeType(Enum):
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"

@dataclass
class KGNode:
    """
    Core data structure for Knowledge Graph nodes.
    Represents a single decision point in the diagnostic workflow.
    """
    # Core identifiers
    node_id: str
    diagnosis_class: str
    node_type: NodeType
    
    # Hierarchical structure
    parent_node: Optional[str] = None
    child_nodes: List[str] = field(default_factory=list)
    
    # CoT-RAG specific attributes
    sub_question: str = ""
    sub_case: str = ""
    sub_description: str = ""  # Populated by RAG
    answer: str = ""           # Generated during execution
    
    # Decision logic
    decision_rule_logic: str = ""
    required_classifier_inputs: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    
    # Clinical metadata
    clinical_significance: str = ""
    evidence_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'diagnosis_class': self.diagnosis_class,
            'node_type': self.node_type.value,
            'parent_node': self.parent_node,
            'child_nodes': self.child_nodes,
            'sub_question': self.sub_question,
            'sub_case': self.sub_case,
            'sub_description': self.sub_description,
            'answer': self.answer,
            'decision_rule_logic': self.decision_rule_logic,
            'required_classifier_inputs': self.required_classifier_inputs,
            'confidence_threshold': self.confidence_threshold,
            'clinical_significance': self.clinical_significance,
            'evidence_sources': self.evidence_sources
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KGNode':
        """Create node from dictionary."""
        data['node_type'] = NodeType(data['node_type'])
        return cls(**data)

class KnowledgeGraph:
    """
    Container for the complete diagnostic knowledge graph.
    Manages nodes, relationships, and provides traversal methods.
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.nodes: Dict[str, KGNode] = {}
        self.root_node_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, node: KGNode) -> None:
        """Add a node to the knowledge graph."""
        self.nodes[node.node_id] = node
        
        # Set root node if this is the first root type node
        if node.node_type == NodeType.ROOT and self.root_node_id is None:
            self.root_node_id = node.node_id
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Retrieve a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[KGNode]:
        """Get all child nodes of a given node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[child_id] for child_id in node.child_nodes if child_id in self.nodes]
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get the path from a node to the root."""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            path.append(current_id)
            current_node = self.nodes[current_id]
            current_id = current_node.parent_node
        
        return path[::-1]  # Reverse to get root-to-node path
    
    def validate_structure(self) -> List[str]:
        """Validate the knowledge graph structure and return any errors."""
        errors = []
        
        # Check for root node
        if not self.root_node_id:
            errors.append("No root node defined")
        
        # Check parent-child consistency
        for node_id, node in self.nodes.items():
            # Check parent references
            if node.parent_node and node.parent_node not in self.nodes:
                errors.append(f"Node {node_id} references non-existent parent {node.parent_node}")
            
            # Check child references
            for child_id in node.child_nodes:
                if child_id not in self.nodes:
                    errors.append(f"Node {node_id} references non-existent child {child_id}")
                else:
                    child_node = self.nodes[child_id]
                    if child_node.parent_node != node_id:
                        errors.append(f"Child node {child_id} doesn't reference parent {node_id}")
        
        return errors
    
    def save_to_json(self, filepath: str) -> None:
        """Save knowledge graph to JSON file."""
        data = {
            'domain': self.domain,
            'root_node_id': self.root_node_id,
            'metadata': self.metadata,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'KnowledgeGraph':
        """Load knowledge graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls(domain=data.get('domain', 'general'))
        kg.root_node_id = data.get('root_node_id')
        kg.metadata = data.get('metadata', {})
        
        # Load nodes
        for node_id, node_data in data.get('nodes', {}).items():
            node = KGNode.from_dict(node_data)
            kg.add_node(node)
        
        return kg
```

### 3.2 Stage 1: Knowledge Graph Generator (`core/stage1_generator.py`)

```python
import yaml
from typing import Dict, List, Any
from ..utils.llm_interface import LLMInterface
from ..utils.prompt_templates import PromptTemplates
from .knowledge_graph import KnowledgeGraph, KGNode, NodeType

class ExpertDecisionTree:
    """
    Represents an expert-defined coarse-grained decision tree.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tree_data = self._load_tree()
    
    def _load_tree(self) -> Dict[str, Any]:
        """Load decision tree from YAML file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @property
    def domain(self) -> str:
        return self.tree_data.get('domain', 'general')
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        return self.tree_data.get('nodes', [])

class KGGenerator:
    """
    Stage 1: Converts expert-defined decision trees into fine-grained knowledge graphs.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.prompt_templates = PromptTemplates()
    
    def generate_from_decision_tree(self, dt_filepath: str) -> KnowledgeGraph:
        """
        Generate a fine-grained knowledge graph from an expert decision tree.
        
        Args:
            dt_filepath: Path to the expert decision tree YAML file
            
        Returns:
            KnowledgeGraph: Generated knowledge graph
        """
        expert_dt = ExpertDecisionTree(dt_filepath)
        kg = KnowledgeGraph(domain=expert_dt.domain)
        
        # Process each node in the decision tree
        for dt_node in expert_dt.nodes:
            self._process_dt_node(dt_node, kg)
        
        # Validate the generated structure
        errors = kg.validate_structure()
        if errors:
            raise ValueError(f"Generated knowledge graph has validation errors: {errors}")
        
        return kg
    
    def _process_dt_node(self, dt_node: Dict[str, Any], kg: KnowledgeGraph) -> None:
        """
        Process a single decision tree node and generate corresponding KG entities.
        """
        node_id = dt_node['node_id']
        question = dt_node['question']
        knowledge_case = dt_node.get('knowledge_case', '')
        
        # Use LLM to decompose the coarse node into fine-grained entities
        decomposition_prompt = self.prompt_templates.get_decomposition_prompt(
            question=question,
            knowledge_case=knowledge_case,
            domain=kg.domain
        )
        
        decomposition_response = self.llm.query(
            prompt=decomposition_prompt,
            model_name="decomposition_model"
        )
        
        # Parse LLM response to extract entities
        entities = self._parse_decomposition_response(decomposition_response)
        
        # Create KG nodes for each entity
        for i, entity in enumerate(entities):
            kg_node_id = f"{node_id}_entity_{i}"
            
            kg_node = KGNode(
                node_id=kg_node_id,
                diagnosis_class=entity.get('diagnosis_class', ''),
                node_type=self._determine_node_type(entity, dt_node),
                parent_node=dt_node.get('parent_node'),
                sub_question=entity.get('sub_question', ''),
                sub_case=entity.get('sub_case', ''),
                decision_rule_logic=entity.get('decision_rule', ''),
                required_classifier_inputs=entity.get('required_inputs', []),
                clinical_significance=entity.get('clinical_significance', '')
            )
            
            kg.add_node(kg_node)
        
        # Establish relationships between entities
        self._establish_entity_relationships(entities, kg, node_id)
    
    def _parse_decomposition_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract entity information.
        Expected format: JSON list of entities with required fields.
        """
        try:
            import json
            # Extract JSON from response (handle potential markdown formatting)
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            entities = json.loads(json_str)
            return entities if isinstance(entities, list) else [entities]
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM decomposition response: {e}")
    
    def _determine_node_type(self, entity: Dict[str, Any], dt_node: Dict[str, Any]) -> NodeType:
        """Determine the type of node based on entity properties."""
        if dt_node.get('is_root', False):
            return NodeType.ROOT
        elif entity.get('is_terminal', False):
            return NodeType.LEAF
        else:
            return NodeType.INTERNAL
    
    def _establish_entity_relationships(self, entities: List[Dict[str, Any]], 
                                      kg: KnowledgeGraph, base_node_id: str) -> None:
        """
        Establish parent-child relationships between entities based on dependencies.
        """
        # Create dependency mapping
        for i, entity in enumerate(entities):
            current_node_id = f"{base_node_id}_entity_{i}"
            dependencies = entity.get('dependencies', [])
            
            # Update child_nodes for parent entities
            for dep_idx in dependencies:
                if 0 <= dep_idx < len(entities):
                    parent_node_id = f"{base_node_id}_entity_{dep_idx}"
                    parent_node = kg.get_node(parent_node_id)
                    current_node = kg.get_node(current_node_id)
                    
                    if parent_node and current_node:
                        parent_node.child_nodes.append(current_node_id)
                        current_node.parent_node = parent_node_id
```

### 3.3 Stage 2: RAG Population (`core/stage2_rag.py`)

```python
from typing import Dict, List, Optional
import re
from ..utils.llm_interface import LLMInterface
from ..utils.prompt_templates import PromptTemplates
from .knowledge_graph import KnowledgeGraph, KGNode

class RAGPopulator:
    """
    Stage 2: Populates knowledge graph with patient-specific information
    using retrieval-augmented generation.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.prompt_templates = PromptTemplates()
    
    def populate_knowledge_graph(self, kg_template: KnowledgeGraph, 
                                patient_data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Populate knowledge graph template with patient-specific information.
        
        Args:
            kg_template: Template knowledge graph from Stage 1
            patient_data: Dictionary containing patient information
                         (clinical_text, ecg_data, demographics, etc.)
        
        Returns:
            KnowledgeGraph: Populated knowledge graph
        """
        # Create a copy of the template
        populated_kg = self._deep_copy_kg(kg_template)
        
        # Extract relevant clinical text
        clinical_text = patient_data.get('clinical_text', '')
        patient_query = patient_data.get('query_description', '')
        
        # Combine all available text data
        full_patient_context = self._combine_patient_text(clinical_text, patient_query)
        
        # Populate each node with relevant information
        for node_id, node in populated_kg.nodes.items():
            if node.sub_question:  # Only populate nodes that have questions
                self._populate_node(node, full_patient_context)
        
        # Update knowledge cases dynamically if new information is found
        self._update_knowledge_cases(populated_kg, full_patient_context)
        
        return populated_kg
    
    def _populate_node(self, node: KGNode, patient_context: str) -> None:
        """
        Populate a single node's sub_description with relevant patient information.
        """
        # Create retrieval prompt
        retrieval_prompt = self.prompt_templates.get_rag_prompt(
            sub_question=node.sub_question,
            sub_case=node.sub_case,
            patient_context=patient_context,
            diagnosis_class=node.diagnosis_class
        )
        
        # Query LLM for information extraction
        response = self.llm.query(
            prompt=retrieval_prompt,
            model_name="rag_model"
        )
        
        # Parse and validate response
        extracted_info = self._parse_rag_response(response)
        
        # Update node with extracted information
        node.sub_description = extracted_info.get('extracted_text', '')
        node.evidence_sources = extracted_info.get('evidence_sources', [])
        
        # Store confidence and relevance scores
        if 'confidence' in extracted_info:
            node.metadata = getattr(node, 'metadata', {})
            node.metadata['rag_confidence'] = extracted_info['confidence']
    
    def _parse_rag_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response from RAG query.
        Expected format includes extracted text and evidence sources.
        """
        result = {
            'extracted_text': '',
            'evidence_sources': [],
            'confidence': 0.0
        }
        
        try:
            # Look for structured response
            if 'EXTRACTED_TEXT:' in response:
                text_match = re.search(r'EXTRACTED_TEXT:\s*(.+?)(?:EVIDENCE:|$)', 
                                     response, re.DOTALL)
                if text_match:
                    result['extracted_text'] = text_match.group(1).strip()
            
            if 'EVIDENCE:' in response:
                evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?:CONFIDENCE:|$)', 
                                         response, re.DOTALL)
                if evidence_match:
                    evidence_text = evidence_match.group(1).strip()
                    result['evidence_sources'] = [e.strip() for e in evidence_text.split('\n') if e.strip()]
            
            if 'CONFIDENCE:' in response:
                conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
                if conf_match:
                    result['confidence'] = float(conf_match.group(1))
            
            # Fallback: if no structured format, use entire response as extracted text
            if not result['extracted_text'] and response.strip():
                result['extracted_text'] = response.strip()
                result['confidence'] = 0.5  # Default confidence
        
        except Exception as e:
            print(f"Warning: Error parsing RAG response: {e}")
            result['extracted_text'] = response.strip()
        
        return result
    
    def _combine_patient_text(self, clinical_text: str, query_description: str) -> str:
        """
        Combine various sources of patient text into a coherent context.
        """
        combined_text = []
        
        if clinical_text:
            combined_text.append(f"Clinical Notes:\n{clinical_text}")
        
        if query_description:
            combined_text.append(f"Current Query:\n{query_description}")
        
        return "\n\n".join(combined_text)
    
    def _update_knowledge_cases(self, kg: KnowledgeGraph, patient_context: str) -> None:
        """
        Dynamically update knowledge cases based on novel patient information.
        """
        # Identify novel patterns or information
        novel_info_prompt = self.prompt_templates.get_novel_info_prompt(
            patient_context=patient_context,
            domain=kg.domain
        )
        
        response = self.llm.query(
            prompt=novel_info_prompt,
            model_name="analysis_model"
        )
        
        # If novel information is found, update relevant knowledge cases
        if self._contains_novel_information(response):
            self._incorporate_novel_information(kg, response, patient_context)
    
    def _contains_novel_information(self, response: str) -> bool:
        """Check if the response indicates novel clinical information."""
        novel_indicators = ['novel', 'new pattern', 'unusual', 'rare', 'atypical']
        return any(indicator in response.lower() for indicator in novel_indicators)
    
    def _incorporate_novel_information(self, kg: KnowledgeGraph, 
                                    analysis: str, patient_context: str) -> None:
        """
        Incorporate novel information into knowledge cases for future use.
        """
        # This would involve updating the knowledge base
        # For now, we'll store it in metadata
        if 'novel_cases' not in kg.metadata:
            kg.metadata['novel_cases'] = []
        
        kg.metadata['novel_cases'].append({
            'analysis': analysis,
            'patient_context': patient_context[:500],  # Truncate for storage
            'timestamp': self._get_timestamp()
        })
    
    def _deep_copy_kg(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Create a deep copy of the knowledge graph."""
        import copy
        return copy.deepcopy(kg)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()
```

### 3.4 Stage 3: Execution Engine (`core/stage3_executor.py`)

```python
from typing import Dict, List, Tuple, Any, Optional
import re
from ..models.base_classifier import BaseClassifier
from ..utils.llm_interface import LLMInterface
from ..utils.prompt_templates import PromptTemplates
from .knowledge_graph import KnowledgeGraph, KGNode

class ReasoningExecutor:
    """
    Stage 3: Executes the reasoning path through the populated knowledge graph
    and generates interpretable diagnostic narratives.
    """
    
    def __init__(self, llm_interface: LLMInterface, classifiers: Dict[str, BaseClassifier]):
        self.llm = llm_interface
        self.classifiers = classifiers
        self.prompt_templates = PromptTemplates()
    
    def execute_reasoning_path(self, populated_kg: KnowledgeGraph, 
                             ecg_data: Optional[Any] = None) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Execute the complete reasoning path through the knowledge graph.
        
        Args:
            populated_kg: Knowledge graph populated with patient data
            ecg_data: ECG signal data for classifier inputs
            
        Returns:
            Tuple of (final_diagnosis, decision_path, execution_metadata)
        """
        if not populated_kg.root_node_id:
            raise ValueError("Knowledge graph has no root node defined")
        
        # Initialize execution state
        execution_state = {
            'classifier_outputs': {},
            'decision_path': [],
            'confidence_scores': {},
            'evidence_chain': []
        }
        
        # Pre-compute classifier outputs if ECG data is provided
        if ecg_data is not None:
            execution_state['classifier_outputs'] = self._compute_classifier_outputs(ecg_data)
        
        # Execute traversal starting from root
        final_diagnosis = self._traverse_knowledge_graph(
            populated_kg, 
            populated_kg.root_node_id, 
            execution_state
        )
        
        return (
            final_diagnosis,
            execution_state['decision_path'],
            execution_state
        )
    
    def _traverse_knowledge_graph(self, kg: KnowledgeGraph, 
                                current_node_id: str, 
                                execution_state: Dict[str, Any]) -> str:
        """
        Recursively traverse the knowledge graph based on decision rules.
        """
        current_node = kg.get_node(current_node_id)
        if not current_node:
            raise ValueError(f"Node {current_node_id} not found in knowledge graph")
        
        # Add current node to decision path
        execution_state['decision_path'].append(current_node_id)
        
        # Execute decision rule for current node
        decision_result = self._execute_decision_rule(current_node, execution_state)
        
        # Store evidence for this decision
        execution_state['evidence_chain'].append({
            'node_id': current_node_id,
            'question': current_node.sub_question,
            'evidence': current_node.sub_description,
            'decision': decision_result['decision'],
            'confidence': decision_result['confidence']
        })
        
        # Check if this is a terminal node
        if not current_node.child_nodes:
            # Generate final answer for leaf node
            return self._generate_final_diagnosis(current_node, execution_state)
        
        # Determine next node based on decision result
        next_node_id = self._select_next_node(
            current_node, 
            decision_result, 
            execution_state
        )
        
        if next_node_id:
            return self._traverse_knowledge_graph(kg, next_node_id, execution_state)
        else:
            # No valid next node found
            return self._generate_uncertainty_diagnosis(current_node, execution_state)
    
    def _execute_decision_rule(self, node: KGNode, 
                             execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the decision rule logic for a given node.
        """
        if not node.decision_rule_logic:
            # If no decision rule, use LLM to make decision based on evidence
            return self._llm_based_decision(node, execution_state)
        
        # Parse and execute the rule
        try:
            decision_context = self._build_decision_context(node, execution_state)
            
            # Execute rule with available context
            rule_result = self._evaluate_rule(node.decision_rule_logic, decision_context)
            
            return {
                'decision': rule_result['outcome'],
                'confidence': rule_result['confidence'],
                'method': 'rule_based'
            }
        
        except Exception as e:
            print(f"Warning: Rule execution failed for node {node.node_id}: {e}")
            # Fallback to LLM-based decision
            return self._llm_based_decision(node, execution_state)
    
    def _build_decision_context(self, node: KGNode, 
                              execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context for decision rule execution.
        """
        context = {
            'classifier_outputs': execution_state['classifier_outputs'],
            'node_evidence': node.sub_description,
            'node_question': node.sub_question,
            'previous_decisions': execution_state['evidence_chain']
        }
        
        return context
    
    def _evaluate_rule(self, rule_logic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a decision rule given the current context.
        Supports rules like: "IF get_prob('classifier_name', 'class') > 0.7 THEN next_node"
        """
        # Extract classifier queries from rule
        classifier_queries = re.findall(r"get_prob\('([^']+)',\s*'([^']+)'\)", rule_logic)
        
        # Replace classifier queries with actual values
        processed_rule = rule_logic
        for classifier_name, class_name in classifier_queries:
            prob_value = self._get_classifier_probability(
                classifier_name, class_name, context['classifier_outputs']
            )
            query_pattern = f"get_prob('{classifier_name}', '{class_name}')"
            processed_rule = processed_rule.replace(query_pattern, str(prob_value))
        
        # Safely evaluate the rule (in a production system, use a more secure approach)
        try:
            # Simple rule evaluation (extend this for more complex logic)
            if 'IF' in processed_rule and 'THEN' in processed_rule:
                condition_part = processed_rule.split('THEN')[0].replace('IF', '').strip()
                outcome_part = processed_rule.split('THEN')[1].strip()
                
                # Evaluate condition
                condition_result = eval(condition_part)  # Note: Use safer evaluation in production
                
                return {
                    'outcome': outcome_part if condition_result else 'continue',
                    'confidence': 0.9 if condition_result else 0.1
                }
            else:
                return {'outcome': 'continue', 'confidence': 0.5}
        
        except Exception as e:
            return {'outcome': 'error', 'confidence': 0.0}
    
    def _get_classifier_probability(self, classifier_name: str, class_name: str, 
                                  classifier_outputs: Dict[str, Any]) -> float:
        """Get probability for a specific class from classifier outputs."""
        if classifier_name in classifier_outputs:
            class_probs = classifier_outputs[classifier_name]
            return class_probs.get(class_name, 0.0)
        return 0.0
    
    def _llm_based_decision(self, node: KGNode, 
                          execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to make decision when rule-based approach is not available.
        """
        decision_prompt = self.prompt_templates.get_decision_prompt(
            node=node,
            evidence_chain=execution_state['evidence_chain'],
            classifier_outputs=execution_state['classifier_outputs']
        )
        
        response = self.llm.query(
            prompt=decision_prompt,
            model_name="reasoning_model"
        )
        
        # Parse LLM decision response
        return self._parse_decision_response(response)
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM decision response."""
        # Look for structured decision indicators
        if 'DECISION:' in response:
            decision_match = re.search(r'DECISION:\s*(.+?)(?:CONFIDENCE:|$)', response)
            decision = decision_match.group(1).strip() if decision_match else 'continue'
        else:
            decision = 'continue'
        
        if 'CONFIDENCE:' in response:
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
        else:
            confidence = 0.5
        
        return {
            'decision': decision,
            'confidence': confidence,
            'method': 'llm_based'
        }
    
    def _compute_classifier_outputs(self, ecg_data: Any) -> Dict[str, Dict[str, float]]:
        """
        Pre-compute outputs from all available classifiers.
        """
        outputs = {}
        
        for classifier_name, classifier in self.classifiers.items():
            try:
                class_probabilities = classifier.predict(ecg_data)
                outputs[classifier_name] = class_probabilities
            except Exception as e:
                print(f"Warning: Classifier {classifier_name} failed: {e}")
                outputs[classifier_name] = {}
        
        return outputs
    
    def _select_next_node(self, current_node: KGNode, 
                         decision_result: Dict[str, Any], 
                         execution_state: Dict[str, Any]) -> Optional[str]:
        """
        Select the next node to visit based on decision result.
        """
        decision = decision_result['decision']
        
        # Simple next node selection (extend this for more sophisticated logic)
        if decision in current_node.child_nodes:
            return decision
        elif current_node.child_nodes:
            # Default to first child if decision doesn't match specific child
            return current_node.child_nodes[0]
        
        return None
    
    def _generate_final_diagnosis(self, leaf_node: KGNode, 
                                execution_state: Dict[str, Any]) -> str:
        """
        Generate final diagnosis for a leaf node.
        """
        # Use LLM to generate final diagnosis based on complete evidence chain
        final_prompt = self.prompt_templates.get_final_diagnosis_prompt(
            leaf_node=leaf_node,
            evidence_chain=execution_state['evidence_chain'],
            classifier_outputs=execution_state['classifier_outputs']
        )
        
        response = self.llm.query(
            prompt=final_prompt,
            model_name="diagnosis_model"
        )
        
        return response.strip()
    
    def _generate_uncertainty_diagnosis(self, node: KGNode, 
                                      execution_state: Dict[str, Any]) -> str:
        """
        Generate diagnosis when reasoning path leads to uncertainty.
        """
        return f"Uncertain diagnosis - requires further evaluation (stopped at {node.diagnosis_class})"
    
    def generate_narrative_report(self, decision_path: List[str], 
                                populated_kg: KnowledgeGraph,
                                execution_metadata: Dict[str, Any]) -> str:
        """
        Generate a comprehensive narrative report of the diagnostic reasoning.
        """
        narrative_prompt = self.prompt_templates.get_narrative_prompt(
            decision_path=decision_path,
            knowledge_graph=populated_kg,
            evidence_chain=execution_metadata.get('evidence_chain', []),
            classifier_outputs=execution_metadata.get('classifier_outputs', {})
        )
        
        narrative = self.llm.query(
            prompt=narrative_prompt,
            model_name="narrative_model"
        )
        
        return self._format_narrative_report(narrative, execution_metadata)
    
    def _format_narrative_report(self, narrative: str, 
                               metadata: Dict[str, Any]) -> str:
        """
        Format the narrative report with additional metadata.
        """
        formatted_report = f"""
# Diagnostic Reasoning Report

## Summary
{narrative}

## Decision Path
{' → '.join(metadata.get('decision_path', []))}

## Confidence Scores
"""
        
        for evidence in metadata.get('evidence_chain', []):
            formatted_report += f"- {evidence['question']}: {evidence['confidence']:.2f}\n"
        
        formatted_report += f"""
## Classifier Outputs
"""
        
        for classifier, outputs in metadata.get('classifier_outputs', {}).items():
            formatted_report += f"### {classifier}\n"
            for class_name, prob in outputs.items():
                formatted_report += f"- {class_name}: {prob:.3f}\n"
        
        return formatted_report
```

## 4. Integration with Existing CoT Framework

### 4.1 Model Interface Extension (`model_interface.py`)

```python
# Extend existing model_interface.py with CoT-RAG capabilities

from typing import Dict, Any, Optional
import numpy as np
from .models.base_classifier import BaseClassifier

class CoTRAGModelInterface:
    """
    Extended model interface that integrates CoT-RAG reasoning with existing models.
    """
    
    def __init__(self, config):
        self.config = config
        self.base_model_interface = get_model_response  # From existing implementation
        self.ecg_classifiers = {}
        self.cot_rag_executor = None
    
    def register_ecg_classifier(self, name: str, classifier: BaseClassifier):
        """Register an ECG classifier for use in reasoning."""
        self.ecg_classifiers[name] = classifier
    
    def setup_cot_rag(self, kg_filepath: str, llm_interface):
        """Setup CoT-RAG reasoning engine."""
        from .core.knowledge_graph import KnowledgeGraph
        from .core.stage3_executor import ReasoningExecutor
        
        # Load knowledge graph
        kg = KnowledgeGraph.load_from_json(kg_filepath)
        
        # Create executor
        self.cot_rag_executor = ReasoningExecutor(llm_interface, self.ecg_classifiers)
    
    def get_cot_rag_response(self, patient_data: Dict[str, Any], 
                           kg_template: KnowledgeGraph) -> Dict[str, Any]:
        """
        Get response using CoT-RAG reasoning.
        
        Returns:
            Dict containing diagnosis, reasoning path, and narrative report
        """
        if not self.cot_rag_executor:
            raise ValueError("CoT-RAG executor not initialized")
        
        from .core.stage2_rag import RAGPopulator
        
        # Stage 2: Populate knowledge graph
        rag_populator = RAGPopulator(self.llm_interface)
        populated_kg = rag_populator.populate_knowledge_graph(kg_template, patient_data)
        
        # Stage 3: Execute reasoning
        final_diagnosis, decision_path, metadata = self.cot_rag_executor.execute_reasoning_path(
            populated_kg, 
            patient_data.get('ecg_data')
        )
        
        # Generate narrative report
        narrative = self.cot_rag_executor.generate_narrative_report(
            decision_path, populated_kg, metadata
        )
        
        return {
            'diagnosis': final_diagnosis,
            'decision_path': decision_path,
            'narrative_report': narrative,
            'confidence_scores': metadata.get('confidence_scores', {}),
            'evidence_chain': metadata.get('evidence_chain', [])
        }
```

### 4.2 Main Execution Script (`main.py`)

```python
#!/usr/bin/env python3
"""
CoT-RAG Main Execution Script
Orchestrates the complete CoT-RAG workflow for ECG diagnosis.
"""

import sys
import json
from pathlib import Path

# Import existing CoT components
from experiment_orchestrator import run_experiment
from model_config import ConfigManager
from model_interface import get_model_response

# Import CoT-RAG components
from core.knowledge_graph import KnowledgeGraph
from core.stage1_generator import KGGenerator
from core.stage2_rag import RAGPopulator
from core.stage3_executor import ReasoningExecutor
from utils.llm_interface import LLMInterface
from data_processing.ecg_loader import ECGLoader
from data_processing.clinical_loader import ClinicalDataLoader
from models.ecg_classifiers import SEResNetClassifier, HANClassifier

def run_cot_rag_experiment(patient_id: str, use_existing_kg: bool = True):
    """
    Run a complete CoT-RAG experiment for a single patient.
    """
    print(f"Starting CoT-RAG experiment for patient {patient_id}")
    
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Initialize LLM interface
    llm_interface = LLMInterface(config)
    
    # Initialize data loaders
    ecg_loader = ECGLoader()
    clinical_loader = ClinicalDataLoader()
    
    # Initialize ECG classifiers
    classifiers = {
        'arrhythmia_model': SEResNetClassifier('models/weights/arrhythmia_model.pth'),
        'ischemia_model': HANClassifier('models/weights/ischemia_model.pth')
    }
    
    # Load or generate knowledge graph
    kg_path = f"output/knowledge_graphs/cardiology_kg.json"
    
    if use_existing_kg and Path(kg_path).exists():
        print("Loading existing knowledge graph...")
        kg_template = KnowledgeGraph.load_from_json(kg_path)
    else:
        print("Generating new knowledge graph from expert decision tree...")
        kg_generator = KGGenerator(llm_interface)
        kg_template = kg_generator.generate_from_decision_tree(
            'expert_knowledge/cardiology_dt.yaml'
        )
        # Save for future use
        Path("output/knowledge_graphs/").mkdir(parents=True, exist_ok=True)
        kg_template.save_to_json(kg_path)
    
    # Load patient data
    print(f"Loading patient data for {patient_id}...")
    patient_data = {
        'ecg_data': ecg_loader.load_patient_ecg(patient_id),
        'clinical_text': clinical_loader.load_clinical_notes(patient_id),
        'query_description': clinical_loader.load_query_description(patient_id)
    }
    
    # Stage 2: Populate knowledge graph with patient data
    print("Populating knowledge graph with patient-specific information...")
    rag_populator = RAGPopulator(llm_interface)
    populated_kg = rag_populator.populate_knowledge_graph(kg_template, patient_data)
    
    # Stage 3: Execute reasoning and generate diagnosis
    print("Executing diagnostic reasoning...")
    executor = ReasoningExecutor(llm_interface, classifiers)
    
    final_diagnosis, decision_path, metadata = executor.execute_reasoning_path(
        populated_kg, 
        patient_data['ecg_data']
    )
    
    # Generate narrative report
    print("Generating narrative report...")
    narrative_report = executor.generate_narrative_report(
        decision_path, populated_kg, metadata
    )
    
    # Display results
    print("\n" + "="*80)
    print("CoT-RAG DIAGNOSTIC RESULTS")
    print("="*80)
    
    print(f"\nPatient ID: {patient_id}")
    print(f"Final Diagnosis: {final_diagnosis}")
    print(f"Decision Path: {' → '.join(decision_path)}")
    
    print(f"\nEvidence Chain:")
    for evidence in metadata.get('evidence_chain', []):
        print(f"  - {evidence['question']}: {evidence['decision']} (conf: {evidence['confidence']:.2f})")
    
    print(f"\nNarrative Report:")
    print(narrative_report)
    
    # Save results
    results = {
        'patient_id': patient_id,
        'final_diagnosis': final_diagnosis,
        'decision_path': decision_path,
        'narrative_report': narrative_report,
        'metadata': metadata
    }
    
    output_path = f"output/results/patient_{patient_id}_cot_rag_results.json"
    Path("output/results/").mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

def run_comparison_experiment():
    """
    Run comparison between standard CoT and CoT-RAG approaches.
    """
    print("Running CoT vs CoT-RAG comparison experiment...")
    
    # Test patients
    test_patients = ['patient_001', 'patient_002', 'patient_003']
    
    results = {
        'standard_cot': {},
        'cot_rag': {}
    }
    
    for patient_id in test_patients:
        print(f"\n--- Processing {patient_id} ---")
        
        # Run standard CoT
        print("Running standard CoT...")
        # Use existing CoT implementation
        standard_result = run_experiment('ecg_diagnosis', use_cot=True)
        results['standard_cot'][patient_id] = standard_result
        
        # Run CoT-RAG
        print("Running CoT-RAG...")
        cot_rag_result = run_cot_rag_experiment(patient_id)
        results['cot_rag'][patient_id] = cot_rag_result
    
    # Save comparison results
    with open('output/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nComparison experiment completed!")
    return results

def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python main.py [single|comparison] [patient_id]")
        print("  single: Run CoT-RAG for a single patient")
        print("  comparison: Run comparison between CoT and CoT-RAG")
        return
    
    mode = sys.argv[1]
    
    if mode == 'single':
        if len(sys.argv) < 3:
            print("Error: Patient ID required for single mode")
            return
        patient_id = sys.argv[2]
        run_cot_rag_experiment(patient_id)
    
    elif mode == 'comparison':
        run_comparison_experiment()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: single, comparison")

if __name__ == '__main__':
    main()
```

## 5. Dependencies and Setup

### 5.1 Additional Requirements (`requirements-cotrag.txt`)

```
# Additional dependencies for CoT-RAG implementation
# Add these to your existing requirements.txt

# Knowledge graph and data processing
networkx>=3.0
pyyaml>=6.0
jsonschema>=4.0

# Medical data processing
wfdb>=4.0  # For ECG data processing
scipy>=1.9.0
scikit-learn>=1.2.0

# Deep learning for ECG classification
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.20.0

# Medical ontology and NLP
spacy>=3.4.0
nltk>=3.8

# Visualization and reporting
matplotlib>=3.6.0
seaborn>=0.11.0
plotly>=5.0.0

# Development and testing
pytest>=7.0.0
black>=22.0.0
mypy>=0.991
```

### 5.2 Setup Script (`setup_cotrag.py`)

```python
#!/usr/bin/env python3
"""
Setup script for CoT-RAG implementation.
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Setup the CoT-RAG development environment."""
    print("Setting up CoT-RAG environment...")
    
    # Install additional requirements
    print("Installing additional dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements-cotrag.txt"
    ], check=True)
    
    # Create necessary directories
    directories = [
        "output/knowledge_graphs",
        "output/results", 
        "output/logs",
        "data/ptb_xl",
        "data/chapman_shaoxing",
        "models/weights",
        "expert_knowledge"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Download spacy models for NLP
    print("Downloading spacy models...")
    subprocess.run([
        sys.executable, "-m", "spacy", "download", "en_core_web_sm"
    ], check=True)
    
    # Create sample expert decision tree
    create_sample_decision_tree()
    
    print("Setup completed successfully!")

def create_sample_decision_tree():
    """Create a sample expert decision tree for testing."""
    sample_dt = {
        'domain': 'cardiology',
        'version': '1.0',
        'expert': 'Dr. Sample Cardiologist',
        'description': 'Basic cardiology decision tree for ECG diagnosis',
        'nodes': [
            {
                'node_id': 'root',
                'question': 'Is the ECG rhythm regular or irregular?',
                'knowledge_case': 'Regular rhythm shows consistent R-R intervals, while irregular rhythm shows varying R-R intervals',
                'is_root': True,
                'children': ['rhythm_regular', 'rhythm_irregular']
            },
            {
                'node_id': 'rhythm_regular',
                'question': 'What is the heart rate category?',
                'knowledge_case': 'Normal: 60-100 bpm, Bradycardia: <60 bpm, Tachycardia: >100 bpm',
                'parent': 'root',
                'children': ['normal_rate', 'bradycardia', 'tachycardia']
            },
            {
                'node_id': 'rhythm_irregular',
                'question': 'Is there evidence of atrial fibrillation?',
                'knowledge_case': 'Atrial fibrillation shows irregularly irregular rhythm with absent P waves',
                'parent': 'root',
                'children': ['atrial_fibrillation', 'other_irregular']
            }
        ]
    }
    
    import yaml
    with open('expert_knowledge/cardiology_dt.yaml', 'w') as f:
        yaml.dump(sample_dt, f, default_flow_style=False, indent=2)
    
    print("Created sample decision tree: expert_knowledge/cardiology_dt.yaml")

if __name__ == '__main__':
    setup_environment()
```

## 6. Testing and Validation Framework

### 6.1 Test Suite (`tests/test_cotrag.py`)

```python
import pytest
import tempfile
import json
from pathlib import Path

from core.knowledge_graph import KnowledgeGraph, KGNode, NodeType
from core.stage1_generator import KGGenerator
from core.stage2_rag import RAGPopulator
from core.stage3_executor import ReasoningExecutor

class TestKnowledgeGraph:
    """Test cases for Knowledge Graph functionality."""
    
    def test_node_creation(self):
        """Test KGNode creation and serialization."""
        node = KGNode(
            node_id="test_node",
            diagnosis_class="Test Condition",
            node_type=NodeType.INTERNAL,
            sub_question="Is condition present?",
            sub_case="Example case description"
        )
        
        assert node.node_id == "test_node"
        assert node.diagnosis_class == "Test Condition"
        assert node.node_type == NodeType.INTERNAL
        
        # Test serialization
        node_dict = node.to_dict()
        assert isinstance(node_dict, dict)
        assert node_dict['node_id'] == "test_node"
        
        # Test deserialization
        restored_node = KGNode.from_dict(node_dict)
        assert restored_node.node_id == node.node_id
        assert restored_node.node_type == node.node_type
    
    def test_knowledge_graph_operations(self):
        """Test KnowledgeGraph operations."""
        kg = KnowledgeGraph(domain="test")
        
        # Create test nodes
        root_node = KGNode("root", "Root", NodeType.ROOT)
        child_node = KGNode("child", "Child", NodeType.LEAF, parent_node="root")
        root_node.child_nodes = ["child"]
        
        # Add nodes
        kg.add_node(root_node)
        kg.add_node(child_node)
        
        # Test retrieval
        assert kg.get_node("root") == root_node
        assert kg.get_node("child") == child_node
        assert kg.root_node_id == "root"
        
        # Test children retrieval
        children = kg.get_children("root")
        assert len(children) == 1
        assert children[0] == child_node
        
        # Test validation
        errors = kg.validate_structure()
        assert len(errors) == 0
    
    def test_kg_serialization(self):
        """Test KnowledgeGraph JSON serialization."""
        kg = KnowledgeGraph(domain="test")
        
        node = KGNode("test", "Test", NodeType.ROOT)
        kg.add_node(node)
        
        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            kg.save_to_json(f.name)
            
            # Load and verify
            loaded_kg = KnowledgeGraph.load_from_json(f.name)
            assert loaded_kg.domain == kg.domain
            assert loaded_kg.root_node_id == kg.root_node_id
            assert len(loaded_kg.nodes) == len(kg.nodes)
            
            Path(f.name).unlink()  # Cleanup

class TestRAGPopulator:
    """Test cases for RAG population functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.kg = KnowledgeGraph(domain="test")
        
        node = KGNode(
            node_id="test_node",
            diagnosis_class="Test Condition",
            node_type=NodeType.LEAF,
            sub_question="What is the patient's heart rate?",
            sub_case="Normal heart rate is 60-100 bpm"
        )
        
        self.kg.add_node(node)
        
        self.patient_data = {
            'clinical_text': 'Patient presents with heart rate of 85 bpm, regular rhythm.',
            'query_description': 'Evaluate cardiac rhythm'
        }
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Mock LLM interface for testing."""
        class MockLLMInterface:
            def query(self, prompt, model_name):
                if "heart rate" in prompt.lower():
                    return "EXTRACTED_TEXT: Heart rate of 85 bpm\nCONFIDENCE: 0.9"
                return "No relevant information found"
        
        return MockLLMInterface()
    
    def test_rag_population(self, mock_llm_interface):
        """Test RAG population process."""
        populator = RAGPopulator(mock_llm_interface)
        
        populated_kg = populator.populate_knowledge_graph(self.kg, self.patient_data)
        
        # Check that node was populated
        node = populated_kg.get_node("test_node")
        assert node.sub_description == "Heart rate of 85 bpm"

class TestReasoningExecutor:
    """Test cases for reasoning execution."""
    
    @pytest.fixture
    def mock_classifiers(self):
        """Mock classifiers for testing."""
        class MockClassifier:
            def predict(self, ecg_data):
                return {
                    'normal': 0.8,
                    'abnormal': 0.2
                }
        
        return {
            'rhythm_classifier': MockClassifier()
        }
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Mock LLM interface."""
        class MockLLMInterface:
            def query(self, prompt, model_name):
                if "final diagnosis" in prompt.lower():
                    return "Normal sinus rhythm"
                return "DECISION: continue\nCONFIDENCE: 0.8"
        
        return MockLLMInterface()
    
    def test_reasoning_execution(self, mock_llm_interface, mock_classifiers):
        """Test complete reasoning execution."""
        # Create simple KG for testing
        kg = KnowledgeGraph(domain="test")
        
        root_node = KGNode(
            node_id="root",
            diagnosis_class="Rhythm Assessment",
            node_type=NodeType.ROOT,
            sub_question="Is the rhythm normal?",
            decision_rule_logic="IF get_prob('rhythm_classifier', 'normal') > 0.5 THEN normal_rhythm"
        )
        
        leaf_node = KGNode(
            node_id="normal_rhythm",
            diagnosis_class="Normal Sinus Rhythm",
            node_type=NodeType.LEAF,
            parent_node="root"
        )
        
        root_node.child_nodes = ["normal_rhythm"]
        
        kg.add_node(root_node)
        kg.add_node(leaf_node)
        
        # Execute reasoning
        executor = ReasoningExecutor(mock_llm_interface, mock_classifiers)
        
        diagnosis, path, metadata = executor.execute_reasoning_path(kg, ecg_data=None)
        
        assert isinstance(diagnosis, str)
        assert len(path) > 0
        assert 'evidence_chain' in metadata

def run_integration_tests():
    """Run integration tests for the complete CoT-RAG pipeline."""
    print("Running CoT-RAG integration tests...")
    
    # This would test the complete workflow
    # from expert decision tree to final diagnosis
    
    pytest.main([__file__, "-v"])

if __name__ == '__main__':
    run_integration_tests()
```

## 7. Deployment and Production Considerations

### 7.1 Configuration Management (`config.py`)

```python
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class CoTRAGConfig:
    """Configuration for CoT-RAG system."""
    
    # LLM Configuration
    llm_provider: str = "openai"  # openai, baidu, zhipuai
    llm_api_key: str = ""
    llm_base_url: str = ""
    
    # Model paths
    ecg_model_dir: str = "models/weights"
    kg_cache_dir: str = "output/knowledge_graphs"
    
    # Expert knowledge
    expert_dt_dir: str = "expert_knowledge"
    
    # Processing parameters
    confidence_threshold: float = 0.7
    max_reasoning_depth: int = 10
    
    # Evaluation settings
    use_hierarchical_metrics: bool = True
    clinical_validation: bool = False
    
    @classmethod
    def from_env(cls) -> 'CoTRAGConfig':
        """Create configuration from environment variables."""
        return cls(
            llm_provider=os.getenv('COTRAG_LLM_PROVIDER', 'openai'),
            llm_api_key=os.getenv('COTRAG_API_KEY', ''),
            llm_base_url=os.getenv('COTRAG_BASE_URL', ''),
            ecg_model_dir=os.getenv('COTRAG_MODEL_DIR', 'models/weights'),
            kg_cache_dir=os.getenv('COTRAG_KG_DIR', 'output/knowledge_graphs'),
            expert_dt_dir=os.getenv('COTRAG_EXPERT_DIR', 'expert_knowledge'),
            confidence_threshold=float(os.getenv('COTRAG_CONFIDENCE', '0.7')),
            max_reasoning_depth=int(os.getenv('COTRAG_MAX_DEPTH', '10')),
            use_hierarchical_metrics=os.getenv('COTRAG_HIERARCHICAL', 'true').lower() == 'true',
            clinical_validation=os.getenv('COTRAG_CLINICAL_VAL', 'false').lower() == 'true'
        )
```

### 7.2 Monitoring and Logging (`utils/monitoring.py`)

```python
import logging
import time
from typing import Dict, Any
from datetime import datetime

class CoTRAGMonitor:
    """Monitoring and logging for CoT-RAG system."""
    
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.metrics = {
            'reasoning_times': [],
            'confidence_scores': [],
            'error_counts': {},
            'stage_performance': {}
        }
    
    def setup_logging(self, log_level):
        """Setup logging configuration."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('output/logs/cotrag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CoTRAG')
    
    def log_reasoning_start(self, patient_id: str, kg_domain: str):
        """Log start of reasoning process."""
        self.logger.info(f"Starting CoT-RAG reasoning for patient {patient_id} in domain {kg_domain}")
        return time.time()
    
    def log_reasoning_complete(self, patient_id: str, start_time: float, 
                             diagnosis: str, confidence: float):
        """Log completion of reasoning process."""
        duration = time.time() - start_time
        self.metrics['reasoning_times'].append(duration)
        self.metrics['confidence_scores'].append(confidence)
        
        self.logger.info(
            f"CoT-RAG reasoning completed for patient {patient_id}: "
            f"diagnosis='{diagnosis}', confidence={confidence:.3f}, duration={duration:.2f}s"
        )
    
    def log_error(self, stage: str, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context."""
        if stage not in self.metrics['error_counts']:
            self.metrics['error_counts'][stage] = 0
        self.metrics['error_counts'][stage] += 1
        
        self.logger.error(
            f"Error in {stage}: {str(error)}",
            extra={'context': context or {}}
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics['reasoning_times']:
            return {'status': 'No data available'}
        
        import statistics
        
        return {
            'total_reasoning_sessions': len(self.metrics['reasoning_times']),
            'avg_reasoning_time': statistics.mean(self.metrics['reasoning_times']),
            'avg_confidence': statistics.mean(self.metrics['confidence_scores']),
            'error_summary': self.metrics['error_counts'],
            'timestamp': datetime.now().isoformat()
        }
```

## 8. Next Steps and Future Enhancements

### 8.1 Research Extensions

1. **Multi-Modal Integration**: Extend to incorporate medical images, lab results, and vital signs
2. **Temporal Reasoning**: Add support for longitudinal patient data and disease progression
3. **Uncertainty Quantification**: Implement Bayesian approaches for better uncertainty handling
4. **Active Learning**: Develop systems that can request specific additional information
5. **Federated Learning**: Enable multi-institutional knowledge sharing while preserving privacy

### 8.2 Clinical Integration

1. **FHIR Compatibility**: Implement HL7 FHIR standards for healthcare interoperability
2. **Clinical Decision Support**: Integration with Electronic Health Record (EHR) systems
3. **Regulatory Compliance**: Ensure compliance with medical device regulations (FDA, CE marking)
4. **Clinical Validation Studies**: Design and execute prospective clinical trials
5. **Expert Feedback Loops**: Implement systems for continuous expert validation and knowledge updates

This comprehensive implementation plan provides a solid foundation for building the CoT-RAG framework, extending your existing CoT implementation with expert knowledge integration, retrieval-augmented reasoning, and clinical validation capabilities.