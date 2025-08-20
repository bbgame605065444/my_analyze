"""
Stage 3: Reasoning Execution and Narrative Generation
Executes diagnostic reasoning through populated knowledge graphs and generates interpretable reports.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import re
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .knowledge_graph import KnowledgeGraph, KGNode, NodeType
from utils.llm_interface import LLMInterface
from utils.prompt_templates import PromptTemplates
from utils.medical_ontology import get_medical_ontology_mapper

class DecisionMethod(Enum):
    """Methods for making decisions at each node."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    CLASSIFIER_BASED = "classifier_based"
    HYBRID = "hybrid"

@dataclass
class DecisionResult:
    """Result of a decision at a knowledge graph node."""
    decision: str
    confidence: float
    method: DecisionMethod
    evidence: str = ""
    next_node_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExecutionStep:
    """Single step in the reasoning execution."""
    node_id: str
    question: str
    evidence: str
    decision: str
    confidence: float
    method: str
    timestamp: str
    next_node: Optional[str] = None

class BaseClassifier:
    """
    Abstract base class for medical classifiers.
    Provides interface for ECG and other medical data classification.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
    
    def predict(self, data: Any) -> Dict[str, float]:
        """
        Predict class probabilities for input data.
        
        Args:
            data: Input data (ECG signals, etc.)
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def load_model(self, model_path: str) -> None:
        """Load the trained model."""
        raise NotImplementedError("Subclasses must implement load_model method")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'type': self.__class__.__name__
        }

class MockECGClassifier(BaseClassifier):
    """
    Mock ECG classifier for testing and development.
    Provides realistic probability distributions for common ECG conditions.
    """
    
    def __init__(self, model_name: str = "mock_ecg_classifier"):
        super().__init__(model_name)
        self.is_loaded = True
        
        # Mock probability distributions for different scenarios
        self.scenarios = {
            'normal': {
                'normal_sinus_rhythm': 0.85,
                'atrial_fibrillation': 0.05,
                'ventricular_tachycardia': 0.02,
                'bradycardia': 0.08
            },
            'atrial_fibrillation': {
                'normal_sinus_rhythm': 0.15,
                'atrial_fibrillation': 0.75,
                'atrial_flutter': 0.08,
                'ventricular_tachycardia': 0.02
            },
            'myocardial_infarction': {
                'normal_sinus_rhythm': 0.25,
                'anterior_mi': 0.40,
                'inferior_mi': 0.25,
                'lateral_mi': 0.10
            }
        }
    
    def predict(self, data: Any) -> Dict[str, float]:
        """Mock prediction based on data characteristics."""
        # Simple heuristic based on data patterns
        if isinstance(data, dict) and 'scenario' in data:
            scenario = data['scenario']
            return self.scenarios.get(scenario, self.scenarios['normal'])
        elif isinstance(data, str):
            # Text-based scenario detection
            if 'atrial fibrillation' in data.lower() or 'irregular' in data.lower():
                return self.scenarios['atrial_fibrillation']
            elif 'myocardial infarction' in data.lower() or 'mi' in data.lower():
                return self.scenarios['myocardial_infarction']
        
        # Default to normal scenario
        return self.scenarios['normal']

class ReasoningExecutor:
    """
    Stage 3: Executes diagnostic reasoning through populated knowledge graphs.
    
    Implements the complete reasoning workflow:
    1. Knowledge graph traversal
    2. Decision rule execution
    3. Classifier integration
    4. Narrative generation
    """
    
    def __init__(self, llm_interface: LLMInterface, classifiers: Dict[str, BaseClassifier] = None):
        """
        Initialize reasoning executor.
        
        Args:
            llm_interface: Interface for LLM queries
            classifiers: Dictionary of available classifiers
        """
        self.llm = llm_interface
        self.classifiers = classifiers or {}
        self.prompt_templates = PromptTemplates()
        self.medical_mapper = get_medical_ontology_mapper()
        
        # Execution state tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_path_length': 0.0,
            'average_confidence': 0.0,
            'decision_method_counts': {method.value: 0 for method in DecisionMethod}
        }
        
        # Add mock classifier if no real classifiers provided
        if not self.classifiers:
            self.classifiers['mock_ecg'] = MockECGClassifier()
    
    def execute_reasoning_path(self, populated_kg: KnowledgeGraph, 
                             patient_data: Dict[str, Any] = None,
                             max_depth: int = 20) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Execute complete reasoning path through populated knowledge graph.
        
        Args:
            populated_kg: Knowledge graph populated with patient data
            patient_data: Optional patient data for classifier inputs
            max_depth: Maximum traversal depth to prevent infinite loops
            
        Returns:
            Tuple of (final_diagnosis, decision_path, execution_metadata)
        """
        self.execution_stats['total_executions'] += 1
        
        print(f"=" * 60)
        print("STAGE 3: REASONING EXECUTION")
        print(f"=" * 60)
        print(f"Domain: {populated_kg.domain}")
        print(f"Total nodes: {len(populated_kg.nodes)}")
        
        # Validate knowledge graph
        if not populated_kg.root_node_id:
            raise ValueError("Knowledge graph has no root node defined")
        
        if populated_kg.root_node_id not in populated_kg.nodes:
            raise ValueError(f"Root node {populated_kg.root_node_id} not found in graph")
        
        # Initialize execution state
        execution_state = {
            'decision_path': [],
            'execution_steps': [],
            'classifier_outputs': {},
            'confidence_scores': [],
            'evidence_chain': [],
            'start_time': self._get_timestamp(),
            'patient_data': patient_data or {}
        }
        
        # Pre-compute classifier outputs if data available
        print(f"\nPre-computing classifier outputs...")
        execution_state['classifier_outputs'] = self._compute_classifier_outputs(patient_data)
        
        print(f"Available classifiers: {list(self.classifiers.keys())}")
        for classifier_name, outputs in execution_state['classifier_outputs'].items():
            print(f"  {classifier_name}: {len(outputs)} class predictions")
        
        # Execute traversal starting from root
        print(f"\nStarting reasoning traversal from root: {populated_kg.root_node_id}")
        
        try:
            final_diagnosis = self._traverse_knowledge_graph(
                populated_kg, 
                populated_kg.root_node_id, 
                execution_state,
                depth=0,
                max_depth=max_depth
            )
            
            # Update success statistics
            self.execution_stats['successful_executions'] += 1
            path_length = len(execution_state['decision_path'])
            avg_confidence = sum(execution_state['confidence_scores']) / max(len(execution_state['confidence_scores']), 1)
            
            self.execution_stats['average_path_length'] = (
                (self.execution_stats['average_path_length'] * (self.execution_stats['successful_executions'] - 1) + path_length) /
                self.execution_stats['successful_executions']
            )
            self.execution_stats['average_confidence'] = (
                (self.execution_stats['average_confidence'] * (self.execution_stats['successful_executions'] - 1) + avg_confidence) /
                self.execution_stats['successful_executions']
            )
            
            print(f"\n✓ Reasoning execution completed successfully")
            print(f"  Final diagnosis: {final_diagnosis}")
            print(f"  Decision path length: {path_length}")
            print(f"  Average confidence: {avg_confidence:.2f}")
            
        except Exception as e:
            self.execution_stats['failed_executions'] += 1
            print(f"\n✗ Reasoning execution failed: {e}")
            
            # Generate fallback diagnosis
            final_diagnosis = self._generate_fallback_diagnosis(populated_kg, execution_state, str(e))
        
        # Finalize execution metadata
        execution_state['end_time'] = self._get_timestamp()
        execution_state['final_diagnosis'] = final_diagnosis
        
        return (
            final_diagnosis,
            execution_state['decision_path'],
            execution_state
        )
    
    def _traverse_knowledge_graph(self, kg: KnowledgeGraph, 
                                current_node_id: str, 
                                execution_state: Dict[str, Any],
                                depth: int = 0,
                                max_depth: int = 20) -> str:
        """
        Recursively traverse knowledge graph based on decision rules.
        
        Args:
            kg: Knowledge graph
            current_node_id: Current node being processed
            execution_state: Execution state tracking
            depth: Current traversal depth
            max_depth: Maximum allowed depth
            
        Returns:
            Final diagnosis string
        """
        if depth >= max_depth:
            print(f"  ⚠  Maximum traversal depth ({max_depth}) reached at node {current_node_id}")
            return self._generate_max_depth_diagnosis(current_node_id, execution_state)
        
        current_node = kg.get_node(current_node_id)
        if not current_node:
            raise ValueError(f"Node {current_node_id} not found in knowledge graph")
        
        print(f"\n{'  ' * depth}Processing node: {current_node_id} (depth {depth})")
        print(f"{'  ' * depth}Question: {current_node.sub_question}")
        
        # Add current node to decision path
        execution_state['decision_path'].append(current_node_id)
        
        # Execute decision for current node
        decision_result = self._execute_node_decision(current_node, execution_state)
        
        print(f"{'  ' * depth}Decision: {decision_result.decision} (confidence: {decision_result.confidence:.2f})")
        print(f"{'  ' * depth}Method: {decision_result.method.value}")
        
        # Track statistics
        self.execution_stats['decision_method_counts'][decision_result.method.value] += 1
        execution_state['confidence_scores'].append(decision_result.confidence)
        
        # Store execution step
        execution_step = ExecutionStep(
            node_id=current_node_id,
            question=current_node.sub_question,
            evidence=current_node.sub_description,
            decision=decision_result.decision,
            confidence=decision_result.confidence,
            method=decision_result.method.value,
            timestamp=self._get_timestamp(),
            next_node=decision_result.next_node_id
        )
        execution_state['execution_steps'].append(execution_step)
        
        # Store evidence for reasoning chain
        execution_state['evidence_chain'].append({
            'node_id': current_node_id,
            'question': current_node.sub_question,
            'evidence': current_node.sub_description,
            'decision': decision_result.decision,
            'confidence': decision_result.confidence,
            'method': decision_result.method.value
        })
        
        # Check if this is a terminal node or decision leads to end
        if current_node.node_type == NodeType.LEAF or not current_node.child_nodes:
            print(f"{'  ' * depth}Reached leaf node: {current_node_id}")
            return self._generate_final_diagnosis(current_node, execution_state)
        
        # Determine next node based on decision result
        next_node_id = decision_result.next_node_id or self._select_next_node(
            current_node, decision_result, execution_state
        )
        
        if next_node_id and next_node_id in kg.nodes:
            print(f"{'  ' * depth}Moving to next node: {next_node_id}")
            return self._traverse_knowledge_graph(kg, next_node_id, execution_state, depth + 1, max_depth)
        else:
            print(f"{'  ' * depth}No valid next node found, generating diagnosis at current node")
            return self._generate_final_diagnosis(current_node, execution_state)
    
    def _execute_node_decision(self, node: KGNode, 
                             execution_state: Dict[str, Any]) -> DecisionResult:
        """
        Execute decision logic for a specific node.
        
        Args:
            node: Current knowledge graph node
            execution_state: Current execution state
            
        Returns:
            DecisionResult with decision information
        """
        print(f"    Executing decision for node: {node.node_id}")
        
        # Check if node has decision rule logic
        if node.decision_rule_logic and node.decision_rule_logic.strip():
            print(f"    Using rule-based decision")
            try:
                return self._execute_rule_based_decision(node, execution_state)
            except Exception as e:
                print(f"    Rule-based decision failed: {e}, falling back to LLM")
                return self._execute_llm_based_decision(node, execution_state)
        
        # Check if node specifies required classifier inputs
        elif node.required_classifier_inputs:
            print(f"    Using classifier-based decision")
            try:
                return self._execute_classifier_based_decision(node, execution_state)
            except Exception as e:
                print(f"    Classifier-based decision failed: {e}, falling back to LLM")
                return self._execute_llm_based_decision(node, execution_state)
        
        # Default to LLM-based decision
        else:
            print(f"    Using LLM-based decision")
            return self._execute_llm_based_decision(node, execution_state)
    
    def _execute_rule_based_decision(self, node: KGNode, 
                                   execution_state: Dict[str, Any]) -> DecisionResult:
        """Execute rule-based decision logic."""
        rule_logic = node.decision_rule_logic
        print(f"      Rule: {rule_logic}")
        
        # Build decision context
        context = self._build_decision_context(node, execution_state)
        
        # Parse and execute rule
        result = self._evaluate_rule(rule_logic, context)
        
        return DecisionResult(
            decision=result['outcome'],
            confidence=result['confidence'],
            method=DecisionMethod.RULE_BASED,
            evidence=f"Rule evaluation: {rule_logic}",
            next_node_id=result.get('next_node'),
            metadata={'rule': rule_logic, 'context': context}
        )
    
    def _execute_classifier_based_decision(self, node: KGNode, 
                                         execution_state: Dict[str, Any]) -> DecisionResult:
        """Execute classifier-based decision logic."""
        required_inputs = node.required_classifier_inputs
        print(f"      Required classifiers: {required_inputs}")
        
        classifier_outputs = execution_state['classifier_outputs']
        confidence_threshold = node.confidence_threshold
        
        # Aggregate classifier results
        max_confidence = 0.0
        predicted_class = "uncertain"
        supporting_evidence = []
        
        for classifier_name in required_inputs:
            if classifier_name in classifier_outputs:
                class_probs = classifier_outputs[classifier_name]
                for class_name, prob in class_probs.items():
                    if prob > max_confidence:
                        max_confidence = prob
                        predicted_class = class_name
                
                supporting_evidence.append(f"{classifier_name}: {dict(list(class_probs.items())[:3])}")
        
        # Make decision based on confidence threshold
        if max_confidence >= confidence_threshold:
            decision = predicted_class
            confidence = max_confidence
        else:
            decision = "uncertain"
            confidence = max_confidence * 0.5  # Reduce confidence for uncertain decisions
        
        return DecisionResult(
            decision=decision,
            confidence=confidence,
            method=DecisionMethod.CLASSIFIER_BASED,
            evidence=f"Classifier predictions: {'; '.join(supporting_evidence)}",
            metadata={'classifier_outputs': dict(classifier_outputs), 'threshold': confidence_threshold}
        )
    
    def _execute_llm_based_decision(self, node: KGNode, 
                                  execution_state: Dict[str, Any]) -> DecisionResult:
        """Execute LLM-based decision when other methods are not available."""
        print(f"      Querying LLM for decision")
        
        # Create decision prompt
        decision_prompt = self.prompt_templates.get_decision_prompt(
            node=node,
            evidence_chain=execution_state['evidence_chain'],
            classifier_outputs=execution_state['classifier_outputs'],
            patient_context=execution_state['patient_data']
        )
        
        try:
            response = self.llm.query(
                prompt=decision_prompt,
                model_name="reasoning_model"
            )
            
            parsed_result = self._parse_decision_response(response)
            
            return DecisionResult(
                decision=parsed_result['decision'],
                confidence=parsed_result['confidence'],
                method=DecisionMethod.LLM_BASED,
                evidence=f"LLM reasoning: {response[:200]}...",
                metadata={'full_response': response, 'prompt': decision_prompt}
            )
            
        except Exception as e:
            print(f"      LLM decision failed: {e}")
            return DecisionResult(
                decision="uncertain",
                confidence=0.3,
                method=DecisionMethod.LLM_BASED,
                evidence=f"LLM decision failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _build_decision_context(self, node: KGNode, 
                              execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for decision rule execution."""
        return {
            'classifier_outputs': execution_state['classifier_outputs'],
            'node_evidence': node.sub_description,
            'node_question': node.sub_question,
            'previous_decisions': execution_state['evidence_chain'],
            'patient_data': execution_state['patient_data'],
            'confidence_threshold': node.confidence_threshold
        }
    
    def _evaluate_rule(self, rule_logic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate decision rule with current context.
        Supports simple IF-THEN rules with classifier probability queries.
        """
        print(f"        Evaluating rule: {rule_logic}")
        
        try:
            # Extract classifier probability queries
            classifier_queries = re.findall(r"get_prob\('([^']+)',\s*'([^']+)'\)", rule_logic)
            
            # Replace classifier queries with actual values
            processed_rule = rule_logic
            for classifier_name, class_name in classifier_queries:
                prob_value = self._get_classifier_probability(
                    classifier_name, class_name, context['classifier_outputs']
                )
                query_pattern = f"get_prob('{classifier_name}', '{class_name}')"
                processed_rule = processed_rule.replace(query_pattern, str(prob_value))
                print(f"        {query_pattern} -> {prob_value}")
            
            # Parse IF-THEN rule structure
            if 'IF' in processed_rule.upper() and 'THEN' in processed_rule.upper():
                parts = processed_rule.upper().split('THEN')
                condition_part = parts[0].replace('IF', '').strip()
                outcome_part = parts[1].strip()
                
                # Safely evaluate condition (limited scope for security)
                condition_result = self._safe_eval_condition(condition_part)
                
                return {
                    'outcome': outcome_part.lower() if condition_result else 'continue',
                    'confidence': 0.9 if condition_result else 0.1,
                    'next_node': outcome_part.lower() if condition_result else None
                }
            else:
                return {'outcome': 'continue', 'confidence': 0.5}
                
        except Exception as e:
            print(f"        Rule evaluation error: {e}")
            return {'outcome': 'error', 'confidence': 0.0}
    
    def _safe_eval_condition(self, condition: str) -> bool:
        """Safely evaluate condition expression."""
        # Only allow specific operators and numeric comparisons
        allowed_chars = set('0123456789.<>=! ()and or')
        if not all(c.lower() in allowed_chars for c in condition):
            raise ValueError(f"Unsafe characters in condition: {condition}")
        
        # Replace logical operators
        condition = condition.replace(' AND ', ' and ').replace(' OR ', ' or ')
        
        try:
            return eval(condition)
        except Exception:
            return False
    
    def _get_classifier_probability(self, classifier_name: str, class_name: str, 
                                  classifier_outputs: Dict[str, Any]) -> float:
        """Get probability for specific class from classifier outputs."""
        if classifier_name in classifier_outputs:
            class_probs = classifier_outputs[classifier_name]
            return class_probs.get(class_name, 0.0)
        return 0.0
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM decision response."""
        result = {
            'decision': 'continue',
            'confidence': 0.5
        }
        
        try:
            # Look for structured decision indicators
            if 'DECISION:' in response.upper():
                decision_match = re.search(r'DECISION:\s*([^\n]+)', response, re.IGNORECASE)
                if decision_match:
                    result['decision'] = decision_match.group(1).strip()
            
            if 'CONFIDENCE:' in response.upper():
                conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    result['confidence'] = max(0.0, min(1.0, confidence))
            
        except Exception as e:
            print(f"        Error parsing decision response: {e}")
        
        return result
    
    def _compute_classifier_outputs(self, patient_data: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
        """Pre-compute outputs from all available classifiers."""
        outputs = {}
        
        for classifier_name, classifier in self.classifiers.items():
            try:
                # Use patient data or default input
                input_data = patient_data if patient_data else {}
                class_probabilities = classifier.predict(input_data)
                outputs[classifier_name] = class_probabilities
                
                print(f"  {classifier_name}: {len(class_probabilities)} predictions")
                
            except Exception as e:
                print(f"  Warning: Classifier {classifier_name} failed: {e}")
                outputs[classifier_name] = {}
        
        return outputs
    
    def _select_next_node(self, current_node: KGNode, 
                         decision_result: DecisionResult, 
                         execution_state: Dict[str, Any]) -> Optional[str]:
        """Select next node based on decision result."""
        decision = decision_result.decision.lower()
        
        # Check if decision matches a specific child node
        for child_id in current_node.child_nodes:
            if decision in child_id.lower():
                return child_id
        
        # Default to first child if available
        if current_node.child_nodes:
            return current_node.child_nodes[0]
        
        return None
    
    def _generate_final_diagnosis(self, final_node: KGNode, 
                                execution_state: Dict[str, Any]) -> str:
        """Generate final diagnosis based on reasoning chain."""
        print(f"    Generating final diagnosis at node: {final_node.node_id}")
        
        try:
            # Create final diagnosis prompt
            final_prompt = self.prompt_templates.get_final_diagnosis_prompt(
                leaf_node=final_node,
                evidence_chain=execution_state['evidence_chain'],
                classifier_outputs=execution_state['classifier_outputs'],
                patient_data=execution_state['patient_data']
            )
            
            response = self.llm.query(
                prompt=final_prompt,
                model_name="diagnosis_model"
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"    Final diagnosis generation failed: {e}")
            # Fallback diagnosis
            return f"Clinical assessment suggests {final_node.diagnosis_class}. Confidence based on available evidence."
    
    def _generate_fallback_diagnosis(self, kg: KnowledgeGraph, 
                                   execution_state: Dict[str, Any], 
                                   error_msg: str) -> str:
        """Generate fallback diagnosis when execution fails."""
        evidence_chain = execution_state.get('evidence_chain', [])
        
        if evidence_chain:
            # Use evidence from completed steps
            last_evidence = evidence_chain[-1]
            return f"Partial assessment: {last_evidence['decision']} (confidence: {last_evidence['confidence']:.2f}). Complete evaluation required due to: {error_msg}"
        else:
            return f"Unable to complete diagnostic assessment. Domain: {kg.domain}. Error: {error_msg}"
    
    def _generate_max_depth_diagnosis(self, current_node_id: str, 
                                    execution_state: Dict[str, Any]) -> str:
        """Generate diagnosis when maximum traversal depth is reached."""
        evidence_chain = execution_state.get('evidence_chain', [])
        
        if evidence_chain:
            # Summarize evidence collected so far
            high_confidence_evidence = [e for e in evidence_chain if e['confidence'] > 0.7]
            
            if high_confidence_evidence:
                last_evidence = high_confidence_evidence[-1]
                return f"Assessment based on available evidence: {last_evidence['decision']} (confidence: {last_evidence['confidence']:.2f}). Further evaluation recommended."
            else:
                return "Complex case requiring additional evaluation. Multiple factors identified but no definitive conclusion reached."
        else:
            return f"Assessment incomplete due to complexity. Stopped at: {current_node_id}"
    
    def generate_narrative_report(self, decision_path: List[str], 
                                populated_kg: KnowledgeGraph,
                                execution_metadata: Dict[str, Any]) -> str:
        """
        Generate comprehensive narrative report of diagnostic reasoning.
        
        Args:
            decision_path: List of node IDs in decision path
            populated_kg: Knowledge graph used for reasoning
            execution_metadata: Metadata from reasoning execution
            
        Returns:
            Formatted narrative report
        """
        print(f"\nGenerating narrative report...")
        
        try:
            # Create narrative prompt
            narrative_prompt = self.prompt_templates.get_narrative_prompt(
                decision_path=decision_path,
                knowledge_graph=populated_kg,
                evidence_chain=execution_metadata.get('evidence_chain', []),
                classifier_outputs=execution_metadata.get('classifier_outputs', {}),
                patient_data=execution_metadata.get('patient_data', {})
            )
            
            narrative = self.llm.query(
                prompt=narrative_prompt,
                model_name="narrative_model"
            )
            
            # Format with additional metadata
            return self._format_narrative_report(narrative, execution_metadata, populated_kg)
            
        except Exception as e:
            print(f"  Warning: Narrative generation failed: {e}")
            return self._generate_fallback_narrative(decision_path, execution_metadata, populated_kg)
    
    def _format_narrative_report(self, narrative: str, 
                               metadata: Dict[str, Any],
                               kg: KnowledgeGraph) -> str:
        """Format narrative report with additional metadata."""
        
        evidence_chain = metadata.get('evidence_chain', [])
        classifier_outputs = metadata.get('classifier_outputs', {})
        
        formatted_report = f"""# CoT-RAG Diagnostic Reasoning Report

## Executive Summary
{narrative}

## Clinical Reasoning Path
**Decision Path**: {' → '.join(metadata.get('decision_path', []))}

## Detailed Evidence Analysis
"""
        
        for i, evidence in enumerate(evidence_chain, 1):
            formatted_report += f"""
### Step {i}: {evidence['question']}
- **Evidence**: {evidence['evidence']}
- **Decision**: {evidence['decision']}
- **Confidence**: {evidence['confidence']:.2f}
- **Method**: {evidence['method']}
"""
        
        formatted_report += f"""
## Diagnostic Support System Results
"""
        
        for classifier, outputs in classifier_outputs.items():
            formatted_report += f"""
### {classifier.replace('_', ' ').title()}
"""
            sorted_outputs = sorted(outputs.items(), key=lambda x: x[1], reverse=True)[:5]
            for class_name, prob in sorted_outputs:
                formatted_report += f"- **{class_name}**: {prob:.3f}\n"
        
        # Add system metadata
        formatted_report += f"""
## Technical Metadata
- **Knowledge Graph Domain**: {kg.domain}
- **Total Nodes Evaluated**: {len(metadata.get('decision_path', []))}
- **Execution Start**: {metadata.get('start_time', 'Unknown')}
- **Execution End**: {metadata.get('end_time', 'Unknown')}
- **Final Diagnosis**: {metadata.get('final_diagnosis', 'Not specified')}

---
*Generated by CoT-RAG Stage 3 Reasoning Executor*
"""
        
        return formatted_report
    
    def _generate_fallback_narrative(self, decision_path: List[str],
                                   metadata: Dict[str, Any],
                                   kg: KnowledgeGraph) -> str:
        """Generate fallback narrative when LLM generation fails."""
        evidence_chain = metadata.get('evidence_chain', [])
        
        report = f"""# Diagnostic Assessment Report

## Summary
Diagnostic reasoning completed through {len(decision_path)} decision points in the {kg.domain} domain.

## Reasoning Chain
"""
        
        for evidence in evidence_chain:
            report += f"- **{evidence['question']}**: {evidence['decision']} (confidence: {evidence['confidence']:.2f})\n"
        
        report += f"""
## Final Assessment
{metadata.get('final_diagnosis', 'Assessment completed based on available evidence.')}

## Technical Notes
- Decision Path: {' → '.join(decision_path)}
- Average Confidence: {sum(e['confidence'] for e in evidence_chain) / max(len(evidence_chain), 1):.2f}

*Note: Detailed narrative generation unavailable. Technical assessment provided.*
"""
        
        return report
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics."""
        return {
            'total_executions': self.execution_stats['total_executions'],
            'successful_executions': self.execution_stats['successful_executions'],
            'failed_executions': self.execution_stats['failed_executions'],
            'success_rate': (
                self.execution_stats['successful_executions'] / 
                max(self.execution_stats['total_executions'], 1)
            ) * 100,
            'average_path_length': self.execution_stats['average_path_length'],
            'average_confidence': self.execution_stats['average_confidence'],
            'decision_method_distribution': self.execution_stats['decision_method_counts'],
            'available_classifiers': list(self.classifiers.keys())
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()

# Convenience functions
def execute_cot_rag_pipeline(kg_template: KnowledgeGraph,
                           patient_data: Dict[str, Any],
                           llm_interface: LLMInterface,
                           classifiers: Dict[str, BaseClassifier] = None) -> Dict[str, Any]:
    """
    Execute complete CoT-RAG pipeline from populated KG to final diagnosis.
    
    Args:
        kg_template: Knowledge graph template (can be from Stage 1)
        patient_data: Patient clinical data
        llm_interface: LLM interface for reasoning
        classifiers: Optional classifiers for decision support
        
    Returns:
        Complete results including diagnosis, path, and narrative
    """
    from .stage2_rag import RAGPopulator
    
    # Stage 2: Populate knowledge graph if needed
    if not any(node.sub_description for node in kg_template.nodes.values()):
        print("Knowledge graph appears unpopulated, running Stage 2 RAG population...")
        rag_populator = RAGPopulator(llm_interface)
        from utils.patient_data_loader import create_patient_data
        
        if isinstance(patient_data, dict) and 'patient_id' not in patient_data:
            patient_data['patient_id'] = 'pipeline_patient'
        
        patient_obj = create_patient_data(**patient_data) if isinstance(patient_data, dict) else patient_data
        populated_kg = rag_populator.populate_knowledge_graph(kg_template, patient_obj)
    else:
        populated_kg = kg_template
    
    # Stage 3: Execute reasoning
    executor = ReasoningExecutor(llm_interface, classifiers)
    
    final_diagnosis, decision_path, metadata = executor.execute_reasoning_path(
        populated_kg, patient_data
    )
    
    # Generate narrative report
    narrative_report = executor.generate_narrative_report(
        decision_path, populated_kg, metadata
    )
    
    return {
        'final_diagnosis': final_diagnosis,
        'decision_path': decision_path,
        'narrative_report': narrative_report,
        'execution_metadata': metadata,
        'populated_knowledge_graph': populated_kg,
        'execution_statistics': executor.get_execution_statistics()
    }