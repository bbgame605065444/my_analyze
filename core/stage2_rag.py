"""
Stage 2: RAG Population Engine
Populates knowledge graphs with patient-specific information using retrieval-augmented generation.

This implements the core CoT-RAG Stage 2 methodology: taking the expert-generated knowledge
graphs from Stage 1 and populating them with patient-specific clinical information.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .knowledge_graph import KnowledgeGraph, KGNode
from utils.llm_interface import LLMInterface
from utils.prompt_templates import PromptTemplates
from utils.medical_ontology import get_medical_ontology_mapper, MedicalStandard

@dataclass
class PatientData:
    """Container for patient clinical data."""
    patient_id: str
    clinical_text: str = ""
    ecg_data: Any = None
    demographics: Dict[str, Any] = None
    query_description: str = ""
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.demographics is None:
            self.demographics = {}
        if self.additional_data is None:
            self.additional_data = {}
    
    def get_combined_text(self) -> str:
        """Get all textual information combined."""
        text_parts = []
        
        if self.clinical_text:
            text_parts.append(f"Clinical Notes:\n{self.clinical_text}")
        
        if self.query_description:
            text_parts.append(f"Current Query:\n{self.query_description}")
        
        if self.demographics:
            demo_text = ", ".join([f"{k}: {v}" for k, v in self.demographics.items()])
            text_parts.append(f"Demographics: {demo_text}")
        
        return "\n\n".join(text_parts)

@dataclass 
class RAGResult:
    """Result of RAG information extraction."""
    extracted_text: str
    evidence_sources: List[str]
    confidence: float
    clinical_relevance: str
    method: str = "llm_rag"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'extracted_text': self.extracted_text,
            'evidence_sources': self.evidence_sources,
            'confidence': self.confidence,
            'clinical_relevance': self.clinical_relevance,
            'method': self.method
        }

class RAGPopulator:
    """
    Stage 2: Populates knowledge graphs with patient-specific information
    using retrieval-augmented generation.
    
    This class implements the core CoT-RAG Stage 2 functionality:
    1. Takes knowledge graph templates from Stage 1
    2. Extracts patient-specific information for each node
    3. Populates nodes with relevant clinical data
    4. Updates knowledge base with novel patterns
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize RAG population engine.
        
        Args:
            llm_interface: LLM interface for information extraction
        """
        self.llm = llm_interface
        self.prompt_templates = PromptTemplates()
        self.medical_mapper = get_medical_ontology_mapper()
        
        # Track RAG statistics
        self.rag_stats = {
            'nodes_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'novel_patterns_detected': 0,
            'llm_calls': 0,
            'total_confidence': 0.0
        }
    
    def populate_knowledge_graph(self, kg_template: KnowledgeGraph, 
                                patient_data: PatientData,
                                save_intermediate: bool = True) -> KnowledgeGraph:
        """
        Populate knowledge graph template with patient-specific information.
        
        Args:
            kg_template: Template knowledge graph from Stage 1
            patient_data: Patient clinical data
            save_intermediate: Whether to save intermediate results
            
        Returns:
            KnowledgeGraph: Populated knowledge graph
        """
        print(f"Starting RAG population for patient: {patient_data.patient_id}")
        print(f"Knowledge graph domain: {kg_template.domain}")
        print(f"Nodes to populate: {len(kg_template.nodes)}")
        
        # Create a deep copy of the template
        populated_kg = self._deep_copy_kg(kg_template)
        
        # Update metadata
        populated_kg.metadata.update({
            'patient_id': patient_data.patient_id,
            'populated_by': 'CoT-RAG Stage 2 RAG Populator',
            'population_timestamp': self._get_timestamp(),
            'source_template': kg_template.metadata.get('source_file', 'unknown')
        })
        
        # Get combined patient context
        patient_context = patient_data.get_combined_text()
        print(f"Patient context length: {len(patient_context)} characters")
        
        # Populate each node with relevant information
        print(f"\nPopulating nodes with patient-specific information...")
        
        nodes_with_questions = [
            (node_id, node) for node_id, node in populated_kg.nodes.items() 
            if node.sub_question and node.sub_question.strip()
        ]
        
        for i, (node_id, node) in enumerate(nodes_with_questions):
            print(f"Processing node {i+1}/{len(nodes_with_questions)}: {node_id}")
            
            try:
                rag_result = self._populate_node(node, patient_context, patient_data)
                
                # Update node with RAG results
                node.sub_description = rag_result.extracted_text
                node.evidence_sources = rag_result.evidence_sources
                
                # Store RAG metadata
                if not hasattr(node, 'execution_metadata'):
                    node.execution_metadata = {}
                node.execution_metadata['rag_result'] = rag_result.to_dict()
                
                self.rag_stats['successful_extractions'] += 1
                self.rag_stats['total_confidence'] += rag_result.confidence
                
                print(f"  ✓ Extracted: {rag_result.extracted_text[:80]}...")
                print(f"  ✓ Confidence: {rag_result.confidence:.2f}")
                
            except Exception as e:
                print(f"  ✗ Failed to populate node {node_id}: {e}")
                self.rag_stats['failed_extractions'] += 1
                continue
            
            self.rag_stats['nodes_processed'] += 1
            
            # Save intermediate results periodically
            if save_intermediate and i % 5 == 0:
                self._save_intermediate_population(populated_kg, patient_data.patient_id, i)
        
        # Check for novel patterns and update knowledge base
        print(f"\nChecking for novel clinical patterns...")
        novel_patterns = self._detect_novel_patterns(populated_kg, patient_context)
        if novel_patterns:
            self._incorporate_novel_patterns(populated_kg, novel_patterns)
        
        # Validate populated knowledge graph
        print(f"\nValidating populated knowledge graph...")
        validation_errors = populated_kg.validate_structure()
        if validation_errors:
            print(f"Warning: Populated KG has validation issues:")
            for error in validation_errors:
                print(f"  - {error}")
        else:
            print("✓ Populated knowledge graph structure is valid")
        
        # Print population summary
        self._print_population_summary(populated_kg, patient_data)
        
        return populated_kg
    
    def _populate_node(self, node: KGNode, patient_context: str, 
                      patient_data: PatientData) -> RAGResult:
        """
        Populate a single node with patient-specific information.
        
        Args:
            node: Knowledge graph node to populate
            patient_context: Combined patient text context
            patient_data: Complete patient data
            
        Returns:
            RAGResult: Extraction results
        """
        # Create RAG extraction prompt
        rag_prompt = self.prompt_templates.get_rag_prompt(
            sub_question=node.sub_question,
            sub_case=node.sub_case,
            patient_context=patient_context,
            diagnosis_class=node.diagnosis_class
        )
        
        # Query LLM for information extraction
        try:
            print(f"    Calling LLM for information extraction...")
            response = self.llm.query(
                prompt=rag_prompt,
                model_name="rag_extraction"
            )
            self.rag_stats['llm_calls'] += 1
            
            # Parse RAG response
            rag_result = self._parse_rag_response(response, node)
            
            # Validate extracted information against medical ontology
            validated_result = self._validate_clinical_information(rag_result, node)
            
            return validated_result
            
        except Exception as e:
            raise Exception(f"RAG extraction failed for node {node.node_id}: {str(e)}")
    
    def _parse_rag_response(self, response: str, node: KGNode) -> RAGResult:
        """
        Parse LLM response from RAG query.
        
        Expected format includes extracted text, evidence sources, and confidence.
        """
        # Initialize default result
        result = RAGResult(
            extracted_text="",
            evidence_sources=[],
            confidence=0.0,
            clinical_relevance=""
        )
        
        try:
            # Look for structured response format
            extracted_text = self._extract_section(response, "EXTRACTED_TEXT")
            evidence = self._extract_section(response, "EVIDENCE")
            confidence_str = self._extract_section(response, "CONFIDENCE")
            relevance = self._extract_section(response, "CLINICAL_RELEVANCE")
            
            # Process extracted text
            if extracted_text:
                result.extracted_text = extracted_text.strip()
            else:
                # Fallback: use entire response if no structured format
                result.extracted_text = response.strip()
            
            # Process evidence sources
            if evidence:
                result.evidence_sources = [
                    e.strip() for e in evidence.split('\n') 
                    if e.strip() and not e.strip().startswith('-')
                ]
            
            # Process confidence
            if confidence_str:
                try:
                    result.confidence = float(re.search(r'([0-9.]+)', confidence_str).group(1))
                    result.confidence = max(0.0, min(1.0, result.confidence))  # Clamp to [0,1]
                except (ValueError, AttributeError):
                    result.confidence = 0.5  # Default confidence
            else:
                result.confidence = 0.5
            
            # Process clinical relevance
            if relevance:
                result.clinical_relevance = relevance.strip()
            else:
                result.clinical_relevance = f"Information relevant to {node.diagnosis_class}"
            
            # Quality check: if no meaningful text extracted, mark as low confidence
            if not result.extracted_text or result.extracted_text.lower() in ['no relevant information found', 'not found', 'none']:
                result.confidence = 0.1
                if not result.extracted_text:
                    result.extracted_text = "No relevant information found in patient data"
            
        except Exception as e:
            print(f"    Warning: Error parsing RAG response: {e}")
            # Fallback to basic extraction
            result.extracted_text = response.strip()[:200] + "..." if len(response) > 200 else response.strip()
            result.confidence = 0.3  # Low confidence for unparseable responses
            result.clinical_relevance = "Automatic extraction"
        
        return result
    
    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a specific section from structured LLM response."""
        pattern = rf"{section_name}:\s*(.+?)(?:[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _validate_clinical_information(self, rag_result: RAGResult, 
                                     node: KGNode) -> RAGResult:
        """
        Validate extracted clinical information against medical knowledge.
        
        Args:
            rag_result: Initial RAG extraction result
            node: Knowledge graph node context
            
        Returns:
            Validated and potentially enhanced RAG result
        """
        # Check for medical term consistency
        extracted_text = rag_result.extracted_text.lower()
        
        # Look for medical terms related to the node's diagnosis class
        diagnosis_terms = self.medical_mapper.map_term_to_codes(node.diagnosis_class)
        
        # Check if extracted information contains related medical concepts
        contains_relevant_terms = False
        for term_code in diagnosis_terms:
            term_description = term_code.description.lower()
            if any(word in extracted_text for word in term_description.split() if len(word) > 3):
                contains_relevant_terms = True
                break
        
        # Adjust confidence based on medical relevance
        if contains_relevant_terms:
            rag_result.confidence = min(1.0, rag_result.confidence + 0.1)
        elif rag_result.confidence > 0.5:
            rag_result.confidence = max(0.3, rag_result.confidence - 0.2)
        
        # Add clinical validation metadata
        rag_result.clinical_relevance += f" (Medical validation: {'relevant' if contains_relevant_terms else 'uncertain'})"
        
        return rag_result
    
    def _detect_novel_patterns(self, populated_kg: KnowledgeGraph, 
                             patient_context: str) -> List[Dict[str, Any]]:
        """
        Detect novel or unusual clinical patterns in patient data.
        
        Args:
            populated_kg: Populated knowledge graph
            patient_context: Patient clinical context
            
        Returns:
            List of detected novel patterns
        """
        novel_patterns = []
        
        try:
            # Create prompt for novel pattern detection
            novel_pattern_prompt = self.prompt_templates.get_novel_info_prompt(
                patient_context=patient_context,
                domain=populated_kg.domain
            )
            
            print(f"    Analyzing for novel clinical patterns...")
            response = self.llm.query(
                prompt=novel_pattern_prompt,
                model_name="pattern_detection"
            )
            
            # Parse response for novel information indicators
            if self._contains_novel_information(response):
                pattern = {
                    'analysis': response,
                    'context_snippet': patient_context[:300] + "...",
                    'detected_timestamp': self._get_timestamp(),
                    'confidence': self._extract_pattern_confidence(response)
                }
                novel_patterns.append(pattern)
                self.rag_stats['novel_patterns_detected'] += 1
                print(f"    ✓ Novel pattern detected (confidence: {pattern['confidence']:.2f})")
        
        except Exception as e:
            print(f"    Warning: Novel pattern detection failed: {e}")
        
        return novel_patterns
    
    def _contains_novel_information(self, response: str) -> bool:
        """Check if the response indicates novel clinical information."""
        novel_indicators = [
            'novel', 'unusual', 'rare', 'atypical', 'uncommon',
            'unexpected', 'unique', 'abnormal pattern', 'interesting finding'
        ]
        response_lower = response.lower()
        
        # Check for explicit novel information indication
        if 'novel_information: yes' in response_lower:
            return True
        
        # Check for novel pattern keywords
        return any(indicator in response_lower for indicator in novel_indicators)
    
    def _extract_pattern_confidence(self, response: str) -> float:
        """Extract confidence score for novel pattern detection."""
        # Look for confidence indicators in response
        confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', response, re.IGNORECASE)
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except ValueError:
                pass
        
        # Default confidence based on strength of language
        strong_indicators = ['definitely', 'clearly', 'obviously', 'certainly']
        moderate_indicators = ['likely', 'probably', 'appears', 'seems']
        
        response_lower = response.lower()
        
        if any(indicator in response_lower for indicator in strong_indicators):
            return 0.8
        elif any(indicator in response_lower for indicator in moderate_indicators):
            return 0.6
        else:
            return 0.4
    
    def _incorporate_novel_patterns(self, kg: KnowledgeGraph, 
                                  patterns: List[Dict[str, Any]]) -> None:
        """
        Incorporate novel patterns into knowledge base for future use.
        
        Args:
            kg: Knowledge graph to update
            patterns: List of novel patterns detected
        """
        if 'novel_patterns' not in kg.metadata:
            kg.metadata['novel_patterns'] = []
        
        for pattern in patterns:
            kg.metadata['novel_patterns'].append(pattern)
            print(f"    ✓ Incorporated novel pattern: {pattern['analysis'][:100]}...")
    
    def _save_intermediate_population(self, kg: KnowledgeGraph, 
                                    patient_id: str, progress: int) -> None:
        """Save intermediate RAG population results."""
        try:
            output_dir = Path('output/rag_population')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            intermediate_file = output_dir / f"intermediate_{patient_id}_{progress}.json"
            kg.save_to_json(intermediate_file)
            print(f"    Saved intermediate results to: {intermediate_file}")
            
        except Exception as e:
            print(f"    Warning: Failed to save intermediate results: {e}")
    
    def _print_population_summary(self, kg: KnowledgeGraph, 
                                patient_data: PatientData) -> None:
        """Print summary of the RAG population process."""
        print(f"\n{'='*60}")
        print("RAG POPULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Patient ID: {patient_data.patient_id}")
        print(f"Domain: {kg.domain}")
        print(f"Total nodes: {len(kg.nodes)}")
        
        # Calculate populated nodes
        populated_nodes = sum(1 for node in kg.nodes.values() if node.sub_description)
        print(f"Populated nodes: {populated_nodes}")
        
        print(f"\nRAG Statistics:")
        print(f"  - Nodes processed: {self.rag_stats['nodes_processed']}")
        print(f"  - Successful extractions: {self.rag_stats['successful_extractions']}")
        print(f"  - Failed extractions: {self.rag_stats['failed_extractions']}")
        print(f"  - LLM calls made: {self.rag_stats['llm_calls']}")
        print(f"  - Novel patterns detected: {self.rag_stats['novel_patterns_detected']}")
        
        if self.rag_stats['successful_extractions'] > 0:
            avg_confidence = self.rag_stats['total_confidence'] / self.rag_stats['successful_extractions']
            print(f"  - Average confidence: {avg_confidence:.2f}")
        
        print(f"{'='*60}")
    
    def _deep_copy_kg(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Create a deep copy of the knowledge graph."""
        import copy
        return copy.deepcopy(kg)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get RAG population statistics."""
        stats = self.rag_stats.copy()
        if stats['successful_extractions'] > 0:
            stats['average_confidence'] = stats['total_confidence'] / stats['successful_extractions']
            stats['success_rate'] = stats['successful_extractions'] / (stats['successful_extractions'] + stats['failed_extractions'])
        else:
            stats['average_confidence'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats

# Utility functions
def create_patient_data(patient_id: str, clinical_text: str = "", 
                       query_description: str = "", **kwargs) -> PatientData:
    """
    Convenience function to create PatientData object.
    
    Args:
        patient_id: Unique patient identifier
        clinical_text: Clinical notes and observations
        query_description: Specific query or reason for analysis
        **kwargs: Additional data (demographics, etc.)
        
    Returns:
        PatientData: Structured patient data object
    """
    return PatientData(
        patient_id=patient_id,
        clinical_text=clinical_text,
        query_description=query_description,
        demographics=kwargs.get('demographics', {}),
        additional_data=kwargs
    )

def populate_kg_with_patient(kg_template: KnowledgeGraph, 
                           patient_data: PatientData,
                           llm_interface: LLMInterface = None) -> KnowledgeGraph:
    """
    Convenience function for knowledge graph population.
    
    Args:
        kg_template: Template knowledge graph from Stage 1
        patient_data: Patient clinical data
        llm_interface: LLM interface (creates new if None)
        
    Returns:
        Populated knowledge graph
    """
    if llm_interface is None:
        from utils.llm_interface import get_global_llm_interface
        llm_interface = get_global_llm_interface()
    
    populator = RAGPopulator(llm_interface)
    return populator.populate_knowledge_graph(kg_template, patient_data)