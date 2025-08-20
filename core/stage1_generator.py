"""
Stage 1: Knowledge Graph Generator
Converts expert-defined decision trees into fine-grained knowledge graphs using LLM decomposition.

This implements the core CoT-RAG Stage 1 methodology, adapted to work with the existing 
model_interface.py and prompt engineering infrastructure.
"""

import yaml
import json
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Import existing infrastructure
from model_interface import get_model_response
from model_config import ModelConfig
from utils.prompt_templates import PromptTemplates
from .knowledge_graph import KnowledgeGraph, KGNode, NodeType

class ExpertDecisionTree:
    """
    Represents an expert-defined coarse-grained decision tree.
    Loads and validates decision trees from YAML files.
    """
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Decision tree file not found: {filepath}")
        
        self.tree_data = self._load_and_validate_tree()
    
    def _load_and_validate_tree(self) -> Dict[str, Any]:
        """Load and validate decision tree from YAML file."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['domain', 'nodes']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Decision tree missing required fields: {missing_fields}")
            
            # Validate nodes structure
            if not isinstance(data['nodes'], list) or not data['nodes']:
                raise ValueError("Decision tree must have a non-empty 'nodes' list")
            
            # Validate each node
            for i, node in enumerate(data['nodes']):
                self._validate_node(node, i, data['nodes'])
            
            return data
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in decision tree file: {e}")
    
    def _validate_node(self, node: Dict[str, Any], index: int, all_nodes: List[Dict[str, Any]]) -> None:
        """Validate a single decision tree node."""
        required_node_fields = ['node_id', 'question']
        missing_fields = [field for field in required_node_fields if field not in node]
        if missing_fields:
            raise ValueError(f"Node {index} missing required fields: {missing_fields}")
        
        # Check for unique node_ids
        node_ids = [n['node_id'] for n in all_nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Duplicate node_ids found in decision tree")
    
    @property
    def domain(self) -> str:
        """Get the medical domain of this decision tree."""
        return self.tree_data.get('domain', 'general')
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the decision tree."""
        return {
            'domain': self.domain,
            'expert': self.tree_data.get('expert', 'Unknown'),
            'version': self.tree_data.get('version', '1.0'),
            'description': self.tree_data.get('description', ''),
            'created_date': self.tree_data.get('created_date', ''),
            'node_count': len(self.nodes)
        }
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the decision tree."""
        return self.tree_data.get('nodes', [])
    
    def get_root_nodes(self) -> List[Dict[str, Any]]:
        """Get all root nodes (nodes with is_root=True or no parent)."""
        return [node for node in self.nodes 
                if node.get('is_root', False) or not node.get('parent')]
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific node by its ID."""
        for node in self.nodes:
            if node['node_id'] == node_id:
                return node
        return None
    
    def get_children(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all child nodes of a given node."""
        return [node for node in self.nodes if node.get('parent') == node_id]

class KGGenerator:
    """
    Stage 1: Converts expert-defined decision trees into fine-grained knowledge graphs.
    
    This class implements the core CoT-RAG Stage 1 methodology:
    1. Load expert decision tree
    2. Use LLM to decompose each node into fine-grained entities
    3. Create knowledge graph with proper relationships
    4. Validate the resulting structure
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the KG Generator.
        
        Args:
            config: Model configuration (uses default if None)
        """
        self.config = config
        self.prompt_templates = PromptTemplates()
        
        # Track generation statistics
        self.generation_stats = {
            'nodes_processed': 0,
            'entities_created': 0,
            'llm_calls': 0,
            'errors': []
        }
    
    def generate_from_decision_tree(self, dt_filepath: Union[str, Path], 
                                  output_path: Optional[Union[str, Path]] = None,
                                  save_intermediate: bool = True) -> KnowledgeGraph:
        """
        Generate a fine-grained knowledge graph from an expert decision tree.
        
        Args:
            dt_filepath: Path to the expert decision tree YAML file
            output_path: Optional path to save the generated KG
            save_intermediate: Whether to save intermediate results
            
        Returns:
            KnowledgeGraph: Generated knowledge graph
            
        Raises:
            FileNotFoundError: If decision tree file doesn't exist
            ValueError: If decision tree is invalid or generation fails
        """
        print(f"Starting KG generation from decision tree: {dt_filepath}")
        
        # Load and validate expert decision tree
        expert_dt = ExpertDecisionTree(dt_filepath)
        print(f"Loaded decision tree for domain: {expert_dt.domain}")
        print(f"Expert: {expert_dt.metadata['expert']}")
        print(f"Nodes to process: {len(expert_dt.nodes)}")
        
        # Initialize knowledge graph
        kg = KnowledgeGraph(domain=expert_dt.domain)
        kg.metadata.update(expert_dt.metadata)
        kg.metadata['source_file'] = str(dt_filepath)
        kg.metadata['generated_by'] = 'CoT-RAG Stage 1 Generator'
        
        # Process each node in the decision tree
        print("\\nProcessing decision tree nodes...")
        
        # First pass: Process all nodes and create entities
        node_to_entities = {}
        for i, dt_node in enumerate(expert_dt.nodes):
            print(f"Processing node {i+1}/{len(expert_dt.nodes)}: {dt_node['node_id']}")
            
            try:
                entities = self._process_dt_node(dt_node, expert_dt)
                node_to_entities[dt_node['node_id']] = entities
                
                # Add entities to knowledge graph
                for entity in entities:
                    kg_node = self._create_kg_node(entity, dt_node, expert_dt)
                    kg.add_node(kg_node)
                    self.generation_stats['entities_created'] += 1
                
                self.generation_stats['nodes_processed'] += 1
                
                if save_intermediate and i % 5 == 0:  # Save every 5 nodes
                    self._save_intermediate_results(kg, dt_filepath, i)
                    
            except Exception as e:
                error_msg = f"Failed to process node {dt_node['node_id']}: {str(e)}"
                print(f"Warning: {error_msg}")
                self.generation_stats['errors'].append(error_msg)
                continue
        
        # Second pass: Establish relationships between entities
        print("\\nEstablishing entity relationships...")
        self._establish_relationships(kg, node_to_entities, expert_dt)
        
        # Validate the generated knowledge graph
        print("\\nValidating knowledge graph structure...")
        validation_errors = kg.validate_structure()
        if validation_errors:
            print(f"Warning: Knowledge graph has validation issues:")
            for error in validation_errors:
                print(f"  - {error}")
            self.generation_stats['errors'].extend(validation_errors)
        else:
            print("✓ Knowledge graph structure is valid")
        
        # Print generation statistics
        self._print_generation_summary(kg)
        
        # Save the final knowledge graph
        if output_path:
            kg.save_to_json(output_path)
            print(f"\\nKnowledge graph saved to: {output_path}")
        
        return kg
    
    def _process_dt_node(self, dt_node: Dict[str, Any], 
                        expert_dt: ExpertDecisionTree) -> List[Dict[str, Any]]:
        """
        Process a single decision tree node using LLM decomposition.
        
        This is the core of CoT-RAG Stage 1: using LLM to break down
        coarse-grained expert knowledge into fine-grained entities.
        """
        node_id = dt_node['node_id']
        question = dt_node['question']
        knowledge_case = dt_node.get('knowledge_case', dt_node.get('example', ''))
        context = dt_node.get('context', f"Clinical decision making in {expert_dt.domain}")
        
        # Create decomposition prompt
        decomposition_prompt = self.prompt_templates.get_decomposition_prompt(
            question=question,
            knowledge_case=knowledge_case,
            domain=expert_dt.domain,
            context=context
        )
        
        # Call LLM for decomposition
        try:
            print(f"  Calling LLM for node decomposition...")
            response = get_model_response(decomposition_prompt, self.config)
            self.generation_stats['llm_calls'] += 1
            
            # Parse LLM response
            entities = self._parse_decomposition_response(response, node_id)
            print(f"  Generated {len(entities)} entities from node {node_id}")
            
            return entities
            
        except Exception as e:
            raise ValueError(f"LLM decomposition failed for node {node_id}: {str(e)}")
    
    def _parse_decomposition_response(self, response: str, node_id: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract entity information.
        
        Expected format: JSON with entities list containing entity definitions.
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\\s*({.*?})\\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'{.*"entities".*}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON structure found in LLM response")
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            
            # Extract entities
            entities = parsed_data.get('entities', [])
            if not entities:
                raise ValueError("No entities found in parsed response")
            
            # Validate and enhance entities
            validated_entities = []
            for i, entity in enumerate(entities):
                # Ensure required fields
                if 'sub_question' not in entity:
                    entity['sub_question'] = f"Sub-question {i+1} for {node_id}"
                
                if 'entity_id' not in entity:
                    entity['entity_id'] = f"{node_id}_entity_{i}"
                
                # Set defaults for optional fields
                entity.setdefault('sub_case', '')
                entity.setdefault('diagnosis_class', '')
                entity.setdefault('required_inputs', [])
                entity.setdefault('clinical_significance', '')
                entity.setdefault('dependencies', [])
                entity.setdefault('is_terminal', False)
                
                validated_entities.append(entity)
            
            return validated_entities
            
        except json.JSONDecodeError as e:
            # Fallback: create a single entity from the original response
            print(f"Warning: Failed to parse JSON from LLM response: {e}")
            print("Creating fallback entity...")
            
            return [{
                'entity_id': f"{node_id}_entity_0",
                'sub_question': f"Evaluate condition for {node_id}",
                'sub_case': response[:200] + "..." if len(response) > 200 else response,
                'diagnosis_class': node_id.replace('_', ' ').title(),
                'required_inputs': [],
                'clinical_significance': 'Generated from fallback parsing',
                'dependencies': [],
                'is_terminal': False
            }]
        
        except Exception as e:
            raise ValueError(f"Failed to parse decomposition response: {str(e)}")
    
    def _create_kg_node(self, entity: Dict[str, Any], dt_node: Dict[str, Any],
                       expert_dt: ExpertDecisionTree) -> KGNode:
        """
        Create a KGNode from an entity definition.
        """
        # Determine node type
        if dt_node.get('is_root', False):
            node_type = NodeType.ROOT
        elif entity.get('is_terminal', False) or not expert_dt.get_children(dt_node['node_id']):
            node_type = NodeType.LEAF
        else:
            node_type = NodeType.INTERNAL
        
        # Create the node
        kg_node = KGNode(
            node_id=entity['entity_id'],
            diagnosis_class=entity.get('diagnosis_class', ''),
            node_type=node_type,
            sub_question=entity.get('sub_question', ''),
            sub_case=entity.get('sub_case', ''),
            clinical_significance=entity.get('clinical_significance', ''),
            required_classifier_inputs=entity.get('required_inputs', []),
            execution_metadata={
                'source_dt_node': dt_node['node_id'],
                'entity_index': entity.get('entity_id', '').split('_')[-1],
                'dependencies': entity.get('dependencies', [])
            }
        )
        
        return kg_node
    
    def _establish_relationships(self, kg: KnowledgeGraph, 
                               node_to_entities: Dict[str, List[Dict[str, Any]]],
                               expert_dt: ExpertDecisionTree) -> None:
        """
        Establish parent-child relationships in the knowledge graph.
        """
        # Build mapping from entity IDs to parent DT nodes
        entity_to_dt_node = {}
        for dt_node_id, entities in node_to_entities.items():
            for entity in entities:
                entity_to_dt_node[entity['entity_id']] = dt_node_id
        
        # Establish relationships based on decision tree structure
        for dt_node in expert_dt.nodes:
            dt_node_id = dt_node['node_id']
            parent_dt_id = dt_node.get('parent')
            
            if parent_dt_id and parent_dt_id in node_to_entities:
                # Connect entities from parent DT node to entities in current DT node
                parent_entities = node_to_entities[parent_dt_id]
                current_entities = node_to_entities.get(dt_node_id, [])
                
                # Simple strategy: connect last parent entity to first current entity
                if parent_entities and current_entities:
                    parent_entity_id = parent_entities[-1]['entity_id']
                    current_entity_id = current_entities[0]['entity_id']
                    
                    parent_node = kg.get_node(parent_entity_id)
                    current_node = kg.get_node(current_entity_id)
                    
                    if parent_node and current_node:
                        parent_node.add_child(current_entity_id)
                        current_node.parent_node = parent_entity_id
        
        # Establish intra-node relationships based on dependencies
        for entities in node_to_entities.values():
            self._establish_intra_node_relationships(kg, entities)
    
    def _establish_intra_node_relationships(self, kg: KnowledgeGraph, 
                                          entities: List[Dict[str, Any]]) -> None:
        """
        Establish relationships between entities within the same DT node.
        """
        for entity in entities:
            entity_id = entity['entity_id']
            dependencies = entity.get('dependencies', [])
            
            current_node = kg.get_node(entity_id)
            if not current_node:
                continue
            
            # Connect to dependent entities
            for dep_index in dependencies:
                if 0 <= dep_index < len(entities):
                    dep_entity_id = entities[dep_index]['entity_id']
                    dep_node = kg.get_node(dep_entity_id)
                    
                    if dep_node:
                        dep_node.add_child(entity_id)
                        current_node.parent_node = dep_entity_id
    
    def _save_intermediate_results(self, kg: KnowledgeGraph, 
                                 source_file: Path, progress: int) -> None:
        """Save intermediate results during generation."""
        try:
            output_dir = Path('output/kg_generation')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            intermediate_file = output_dir / f"intermediate_{source_file.stem}_{progress}.json"
            kg.save_to_json(intermediate_file)
            print(f"  Saved intermediate results to: {intermediate_file}")
            
        except Exception as e:
            print(f"  Warning: Failed to save intermediate results: {e}")
    
    def _print_generation_summary(self, kg: KnowledgeGraph) -> None:
        """Print a summary of the generation process."""
        stats = kg.get_statistics()
        
        print(f"\\n{'='*60}")
        print("KNOWLEDGE GRAPH GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Domain: {kg.domain}")
        print(f"Total nodes created: {stats['total_nodes']}")
        print(f"Node types: {stats['node_types']}")
        print(f"Maximum depth: {stats['max_depth']}")
        print(f"Leaf nodes: {stats['leaf_nodes']}")
        print(f"\\nGeneration Statistics:")
        print(f"  - DT nodes processed: {self.generation_stats['nodes_processed']}")
        print(f"  - Entities created: {self.generation_stats['entities_created']}")
        print(f"  - LLM calls made: {self.generation_stats['llm_calls']}")
        print(f"  - Errors encountered: {len(self.generation_stats['errors'])}")
        
        if self.generation_stats['errors']:
            print(f"\\nErrors:")
            for error in self.generation_stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.generation_stats['errors']) > 5:
                print(f"  ... and {len(self.generation_stats['errors']) - 5} more")
        
        print(f"{'='*60}")

def create_sample_decision_tree(output_path: Union[str, Path]) -> None:
    """
    Create a sample expert decision tree for testing and demonstration.
    """
    sample_dt = {
        'domain': 'cardiology',
        'expert': 'Dr. Sample Cardiologist',
        'version': '1.0',
        'description': 'Basic cardiology decision tree for ECG diagnosis demonstration',
        'created_date': '2025-01-01',
        'nodes': [
            {
                'node_id': 'rhythm_assessment',
                'question': 'What is the fundamental rhythm pattern of this ECG?',
                'knowledge_case': 'Rhythm assessment involves evaluating R-R interval regularity, P-wave presence and morphology, and relationship between P-waves and QRS complexes. Regular rhythm shows consistent R-R intervals, while irregular rhythm shows varying intervals.',
                'context': 'Initial rhythm classification is the foundation of ECG interpretation',
                'is_root': True,
                'clinical_significance': 'Rhythm disturbances can indicate serious cardiac conditions requiring immediate intervention'
            },
            {
                'node_id': 'rate_evaluation',
                'question': 'What is the heart rate category and are there rate-related abnormalities?',
                'knowledge_case': 'Normal adult heart rate: 60-100 bpm. Bradycardia: <60 bpm (may indicate conduction blocks, medications, or athletic conditioning). Tachycardia: >100 bpm (may indicate fever, anxiety, heart failure, or arrhythmias).',
                'context': 'Rate assessment helps differentiate normal from pathological conditions',
                'parent': 'rhythm_assessment',
                'clinical_significance': 'Extreme heart rates can compromise cardiac output and require intervention'
            },
            {
                'node_id': 'morphology_analysis',
                'question': 'Are there morphological abnormalities in the ECG waveforms?',
                'knowledge_case': 'Morphology analysis includes P-wave shape and duration, QRS width and configuration, ST-segment elevation or depression, and T-wave inversions. Each component can indicate specific pathologies.',
                'context': 'Waveform morphology reveals structural and functional cardiac abnormalities',
                'parent': 'rhythm_assessment',
                'clinical_significance': 'Morphological changes can indicate ischemia, infarction, hypertrophy, or conduction defects'
            },
            {
                'node_id': 'conduction_assessment',
                'question': 'Are there conduction abnormalities or blocks present?',
                'knowledge_case': 'Conduction abnormalities include AV blocks (1st, 2nd, 3rd degree), bundle branch blocks (RBBB, LBBB), fascicular blocks, and intraventricular conduction delays. Each has distinct ECG patterns and clinical implications.',
                'context': 'Conduction system evaluation for electrical pathway abnormalities',
                'parent': 'rate_evaluation',
                'clinical_significance': 'Conduction blocks can progress to complete heart block requiring pacemaker intervention'
            },
            {
                'node_id': 'ischemia_evaluation',
                'question': 'Is there evidence of myocardial ischemia or infarction?',
                'knowledge_case': 'Ischemia signs include ST-depression, T-wave inversions, and poor R-wave progression. Acute MI shows ST-elevation in contiguous leads, Q-wave development, and reciprocal changes. Location is determined by lead patterns.',
                'context': 'Assessment for acute coronary syndromes requiring emergency intervention',
                'parent': 'morphology_analysis',
                'clinical_significance': 'Acute MI requires immediate reperfusion therapy to minimize myocardial damage',
                'is_terminal': True
            }
        ]
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_dt, f, default_flow_style=False, indent=2, allow_unicode=True)
    
    print(f"Sample decision tree created: {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample decision tree for testing
    sample_dt_path = Path("expert_knowledge/cardiology_sample.yaml")
    create_sample_decision_tree(sample_dt_path)
    
    # Test KG generation
    try:
        generator = KGGenerator()
        kg = generator.generate_from_decision_tree(
            sample_dt_path,
            output_path="output/sample_cardiology_kg.json"
        )
        
        print(f"\\nGenerated knowledge graph: {kg}")
        print(f"Validation errors: {kg.validate_structure()}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()