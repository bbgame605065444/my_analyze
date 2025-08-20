"""
Knowledge Graph Framework for CoT-RAG
Core data structures for representing diagnostic decision trees and reasoning chains.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
from enum import Enum
from pathlib import Path

class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    ROOT = "root"
    INTERNAL = "internal" 
    LEAF = "leaf"

@dataclass
class KGNode:
    """
    Core data structure for Knowledge Graph nodes.
    Represents a single decision point in the diagnostic workflow.
    
    Based on CoT-RAG methodology with adaptations for medical reasoning.
    """
    # Core identifiers
    node_id: str
    diagnosis_class: str
    node_type: NodeType
    
    # Hierarchical structure
    parent_node: Optional[str] = None
    child_nodes: List[str] = field(default_factory=list)
    
    # CoT-RAG specific attributes (from original paper)
    sub_question: str = ""          # Decomposed question from expert DT
    sub_case: str = ""              # Example case from expert knowledge
    sub_description: str = ""       # Populated by RAG in Stage 2
    answer: str = ""               # Generated during execution in Stage 3
    
    # Decision logic for classifier-driven execution
    decision_rule_logic: str = ""
    required_classifier_inputs: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    
    # Medical/Clinical metadata
    clinical_significance: str = ""
    evidence_sources: List[str] = field(default_factory=list)
    medical_codes: Dict[str, str] = field(default_factory=dict)  # ICD-10, SNOMED, etc.
    
    # Execution metadata (populated during reasoning)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
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
            'evidence_sources': self.evidence_sources,
            'medical_codes': self.medical_codes,
            'execution_metadata': self.execution_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KGNode':
        """Create node from dictionary (for JSON deserialization)."""
        # Convert node_type string back to enum
        if isinstance(data.get('node_type'), str):
            data['node_type'] = NodeType(data['node_type'])
        
        # Ensure all required fields have defaults
        defaults = {
            'parent_node': None,
            'child_nodes': [],
            'sub_question': "",
            'sub_case': "",
            'sub_description': "",
            'answer': "",
            'decision_rule_logic': "",
            'required_classifier_inputs': [],
            'confidence_threshold': 0.7,
            'clinical_significance': "",
            'evidence_sources': [],
            'medical_codes': {},
            'execution_metadata': {}
        }
        
        # Apply defaults for missing fields
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        return cls(**data)
    
    def add_child(self, child_node_id: str) -> None:
        """Add a child node ID to this node."""
        if child_node_id not in self.child_nodes:
            self.child_nodes.append(child_node_id)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.node_type == NodeType.LEAF or len(self.child_nodes) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.node_type == NodeType.ROOT or self.parent_node is None

class KnowledgeGraph:
    """
    Container for the complete diagnostic knowledge graph.
    Manages nodes, relationships, and provides traversal methods.
    
    Implements the knowledge graph structure from CoT-RAG Stage 1.
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.nodes: Dict[str, KGNode] = {}
        self.root_node_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {
            'created_by': 'CoT-RAG Stage 1 Generator',
            'version': '1.0',
            'domain': domain
        }
    
    def add_node(self, node: KGNode) -> None:
        """Add a node to the knowledge graph."""
        self.nodes[node.node_id] = node
        
        # Auto-set root node if this is the first root type node
        if node.is_root() and self.root_node_id is None:
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
    
    def get_parent(self, node_id: str) -> Optional[KGNode]:
        """Get the parent node of a given node."""
        node = self.get_node(node_id)
        if not node or not node.parent_node:
            return None
        return self.get_node(node.parent_node)
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get the path from a node to the root (root-to-node order)."""
        path = []
        current_id = node_id
        
        # Build path from node to root
        while current_id and current_id in self.nodes:
            path.append(current_id)
            current_node = self.nodes[current_id]
            current_id = current_node.parent_node
        
        return path[::-1]  # Reverse to get root-to-node path
    
    def get_leaf_nodes(self) -> List[KGNode]:
        """Get all leaf nodes in the graph."""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_depth(self, node_id: str) -> int:
        """Get the depth of a node from the root."""
        path = self.get_path_to_root(node_id)
        return len(path) - 1  # Root has depth 0
    
    def validate_structure(self) -> List[str]:
        """
        Validate the knowledge graph structure and return any errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check for root node
        if not self.root_node_id:
            errors.append("No root node defined")
        elif self.root_node_id not in self.nodes:
            errors.append(f"Root node {self.root_node_id} not found in nodes")
        
        # Check parent-child consistency
        for node_id, node in self.nodes.items():
            # Check parent references
            if node.parent_node:
                if node.parent_node not in self.nodes:
                    errors.append(f"Node {node_id} references non-existent parent {node.parent_node}")
                else:
                    parent = self.nodes[node.parent_node]
                    if node_id not in parent.child_nodes:
                        errors.append(f"Parent {node.parent_node} doesn't list {node_id} as child")
            
            # Check child references
            for child_id in node.child_nodes:
                if child_id not in self.nodes:
                    errors.append(f"Node {node_id} references non-existent child {child_id}")
                else:
                    child_node = self.nodes[child_id]
                    if child_node.parent_node != node_id:
                        errors.append(f"Child node {child_id} doesn't reference parent {node_id}")
        
        # Check for cycles
        if self._has_cycles():
            errors.append("Knowledge graph contains cycles")
        
        # Check for orphaned nodes (except root)
        if self.root_node_id:
            reachable = self._get_reachable_nodes(self.root_node_id)
            for node_id in self.nodes:
                if node_id not in reachable and node_id != self.root_node_id:
                    errors.append(f"Node {node_id} is not reachable from root")
        
        return errors
    
    def _has_cycles(self) -> bool:
        """Check if the graph has cycles using DFS."""
        if not self.root_node_id:
            return False
        
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.get_node(node_id)
            if node:
                for child_id in node.child_nodes:
                    if child_id not in visited:
                        if dfs(child_id):
                            return True
                    elif child_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        return dfs(self.root_node_id)
    
    def _get_reachable_nodes(self, start_node_id: str) -> set:
        """Get all nodes reachable from the start node."""
        reachable = set()
        stack = [start_node_id]
        
        while stack:
            current_id = stack.pop()
            if current_id in reachable:
                continue
            
            reachable.add(current_id)
            node = self.get_node(current_id)
            if node:
                stack.extend(node.child_nodes)
        
        return reachable
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        if not self.nodes:
            return {'total_nodes': 0}
        
        node_types = {}
        depths = []
        
        for node in self.nodes.values():
            node_type = node.node_type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
            depths.append(self.get_depth(node.node_id))
        
        return {
            'total_nodes': len(self.nodes),
            'node_types': node_types,
            'max_depth': max(depths) if depths else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'leaf_nodes': len(self.get_leaf_nodes()),
            'domain': self.domain,
            'has_root': self.root_node_id is not None
        }
    
    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save knowledge graph to JSON file."""
        filepath = Path(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'domain': self.domain,
            'root_node_id': self.root_node_id,
            'metadata': self.metadata,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> 'KnowledgeGraph':
        """Load knowledge graph from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls(domain=data.get('domain', 'general'))
        kg.root_node_id = data.get('root_node_id')
        kg.metadata = data.get('metadata', {})
        
        # Load nodes
        for node_id, node_data in data.get('nodes', {}).items():
            try:
                node = KGNode.from_dict(node_data)
                kg.add_node(node)
            except Exception as e:
                print(f"Warning: Failed to load node {node_id}: {e}")
        
        # Validate loaded structure
        errors = kg.validate_structure()
        if errors:
            print(f"Warning: Loaded knowledge graph has validation errors: {errors}")
        
        return kg
    
    def copy(self) -> 'KnowledgeGraph':
        """Create a deep copy of the knowledge graph."""
        import copy
        return copy.deepcopy(self)
    
    def merge(self, other: 'KnowledgeGraph') -> 'KnowledgeGraph':
        """
        Merge another knowledge graph into this one.
        Useful for combining domain-specific knowledge.
        """
        merged = self.copy()
        
        for node_id, node in other.nodes.items():
            if node_id not in merged.nodes:
                merged.add_node(node)
            else:
                print(f"Warning: Node {node_id} already exists, skipping")
        
        return merged
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID exists in the graph."""
        return node_id in self.nodes
    
    def __str__(self) -> str:
        """String representation of the knowledge graph."""
        stats = self.get_statistics()
        return (f"KnowledgeGraph(domain='{self.domain}', "
                f"nodes={stats['total_nodes']}, "
                f"depth={stats['max_depth']}, "
                f"root='{self.root_node_id}')")
    
    def __repr__(self) -> str:
        """Detailed representation of the knowledge graph."""
        return self.__str__()