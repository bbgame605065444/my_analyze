#!/usr/bin/env python3
"""
Test Script for CoT-RAG Stage 1 Knowledge Graph Generation
Tests the complete Stage 1 pipeline using existing LLM infrastructure.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import existing infrastructure
from model_config import ConfigManager
from core.stage1_generator import KGGenerator, ExpertDecisionTree, create_sample_decision_tree
from core.knowledge_graph import KnowledgeGraph
from utils.prompt_templates import PromptTemplates

def test_expert_decision_tree_loading():
    """Test loading and validation of expert decision trees."""
    print("=" * 60)
    print("TESTING EXPERT DECISION TREE LOADING")
    print("=" * 60)
    
    # Create sample decision tree if it doesn't exist
    sample_dt_path = Path("expert_knowledge/test_cardiology.yaml")
    if not sample_dt_path.exists():
        print("Creating sample decision tree...")
        create_sample_decision_tree(sample_dt_path)
    
    try:
        # Load the decision tree
        expert_dt = ExpertDecisionTree(sample_dt_path)
        
        print(f"✓ Successfully loaded decision tree: {expert_dt.domain}")
        print(f"  Expert: {expert_dt.metadata['expert']}")
        print(f"  Nodes: {len(expert_dt.nodes)}")
        print(f"  Root nodes: {len(expert_dt.get_root_nodes())}")
        
        # Print node structure
        print("\nDecision Tree Structure:")
        for i, node in enumerate(expert_dt.nodes):
            indent = "  " if node.get('parent') else ""
            print(f"{indent}{i+1}. {node['node_id']}: {node['question'][:60]}...")
        
        return expert_dt
        
    except Exception as e:
        print(f"✗ Failed to load decision tree: {e}")
        return None

def test_prompt_templates():
    """Test prompt template generation."""
    print("\n" + "=" * 60)
    print("TESTING PROMPT TEMPLATES")
    print("=" * 60)
    
    try:
        templates = PromptTemplates()
        
        # Test decomposition prompt
        sample_prompt = templates.get_decomposition_prompt(
            question="What is the fundamental rhythm pattern of this ECG?",
            knowledge_case="Normal sinus rhythm shows regular R-R intervals...",
            domain="cardiology",
            context="Initial ECG assessment"
        )
        
        print("✓ Generated decomposition prompt:")
        print(f"  Length: {len(sample_prompt)} characters")
        print(f"  Contains required elements: {'JSON' in sample_prompt and 'entities' in sample_prompt}")
        
        # Show first 200 characters
        print(f"  Preview: {sample_prompt[:200]}...")
        
        return templates
        
    except Exception as e:
        print(f"✗ Failed to generate prompt templates: {e}")
        return None

def test_knowledge_graph_creation():
    """Test basic knowledge graph operations."""
    print("\n" + "=" * 60)
    print("TESTING KNOWLEDGE GRAPH CREATION")
    print("=" * 60)
    
    try:
        from core.knowledge_graph import KGNode, NodeType
        
        # Create test knowledge graph
        kg = KnowledgeGraph(domain="test_cardiology")
        
        # Create test nodes
        root_node = KGNode(
            node_id="rhythm_root",
            diagnosis_class="Rhythm Assessment",
            node_type=NodeType.ROOT,
            sub_question="Is the ECG rhythm normal or abnormal?",
            sub_case="Normal sinus rhythm shows regular R-R intervals with normal P-waves"
        )
        
        child_node = KGNode(
            node_id="rate_analysis",
            diagnosis_class="Rate Evaluation",
            node_type=NodeType.INTERNAL,
            parent_node="rhythm_root",
            sub_question="What is the heart rate category?"
        )
        
        root_node.add_child("rate_analysis")
        
        # Add nodes to graph
        kg.add_node(root_node)
        kg.add_node(child_node)
        
        # Test operations
        print(f"✓ Created knowledge graph: {kg}")
        print(f"  Root node: {kg.root_node_id}")
        print(f"  Total nodes: {len(kg.nodes)}")
        print(f"  Children of root: {len(kg.get_children('rhythm_root'))}")
        
        # Test validation
        errors = kg.validate_structure()
        print(f"  Validation errors: {len(errors)}")
        if errors:
            for error in errors:
                print(f"    - {error}")
        
        # Test statistics
        stats = kg.get_statistics()
        print(f"  Statistics: {stats}")
        
        return kg
        
    except Exception as e:
        print(f"✗ Failed to create knowledge graph: {e}")
        return None

def test_full_kg_generation():
    """Test complete knowledge graph generation using real decision tree."""
    print("\n" + "=" * 60)
    print("TESTING FULL KG GENERATION")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"Using model: {config.get('MODEL_TYPE', 'default')}")
        
        # Use the comprehensive cardiology decision tree
        dt_path = Path("expert_knowledge/cardiology_decision_tree.yaml")
        
        if not dt_path.exists():
            print(f"Decision tree not found: {dt_path}")
            print("Using fallback simple decision tree...")
            dt_path = Path("expert_knowledge/test_cardiology.yaml")
            if not dt_path.exists():
                create_sample_decision_tree(dt_path)
        
        # Initialize KG generator
        generator = KGGenerator(config)
        
        print(f"Generating knowledge graph from: {dt_path}")
        print("This will make LLM calls - please wait...")
        
        # Generate knowledge graph
        output_path = Path("output/test_generated_kg.json")
        kg = generator.generate_from_decision_tree(
            dt_path,
            output_path=output_path,
            save_intermediate=True
        )
        
        print(f"\n✓ Knowledge graph generation completed!")
        print(f"  Domain: {kg.domain}")
        print(f"  Total nodes: {len(kg.nodes)}")
        print(f"  Validation errors: {len(kg.validate_structure())}")
        print(f"  Output saved to: {output_path}")
        
        # Show sample nodes
        print(f"\nSample generated nodes:")
        for i, (node_id, node) in enumerate(list(kg.nodes.items())[:3]):
            print(f"  {i+1}. {node_id}")
            print(f"     Question: {node.sub_question}")
            print(f"     Type: {node.node_type.value}")
            if node.sub_case:
                print(f"     Case: {node.sub_case[:100]}...")
        
        return kg
        
    except Exception as e:
        print(f"✗ Knowledge graph generation failed: {e}")
        import traceback
        print("Full error:")
        traceback.print_exc()
        return None

def test_kg_serialization():
    """Test knowledge graph save/load functionality."""
    print("\n" + "=" * 60)
    print("TESTING KG SERIALIZATION")
    print("=" * 60)
    
    try:
        # Create test KG
        kg = test_knowledge_graph_creation()
        if not kg:
            return False
        
        # Test save
        test_file = Path("output/test_kg_serialization.json")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        kg.save_to_json(test_file)
        print(f"✓ Saved KG to: {test_file}")
        
        # Test load
        loaded_kg = KnowledgeGraph.load_from_json(test_file)
        print(f"✓ Loaded KG from file")
        
        # Compare
        print(f"  Original nodes: {len(kg.nodes)}")
        print(f"  Loaded nodes: {len(loaded_kg.nodes)}")
        print(f"  Domains match: {kg.domain == loaded_kg.domain}")
        print(f"  Root nodes match: {kg.root_node_id == loaded_kg.root_node_id}")
        
        # Cleanup
        test_file.unlink()
        print(f"✓ Cleaned up test file")
        
        return True
        
    except Exception as e:
        print(f"✗ Serialization test failed: {e}")
        return False

def main():
    """Run all Stage 1 tests."""
    print("CoT-RAG Stage 1 Knowledge Graph Generation Test Suite")
    print("Using existing LLM infrastructure from model_interface.py")
    
    # Ensure output directories exist
    Path("output").mkdir(exist_ok=True)
    Path("expert_knowledge").mkdir(exist_ok=True)
    
    test_results = {}
    
    # Run tests
    test_results['dt_loading'] = test_expert_decision_tree_loading() is not None
    test_results['prompt_templates'] = test_prompt_templates() is not None
    test_results['kg_creation'] = test_knowledge_graph_creation() is not None
    test_results['serialization'] = test_kg_serialization()
    
    # Full generation test (may take longer due to LLM calls)
    print(f"\n{'='*60}")
    print("OPTIONAL: Full KG Generation Test")
    print("This test makes actual LLM calls and may take several minutes.")
    
    user_input = input("Run full generation test? (y/N): ").strip().lower()
    if user_input in ['y', 'yes']:
        test_results['full_generation'] = test_full_kg_generation() is not None
    else:
        test_results['full_generation'] = None
        print("Skipped full generation test")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in test_results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    # Overall status
    passed_tests = sum(1 for r in test_results.values() if r is True)
    total_tests = sum(1 for r in test_results.values() if r is not None)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests and total_tests > 0:
        print("🎉 All tests passed! Stage 1 implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)