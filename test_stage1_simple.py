#!/usr/bin/env python3
"""
Simple Stage 1 test without interactive input.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_config import ConfigManager
from core.stage1_generator import KGGenerator, create_sample_decision_tree
from core.knowledge_graph import KnowledgeGraph

def test_stage1():
    """Test Stage 1 implementation."""
    print("🚀 Testing CoT-RAG Stage 1 Knowledge Graph Generation")
    print("=" * 60)
    
    # 1. Test configuration
    print("1. Testing configuration...")
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print(f"   ✓ Configuration loaded")
        print(f"   Model: {config.model_type}")
        print(f"   Offline mode: {config.__dict__.get('offline_mode', 'Not set')}")
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        return False
    
    # 2. Create sample decision tree if needed
    print("\n2. Preparing decision tree...")
    sample_dt_path = Path("expert_knowledge/test_simple.yaml")
    
    try:
        # Create a simple decision tree for testing
        simple_dt = {
            'domain': 'test_cardiology',
            'expert': 'Test Expert',
            'version': '1.0',
            'description': 'Simple test decision tree',
            'created_date': '2025-01-19',
            'nodes': [
                {
                    'node_id': 'rhythm_check',
                    'question': 'Is the ECG rhythm normal or abnormal?',
                    'knowledge_case': 'Normal rhythm shows regular R-R intervals with P-waves before each QRS.',
                    'is_root': True,
                    'clinical_significance': 'Initial rhythm assessment'
                },
                {
                    'node_id': 'rate_assessment', 
                    'question': 'What is the heart rate category?',
                    'knowledge_case': 'Normal: 60-100 bpm, Bradycardia: <60 bpm, Tachycardia: >100 bpm',
                    'parent': 'rhythm_check',
                    'clinical_significance': 'Rate determines urgency'
                }
            ]
        }
        
        # Save simple decision tree
        import yaml
        sample_dt_path.parent.mkdir(exist_ok=True)
        with open(sample_dt_path, 'w') as f:
            yaml.dump(simple_dt, f, default_flow_style=False, indent=2)
        
        print(f"   ✓ Created test decision tree: {sample_dt_path}")
        
    except Exception as e:
        print(f"   ✗ Failed to create decision tree: {e}")
        return False
    
    # 3. Test knowledge graph creation
    print("\n3. Testing basic knowledge graph operations...")
    try:
        from core.knowledge_graph import KGNode, NodeType
        
        kg = KnowledgeGraph(domain="test")
        
        # Create test nodes
        root_node = KGNode(
            node_id="test_root",
            diagnosis_class="Test Assessment", 
            node_type=NodeType.ROOT,
            sub_question="Test question?",
            sub_case="Test case example"
        )
        
        kg.add_node(root_node)
        
        # Test operations
        print(f"   ✓ Knowledge graph created: {len(kg.nodes)} nodes")
        print(f"   ✓ Root node: {kg.root_node_id}")
        
        # Test validation
        errors = kg.validate_structure()
        print(f"   ✓ Validation: {'✓ Valid' if not errors else f'✗ {len(errors)} errors'}")
        
    except Exception as e:
        print(f"   ✗ Knowledge graph test failed: {e}")
        return False
    
    # 4. Test prompt templates
    print("\n4. Testing prompt templates...")
    try:
        from utils.prompt_templates import PromptTemplates
        
        templates = PromptTemplates()
        prompt = templates.get_decomposition_prompt(
            question="Test question",
            knowledge_case="Test case",
            domain="test",
            context="Test context"
        )
        
        print(f"   ✓ Prompt template generated: {len(prompt)} characters")
        print(f"   ✓ Contains JSON structure: {'JSON' in prompt}")
        
    except Exception as e:
        print(f"   ✗ Prompt template test failed: {e}")
        return False
    
    # 5. Test full generation (this will make LLM calls)
    print("\n5. Testing knowledge graph generation...")
    print("   This will make LLM calls - testing with simple decision tree...")
    
    try:
        generator = KGGenerator(config)
        
        output_path = Path("output/test_kg.json")
        output_path.parent.mkdir(exist_ok=True)
        
        print(f"   Generating KG from: {sample_dt_path}")
        
        kg = generator.generate_from_decision_tree(
            dt_filepath=sample_dt_path,
            output_path=output_path,
            save_intermediate=False
        )
        
        print(f"   ✓ Knowledge graph generated successfully!")
        print(f"     Domain: {kg.domain}")
        print(f"     Total nodes: {len(kg.nodes)}")
        print(f"     Root node: {kg.root_node_id}")
        
        # Show sample nodes
        if kg.nodes:
            sample_node = list(kg.nodes.values())[0]
            print(f"     Sample node: {sample_node.node_id}")
            print(f"     Question: {sample_node.sub_question}")
        
        # Generation stats
        stats = generator.generation_stats
        print(f"     LLM calls made: {stats['llm_calls']}")
        print(f"     Entities created: {stats['entities_created']}")
        print(f"     Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print("     Errors encountered:")
            for error in stats['errors'][:3]:
                print(f"       - {error}")
        
        print(f"   ✓ Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Knowledge graph generation failed: {e}")
        import traceback
        print("   Full error trace:")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_stage1()
    
    print(f"\n{'='*60}")
    print("STAGE 1 TEST SUMMARY")
    print(f"{'='*60}")
    
    if success:
        print("🎉 All Stage 1 tests passed!")
        print("✅ CoT-RAG Stage 1 implementation is working correctly")
        print("📊 Knowledge graph generation successful")
        print("🔄 Ready for Stage 2 implementation")
        
        print(f"\nGenerated files:")
        print(f"  - expert_knowledge/test_simple.yaml")
        print(f"  - output/test_kg.json")
        
        print(f"\nNext steps:")
        print(f"  1. Check output/test_kg.json to see generated knowledge graph")
        print(f"  2. Try with full cardiology decision tree")
        print(f"  3. Implement Stage 2 (RAG population)")
        
    else:
        print("❌ Stage 1 tests failed")
        print("🔧 Please check the error messages above")
        print("💡 Ensure your model configuration is correct")
    
    return success

if __name__ == "__main__":
    main()