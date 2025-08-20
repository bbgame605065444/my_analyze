#!/usr/bin/env python3
"""
CoT-RAG Stage 1 Demo Script
Demonstrates the complete Stage 1 knowledge graph generation process.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_config import ConfigManager
from core.stage1_generator import KGGenerator
from core.knowledge_graph import KnowledgeGraph

def run_stage1_demo():
    """Run a complete Stage 1 demonstration."""
    print("🚀 CoT-RAG Stage 1 Knowledge Graph Generation Demo")
    print("=" * 60)
    
    # Initialize configuration
    print("1. Initializing configuration...")
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(f"   Model: {config.get('MODEL_TYPE', 'default')}")
    print(f"   Offline mode: {config.get('OFFLINE_MODE', False)}")
    
    # Initialize KG generator
    print("\n2. Initializing KG generator...")
    generator = KGGenerator(config)
    
    # Choose decision tree
    print("\n3. Available decision trees:")
    dt_options = {
        "1": "expert_knowledge/cardiology_decision_tree.yaml",
        "2": "expert_knowledge/arrhythmia_decision_tree.yaml"
    }
    
    for key, path in dt_options.items():
        exists = "✓" if Path(path).exists() else "✗"
        print(f"   {key}. {Path(path).stem.replace('_', ' ').title()} {exists}")
    
    choice = input("\nSelect decision tree (1-2): ").strip()
    dt_path = dt_options.get(choice, dt_options["1"])
    
    if not Path(dt_path).exists():
        print(f"❌ Decision tree not found: {dt_path}")
        return False
    
    print(f"   Selected: {dt_path}")
    
    # Generate knowledge graph
    print(f"\n4. Generating knowledge graph...")
    print("   This will make LLM calls for decomposition...")
    
    try:
        output_path = Path(f"output/demo_{Path(dt_path).stem}_kg.json")
        
        kg = generator.generate_from_decision_tree(
            dt_path=dt_path,
            output_path=output_path,
            save_intermediate=True
        )
        
        print(f"\n✅ Knowledge graph generated successfully!")
        
        # Display results
        print(f"\n5. Knowledge Graph Summary:")
        print(f"   Domain: {kg.domain}")
        print(f"   Total nodes: {len(kg.nodes)}")
        print(f"   Root node: {kg.root_node_id}")
        
        stats = kg.get_statistics()
        print(f"   Node types: {stats['node_types']}")
        print(f"   Max depth: {stats['max_depth']}")
        print(f"   Validation: {'✓ Valid' if not kg.validate_structure() else '⚠ Has issues'}")
        
        # Show sample nodes
        print(f"\n6. Sample Generated Entities:")
        sample_nodes = list(kg.nodes.items())[:3]
        
        for i, (node_id, node) in enumerate(sample_nodes, 1):
            print(f"\n   Entity {i}: {node_id}")
            print(f"   ├─ Question: {node.sub_question}")
            print(f"   ├─ Type: {node.node_type.value}")
            print(f"   ├─ Parent: {node.parent_node or 'None'}")
            print(f"   └─ Children: {len(node.child_nodes)}")
            
            if node.sub_case:
                case_preview = node.sub_case[:100] + "..." if len(node.sub_case) > 100 else node.sub_case
                print(f"      Case example: {case_preview}")
        
        print(f"\n7. Output saved to: {output_path}")
        
        # Generation statistics
        print(f"\n8. Generation Statistics:")
        stats = generator.generation_stats
        print(f"   DT nodes processed: {stats['nodes_processed']}")
        print(f"   Entities created: {stats['entities_created']}")
        print(f"   LLM calls made: {stats['llm_calls']}")
        print(f"   Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print(f"   First few errors:")
            for error in stats['errors'][:3]:
                print(f"     - {error}")
        
        print(f"\n🎉 Stage 1 Demo completed successfully!")
        print(f"   Next steps: Use this KG for Stage 2 (RAG population) and Stage 3 (execution)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_kg_explorer():
    """Interactive knowledge graph explorer."""
    print(f"\n📊 Interactive KG Explorer")
    print("=" * 40)
    
    # Load existing KG
    kg_files = list(Path("output").glob("*_kg.json"))
    
    if not kg_files:
        print("No knowledge graphs found in output directory.")
        return
    
    print("Available knowledge graphs:")
    for i, kg_file in enumerate(kg_files, 1):
        print(f"  {i}. {kg_file.name}")
    
    try:
        choice = int(input("Select KG to explore (number): ")) - 1
        kg_file = kg_files[choice]
        
        print(f"Loading: {kg_file}")
        kg = KnowledgeGraph.load_from_json(kg_file)
        
        print(f"\nKnowledge Graph: {kg.domain}")
        print(f"Nodes: {len(kg.nodes)}")
        
        while True:
            print(f"\nCommands:")
            print("  1. List all nodes")
            print("  2. Show node details") 
            print("  3. Show statistics")
            print("  4. Validate structure")
            print("  5. Exit")
            
            cmd = input("Choose command: ").strip()
            
            if cmd == "1":
                print(f"\nAll nodes:")
                for node_id, node in kg.nodes.items():
                    print(f"  {node_id} ({node.node_type.value}): {node.sub_question}")
            
            elif cmd == "2":
                node_id = input("Enter node ID: ").strip()
                node = kg.get_node(node_id)
                if node:
                    print(f"\nNode Details: {node_id}")
                    print(f"  Type: {node.node_type.value}")
                    print(f"  Question: {node.sub_question}")
                    print(f"  Diagnosis: {node.diagnosis_class}")
                    print(f"  Parent: {node.parent_node}")
                    print(f"  Children: {node.child_nodes}")
                    if node.sub_case:
                        print(f"  Case: {node.sub_case[:200]}...")
                else:
                    print(f"Node not found: {node_id}")
            
            elif cmd == "3":
                stats = kg.get_statistics()
                print(f"\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            elif cmd == "4":
                errors = kg.validate_structure()
                if errors:
                    print(f"\nValidation errors:")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print(f"\n✓ Knowledge graph structure is valid")
            
            elif cmd == "5":
                break
            
            else:
                print("Invalid command")
                
    except (ValueError, IndexError):
        print("Invalid selection")
    except Exception as e:
        print(f"Explorer error: {e}")

def main():
    """Main demo function."""
    print("CoT-RAG Stage 1 Implementation")
    print("Choose an option:")
    print("  1. Run full Stage 1 demo")
    print("  2. Explore existing knowledge graphs")
    print("  3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        success = run_stage1_demo()
        if success:
            print(f"\n✨ Demo completed! You can now explore the generated KG or proceed to Stage 2.")
    
    elif choice == "2":
        interactive_kg_explorer()
    
    elif choice == "3":
        print("Goodbye!")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()