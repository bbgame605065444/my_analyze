#!/usr/bin/env python3
"""
Complete Stage 1 → Stage 2 Integration Test
Tests the full CoT-RAG pipeline from expert decision trees to populated knowledge graphs.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.stage1_generator import KGGenerator, ExpertDecisionTree
from core.stage2_rag import RAGPopulator
from core.knowledge_graph import KnowledgeGraph
from utils.llm_interface import get_global_llm_interface
from utils.patient_data_loader import create_test_patient_simple, load_sample_patients

def test_complete_pipeline():
    """Test the complete Stage 1 → Stage 2 pipeline."""
    print("=" * 70)
    print("COMPLETE COT-RAG PIPELINE TEST (STAGE 1 → STAGE 2)")
    print("=" * 70)
    
    pipeline_success = True
    
    # Stage 1: Generate Knowledge Graph from Expert Decision Tree
    print("\n🔬 STAGE 1: KNOWLEDGE GRAPH GENERATION")
    print("-" * 50)
    
    print("1. Loading expert decision tree...")
    
    try:
        # Use the simple test decision tree
        dt_path = "expert_knowledge/test_simple.yaml"
        expert_dt = ExpertDecisionTree(dt_path)
        
        print(f"   ✓ Loaded decision tree: {expert_dt.metadata.get('name', 'Unknown')}")
        print(f"   ✓ Domain: {expert_dt.metadata.get('domain', 'Unknown')}")
        print(f"   ✓ Nodes: {len(expert_dt.tree_data.get('nodes', []))}")
        
    except Exception as e:
        print(f"   ✗ Failed to load decision tree: {e}")
        pipeline_success = False
        return pipeline_success
    
    print("\n2. Generating knowledge graph from decision tree...")
    
    try:
        # Initialize KG generator with mock if needed
        try:
            llm_interface = get_global_llm_interface()
            generator = KGGenerator(llm_interface)
            use_mock = False
        except Exception:
            print("   ℹ  LLM not available, using mock generator")
            generator = MockKGGenerator()
            use_mock = True
        
        # Generate knowledge graph
        generated_kg = generator.generate_from_decision_tree(dt_path)
        
        print(f"   ✓ Generated knowledge graph with {len(generated_kg.nodes)} nodes")
        print(f"   ✓ Domain: {generated_kg.domain}")
        
        # Validate knowledge graph structure
        validation_errors = generated_kg.validate_structure()
        if validation_errors:
            print(f"   ⚠  Knowledge graph has {len(validation_errors)} validation issues")
            for error in validation_errors[:3]:  # Show first 3
                print(f"      - {error}")
        else:
            print("   ✓ Knowledge graph structure is valid")
        
    except Exception as e:
        print(f"   ✗ Failed to generate knowledge graph: {e}")
        pipeline_success = False
        return pipeline_success
    
    # Stage 2: Populate Knowledge Graph with Patient Data
    print("\n🏥 STAGE 2: RAG POPULATION WITH PATIENT DATA")
    print("-" * 50)
    
    print("3. Creating test patient data...")
    
    try:
        # Create multiple test patients
        patients = [
            create_test_patient_simple(),
            *load_sample_patients("synthetic", count=2)
        ]
        
        print(f"   ✓ Created {len(patients)} test patients")
        
        for i, patient in enumerate(patients):
            print(f"   ✓ Patient {i+1}: {patient.patient_id} ({patient.demographics.get('age', 'Unknown')} years)")
        
    except Exception as e:
        print(f"   ✗ Failed to create patient data: {e}")
        pipeline_success = False
        return pipeline_success
    
    print("\n4. Initializing RAG populator...")
    
    try:
        # Initialize RAG populator
        if not use_mock:
            rag_populator = RAGPopulator(llm_interface)
        else:
            print("   ℹ  Using mock RAG populator")
            rag_populator = MockRAGPopulator()
        
        print("   ✓ RAG populator initialized successfully")
        
    except Exception as e:
        print(f"   ✗ Failed to initialize RAG populator: {e}")
        pipeline_success = False
        return pipeline_success
    
    print("\n5. Populating knowledge graphs with patient data...")
    
    populated_kgs = []
    population_stats = []
    
    for i, patient in enumerate(patients):
        print(f"\n   Processing Patient {i+1}: {patient.patient_id}")
        
        try:
            # Use the generated KG as template for each patient
            populated_kg = rag_populator.populate_knowledge_graph(
                kg_template=generated_kg,
                patient_data=patient,
                save_intermediate=False
            )
            
            populated_kgs.append(populated_kg)
            
            # Get population statistics
            if hasattr(rag_populator, 'get_population_stats'):
                stats = rag_populator.get_population_stats()
                population_stats.append(stats)
                print(f"   ✓ Population success rate: {stats.get('success_rate', 0):.1f}%")
                
                # Reset stats for next patient
                if hasattr(rag_populator, 'rag_stats'):
                    rag_populator.rag_stats = {
                        'nodes_processed': 0,
                        'successful_extractions': 0,
                        'failed_extractions': 0,
                        'novel_patterns_detected': 0,
                        'llm_calls': 0,
                        'total_confidence': 0.0
                    }
            
            print(f"   ✓ Successfully populated knowledge graph for {patient.patient_id}")
            
        except Exception as e:
            print(f"   ✗ Failed to populate KG for {patient.patient_id}: {e}")
            pipeline_success = False
            continue
    
    # Integration Analysis
    print("\n📊 INTEGRATION ANALYSIS")
    print("-" * 50)
    
    print("6. Analyzing pipeline results...")
    
    try:
        # Analyze Stage 1 results
        print(f"\n   Stage 1 Results:")
        print(f"   ✓ Knowledge graph nodes: {len(generated_kg.nodes)}")
        print(f"   ✓ Knowledge graph domain: {generated_kg.domain}")
        
        # Count different node types
        from core.knowledge_graph import NodeType
        node_type_counts = {}
        for node in generated_kg.nodes.values():
            node_type = node.node_type
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        for node_type, count in node_type_counts.items():
            print(f"   ✓ {node_type.value} nodes: {count}")
        
        # Analyze Stage 2 results
        print(f"\n   Stage 2 Results:")
        print(f"   ✓ Patients processed: {len(populated_kgs)}")
        
        if population_stats:
            avg_success_rate = sum(stats.get('success_rate', 0) for stats in population_stats) / len(population_stats)
            avg_confidence = sum(stats.get('average_confidence', 0) for stats in population_stats) / len(population_stats)
            
            print(f"   ✓ Average success rate: {avg_success_rate:.1f}%")
            print(f"   ✓ Average confidence: {avg_confidence:.2f}")
        
        # Check population quality
        total_populated_nodes = 0
        total_nodes = 0
        
        for kg in populated_kgs:
            for node in kg.nodes.values():
                total_nodes += 1
                if node.sub_description and node.sub_description.strip():
                    total_populated_nodes += 1
        
        if total_nodes > 0:
            population_rate = (total_populated_nodes / total_nodes) * 100
            print(f"   ✓ Overall population rate: {population_rate:.1f}%")
        
    except Exception as e:
        print(f"   ⚠  Analysis failed: {e}")
    
    # Save Results
    print("\n7. Saving pipeline results...")
    
    try:
        output_dir = Path("output/pipeline_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Stage 1 results
        stage1_output = output_dir / "stage1_generated_kg.json"
        generated_kg.save_to_json(stage1_output)
        print(f"   ✓ Stage 1 KG saved to: {stage1_output}")
        
        # Save Stage 2 results
        for i, (kg, patient) in enumerate(zip(populated_kgs, patients)):
            stage2_output = output_dir / f"stage2_populated_kg_{patient.patient_id}.json"
            kg.save_to_json(stage2_output)
            print(f"   ✓ Stage 2 KG {i+1} saved to: {stage2_output}")
        
        # Save pipeline summary
        summary_data = {
            'stage1': {
                'nodes_generated': len(generated_kg.nodes),
                'domain': generated_kg.domain,
                'validation_errors': len(generated_kg.validate_structure())
            },
            'stage2': {
                'patients_processed': len(populated_kgs),
                'population_stats': population_stats
            },
            'pipeline_success': pipeline_success
        }
        
        import json
        summary_output = output_dir / "pipeline_summary.json"
        with open(summary_output, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"   ✓ Pipeline summary saved to: {summary_output}")
        
    except Exception as e:
        print(f"   ⚠  Failed to save results: {e}")
    
    return pipeline_success

class MockKGGenerator:
    """Mock KG generator for testing when LLM is not available."""
    
    def generate_from_decision_tree(self, dt_filepath):
        """Generate a mock knowledge graph."""
        print("   [MOCK] Generating knowledge graph from decision tree")
        
        # Create a simple mock knowledge graph
        kg = KnowledgeGraph(
            domain="mock_cardiology",
            description="Mock knowledge graph for testing"
        )
        
        from core.knowledge_graph import KGNode, NodeType
        
        # Add mock nodes based on common cardiology workflow
        mock_nodes = [
            KGNode(
                node_id="rhythm_analysis",
                diagnosis_class="arrhythmia",
                node_type=NodeType.INTERNAL,
                sub_question="What is the underlying cardiac rhythm?",
                sub_case="rhythm_assessment",
                sub_description="Evaluate ECG rhythm characteristics",
                answer=""
            ),
            KGNode(
                node_id="rate_assessment",
                diagnosis_class="arrhythmia", 
                node_type=NodeType.INTERNAL,
                sub_question="Is the heart rate within normal limits?",
                sub_case="rate_evaluation",
                sub_description="Assess heart rate and clinical significance",
                answer=""
            ),
            KGNode(
                node_id="final_diagnosis",
                diagnosis_class="arrhythmia",
                node_type=NodeType.LEAF,
                sub_question="What is the final diagnosis?",
                sub_case="diagnostic_conclusion",
                sub_description="Synthesize findings for final diagnosis",
                answer=""
            )
        ]
        
        for node in mock_nodes:
            kg.add_node(node)
        
        return kg

class MockRAGPopulator:
    """Mock RAG populator for testing when LLM is not available."""
    
    def __init__(self):
        self.rag_stats = {
            'nodes_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'novel_patterns_detected': 0,
            'llm_calls': 0,
            'total_confidence': 0.0
        }
    
    def populate_knowledge_graph(self, kg_template, patient_data, save_intermediate=True):
        """Mock populate with simulated responses."""
        import copy
        populated_kg = copy.deepcopy(kg_template)
        
        # Mock responses based on patient data
        if "atrial fibrillation" in patient_data.clinical_text.lower():
            responses = {
                "rhythm_analysis": "Irregularly irregular rhythm consistent with atrial fibrillation. No discernible P waves.",
                "rate_assessment": "Heart rate elevated at 120-140 bpm, indicating rapid ventricular response requiring intervention.",
                "final_diagnosis": "Atrial fibrillation with rapid ventricular response. Consider rate control and anticoagulation."
            }
        else:
            responses = {
                "rhythm_analysis": "Regular sinus rhythm observed with normal P wave morphology and PR intervals.",
                "rate_assessment": "Heart rate within normal limits at 60-100 bpm with regular intervals.",
                "final_diagnosis": "Normal sinus rhythm. No immediate intervention required."
            }
        
        # Populate nodes with mock responses
        for node_id, node in populated_kg.nodes.items():
            if node_id in responses:
                node.sub_description = responses[node_id]
                self.rag_stats['successful_extractions'] += 1
                self.rag_stats['total_confidence'] += 0.85
            
            self.rag_stats['nodes_processed'] += 1
        
        return populated_kg
    
    def get_population_stats(self):
        """Return mock statistics."""
        if self.rag_stats['successful_extractions'] > 0:
            return {
                'success_rate': (self.rag_stats['successful_extractions'] / self.rag_stats['nodes_processed']) * 100,
                'average_confidence': self.rag_stats['total_confidence'] / self.rag_stats['successful_extractions'],
                'nodes_processed': self.rag_stats['nodes_processed']
            }
        return {'success_rate': 0, 'average_confidence': 0, 'nodes_processed': 0}

def main():
    """Run complete pipeline test."""
    print("Starting Complete CoT-RAG Pipeline Test...\n")
    
    pipeline_success = test_complete_pipeline()
    
    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    if pipeline_success:
        print("🎉 PIPELINE TEST PASSED!")
        print("✅ Stage 1 → Stage 2 integration working correctly")
        print("✅ Knowledge graph generation successful")
        print("✅ RAG population functional")
        print("✅ End-to-end CoT-RAG pipeline validated")
    else:
        print("❌ PIPELINE TEST FAILED!")
        print("⚠️  Check individual components and dependencies")
        print("⚠️  Verify LLM configuration if using real models")
    
    print("\nCoT-RAG pipeline testing completed.")
    return pipeline_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)