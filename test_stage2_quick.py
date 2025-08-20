#!/usr/bin/env python3
"""
Quick Stage 2 Test - Non-interactive version that works without full LLM integration
Tests Stage 2 components with mock data to validate the framework.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_stage2_framework():
    """Test Stage 2 framework components without requiring LLM."""
    print("=" * 60)
    print("STAGE 2 FRAMEWORK VALIDATION")
    print("=" * 60)
    
    success = True
    
    print("\n1. Testing PatientData class...")
    try:
        from core.stage2_rag import PatientData, create_patient_data
        
        patient = create_patient_data(
            patient_id="test_001",
            clinical_text="Test clinical notes",
            query_description="Test query",
            demographics={"age": 70, "sex": "M"}
        )
        
        print(f"   ✓ Created patient: {patient.patient_id}")
        print(f"   ✓ Combined text length: {len(patient.get_combined_text())}")
        
    except Exception as e:
        print(f"   ✗ PatientData test failed: {e}")
        success = False
    
    print("\n2. Testing RAG data structures...")
    try:
        from core.stage2_rag import RAGResult
        
        result = RAGResult(
            extracted_text="Test extraction",
            evidence_sources=["test source"],
            confidence=0.8,
            clinical_relevance="test relevance"
        )
        
        result_dict = result.to_dict()
        print(f"   ✓ RAGResult created and serialized")
        print(f"   ✓ Confidence: {result.confidence}")
        
    except Exception as e:
        print(f"   ✗ RAGResult test failed: {e}")
        success = False
    
    print("\n3. Testing patient data loader...")
    try:
        from utils.patient_data_loader import load_sample_patients, create_test_patient_simple
        
        # Test synthetic patient creation
        patients = load_sample_patients("synthetic", count=2)
        print(f"   ✓ Created {len(patients)} synthetic patients")
        
        # Test simple patient creation
        simple_patient = create_test_patient_simple()
        print(f"   ✓ Created simple test patient: {simple_patient.patient_id}")
        
    except Exception as e:
        print(f"   ✗ Patient data loader test failed: {e}")
        success = False
    
    print("\n4. Testing knowledge graph integration...")
    try:
        from core.knowledge_graph import KnowledgeGraph, KGNode, NodeType
        
        # Create test knowledge graph
        kg = KnowledgeGraph(domain="test")
        
        test_node = KGNode(
            node_id="test_node",
            diagnosis_class="test",
            node_type=NodeType.INTERNAL,
            sub_question="Test question?",
            sub_case="test_case"
        )
        
        kg.add_node(test_node)
        print(f"   ✓ Created knowledge graph with {len(kg.nodes)} nodes")
        
        # Test serialization
        output_path = Path("output/test_kg.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kg.save_to_json(output_path)
        print(f"   ✓ Saved knowledge graph to: {output_path}")
        
    except Exception as e:
        print(f"   ✗ Knowledge graph integration test failed: {e}")
        success = False
    
    print("\n5. Testing medical ontology...")
    try:
        from utils.medical_ontology import get_medical_ontology_mapper, map_diagnosis_to_scp_ecg
        
        mapper = get_medical_ontology_mapper()
        
        # Test term mapping
        codes = mapper.map_term_to_codes("atrial fibrillation")
        print(f"   ✓ Mapped 'atrial fibrillation' to {len(codes)} codes")
        
        # Test convenience function
        scp_codes = map_diagnosis_to_scp_ecg("anterior mi")
        print(f"   ✓ Mapped 'anterior mi' to SCP-ECG codes: {scp_codes}")
        
    except Exception as e:
        print(f"   ✗ Medical ontology test failed: {e}")
        success = False
    
    return success

def test_mock_rag_population():
    """Test RAG population with mock LLM."""
    print("\n6. Testing mock RAG population...")
    
    try:
        from core.knowledge_graph import KnowledgeGraph, KGNode, NodeType
        from utils.patient_data_loader import create_test_patient_simple
        
        # Create test setup
        kg = KnowledgeGraph(domain="mock_test")
        
        test_node = KGNode(
            node_id="mock_assessment",
            diagnosis_class="test",
            node_type=NodeType.INTERNAL,
            sub_question="What is the clinical finding?",
            sub_case="mock_case"
        )
        
        kg.add_node(test_node)
        patient = create_test_patient_simple()
        
        # Mock RAG populator
        class MockRAGPopulator:
            def __init__(self):
                self.stats = {'nodes_processed': 0, 'successful_extractions': 0}
            
            def populate_knowledge_graph(self, kg_template, patient_data, **kwargs):
                import copy
                populated_kg = copy.deepcopy(kg_template)
                
                for node in populated_kg.nodes.values():
                    node.sub_description = f"Mock extraction for {patient_data.patient_id}"
                    self.stats['nodes_processed'] += 1
                    self.stats['successful_extractions'] += 1
                
                return populated_kg
            
            def get_population_stats(self):
                return {'success_rate': 100.0, 'average_confidence': 0.9}
        
        # Test mock population
        populator = MockRAGPopulator()
        populated_kg = populator.populate_knowledge_graph(kg, patient)
        
        stats = populator.get_population_stats()
        print(f"   ✓ Mock RAG population completed")
        print(f"   ✓ Success rate: {stats['success_rate']}%")
        print(f"   ✓ Nodes populated: {stats.get('nodes_processed', len(populated_kg.nodes))}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Mock RAG population test failed: {e}")
        return False

def main():
    """Run quick Stage 2 tests."""
    print("Starting Quick Stage 2 Framework Tests...\n")
    
    # Run framework tests
    framework_success = test_stage2_framework()
    
    # Run mock population test
    population_success = test_mock_rag_population()
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUICK STAGE 2 TEST SUMMARY")
    print("=" * 60)
    
    if framework_success and population_success:
        print("🎉 All Stage 2 framework tests passed!")
        print("✅ Core data structures working")
        print("✅ Patient data loading functional")
        print("✅ Knowledge graph integration successful")
        print("✅ Medical ontology mappings working")
        print("✅ Mock RAG population validated")
    else:
        print("⚠️  Some framework tests failed")
        print(f"Framework tests: {'PASS' if framework_success else 'FAIL'}")
        print(f"Population tests: {'PASS' if population_success else 'FAIL'}")
    
    print("\nStage 2 framework validation completed.")
    return framework_success and population_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)