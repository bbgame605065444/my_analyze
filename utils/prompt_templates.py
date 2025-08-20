"""
Prompt Templates for CoT-RAG Framework
Structured prompts for each stage of the CoT-RAG pipeline.
"""

from typing import Dict, List, Any, Optional

class PromptTemplates:
    """
    Collection of prompt templates for CoT-RAG operations.
    Templates are designed to work with the existing LLM interface.
    """
    
    def __init__(self):
        self.templates = {
            'decomposition': self._get_decomposition_template(),
            'rag_extraction': self._get_rag_template(),
            'decision_making': self._get_decision_template(),
            'final_diagnosis': self._get_diagnosis_template(),
            'narrative_generation': self._get_narrative_template()
        }
    
    def _get_decomposition_template(self) -> str:
        """
        Template for Stage 1: Decomposing expert decision tree nodes into fine-grained entities.
        This is the core of CoT-RAG Stage 1 implementation.
        """
        return """
You are a medical expert assistant helping to decompose clinical decision-making processes.

TASK: Break down a high-level clinical question into specific, actionable sub-questions that can guide step-by-step reasoning.

EXPERT DECISION TREE NODE:
Domain: {domain}
Clinical Question: {question}
Knowledge Case: {knowledge_case}
Context: {context}

REQUIREMENTS:
1. Decompose the main question into 3-5 specific sub-questions
2. Each sub-question should be answerable with specific clinical data
3. Create realistic clinical examples for each sub-question
4. Specify what type of medical data or classifier output is needed
5. Establish logical dependencies between sub-questions

OUTPUT FORMAT (JSON):
```json
{{
  "entities": [
    {{
      "entity_id": "entity_1",
      "sub_question": "Specific clinical question",
      "sub_case": "Example case showing how to answer this question",
      "diagnosis_class": "Clinical concept this addresses",
      "required_inputs": ["classifier_name", "data_type"],
      "clinical_significance": "Why this question matters clinically",
      "dependencies": [0, 1], 
      "is_terminal": false
    }}
  ],
  "relationships": [
    {{
      "from_entity": "entity_1",
      "to_entity": "entity_2", 
      "relationship_type": "answer_provision",
      "description": "How entity_1 output feeds into entity_2"
    }}
  ]
}}
```

EXAMPLE MEDICAL CONTEXT:
If the main question is "Does the patient have a cardiac arrhythmia?", break it down into:
- "What is the heart rate category?" (bradycardia/normal/tachycardia)
- "Is the rhythm regular or irregular?"
- "Are there abnormal P-wave patterns?"
- "Is there evidence of conduction blocks?"

Focus on creating a logical flow where each answer builds toward the final diagnosis.

Generate the decomposition now:
"""
    
    def _get_rag_template(self) -> str:
        """
        Template for Stage 2: RAG-based information extraction from patient data.
        """
        return """
You are a clinical information extraction specialist.

TASK: Extract specific clinical information from patient data to answer a focused medical question.

CLINICAL QUESTION: {sub_question}
EXAMPLE CASE: {sub_case}
DIAGNOSIS CONTEXT: {diagnosis_class}

PATIENT DATA:
{patient_context}

INSTRUCTIONS:
1. Search the patient data for information relevant to the clinical question
2. Extract the most relevant clinical findings
3. Provide specific evidence sources from the patient data
4. Rate your confidence in the extracted information

OUTPUT FORMAT:
EXTRACTED_TEXT: [The specific clinical information found, or "No relevant information found"]
EVIDENCE: [Direct quotes or references from patient data that support this finding]
CONFIDENCE: [0.0-1.0 confidence score]
CLINICAL_RELEVANCE: [Why this finding is important for the diagnosis]

EXAMPLE:
EXTRACTED_TEXT: Heart rate 120 bpm with irregular rhythm
EVIDENCE: "ECG shows heart rate of 120 with irregularly irregular pattern, no discernible P waves"
CONFIDENCE: 0.9
CLINICAL_RELEVANCE: High heart rate with irregular rhythm suggests possible atrial fibrillation

Extract the information now:
"""
    
    def _get_decision_template(self) -> str:
        """
        Template for Stage 3: Decision-making during knowledge graph traversal.
        """
        return """
You are a clinical reasoning engine making diagnostic decisions.

CURRENT DECISION POINT:
Question: {node_question}
Clinical Context: {diagnosis_class}
Available Evidence: {node_evidence}

PREVIOUS REASONING CHAIN:
{evidence_chain}

CLASSIFIER OUTPUTS:
{classifier_outputs}

DECISION OPTIONS:
{child_options}

TASK: 
1. Analyze the available evidence and classifier outputs
2. Make a logical decision about which path to follow
3. Provide reasoning for your decision
4. Rate your confidence

DECISION RULES:
- Use classifier probabilities when available (threshold > 0.7 for high confidence)
- Consider clinical evidence from patient data
- Follow established medical decision-making practices
- If uncertain, indicate need for additional information

OUTPUT FORMAT:
DECISION: [chosen_path or "uncertain" or "continue"]
REASONING: [Clinical reasoning for this decision]
CONFIDENCE: [0.0-1.0 confidence score]
NEXT_STEPS: [What should be evaluated next]

Make your decision now:
"""
    
    def _get_diagnosis_template(self) -> str:
        """
        Template for generating final diagnosis at leaf nodes.
        """
        return """
You are a clinical diagnostic specialist providing final medical conclusions.

DIAGNOSTIC CONTEXT:
Final Diagnosis Category: {leaf_diagnosis}
Domain: {domain}

COMPLETE EVIDENCE CHAIN:
{evidence_chain}

CLASSIFIER OUTPUTS SUMMARY:
{classifier_summary}

TASK:
1. Synthesize all evidence into a coherent final diagnosis
2. Provide confidence assessment
3. Suggest follow-up actions if needed
4. Note any limitations or uncertainties

OUTPUT FORMAT:
FINAL_DIAGNOSIS: [Specific medical diagnosis or conclusion]
CONFIDENCE: [0.0-1.0 overall confidence]
SUPPORTING_EVIDENCE: [Key evidence supporting this diagnosis]
LIMITATIONS: [Any uncertainties or limitations in the assessment]
RECOMMENDATIONS: [Suggested follow-up actions or additional tests]

MEDICAL STANDARDS:
- Use standard medical terminology
- Be specific about diagnostic criteria met
- Acknowledge uncertainty when appropriate
- Follow evidence-based medicine principles

Generate the final diagnosis now:
"""
    
    def _get_narrative_template(self) -> str:
        """
        Template for generating human-readable narrative reports.
        """
        return """
You are a medical reporting specialist creating clear, professional diagnostic reports.

DIAGNOSTIC CASE SUMMARY:
Decision Path: {decision_path}
Final Diagnosis: {final_diagnosis}
Confidence: {overall_confidence}

REASONING EVIDENCE:
{evidence_chain}

TECHNICAL ANALYSIS:
{classifier_outputs}

TASK: Create a clear, professional medical report that explains:
1. The step-by-step diagnostic reasoning process
2. Key clinical findings that led to the diagnosis
3. How different pieces of evidence support the conclusion
4. Any limitations or areas of uncertainty

REPORT STRUCTURE:
## Clinical Assessment Summary
[Opening statement with final diagnosis and confidence]

## Diagnostic Reasoning Process
[Step-by-step explanation of how the diagnosis was reached]

## Supporting Evidence
[Key clinical findings and their significance]

## Technical Analysis
[How AI classifiers and data analysis contributed]

## Limitations and Recommendations
[Any uncertainties and suggested follow-up]

WRITING STYLE:
- Professional medical terminology
- Clear, logical flow
- Accessible to healthcare professionals
- Evidence-based reasoning
- Appropriate caveats about AI-assisted diagnosis

Generate the narrative report now:
"""
    
    def get_decomposition_prompt(self, question: str, knowledge_case: str, 
                                domain: str, context: str = "") -> str:
        """
        Generate a decomposition prompt for Stage 1 KG generation.
        
        Args:
            question: High-level clinical question from expert DT
            knowledge_case: Example case from expert knowledge
            domain: Medical domain (e.g., 'cardiology', 'general')
            context: Additional context about the decision tree
            
        Returns:
            Formatted prompt for LLM
        """
        return self.templates['decomposition'].format(
            question=question,
            knowledge_case=knowledge_case,
            domain=domain,
            context=context or f"Clinical decision-making in {domain}"
        )
    
    def get_rag_prompt(self, sub_question: str, sub_case: str, 
                      patient_context: str, diagnosis_class: str) -> str:
        """
        Generate a RAG extraction prompt for Stage 2.
        
        Args:
            sub_question: Specific question to answer
            sub_case: Example case for context
            patient_context: Patient clinical data
            diagnosis_class: Clinical concept being evaluated
            
        Returns:
            Formatted prompt for information extraction
        """
        return self.templates['rag_extraction'].format(
            sub_question=sub_question,
            sub_case=sub_case,
            patient_context=patient_context,
            diagnosis_class=diagnosis_class
        )
    
    def get_decision_prompt(self, node, evidence_chain: List[Dict] = None,
                           classifier_outputs: Dict = None, patient_context: Dict = None) -> str:
        """
        Generate a decision-making prompt for Stage 3 execution.
        
        Args:
            node: KGNode with question and evidence
            evidence_chain: Previous reasoning steps
            classifier_outputs: Classifier predictions
            patient_context: Patient data context
            
        Returns:
            Formatted decision prompt
        """
        # Get evidence chain summary
        previous_steps = ""
        if evidence_chain:
            previous_steps = "\n".join([
                f"- {step['question']}: {step['decision']} (confidence: {step['confidence']:.2f})"
                for step in evidence_chain[-3:]  # Last 3 steps
            ])
        
        # Get classifier summary
        classifier_summary = ""
        if classifier_outputs:
            for classifier, outputs in classifier_outputs.items():
                top_predictions = sorted(outputs.items(), key=lambda x: x[1], reverse=True)[:3]
                classifier_summary += f"\n{classifier}: {dict(top_predictions)}"
        
        # Get patient context
        patient_summary = ""
        if patient_context:
            patient_summary = str(patient_context).get('clinical_text', '')[:200] + "..."
        
        return f"""You are a clinical reasoning expert making diagnostic decisions.

Current Question: {node.sub_question}
Available Evidence: {node.sub_description}
Diagnosis Class: {node.diagnosis_class}

Previous Reasoning Steps:
{previous_steps or "None (this is the first step)"}

AI Classifier Predictions:
{classifier_summary or "No classifier outputs available"}

Patient Context: {patient_summary}

Based on the evidence and previous reasoning, make a decision for this step.

Format your response as:
DECISION: [your decision - can be a medical conclusion, next step, or "continue"]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation of your decision]
"""

    def get_final_diagnosis_prompt(self, leaf_node, evidence_chain: List[Dict] = None,
                                  classifier_outputs: Dict = None, patient_data: Dict = None) -> str:
        """
        Generate final diagnosis prompt for Stage 3.
        
        Args:
            leaf_node: Final KG node reached
            evidence_chain: Complete reasoning chain
            classifier_outputs: All classifier results
            patient_data: Patient information
            
        Returns:
            Formatted final diagnosis prompt
        """
        # Summarize evidence chain
        reasoning_summary = ""
        if evidence_chain:
            reasoning_summary = "\n".join([
                f"{i+1}. {step['question']}: {step['decision']} (confidence: {step['confidence']:.2f})"
                for i, step in enumerate(evidence_chain)
            ])
        
        # Summarize classifier outputs
        classifier_summary = ""
        if classifier_outputs:
            for classifier, outputs in classifier_outputs.items():
                top_prediction = max(outputs.items(), key=lambda x: x[1]) if outputs else ("none", 0.0)
                classifier_summary += f"\n- {classifier}: {top_prediction[0]} ({top_prediction[1]:.2f})"
        
        return f"""You are a clinical expert providing a final diagnostic conclusion.

Diagnostic Category: {leaf_node.diagnosis_class}
Final Evidence: {leaf_node.sub_description}

Complete Reasoning Chain:
{reasoning_summary}

AI Diagnostic Support:
{classifier_summary or "No AI classifier results"}

Patient Information: {patient_data.get('clinical_text', 'Limited information') if patient_data else 'Not available'}

Provide a comprehensive final diagnosis that synthesizes all available evidence.
Include:
1. Primary diagnosis with confidence level
2. Key supporting evidence
3. Any uncertainties or recommendations for further evaluation
4. Clinical reasoning summary

Format as a clear, professional medical assessment.
"""

    def get_narrative_prompt(self, decision_path: List[str], knowledge_graph,
                           evidence_chain: List[Dict], classifier_outputs: Dict,
                           patient_data: Dict = None) -> str:
        """
        Generate narrative report prompt for Stage 3.
        
        Args:
            decision_path: Sequence of nodes traversed
            knowledge_graph: KG used for reasoning
            evidence_chain: Detailed reasoning steps
            classifier_outputs: AI system results
            patient_data: Patient information
            
        Returns:
            Formatted narrative prompt
        """
        # Create reasoning pathway description
        pathway_description = " → ".join(decision_path)
        
        # Summarize key evidence
        key_evidence = []
        if evidence_chain:
            for step in evidence_chain:
                if step['confidence'] > 0.7:  # High confidence evidence
                    key_evidence.append(f"- {step['question']}: {step['decision']}")
        
        return f"""You are creating a comprehensive diagnostic report for medical professionals.

Domain: {knowledge_graph.domain}
Decision Pathway: {pathway_description}

High-Confidence Clinical Findings:
{chr(10).join(key_evidence) if key_evidence else "No high-confidence findings identified"}

Complete Evidence Analysis:
{chr(10).join([f"Step {i+1}: {step['question']} → {step['decision']} (confidence: {step['confidence']:.2f})" for i, step in enumerate(evidence_chain)])}

AI Diagnostic Support Summary:
{chr(10).join([f"{classifier}: {max(outputs.items(), key=lambda x: x[1])[0] if outputs else 'No predictions'}" for classifier, outputs in classifier_outputs.items()])}

Create a professional medical narrative report that:
1. Summarizes the clinical presentation
2. Explains the diagnostic reasoning process
3. Discusses key evidence and decision points
4. Integrates AI system findings appropriately
5. Provides clear conclusions and recommendations

Write in a professional medical tone suitable for clinical documentation.
"""
    
    def get_novel_info_prompt(self, patient_context: str, domain: str) -> str:
        """
        Generate prompt for identifying novel clinical information.
        Used in Stage 2 for dynamic knowledge base updates.
        """
        return f"""
You are a clinical knowledge analyst identifying novel or unusual clinical patterns.

MEDICAL DOMAIN: {domain}
PATIENT CONTEXT:
{patient_context}

TASK: Analyze the patient data for:
1. Unusual clinical presentations
2. Rare symptom combinations  
3. Novel diagnostic patterns
4. Information that doesn't fit standard clinical categories

OUTPUT:
NOVEL_INFORMATION: [Yes/No - whether novel patterns are present]
PATTERNS: [Description of any unusual patterns found]
CLINICAL_SIGNIFICANCE: [Why these patterns are important]
RECOMMENDATIONS: [How to incorporate this into knowledge base]

Analyze now:
"""