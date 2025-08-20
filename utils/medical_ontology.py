"""
Medical Ontology and Standard Mappings
Provides mappings between diagnostic terms and medical standards (ICD-10, SNOMED-CT, etc.)
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class MedicalStandard(Enum):
    """Supported medical coding standards."""
    ICD10 = "ICD-10"
    SNOMED_CT = "SNOMED-CT"
    SCP_ECG = "SCP-ECG"
    LOINC = "LOINC"

@dataclass
class MedicalCode:
    """Represents a medical code in a specific standard."""
    code: str
    description: str
    standard: MedicalStandard
    hierarchy_level: int = 0
    parent_codes: List[str] = None
    
    def __post_init__(self):
        if self.parent_codes is None:
            self.parent_codes = []

class MedicalOntologyMapper:
    """
    Maps between diagnostic terms and standard medical codes.
    Provides hierarchical relationships for clinical reasoning.
    """
    
    def __init__(self):
        """Initialize the medical ontology mapper."""
        self.code_mappings = self._initialize_code_mappings()
        self.term_to_codes = self._build_term_mappings()
        self.hierarchies = self._build_hierarchies()
    
    def _initialize_code_mappings(self) -> Dict[str, MedicalCode]:
        """Initialize standard medical code mappings."""
        codes = {}
        
        # ICD-10 Cardiovascular codes
        cardiovascular_codes = [
            # Main categories
            MedicalCode("I00-I99", "Diseases of the circulatory system", MedicalStandard.ICD10, 0),
            MedicalCode("I20-I25", "Ischaemic heart diseases", MedicalStandard.ICD10, 1, ["I00-I99"]),
            MedicalCode("I30-I52", "Other forms of heart disease", MedicalStandard.ICD10, 1, ["I00-I99"]),
            MedicalCode("I44-I49", "Conduction disorders and cardiac arrhythmias", MedicalStandard.ICD10, 1, ["I00-I99"]),
            
            # Specific conditions
            MedicalCode("I21", "Acute myocardial infarction", MedicalStandard.ICD10, 2, ["I20-I25"]),
            MedicalCode("I21.0", "Acute transmural myocardial infarction of anterior wall", MedicalStandard.ICD10, 3, ["I21"]),
            MedicalCode("I21.1", "Acute transmural myocardial infarction of inferior wall", MedicalStandard.ICD10, 3, ["I21"]),
            MedicalCode("I21.2", "Acute transmural myocardial infarction of other sites", MedicalStandard.ICD10, 3, ["I21"]),
            
            # Arrhythmias
            MedicalCode("I48", "Atrial fibrillation and flutter", MedicalStandard.ICD10, 2, ["I44-I49"]),
            MedicalCode("I48.0", "Paroxysmal atrial fibrillation", MedicalStandard.ICD10, 3, ["I48"]),
            MedicalCode("I48.1", "Persistent atrial fibrillation", MedicalStandard.ICD10, 3, ["I48"]),
            MedicalCode("I48.2", "Chronic atrial fibrillation", MedicalStandard.ICD10, 3, ["I48"]),
            
            # Conduction disorders
            MedicalCode("I44", "Atrioventricular and left bundle-branch block", MedicalStandard.ICD10, 2, ["I44-I49"]),
            MedicalCode("I44.0", "Atrioventricular block, first degree", MedicalStandard.ICD10, 3, ["I44"]),
            MedicalCode("I44.1", "Atrioventricular block, second degree", MedicalStandard.ICD10, 3, ["I44"]),
            MedicalCode("I44.2", "Atrioventricular block, complete", MedicalStandard.ICD10, 3, ["I44"]),
            MedicalCode("I45", "Other conduction disorders", MedicalStandard.ICD10, 2, ["I44-I49"]),
            MedicalCode("I45.0", "Right fascicular block", MedicalStandard.ICD10, 3, ["I45"]),
            MedicalCode("I45.1", "Left fascicular block", MedicalStandard.ICD10, 3, ["I45"]),
        ]
        
        # SCP-ECG diagnostic codes (based on PTB-XL dataset)
        scp_ecg_codes = [
            # Superclasses
            MedicalCode("NORM", "Normal ECG", MedicalStandard.SCP_ECG, 0),
            MedicalCode("MI", "Myocardial Infarction", MedicalStandard.SCP_ECG, 0),
            MedicalCode("CD", "Conduction Disturbance", MedicalStandard.SCP_ECG, 0),
            MedicalCode("STTC", "ST/T Change", MedicalStandard.SCP_ECG, 0),
            MedicalCode("HYP", "Hypertrophy", MedicalStandard.SCP_ECG, 0),
            
            # MI subcategories
            MedicalCode("AMI", "Anterior Myocardial Infarction", MedicalStandard.SCP_ECG, 1, ["MI"]),
            MedicalCode("IMI", "Inferior Myocardial Infarction", MedicalStandard.SCP_ECG, 1, ["MI"]),
            MedicalCode("LMI", "Lateral Myocardial Infarction", MedicalStandard.SCP_ECG, 1, ["MI"]),
            MedicalCode("PMI", "Posterior Myocardial Infarction", MedicalStandard.SCP_ECG, 1, ["MI"]),
            
            # Conduction disturbances
            MedicalCode("1AVB", "First degree AV block", MedicalStandard.SCP_ECG, 1, ["CD"]),
            MedicalCode("2AVB", "Second degree AV block", MedicalStandard.SCP_ECG, 1, ["CD"]),
            MedicalCode("3AVB", "Third degree AV block", MedicalStandard.SCP_ECG, 1, ["CD"]),
            MedicalCode("LBBB", "Left bundle branch block", MedicalStandard.SCP_ECG, 1, ["CD"]),
            MedicalCode("RBBB", "Right bundle branch block", MedicalStandard.SCP_ECG, 1, ["CD"]),
            
            # Hypertrophy
            MedicalCode("LVH", "Left ventricular hypertrophy", MedicalStandard.SCP_ECG, 1, ["HYP"]),
            MedicalCode("RVH", "Right ventricular hypertrophy", MedicalStandard.SCP_ECG, 1, ["HYP"]),
            MedicalCode("LAO", "Left atrial overload", MedicalStandard.SCP_ECG, 1, ["HYP"]),
            MedicalCode("RAO", "Right atrial overload", MedicalStandard.SCP_ECG, 1, ["HYP"]),
        ]
        
        # Add all codes to mapping
        for code_list in [cardiovascular_codes, scp_ecg_codes]:
            for code in code_list:
                codes[f"{code.standard.value}:{code.code}"] = code
        
        return codes
    
    def _build_term_mappings(self) -> Dict[str, List[str]]:
        """Build mappings from diagnostic terms to codes."""
        term_mappings = {}
        
        # Common term variations for cardiovascular conditions
        term_variations = {
            # Myocardial Infarction variations
            "myocardial infarction": ["ICD-10:I21", "SCP-ECG:MI"],
            "heart attack": ["ICD-10:I21", "SCP-ECG:MI"],
            "mi": ["ICD-10:I21", "SCP-ECG:MI"],
            "anterior mi": ["ICD-10:I21.0", "SCP-ECG:AMI"],
            "anterior myocardial infarction": ["ICD-10:I21.0", "SCP-ECG:AMI"],
            "inferior mi": ["ICD-10:I21.1", "SCP-ECG:IMI"],
            "inferior myocardial infarction": ["ICD-10:I21.1", "SCP-ECG:IMI"],
            
            # Atrial Fibrillation variations
            "atrial fibrillation": ["ICD-10:I48", "SCP-ECG:AFIB"],
            "afib": ["ICD-10:I48", "SCP-ECG:AFIB"],
            "a-fib": ["ICD-10:I48", "SCP-ECG:AFIB"],
            "paroxysmal atrial fibrillation": ["ICD-10:I48.0"],
            "persistent atrial fibrillation": ["ICD-10:I48.1"],
            "chronic atrial fibrillation": ["ICD-10:I48.2"],
            
            # AV Blocks
            "av block": ["ICD-10:I44", "SCP-ECG:CD"],
            "atrioventricular block": ["ICD-10:I44", "SCP-ECG:CD"],
            "first degree av block": ["ICD-10:I44.0", "SCP-ECG:1AVB"],
            "second degree av block": ["ICD-10:I44.1", "SCP-ECG:2AVB"],
            "complete heart block": ["ICD-10:I44.2", "SCP-ECG:3AVB"],
            "third degree av block": ["ICD-10:I44.2", "SCP-ECG:3AVB"],
            
            # Bundle Branch Blocks
            "bundle branch block": ["ICD-10:I45", "SCP-ECG:CD"],
            "left bundle branch block": ["ICD-10:I45.1", "SCP-ECG:LBBB"],
            "lbbb": ["ICD-10:I45.1", "SCP-ECG:LBBB"],
            "right bundle branch block": ["ICD-10:I45.0", "SCP-ECG:RBBB"],
            "rbbb": ["ICD-10:I45.0", "SCP-ECG:RBBB"],
            
            # Hypertrophy
            "left ventricular hypertrophy": ["SCP-ECG:LVH"],
            "lvh": ["SCP-ECG:LVH"],
            "right ventricular hypertrophy": ["SCP-ECG:RVH"],
            "rvh": ["SCP-ECG:RVH"],
            
            # Normal
            "normal": ["SCP-ECG:NORM"],
            "normal ecg": ["SCP-ECG:NORM"],
            "normal sinus rhythm": ["SCP-ECG:NORM"],
        }
        
        # Add variations and synonyms
        for term, codes in term_variations.items():
            term_mappings[term.lower()] = codes
        
        return term_mappings
    
    def _build_hierarchies(self) -> Dict[MedicalStandard, Dict[str, List[str]]]:
        """Build hierarchical relationships for each standard."""
        hierarchies = {}
        
        for standard in MedicalStandard:
            hierarchy = {}
            
            # Get all codes for this standard
            standard_codes = [
                code for code in self.code_mappings.values() 
                if code.standard == standard
            ]
            
            # Build parent-child relationships
            for code in standard_codes:
                code_id = code.code
                children = [
                    child.code for child in standard_codes
                    if code_id in child.parent_codes
                ]
                if children:
                    hierarchy[code_id] = children
            
            hierarchies[standard] = hierarchy
        
        return hierarchies
    
    def map_term_to_codes(self, term: str) -> List[MedicalCode]:
        """
        Map a diagnostic term to medical codes.
        
        Args:
            term: Diagnostic term (e.g., "atrial fibrillation")
            
        Returns:
            List of matching medical codes
        """
        term_lower = term.lower().strip()
        
        if term_lower in self.term_to_codes:
            code_ids = self.term_to_codes[term_lower]
            return [self.code_mappings[code_id] for code_id in code_ids if code_id in self.code_mappings]
        
        # Fuzzy matching for partial terms
        matches = []
        for mapped_term, code_ids in self.term_to_codes.items():
            if term_lower in mapped_term or mapped_term in term_lower:
                matches.extend([self.code_mappings[code_id] for code_id in code_ids if code_id in self.code_mappings])
        
        return matches
    
    def get_hierarchy_for_standard(self, standard: MedicalStandard) -> Dict[str, List[str]]:
        """Get hierarchical relationships for a specific standard."""
        return self.hierarchies.get(standard, {})
    
    def get_parent_codes(self, code: str, standard: MedicalStandard) -> List[str]:
        """Get parent codes for a given code."""
        full_code_id = f"{standard.value}:{code}"
        if full_code_id in self.code_mappings:
            medical_code = self.code_mappings[full_code_id]
            return medical_code.parent_codes
        return []
    
    def get_child_codes(self, code: str, standard: MedicalStandard) -> List[str]:
        """Get child codes for a given code."""
        hierarchy = self.hierarchies.get(standard, {})
        return hierarchy.get(code, [])
    
    def validate_hierarchy_consistency(self, predicted_codes: List[str], 
                                     standard: MedicalStandard) -> Tuple[bool, List[str]]:
        """
        Validate that predicted codes are hierarchically consistent.
        
        Args:
            predicted_codes: List of predicted diagnostic codes
            standard: Medical standard to validate against
            
        Returns:
            Tuple of (is_consistent, list_of_errors)
        """
        errors = []
        
        for code in predicted_codes:
            parent_codes = self.get_parent_codes(code, standard)
            
            # Check if all required parent codes are present
            for parent_code in parent_codes:
                if parent_code not in predicted_codes:
                    errors.append(f"Code {code} requires parent {parent_code} but it's not predicted")
        
        return len(errors) == 0, errors
    
    def get_code_description(self, code: str, standard: MedicalStandard) -> Optional[str]:
        """Get description for a medical code."""
        full_code_id = f"{standard.value}:{code}"
        if full_code_id in self.code_mappings:
            return self.code_mappings[full_code_id].description
        return None
    
    def get_scp_ecg_hierarchy(self) -> Dict[str, List[str]]:
        """Get the PTB-XL SCP-ECG diagnostic hierarchy."""
        return {
            "MI": ["AMI", "IMI", "LMI", "PMI", "ALMI", "IPMI", "ILMI"],
            "CD": ["1AVB", "2AVB", "3AVB", "LBBB", "RBBB", "CLBBB", "CRBBB", "ILBBB", "IRBBB", "LAFB", "LPFB", "WPW", "IVCD"],
            "STTC": ["ISCA", "ISCI", "ISCIN", "ISCIL", "ISCAL", "ISCAS", "ISC_", "NST_", "STD_", "STE_", "STTC"],
            "HYP": ["LVH", "RVH", "LAO", "RAO", "LAE", "RAE", "SEHYP"]
        }

# Global instance for easy access
_global_mapper = None

def get_medical_ontology_mapper() -> MedicalOntologyMapper:
    """Get or create global medical ontology mapper."""
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = MedicalOntologyMapper()
    return _global_mapper

# Convenience functions
def map_diagnosis_to_icd10(diagnosis: str) -> List[str]:
    """Map diagnosis term to ICD-10 codes."""
    mapper = get_medical_ontology_mapper()
    codes = mapper.map_term_to_codes(diagnosis)
    return [code.code for code in codes if code.standard == MedicalStandard.ICD10]

def map_diagnosis_to_scp_ecg(diagnosis: str) -> List[str]:
    """Map diagnosis term to SCP-ECG codes."""
    mapper = get_medical_ontology_mapper()
    codes = mapper.map_term_to_codes(diagnosis)
    return [code.code for code in codes if code.standard == MedicalStandard.SCP_ECG]

def validate_scp_ecg_predictions(predictions: List[str]) -> Tuple[bool, List[str]]:
    """Validate SCP-ECG predictions for hierarchical consistency."""
    mapper = get_medical_ontology_mapper()
    return mapper.validate_hierarchy_consistency(predictions, MedicalStandard.SCP_ECG)