class MedicalSafetyValidator:
    """
    Production-grade medical safety validation system
    Implements comprehensive drug interaction checking, emergency detection,
    and clinical guideline compliance
    """
    
    def __init__(self):
        # Comprehensive drug interaction database
        self.drug_interactions = {
            'warfarin': ['aspirin', 'ibuprofen', 'nsaids', 'vitamin k', 'amiodarone'],
            'metformin': ['contrast dye', 'alcohol', 'cimetidine', 'furosemide'],
            'ssri': ['maoi', 'tramadol', 'linezolid', 'st johns wort'],
            'ace_inhibitor': ['potassium', 'nsaids', 'lithium', 'aliskiren'],
            'statins': ['grapefruit', 'gemfibrozil', 'niacin', 'cyclosporine'],
            'beta_blockers': ['verapamil', 'diltiazem', 'clonidine'],
            'digoxin': ['quinidine', 'verapamil', 'amiodarone', 'spironolactone']
        }
        
        # Critical emergency indicators
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe bleeding',
            'stroke', 'unconscious', 'seizure', 'anaphylaxis',
            'heart attack', 'severe headache', 'vision loss',
            'confusion', 'weakness', 'numbness', 'paralysis'
        ]
        
        # FDA-approved dosage ranges
        self.dosage_ranges = {
            'aspirin': (81, 325),
            'metformin': (500, 2550),
            'lisinopril': (5, 40),
            'atorvastatin': (10, 80),
            'levothyroxine': (25, 300),
            'amlodipine': (2.5, 10),
            'metoprolol': (25, 400)
        }
        
        # Clinical red flags
        self.clinical_red_flags = [
            'sudden onset', 'worst ever', 'progressive', 'bilateral',
            'weight loss', 'night sweats', 'fever', 'bleeding'
        ]
    
    def validate_medical_response(self, response: str, patient_context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Comprehensive medical validation with risk stratification
        
        Args:
            response: Generated medical response
            patient_context: Optional patient-specific information
            
        Returns:
            Validated response with safety disclaimers and validation report
        """
        validations = {
            'drug_safety': self._check_drug_interactions(response),
            'emergency_detection': self._detect_emergencies(response),
            'dosage_accuracy': self._verify_dosages(response),
            'contraindications': self._check_contraindications(response, patient_context),
            'clinical_red_flags': self._detect_red_flags(response)
        }
        
        # Calculate aggregate risk score
        risk_score = self._calculate_risk_score(validations)
        
        # Add appropriate disclaimers based on risk level
        if risk_score >= 3:
            response = self._add_safety_disclaimer(response, risk_score)
        
        # Add clinical decision support
        if risk_score >= 4:
            response = self._add_clinical_guidelines(response, validations)
            
        return response, validations
    
    def _check_drug_interactions(self, response: str) -> Dict:
        """Check for potential drug interactions"""
        response_lower = response.lower()
        interactions_found = []
        severity_scores = []
        
        for drug, interactions in self.drug_interactions.items():
            if drug in response_lower:
                for interaction in interactions:
                    if interaction in response_lower:
                        interactions_found.append(f"{drug} + {interaction}")
                        # Assign severity based on known interaction severity
                        severity = self._get_interaction_severity(drug, interaction)
                        severity_scores.append(severity)
        
        return {
            'interactions': interactions_found,
            'risk_level': max(severity_scores) if severity_scores else 0,
            'count': len(interactions_found),
            'recommendation': self._get_interaction_recommendation(interactions_found)
        }
    
    def _get_interaction_severity(self, drug1: str, drug2: str) -> int:
        """Determine interaction severity (1-5 scale)"""
        # Critical interactions (level 5)
        critical_pairs = [('warfarin', 'aspirin'), ('ssri', 'maoi'), ('digoxin', 'quinidine')]
        for pair in critical_pairs:
            if drug1 in pair and drug2 in pair:
                return 5
        
        # Major interactions (level 4)
        major_pairs = [('ace_inhibitor', 'potassium'), ('statins', 'gemfibrozil')]
        for pair in major_pairs:
            if drug1 in pair and drug2 in pair:
                return 4
                
        # Moderate interactions (level 3)
        return 3
    
    def _detect_emergencies(self, response: str) -> Dict:
        """Detect emergency conditions requiring immediate attention"""
        response_lower = response.lower()
        emergencies = []
        
        for keyword in self.emergency_keywords:
            if keyword in response_lower:
                emergencies.append(keyword)
                # Check for qualifying terms that increase urgency
                if any(qualifier in response_lower for qualifier in ['severe', 'sudden', 'acute']):
                    emergencies.append(f"URGENT: {keyword}")
        
        return {
            'emergencies': emergencies,
            'risk_level': 5 if emergencies else 0,
            'action': 'CALL 911 IMMEDIATELY' if emergencies else None,
            'triage_category': self._determine_triage_category(emergencies)
        }
    
    def _determine_triage_category(self, emergencies: List[str]) -> str:
        """Assign triage category based on emergency type"""
        if not emergencies:
            return 'routine'
        
        critical_emergencies = ['chest pain', 'difficulty breathing', 'unconscious', 'stroke']
        if any(em in emergencies for em in critical_emergencies):
            return 'critical - immediate'
        
        return 'urgent - within 1 hour'
    
    def _verify_dosages(self, response: str) -> Dict:
        """Verify medication dosages are within safe ranges"""
        import re
        dosage_issues = []
        recommendations = []
        
        for drug, (min_dose, max_dose) in self.dosage_ranges.items():
            # More sophisticated regex to catch various dosage formats
            patterns = [
                rf"{drug}.*?(\d+\.?\d*)\s*mg",
                rf"(\d+\.?\d*)\s*mg.*?{drug}",
                rf"{drug}.*?(\d+\.?\d*)\s*milligrams"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response.lower())
                for match in matches:
                    try:
                        dose = float(match)
                        if dose < min_dose:
                            issue = f"{drug}: {dose}mg (below minimum {min_dose}mg)"
                            dosage_issues.append(issue)
                            recommendations.append(f"Consider increasing {drug} dose")
                        elif dose > max_dose:
                            issue = f"{drug}: {dose}mg (exceeds maximum {max_dose}mg)"
                            dosage_issues.append(issue)
                            recommendations.append(f"Reduce {drug} dose to safe range")
                    except ValueError:
                        continue
        
        return {
            'dosage_issues': dosage_issues,
            'risk_level': 4 if dosage_issues else 0,
            'recommendations': recommendations
        }
    
    def _check_contraindications(self, response: str, patient_context: Optional[Dict]) -> Dict:
        """Check for medication contraindications based on patient context"""
        contraindications = []
        recommendations = []
        
        if not patient_context:
            return {'contraindications': [], 'risk_level': 0, 'recommendations': []}
        
        response_lower = response.lower()
        
        # Pregnancy contraindications
        if patient_context.get('pregnant'):
            pregnancy_contraindicated = [
                'methotrexate', 'warfarin', 'isotretinoin', 'valproic acid',
                'lithium', 'ace inhibitor', 'nsaid'
            ]
            for drug in pregnancy_contraindicated:
                if drug in response_lower:
                    contraindications.append(f"{drug} - Category X in pregnancy")
                    recommendations.append(f"Replace {drug} with pregnancy-safe alternative")
        
        # Renal impairment contraindications
        if patient_context.get('kidney_disease'):
            renal_risk_drugs = ['nsaids', 'metformin', 'lithium', 'digoxin']
            for drug in renal_risk_drugs:
                if drug in response_lower:
                    contraindications.append(f"{drug} - requires renal dose adjustment")
                    recommendations.append(f"Adjust {drug} dose for GFR")
        
        # Hepatic impairment contraindications
        if patient_context.get('liver_disease'):
            hepatic_risk_drugs = ['acetaminophen', 'statins', 'methotrexate']
            for drug in hepatic_risk_drugs:
                if drug in response_lower:
                    contraindications.append(f"{drug} - use with caution in liver disease")
                    recommendations.append(f"Monitor LFTs if using {drug}")
        
        return {
            'contraindications': contraindications,
            'risk_level': 4 if contraindications else 0,
            'recommendations': recommendations
        }
    
    def _detect_red_flags(self, response: str) -> Dict:
        """Detect clinical red flags requiring further investigation"""
        response_lower = response.lower()
        red_flags = [flag for flag in self.clinical_red_flags if flag in response_lower]
        
        return {
            'red_flags': red_flags,
            'risk_level': 3 if red_flags else 0,
            'action': 'Requires clinical evaluation' if red_flags else None
        }
    
    def _calculate_risk_score(self, validations: Dict) -> int:
        """Calculate aggregate risk score from all validations"""
        scores = [v.get('risk_level', 0) for v in validations.values()]
        return max(scores) if scores else 0
    
    def _add_safety_disclaimer(self, response: str, risk_level: int) -> str:
        """Add appropriate safety disclaimers based on risk level"""
        disclaimers = {
            3: "\n\n⚠️ **MEDICAL NOTICE**: This information is for educational purposes only. Please consult your healthcare provider before making medical decisions.",
            4: "\n\n⚠️ **WARNING**: Potential drug interactions or contraindications detected. Consult your healthcare provider immediately for personalized medical advice.",
            5: "\n\n🚨 **EMERGENCY WARNING**: If experiencing these symptoms, seek immediate medical attention or call 911. Do not delay emergency care."
        }
        
        disclaimer = disclaimers.get(risk_level, "")
        return response + disclaimer
    
    def _add_clinical_guidelines(self, response: str, validations: Dict) -> str:
        """Add relevant clinical guidelines and recommendations"""
        guidelines = "\n\n📋 **Clinical Recommendations**:\n"
        
        for validation_type, results in validations.items():
            if results.get('recommendations'):
                guidelines += f"• {', '.join(results['recommendations'])}\n"
        
        return response + guidelines if guidelines != "\n\n📋 **Clinical Recommendations**:\n" else response
    
    def _get_interaction_recommendation(self, interactions: List[str]) -> str:
        """Generate specific recommendations for drug interactions"""
        if not interactions:
            return "No significant interactions detected"
        
        recommendations = []
        for interaction in interactions:
            if 'warfarin' in interaction:
                recommendations.append("Monitor INR closely")
            if 'ssri' in interaction and 'maoi' in interaction:
                recommendations.append("Contraindicated - risk of serotonin syndrome")
            if 'statin' in interaction:
                recommendations.append("Monitor for myopathy")
                
        return "; ".join(recommendations) if recommendations else "Consult pharmacist for interaction management"
