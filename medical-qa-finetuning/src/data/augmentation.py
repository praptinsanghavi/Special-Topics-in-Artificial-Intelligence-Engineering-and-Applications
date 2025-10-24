class MedicalDataAugmenter:
    """
    Advanced data augmentation for medical QA
    Implements paraphrasing, synonym replacement, and context injection
    """
    
    def __init__(self):
        # Comprehensive medical synonym database
        self.medical_synonyms = {
            'hypertension': ['high blood pressure', 'HTN', 'elevated BP', 'raised blood pressure'],
            'diabetes': ['diabetes mellitus', 'DM', 'high blood sugar', 'hyperglycemia', 'sugar disease'],
            'myocardial infarction': ['heart attack', 'MI', 'acute MI', 'cardiac infarction'],
            'pneumonia': ['lung infection', 'chest infection', 'lower respiratory infection'],
            'stroke': ['cerebrovascular accident', 'CVA', 'brain attack', 'cerebral infarction'],
            'COPD': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
            'cancer': ['malignancy', 'neoplasm', 'tumor', 'carcinoma'],
            'infection': ['sepsis', 'bacteremia', 'infectious disease'],
            'pain': ['discomfort', 'ache', 'soreness', 'tenderness'],
            'fever': ['pyrexia', 'elevated temperature', 'hyperthermia', 'febrile']
        }
        
        # Question templates for augmentation
        self.question_templates = [
            "What are the symptoms of {condition}?",
            "How is {condition} diagnosed?",
            "What causes {condition}?",
            "What are treatment options for {condition}?",
            "What are complications of {condition}?",
            "How to prevent {condition}?",
            "What is the prognosis for {condition}?",
            "What are risk factors for {condition}?"
        ]
    
    def augment_dataset(self, qa_pairs: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """
        Augment QA pairs with multiple strategies
        
        Args:
            qa_pairs: Original QA pairs
            augmentation_factor: How many augmented versions to create
            
        Returns:
            Augmented dataset
        """
        augmented_data = []
        
        for pair in qa_pairs:
            # Add original
            augmented_data.append(pair)
            
            # Generate augmented versions
            for i in range(augmentation_factor - 1):
                augmented_pair = self._augment_single_pair(pair, strategy=i % 3)
                augmented_data.append(augmented_pair)
        
        logger.info(f"Augmented {len(qa_pairs)} pairs to {len(augmented_data)} pairs")
        return augmented_data
    
    def _augment_single_pair(self, pair: Dict, strategy: int = 0) -> Dict:
        """Apply augmentation strategy to a single QA pair"""
        augmented = pair.copy()
        
        if strategy == 0:
            # Synonym replacement
            augmented = self._apply_synonym_replacement(augmented)
        elif strategy == 1:
            # Paraphrasing
            augmented = self._apply_paraphrasing(augmented)
        else:
            # Context injection
            augmented = self._apply_context_injection(augmented)
        
        return augmented
    
    def _apply_synonym_replacement(self, pair: Dict) -> Dict:
        """Replace medical terms with synonyms"""
        text = pair.get('instruction', '') + ' ' + pair.get('response', '')
        
        for term, synonyms in self.medical_synonyms.items():
            if term.lower() in text.lower():
                synonym = random.choice(synonyms)
                # Use word boundaries for accurate replacement
                text = re.sub(rf'\b{term}\b', synonym, text, flags=re.IGNORECASE)
        
        # Split back into instruction and response
        if 'Answer:' in text:
            parts = text.split('Answer:', 1)
            if len(parts) == 2:
                pair['instruction'] = parts[0].strip()
                pair['response'] = 'Answer: ' + parts[1].strip()
        
        return pair
    
    def _apply_paraphrasing(self, pair: Dict) -> Dict:
        """Apply template-based paraphrasing"""
        instruction = pair.get('instruction', '')
        
        # Extract condition if present
        condition_match = re.search(r'about (\w+)', instruction.lower())
        if condition_match:
            condition = condition_match.group(1)
            # Use a random template
            new_question = random.choice(self.question_templates).format(condition=condition)
            pair['instruction'] = f"Medical Question: {new_question}"
        
        return pair
    
    def _apply_context_injection(self, pair: Dict) -> Dict:
        """Inject clinical context into questions"""
        contexts = [
            "In a clinical setting, ",
            "For patient education, ",
            "From a medical perspective, ",
            "In emergency situations, ",
            "For differential diagnosis, "
        ]
        
        instruction = pair.get('instruction', '')
        if 'Medical Question:' in instruction:
            instruction = instruction.replace('Medical Question:', 
                                            f"Medical Question: {random.choice(contexts)}")
            pair['instruction'] = instruction
        
        return pair
