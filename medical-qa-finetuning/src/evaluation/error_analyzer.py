# ============================================================================
# SECTION 9: ERROR ANALYSIS
# ============================================================================

class ErrorAnalyzer:
    """
    Comprehensive error analysis with pattern identification
    Implements detailed error categorization and improvement suggestions
    """
    
    def __init__(self, model, tokenizer, test_dataset, original_responses):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.original_responses = original_responses
        self.errors = []
        self.error_patterns = Counter()
        
    def analyze_errors(self) -> Dict:
        """
        Comprehensive error analysis with pattern detection
        
        Returns:
            Dictionary containing error analysis results
        """
        logger.info("🔍 Starting error analysis...")
        
        # Analyze a subset of test examples
        num_examples = min(200, len(self.test_dataset))
        
        for i in range(num_examples):
            example = self.test_dataset[i]
            reference = self.original_responses['test'][i] if i < len(self.original_responses['test']) else ""
            
            # Generate prediction
            prediction = self._generate_prediction(example)
            
            # Analyze error
            error_info = self._analyze_single_error(prediction, reference, example)
            
            if error_info['is_error']:
                self.errors.append(error_info)
                self.error_patterns[error_info['error_type']] += 1
        
        # Identify patterns and generate suggestions
        patterns = self._identify_error_patterns()
        suggestions = self._generate_improvement_suggestions()
        
        # Create confusion matrix
        cm = self._create_error_confusion_matrix()
        
        results = {
            'total_errors': len(self.errors),
            'error_rate': len(self.errors) / num_examples if num_examples > 0 else 0,
            'error_distribution': dict(self.error_patterns),
            'patterns': patterns,
            'suggestions': suggestions,
            'confusion_matrix': cm
        }
        
        # Save error analysis
        with open('error_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Error analysis complete. Found {len(self.errors)} errors")
        
        return results
    
    def _generate_prediction(self, example: Dict) -> str:
        """Generate prediction for a single example"""
        instruction = example.get('instruction', '')
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in generated:
            return generated.split("### Response:")[1].strip()
        
        return generated
    
    def _analyze_single_error(self, prediction: str, reference: str, example: Dict) -> Dict:
        """Analyze a single prediction for errors"""
        error_info = {
            'instruction': example.get('instruction', ''),
            'prediction': prediction,
            'reference': reference,
            'is_error': False,
            'error_type': None,
            'error_severity': 0
        }
        
        # Determine error type
        if len(prediction) < 20:
            error_info['is_error'] = True
            error_info['error_type'] = 'too_short'
            error_info['error_severity'] = 3
        elif len(prediction) > len(reference) * 2 and len(reference) > 0:
            error_info['is_error'] = True
            error_info['error_type'] = 'too_verbose'
            error_info['error_severity'] = 2
        elif 'Answer:' not in prediction and 'Answer:' in reference:
            error_info['is_error'] = True
            error_info['error_type'] = 'format_error'
            error_info['error_severity'] = 2
        elif self._check_medical_accuracy(prediction, reference) < 0.5:
            error_info['is_error'] = True
            error_info['error_type'] = 'content_error'
            error_info['error_severity'] = 4
        
        return error_info
    
    def _check_medical_accuracy(self, prediction: str, reference: str) -> float:
        """Check medical accuracy of prediction"""
        if not reference:
            return 1.0
        
        # Extract medical terms from reference
        medical_terms = re.findall(
            r'\b(diagnosis|treatment|symptom|medication|disease|condition|therapy|clinical)\b',
            reference.lower()
        )
        
        if not medical_terms:
            # Simple overlap check for non-medical content
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            overlap = len(pred_words & ref_words)
            return overlap / max(len(ref_words), 1)
        
        # Check how many medical terms are in prediction
        matches = sum(1 for term in medical_terms if term in prediction.lower())
        return matches / len(medical_terms)
    
    def _identify_error_patterns(self) -> List[str]:
        """Identify patterns in errors"""
        patterns = []
        
        if self.error_patterns['too_short'] > len(self.errors) * 0.3:
            patterns.append("Model frequently generates responses that are too brief")
        
        if self.error_patterns['too_verbose'] > len(self.errors) * 0.3:
            patterns.append("Model tends to be overly verbose in responses")
        
        if self.error_patterns['format_error'] > len(self.errors) * 0.2:
            patterns.append("Model struggles with maintaining proper response format")
        
        if self.error_patterns['content_error'] > len(self.errors) * 0.3:
            patterns.append("Model has difficulty with medical content accuracy")
        
        # Check for specific medical error patterns
        medical_errors = [e for e in self.errors if e['error_type'] == 'content_error']
        if medical_errors:
            # Analyze common missing elements
            missing_elements = Counter()
            for error in medical_errors:
                if 'diagnosis' in error['reference'].lower() and 'diagnosis' not in error['prediction'].lower():
                    missing_elements['diagnosis'] += 1
                if 'treatment' in error['reference'].lower() and 'treatment' not in error['prediction'].lower():
                    missing_elements['treatment'] += 1
            
            for element, count in missing_elements.most_common(3):
                if count > len(medical_errors) * 0.2:
                    patterns.append(f"Model frequently misses {element} information")
        
        return patterns
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate targeted improvement suggestions"""
        suggestions = []
        
        # Based on error distribution
        if self.error_patterns['too_short'] > len(self.errors) * 0.3:
            suggestions.append(
                "Increase minimum generation length or adjust temperature to encourage more detailed responses"
            )
        
        if self.error_patterns['too_verbose'] > len(self.errors) * 0.3:
            suggestions.append(
                "Implement length penalty during generation or add conciseness examples to training data"
            )
        
        if self.error_patterns['format_error'] > len(self.errors) * 0.2:
            suggestions.append(
                "Add more format-specific training examples or use a structured generation template"
            )
        
        if self.error_patterns['content_error'] > len(self.errors) * 0.3:
            suggestions.append(
                "Increase medical domain training data or implement domain-specific fine-tuning strategies"
            )
        
        # General suggestions
        suggestions.extend([
            "Consider implementing beam search for more stable generation",
            "Add medical terminology post-processing to ensure accuracy",
            "Implement confidence scoring to identify uncertain predictions",
            "Use ensemble methods to combine multiple model predictions"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _create_error_confusion_matrix(self):
        """Create confusion matrix for error types"""
        # Simplified confusion matrix for error categories
        categories = ['correct', 'too_short', 'too_verbose', 'format_error', 'content_error']
        matrix = np.zeros((len(categories), len(categories)))
        
        # This is a simplified version - in production, you'd track actual vs predicted categories
        for error in self.errors:
            error_idx = categories.index(error['error_type']) if error['error_type'] in categories else 0
            # For simplicity, assuming 'correct' as the expected category
            matrix[0][error_idx] += 1
        
        # Add some correct predictions
        matrix[0][0] = max(len(self.test_dataset) - len(self.errors), 0)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        plt.title('Error Type Confusion Matrix')
        plt.ylabel('Expected')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('error_confusion_matrix.png', dpi=300)
        
        return matrix.tolist()
