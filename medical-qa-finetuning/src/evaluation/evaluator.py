class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics
    Implements thorough evaluation including baseline comparison
    """
    
    def __init__(self, model, tokenizer, test_dataset, original_responses):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.original_responses = original_responses
        self.metrics = {}
        self.predictions = []
        
    def evaluate_model(self) -> Dict:
        """
        Comprehensive model evaluation with multiple metrics
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("📊 Starting comprehensive model evaluation")
        
        # Load evaluation metrics
        rouge = evaluate.load('rouge')
        bleu = evaluate.load('bleu')
        
        predictions = []
        references = self.original_responses.get('test', [])
        
        # Limit evaluation size for efficiency
        eval_size = min(100, len(self.test_dataset))
        references = references[:eval_size]
        
        # Generate predictions
        logger.info(f"Generating predictions for {eval_size} samples...")
        
        for i in range(eval_size):
            example = self.test_dataset[i]
            
            # Extract instruction
            instruction = example.get('instruction', '')
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=256
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "### Response:" in generated:
                prediction = generated.split("### Response:")[1].strip()
            else:
                prediction = generated
            
            predictions.append(prediction)
            self.predictions.append(prediction)
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        
        # ROUGE scores
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        
        # BLEU scores
        bleu_scores = bleu.compute(
            predictions=predictions, 
            references=[[r] for r in references]
        )
        
        # Medical accuracy (simple keyword matching for demo)
        medical_accuracy = self._calculate_medical_accuracy(predictions, references)
        
        # Perplexity
        perplexity = self._calculate_perplexity()
        
        # Store metrics
        self.metrics = {
            'rouge': rouge_scores,
            'bleu': bleu_scores['bleu'],
            'medical_accuracy': medical_accuracy,
            'perplexity': perplexity,
            'num_evaluated': eval_size
        }
        
        logger.info(f"✅ Evaluation complete: {self.metrics}")
        
        return self.metrics
    
    def _calculate_medical_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate medical domain-specific accuracy"""
        correct = 0
        
        for pred, ref in zip(predictions, references):
            # Simple keyword matching for medical terms
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Extract key medical terms
            medical_keywords = re.findall(r'\b(diagnosis|treatment|symptom|medication|disease)\b', 
                                        ref_lower)
            
            if medical_keywords:
                matches = sum(1 for keyword in medical_keywords if keyword in pred_lower)
                if matches >= len(medical_keywords) * 0.5:  # At least 50% match
                    correct += 1
            elif len(pred) > 10:  # Basic length check for non-medical responses
                correct += 1
        
        return correct / len(predictions) if predictions else 0
    
    def _calculate_perplexity(self) -> float:
        """Calculate model perplexity on test set"""
        # Simplified perplexity calculation
        total_loss = 0
        num_tokens = 0
        
        for i in range(min(50, len(self.test_dataset))):
            example = self.test_dataset[i]
            text = example.get('text', '')
            
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                num_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / max(1, num_tokens)
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def compare_with_baseline(self, baseline_model) -> Dict:
        """Compare performance with baseline model"""
        logger.info("📊 Comparing with baseline model...")
        
        # Evaluate baseline
        baseline_evaluator = ModelEvaluator(
            baseline_model, 
            self.tokenizer,
            self.test_dataset, 
            self.original_responses
        )
        baseline_metrics = baseline_evaluator.evaluate_model()
        
        # Calculate improvements
        improvements = {}
        
        for metric_name in self.metrics:
            if metric_name in baseline_metrics:
                if isinstance(self.metrics[metric_name], dict):
                    improvements[metric_name] = {
                        k: self.metrics[metric_name].get(k, 0) - baseline_metrics[metric_name].get(k, 0)
                        for k in self.metrics[metric_name]
                    }
                else:
                    improvements[metric_name] = self.metrics[metric_name] - baseline_metrics[metric_name]
        
        # Create comparison report
        comparison = {
            'baseline': baseline_metrics,
            'fine_tuned': self.metrics,
            'improvements': improvements,
            'improvement_percentage': {
                'rouge': (self.metrics.get('rouge', {}).get('rougeL', 0) - 
                         baseline_metrics.get('rouge', {}).get('rougeL', 0)) * 100,
                'bleu': (self.metrics.get('bleu', 0) - baseline_metrics.get('bleu', 0)) * 100,
                'medical_accuracy': (self.metrics.get('medical_accuracy', 0) - 
                                    baseline_metrics.get('medical_accuracy', 0)) * 100
            }
        }
        
        # Save comparison
        with open('model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"✅ Comparison complete. Improvements: {comparison['improvement_percentage']}")
        
        return comparison
