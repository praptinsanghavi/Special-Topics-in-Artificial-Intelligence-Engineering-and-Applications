#!/usr/bin/env python
"""Evaluation script"""

from src.evaluation.evaluator import ModelEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

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
