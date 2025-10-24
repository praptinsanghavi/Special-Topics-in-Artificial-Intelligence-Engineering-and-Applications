#!/usr/bin/env python3
"""
Main training script for Medical QA Fine-Tuning
Execute with: python scripts/train.py --dataset medmcqa --model microsoft/phi-2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import random
import numpy as np
import torch
import wandb
from datetime import datetime
from transformers import AutoTokenizer, TrainingArguments

# Import project modules
from src.data.preprocessor import DatasetPreparator
from src.models.model_selector import ModelSelector
from src.training.trainer import FinetuningSetup
from src.training.hyperparameter_optimizer import HyperparameterOptimizer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.inference.pipeline import InferencePipeline
from src.utils.metrics_tracker import MetricsTracker
from src.utils.documentation import DocumentationGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    """Main training pipeline"""
    
    logger.info("="*80)
    logger.info("MEDICAL QA FINE-TUNING SYSTEM")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.samples}")
    logger.info("="*80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize W&B if available
    use_wandb = False
    if not args.no_wandb:
        try:
            wandb.init(
                project="medical-qa-finetuning",
                config=vars(args),
                tags=["production", "medical", "safety"]
            )
            use_wandb = True
            logger.info("W&B tracking initialized")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(project_name="Medical QA Fine-Tuning")
    final_results = {}
    
    # Step 1: Dataset Preparation
    logger.info("\\nStep 1: Dataset Preparation")
    data_prep = DatasetPreparator(
        dataset_name=args.dataset,
        max_samples=args.samples,
        use_augmentation=args.augmentation
    )
    dataset_dict, original_responses = data_prep.prepare_dataset()
    final_results['dataset_size'] = len(dataset_dict['train'])
    
    # Step 2: Tokenization
    logger.info("\\nStep 2: Tokenization")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
        return tokenizer(texts, truncation=True, max_length=256, padding="max_length")
    
    # Remove unnecessary columns and tokenize
    columns_to_remove = [col for col in dataset_dict["train"].column_names
                         if col not in ['text', 'instruction', 'response']]
    
    for split in dataset_dict.keys():
        if columns_to_remove:
            dataset_dict[split] = dataset_dict[split].remove_columns(columns_to_remove)
    
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Step 3: Model Selection
    logger.info("\\nStep 3: Model Selection and Setup")
    model_selector = ModelSelector()
    model, _ = model_selector.select_and_setup_model(args.model)
    
    # Step 4: Hyperparameter Optimization (optional)
    if args.hp_search:
        logger.info("\\nStep 4: Hyperparameter Optimization")
        hp_optimizer = HyperparameterOptimizer(
            model, tokenizer,
            tokenized_datasets['train'][:500],
            tokenized_datasets['validation'][:100]
        )
        hp_optimizer.run_hyperparameter_search(num_trials=args.hp_trials)
        
        with open('best_hyperparameters.json', 'r') as f:
            best_config = json.load(f)
    else:
        # Use default configuration
        best_config = {
            'config': {
                'num_epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'lora_r': 64
            }
        }
    
    # Step 5: Fine-tuning
    logger.info("\\nStep 5: Fine-tuning with Optimal Configuration")
    finetuning_setup = FinetuningSetup(model, tokenizer)
    model = finetuning_setup.setup_training_environment()
    
    # Get trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=best_config['config']['num_epochs'],
        per_device_train_batch_size=best_config['config']['batch_size'],
        per_device_eval_batch_size=4,
        learning_rate=best_config['config']['learning_rate'],
        warmup_ratio=best_config['config']['warmup_ratio'],
        weight_decay=best_config['config']['weight_decay'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if use_wandb else "none",
        gradient_accumulation_steps=max(1, 8 // best_config['config']['batch_size']),
        remove_unused_columns=False,
        save_total_limit=2
    )
    
    # Create trainer
    trainer = finetuning_setup.create_trainer(
        tokenized_datasets['train'],
        tokenized_datasets['validation'],
        training_args,
        metrics_tracker
    )
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Step 6: Evaluation
    if args.evaluate:
        logger.info("\\nStep 6: Model Evaluation")
        evaluator = ModelEvaluator(
            model, tokenizer,
            tokenized_datasets['test'],
            original_responses
        )
        metrics = evaluator.evaluate_model()
        final_results.update(metrics)
    
    # Step 7: Error Analysis
    if args.error_analysis:
        logger.info("\\nStep 7: Error Analysis")
        error_analyzer = ErrorAnalyzer(
            model, tokenizer,
            tokenized_datasets['test'],
            original_responses
        )
        error_results = error_analyzer.analyze_errors()
        final_results['error_analysis'] = error_results
    
    # Step 8: Create Inference Pipeline
    if args.create_interface:
        logger.info("\\nStep 8: Creating Inference Pipeline")
        pipeline = InferencePipeline(model, tokenizer)
        interface = pipeline.create_interface()
        
        if args.launch_interface:
            logger.info("Launching Gradio interface...")
            interface.launch(share=True)
    
    # Step 9: Generate Documentation
    if args.generate_docs:
        logger.info("\\nStep 9: Generating Documentation")
        doc_generator = DocumentationGenerator(final_results)
        doc_generator.generate_technical_report()
        doc_generator.generate_readme()
    
    # Step 10: Create Dashboard
    logger.info("\\nStep 10: Creating Performance Dashboard")
    metrics_tracker.create_comprehensive_dashboard()
    
    # Save final results
    with open('final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    if use_wandb:
        wandb.log(final_results)
        wandb.finish()
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical QA Model")
    
    # Model and dataset arguments
    parser.add_argument("--model", type=str, default="microsoft/phi-2",
                        help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="medmcqa",
                        help="Dataset to use for training")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Maximum number of samples to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./models/final",
                        help="Output directory for model")
    
    # Optional features
    parser.add_argument("--augmentation", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--hp_search", action="store_true",
                        help="Perform hyperparameter search")
    parser.add_argument("--hp_trials", type=int, default=3,
                        help="Number of hyperparameter search trials")
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="Perform evaluation after training")
    parser.add_argument("--error_analysis", action="store_true", default=True,
                        help="Perform error analysis")
    parser.add_argument("--create_interface", action="store_true", default=True,
                        help="Create Gradio interface")
    parser.add_argument("--launch_interface", action="store_true",
                        help="Launch Gradio interface after training")
    parser.add_argument("--generate_docs", action="store_true", default=True,
                        help="Generate documentation")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    main(args)
