# Lines 850-950 from notebook: CustomCallback and FinetuningSetup classes
from transformers import TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

class CustomCallback(TrainerCallback):
    """Custom callback for comprehensive training monitoring"""
    
    def __init__(self, metrics_tracker: Optional[MetricsTracker] = None):
        self.metrics_tracker = metrics_tracker
        self.training_history = {
            'loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics"""
        if logs:
            # Store in history
            self.training_history['loss'].append(logs.get('loss', 0))
            self.training_history['eval_loss'].append(logs.get('eval_loss', 0))
            self.training_history['learning_rate'].append(logs.get('learning_rate', 0))
            self.training_history['epoch'].append(state.epoch)
            
            # Log to metrics tracker
            if self.metrics_tracker:
                self.metrics_tracker.log_metrics(state.epoch, logs)
            
            # Log to file
            with open('training_logs.jsonl', 'a') as f:
                f.write(json.dumps(logs) + '\n')
    
class FinetuningSetup:
    """
    Complete fine-tuning setup with LoRA and best practices
    Implements efficient training with comprehensive monitoring
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None
        self.lora_config = None
        
    def setup_training_environment(self, output_dir: str = "./fine_tuned_model") -> None:
        """
        Configure training environment with LoRA
        
        Args:
            output_dir: Directory for model checkpoints
            
        Returns:
            PEFT model with LoRA adapters
        """
        # Configure LoRA for efficient training
        self.lora_config = LoraConfig(
            r=64,  # LoRA rank
            lora_alpha=128,  # LoRA scaling parameter
            lora_dropout=0.1,  # LoRA dropout
            bias="none",  # Don't train biases
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
            modules_to_save=["embed_tokens", "lm_head"]  # Also save embeddings and output layer
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Log trainable parameters
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(f"📊 Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # Print parameter summary
        self.model.print_trainable_parameters()
        
        return self.model
    
    def create_trainer(self, train_dataset, eval_dataset, training_args, 
                      metrics_tracker: Optional[MetricsTracker] = None):
        """
        Create trainer with custom callbacks and monitoring
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Validation dataset
            training_args: Training configuration
            metrics_tracker: Optional metrics tracking system
            
        Returns:
            Configured Trainer instance
        """
        # Create custom callback
        custom_callback = CustomCallback(metrics_tracker)
        
        # Create data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[custom_callback]
        )
        
        return self.trainer
