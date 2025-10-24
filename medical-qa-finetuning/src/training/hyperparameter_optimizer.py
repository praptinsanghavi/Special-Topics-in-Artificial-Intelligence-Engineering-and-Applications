# Lines 950-1100 from notebook: HyperparameterOptimizer class
class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization with grid/random search
    Implements systematic hyperparameter tuning with visualization
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.results = []
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    def define_search_space(self) -> Dict:
        """Define hyperparameter search space"""
        return {
            'learning_rate': [5e-5, 1e-4, 2e-4],
            'batch_size': [2, 4],
            'num_epochs': [2, 3],
            'warmup_ratio': [0.05, 0.1],
            'weight_decay': [0.01, 0.05],
            'lora_r': [32, 64],
            'lora_dropout': [0.05, 0.1]
        }

    def run_hyperparameter_search(self, num_trials: int = 3):
        """
        Run hyperparameter search with specified number of trials

        Args:
            num_trials: Number of configurations to test
        """
        logger.info(f"🔍 Starting hyperparameter search with {num_trials} trials")

        search_space = self.define_search_space()

        # Generate configurations
        all_configs = list(product(*search_space.values()))
        random.shuffle(all_configs)

        # Test configurations
        configs = all_configs[:min(num_trials, len(all_configs))]

        for i, config in enumerate(configs):
            hp_dict = dict(zip(search_space.keys(), config))
            logger.info(f"Testing configuration {i+1}/{len(configs)}: {hp_dict}")

            # Create LoRA config for this trial
            lora_config = LoraConfig(
                r=hp_dict['lora_r'],
                lora_alpha=hp_dict['lora_r'] * 2,
                lora_dropout=hp_dict['lora_dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )

            # Apply LoRA
            peft_model = get_peft_model(self.model, lora_config)

            # Determine if we can use mixed precision
            use_fp16 = False
            use_bf16 = False

            # Check if model dtype is already float16
            if hasattr(self.model, 'dtype'):
                if self.model.dtype == torch.float16:
                    # Model is already in FP16, don't use mixed precision training
                    use_fp16 = False
                elif torch.cuda.is_bf16_supported():
                    # Use BF16 if available
                    use_bf16 = True
            elif torch.cuda.is_bf16_supported():
                # Default to BF16 if available
                use_bf16 = True

            # Create training arguments with fixed FP16 handling
            training_args = TrainingArguments(
                output_dir=f"./hp_search/config_{i}",
                num_train_epochs=hp_dict['num_epochs'],
                per_device_train_batch_size=hp_dict['batch_size'],
                learning_rate=hp_dict['learning_rate'],
                warmup_ratio=hp_dict['warmup_ratio'],
                weight_decay=hp_dict['weight_decay'],
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_steps=50,
                fp16=use_fp16,  # Only use if safe
                bf16=use_bf16,  # Prefer BF16 when available
                report_to="none",
                remove_unused_columns=False,
                gradient_accumulation_steps=max(1, 8 // hp_dict['batch_size']),
                max_grad_norm=1.0  # Add gradient clipping
            )

            # Create trainer
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator
            )

            try:
                # Train and evaluate
                train_result = trainer.train()
                eval_result = trainer.evaluate()

                # Store results
                result = {
                    'config': hp_dict,
                    'train_loss': train_result.training_loss,
                    'eval_loss': eval_result['eval_loss'],
                    'eval_runtime': eval_result['eval_runtime'],
                    'eval_samples_per_second': eval_result.get('eval_samples_per_second', 0)
                }
                self.results.append(result)

                # Log to W&B if available
                if WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        f"hp_trial_{i}/train_loss": train_result.training_loss,
                        f"hp_trial_{i}/eval_loss": eval_result['eval_loss']
                    })

            except Exception as e:
                logger.warning(f"Trial {i} failed with error: {e}")
                # Skip this configuration
                continue

        # Document and visualize results
        if self.results:
            self._document_results()
            self.visualize_hyperparameter_search()
        else:
            logger.error("All hyperparameter trials failed!")

    def _document_results(self):
        """Save hyperparameter search results"""
        # Save all results
        with open('hyperparameter_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        # Create comparison table
        df = pd.DataFrame(self.results)
        df.to_csv('hyperparameter_comparison.csv', index=False)

        # Find best configuration
        best_config = min(self.results, key=lambda x: x['eval_loss'])
        logger.info(f"🏆 Best configuration: {best_config}")

        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_config, f, indent=2)

    def visualize_hyperparameter_search(self):
        """Create comprehensive hyperparameter analysis visualization"""
        if not self.results:
            logger.warning("No results to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Learning rate impact
        learning_rates = [r['config']['learning_rate'] for r in self.results]
        eval_losses = [r['eval_loss'] for r in self.results]

        axes[0, 0].scatter(learning_rates, eval_losses, s=100, alpha=0.6, color='blue')
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('Evaluation Loss')
        axes[0, 0].set_title('Learning Rate Impact on Performance')
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        # Batch size impact
        batch_sizes = [r['config']['batch_size'] for r in self.results]
        axes[0, 1].scatter(batch_sizes, eval_losses, s=100, alpha=0.6, color='green')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Evaluation Loss')
        axes[0, 1].set_title('Batch Size Impact on Performance')
        axes[0, 1].grid(True, alpha=0.3)

        # LoRA rank impact
        lora_ranks = [r['config'].get('lora_r', 64) for r in self.results]
        axes[1, 0].scatter(lora_ranks, eval_losses, s=100, alpha=0.6, color='red')
        axes[1, 0].set_xlabel('LoRA Rank (r)')
        axes[1, 0].set_ylabel('Evaluation Loss')
        axes[1, 0].set_title('LoRA Rank Impact on Performance')
        axes[1, 0].grid(True, alpha=0.3)

        # Configuration comparison
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')

        # Create comparison table
        sorted_results = sorted(self.results, key=lambda x: x['eval_loss'])[:3]
        table_data = []
        for i, r in enumerate(sorted_results):
            table_data.append([
                f"#{i+1}",
                f"{r['config']['learning_rate']:.1e}",
                f"{r['config']['batch_size']}",
                f"{r['config']['lora_r']}",
                f"{r['eval_loss']:.4f}"
            ])

        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Rank', 'LR', 'BS', 'LoRA_r', 'Loss'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 1].set_title('Top 3 Configurations')

        plt.suptitle('Hyperparameter Search Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save figure
        plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Hyperparameter analysis saved")
