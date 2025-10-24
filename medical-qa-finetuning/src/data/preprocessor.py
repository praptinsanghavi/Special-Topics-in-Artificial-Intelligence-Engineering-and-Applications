"""Dataset preparation and preprocessing module"""

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
import logging
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DatasetPreparator:
    """
    Comprehensive dataset preparation with all required features
    Implements thorough preprocessing, quality checks, and stratified splitting
    """
    
    def __init__(self, dataset_name: str = "medmcqa", max_samples: int = 5000, use_augmentation: bool = False):
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.use_augmentation = use_augmentation
        self.stats = {}
        self.original_responses = {}
        # Import safety validator and augmenter when needed
        from src.medical.safety_validator import MedicalSafetyValidator
        from src.data.augmentation import MedicalDataAugmenter
        self.safety_validator = MedicalSafetyValidator()
        self.augmenter = MedicalDataAugmenter() if use_augmentation else None
    
    def prepare_dataset(self) -> Tuple[DatasetDict, Dict]:
        """Complete dataset preparation pipeline"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split="train")
        logger.info(f"Loaded {len(dataset)} samples")
        
        df = self._preprocess_data(dataset)
        
        if self.augmenter:
            logger.info("Applying data augmentation...")
            qa_pairs = df.to_dict('records')
            augmented_pairs = self.augmenter.augment_dataset(qa_pairs, augmentation_factor=2)
            df = pd.DataFrame(augmented_pairs[:min(self.max_samples * 2, 10000)])
            logger.info(f"Augmented to {len(df)} samples")
        
        train_df, val_df, test_df = self._create_splits(df)
        formatted_data = self._format_for_finetuning(train_df, val_df, test_df)
        
        self.original_responses = {
            'train': train_df['response'].tolist() if 'response' in train_df.columns else [],
            'validation': val_df['response'].tolist() if 'response' in val_df.columns else [],
            'test': test_df['response'].tolist() if 'response' in test_df.columns else []
        }
        
        self._save_statistics()
        return formatted_data, self.original_responses
    
    def _preprocess_data(self, dataset) -> pd.DataFrame:
        """Thorough data preprocessing and cleaning"""
        df = pd.DataFrame(dataset)
        initial_size = len(df)
        
        df = df.drop_duplicates(subset=['question']) if 'question' in df.columns else df
        logger.info(f"Removed {initial_size - len(df)} duplicates")
        
        if 'question' in df.columns:
            df['question'] = df['question'].apply(self._clean_medical_text)
        
        qa_pairs = []
        for _, row in df.iterrows():
            qa_pair = self._create_qa_pair(row)
            if qa_pair and self._quality_check(qa_pair):
                if 'response' in qa_pair:
                    validated_response, safety_checks = self.safety_validator.validate_medical_response(
                        qa_pair['response']
                    )
                    qa_pair['response'] = validated_response
                    qa_pair['safety_validation'] = safety_checks
                qa_pairs.append(qa_pair)
        
        df = pd.DataFrame(qa_pairs[:self.max_samples])
        self.stats['preprocessing'] = {
            'initial_samples': initial_size,
            'after_dedup': len(df),
            'final_samples': len(df)
        }
        
        return df
    
    def _clean_medical_text(self, text):
        """Clean medical text with domain-specific rules"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('ml_', 'mL ')
        text = text.replace('mg_', 'mg ')
        text = text.replace('_', ' ')
        text = ' '.join(text.split())
        return text
    
    def _create_qa_pair(self, row) -> Optional[Dict]:
        """Create high-quality QA pairs"""
        question = row.get('question', '')
        
        if 'cop' in row and row['cop'] in [1, 2, 3, 4]:
            options = ['opa', 'opb', 'opc', 'opd']
            correct_idx = int(row['cop']) - 1
            if correct_idx < len(options):
                correct_answer = row.get(options[correct_idx], '')
                explanation = row.get('exp', '')
                
                if explanation:
                    response = f"Answer: {correct_answer}\\n\\nExplanation: {explanation}"
                else:
                    response = f"Answer: {correct_answer}"
                
                return {
                    'instruction': f"Medical Question: {question}",
                    'response': response,
                    'subject': row.get('subject_name', 'General Medicine'),
                    'difficulty': self._calculate_difficulty(row)
                }
        return None
    
    def _quality_check(self, qa_pair: Dict) -> bool:
        """Quality filtering for QA pairs"""
        if len(qa_pair.get('instruction', '')) < 20 or len(qa_pair.get('response', '')) < 10:
            return False
        
        medical_terms = [
            'diagnosis', 'treatment', 'symptoms', 'patient', 'medical',
            'disease', 'condition', 'therapy', 'medication', 'clinical'
        ]
        text = (qa_pair.get('instruction', '') + ' ' + qa_pair.get('response', '')).lower()
        return any(term in text for term in medical_terms)
    
    def _calculate_difficulty(self, row) -> str:
        """Calculate question difficulty based on complexity"""
        text_len = len(str(row.get('question', ''))) + len(str(row.get('exp', '')))
        if text_len > 800:
            return 'hard'
        elif text_len > 400:
            return 'medium'
        else:
            return 'easy'
    
    def _create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits"""
        stratify_col = df['subject'] if 'subject' in df.columns else None
        
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=stratify_col
        )
        
        stratify_temp = temp_df['subject'] if 'subject' in temp_df.columns else None
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=stratify_temp
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        self.stats['splits'] = {
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df)
        }
        
        return train_df, val_df, test_df
    
    def _format_for_finetuning(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                test_df: pd.DataFrame) -> DatasetDict:
        """Format data appropriately for fine-tuning"""
        def format_example(example):
            instruction = example.get('instruction', '')
            response = example.get('response', '')
            return {
                'text': f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}",
                'instruction': instruction,
                'response': response
            }
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        train_dataset = train_dataset.map(format_example)
        val_dataset = val_dataset.map(format_example)
        test_dataset = test_dataset.map(format_example)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    
    def _save_statistics(self):
        """Save dataset statistics for documentation"""
        stats_file = 'dataset_statistics.json'
        
        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['configuration'] = {
            'dataset_name': self.dataset_name,
            'max_samples': self.max_samples,
            'augmentation_enabled': self.use_augmentation
        }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
