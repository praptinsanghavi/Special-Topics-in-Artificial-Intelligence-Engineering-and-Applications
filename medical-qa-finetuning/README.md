# Medical QA Fine-Tuning System

Production-grade medical question-answering system using QLoRA fine-tuning with comprehensive safety validation.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance Metrics](#performance-metrics)
- [Error Analysis & Improvement Roadmap](#error-analysis-&-improvement-roadmap)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Video](#video)
- [License](#license)

## Overview

This system fine-tunes large language models for medical question-answering using Quantized Low-Rank Adaptation (QLoRA), achieving 90% memory reduction while maintaining performance with only 9.55%  trainable parameters.

### Key Features
- **QLoRA Implementation**: 9.55% trainable parameters (293M params)  
- **Training Efficiency**: BFloat16 precision with gradient accumulation
- **Comprehensive Evaluation**: Automated error pattern analysis with improvement suggestions
- **Production Ready**: Sub-150ms P95 inference latency

### Technical Stack
- **Base Model**: Microsoft Phi-2 (2.7B parameters)
- **Framework**: PyTorch 2.2.0, Transformers 4.36.0
- **Fine-tuning**: PEFT 0.7.1 with LoRA
- **Dataset**: MedMCQA (5000 samples with augmentation)
- **Deployment**: Gradio 4.8.0, Docker, CUDA 11.8

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Pipeline                        │
├─────────────────────────────────────────────────────────┤
│  Dataset Loading → Preprocessing → Augmentation → Split  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Model Architecture                      │
├─────────────────────────────────────────────────────────┤
│  Phi-2 Base (2.7B) → LoRA Adapters → QLoRA (4-bit)     │
│  Target Modules: q_proj, v_proj, k_proj, o_proj         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Training Pipeline                       │
├─────────────────────────────────────────────────────────┤
│  Hyperparameter Search → Fine-tuning → Checkpointing    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Safety Validation                        │
├─────────────────────────────────────────────────────────┤
│  Drug Interactions → Emergency Detection → Dosage Check  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Inference & Deployment                   │
├─────────────────────────────────────────────────────────┤
│  Gradio Interface → REST API → Docker Container         │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ compatible GPU (16GB+ VRAM recommended)
- 32GB RAM minimum
- 50GB free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/medical-qa-finetuning.git
cd medical-qa-finetuning
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 4: Download Data
```bash
python scripts/download_data.py --dataset medmcqa --output data/raw/
```

## Quick Start

### Training a Model
```bash
# Basic training
python scripts/train.py \
    --dataset medmcqa \
    --model microsoft/phi-2 \
    --samples 7000 \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --lora-r 32 \
    --precision bfloat16

# Advanced training with custom config
python scripts/train.py --config configs/training_config.yaml
```

### Running Inference
```bash
# Start Gradio interface
python scripts/deploy_gradio.py --model models/final/best_model

# API server
python scripts/api_server.py --model models/final/best_model --port 8080
```

### Evaluation
```bash
python scripts/evaluate.py \
    --model models/final/best_model \
    --test-data data/processed/test.jsonl \
    --output results/evaluation_report.json
```

## Project Structure

```
medical-qa-finetuning/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
│
├── configs/                    # Configuration files
│   └── training_config.yaml   # Training hyperparameters
│
├── src/                        # Source code
│   ├── data/                  # Data processing modules
│   │   ├── preprocessor.py   # Dataset preparation
│   │   └── augmentation.py   # Data augmentation
│   │
│   ├── models/                # Model modules
│   │   ├── model_selector.py # Model selection logic
│   │   └── qlora_model.py    # QLoRA implementation
│   │
│   ├── training/              # Training modules
│   │   ├── trainer.py        # Main training logic
│   │   └── hyperparameter_optimizer.py
│   │
│   ├── evaluation/            # Evaluation modules
│   │   ├── evaluator.py      # Model evaluation
│   │   └── error_analyzer.py # Error analysis
│   │
│   ├── medical/               # Medical-specific modules
│   │   ├── safety_validator.py  # Safety validation
│   │   └── drug_interactions.py # Drug interaction DB
│   │
│   ├── inference/             # Inference modules
│   │   └── pipeline.py       # Inference pipeline
│   │
│   └── utils/                 # Utility modules
│       ├── metrics_tracker.py # Metrics tracking
│       └── documentation.py   # Doc generation
│
├── scripts/                    # Executable scripts
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   └── deploy_gradio.py      # Deploy Gradio app
│
├── deployment/                 # Deployment files
│   └── Dockerfile             # Docker container
│
├── results/                     # Results and outputs
│   └── electric-mountain-21/    # Model ran on WANDB
│       ├── config.yaml          # Config parameters are your model's inputs
│       ├── output.log           # Console output/training logs
│       ├── wandb-metadata.json  # Comprehensive system and environment information
│       ├── wandb-summary.json   # Summary metrics are your model's outputs
│
├── docs/                       # Documentation
│   └── api.md                 # API documentation
└──
```

## Usage

### Data Preparation
```python
from src.data.preprocessor import DatasetPreparator

# Initialize preprocessor
prep = DatasetPreparator(
    dataset_name="medmcqa",
    max_samples=5000,
    use_augmentation=True
)

# Prepare dataset
dataset_dict, original_responses = prep.prepare_dataset()
```

### Model Training
```python
from src.models.model_selector import ModelSelector
from src.training.trainer import FinetuningSetup

# Select model
selector = ModelSelector()
model, tokenizer = selector.select_and_setup_model("microsoft/phi-2")

# Setup training
trainer_setup = FinetuningSetup(model, tokenizer)
model = trainer_setup.setup_training_environment()

# Train
trainer = trainer_setup.create_trainer(train_dataset, eval_dataset, training_args)
trainer.train()
```

### Safety Validation
```python
from src.medical.safety_validator import MedicalSafetyValidator

validator = MedicalSafetyValidator()
validated_response, safety_report = validator.validate_medical_response(
    response="Take warfarin and aspirin together",
    patient_context={"conditions": ["atrial_fibrillation"]}
)
```

### Inference
```python
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline(model, tokenizer)
response = pipeline.generate(
    question="What are the symptoms of diabetes?",
    temperature=0.7,
    max_length=100
)
```

## Configuration

### Training Configuration (configs/training_config.yaml)
```yaml
model:
  dtype: "bfloat16"  # Changed from float16

dataset:
  name: "medmcqa"
  max_samples: 5000
  augmentation_factor: 2

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 1e-4
  warmup_ratio: 0.05
  weight_decay: 0.05

lora:
  r: 32  # Changed from 64
  lora_alpha: 64  # Adjusted for r=32
  lora_dropout: 0.1

evaluation:
  metrics: ["rouge", "bleu", "accuracy", "perplexity"]
  test_size: 100
```

## Performance Metrics

### Training Results
| Metric | Value | Notes |
|--------|-------|-------|
| Final Training Loss | 1.797 | Converged after 3 epochs |
| Evaluation Loss | 1.754 | Stable validation performance |
| Accuracy Improvement | +22.3% | Relative improvement from baseline |
| BLEU Improvement | +18.5% | Enhanced response quality |
| ROUGE Improvement | +15.2% | Better content overlap |
| Perplexity | 100 | Room for improvement |

### Resource Usage
| Resource | Usage |
|----------|-------|
| Trainable Parameters | 293.65M (9.55%) |
| Training Time | 11.7 hours (42,053s) |
| Training Steps/Second | 0.062 |
| Evaluation Steps/Second | 0.432 |
| Inference Latency (P50) | 87ms |
| Inference Latency (P95) | 145ms |
| Total FLOPs | 9.07e16 |

### Identified Issues & Improvements
| Issue | Frequency | Proposed Solution |
|-------|-----------|-------------------|
| Overly Verbose Responses | 98.5% | Implement length penalty during generation |
| Medical Content Accuracy | Variable | Increase domain-specific training data |
| High Perplexity | N/A | Add medical terminology post-processing |

## Error Analysis & Improvement Roadmap

### Common Error Patterns
1. **Verbosity Issue (98.5% of responses)**
   - Model generates unnecessarily long explanations
   - Solution: Implement response length constraints

2. **Medical Accuracy Concerns**
   - Difficulty with technical medical terminology
   - Solution: Domain-specific fine-tuning strategies

### Recommended Improvements
- [ ] Implement beam search for more stable generation
- [ ] Add confidence scoring to identify uncertain predictions
- [ ] Integrate medical terminology validation
- [ ] Add conciseness examples to training data
- [ ] Implement length penalty λ=0.8 during inference

## API Documentation

### REST API Endpoints

#### POST /generate
Generate medical response
```json
{
  "question": "What are the symptoms of diabetes?",
  "temperature": 0.7,
  "max_length": 100,
  "use_safety": true
}
```

Response:
```json
{
  "response": "Diabetes symptoms include...",
  "safety_report": {
    "risk_level": 0,
    "warnings": []
  },
  "inference_time_ms": 87
}
```

#### GET /health
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true
}
```

## Testing

### Run Unit Tests
```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_safety_validator.py

# With coverage
pytest --cov=src tests/
```

### Run Integration Tests
```bash
python tests/integration/test_end_to_end.py
```

### Performance Tests
```bash
python tests/performance/benchmark.py --samples 1000
```

## Known Limitations

Based on current evaluation:
- **Response Length**: Model tends to be overly verbose (mitigation in progress)
- **Medical Accuracy**: Requires additional domain-specific training
- **Perplexity**: Higher than target (100 vs goal of <20)
- **Training Time**: ~12 hours on single GPU (consider distributed training)

These limitations are actively being addressed in the development roadmap.

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t medical-qa:latest .

# Run container
docker run -p 7860:7860 --gpus all medical-qa:latest
```

### Kubernetes Deployment
```bash
kubectl apply -f deployment/kubernetes/
```

### Cloud Deployment
```bash
# AWS
./deployment/scripts/deploy_aws.sh

# Google Cloud
./deployment/scripts/deploy_gcp.sh
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
mypy src/
```

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Video

Screen recording (3-5 minutes) showcasing functions: [[Demo Video Link](https://youtu.be/EJrIMasY6fo?si=Yf_I243lTpz7blr8)](#)

## License

MIT License - see LICENSE file for details

## Disclaimer

This system is for research and educational purposes only. Not intended for clinical use. Always consult qualified healthcare professionals for medical advice.

## Contact

- Email: your.email@northeastern.edu
- GitHub: [@yourusername](https://github.com/yourusername)
- Project Link: [https://github.com/yourusername/medical-qa-finetuning](https://github.com/yourusername/medical-qa-finetuning)
