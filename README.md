# TCGA Multi-modal Cancer Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible MLOps pipeline for multi-modal cancer classification using TCGA TITAN visual embeddings and clinical text reports. This project implements a late-fusion MLP classifier that combines histopathology image features with processed diagnostic text.

##  Project Overview

This project classifies **32 cancer types** from The Cancer Genome Atlas (TCGA) dataset using:
- **Visual modality**: Pre-extracted TITAN embeddings (768-dim) from Whole Slide Images
- **Text modality**: Clinical reports processed through vLLM (Qwen3) for cleaning and embedding extraction (896-dim)

## Repository Structure
```
tcga-multimodal-mlops/
├── configs/
│   └── config.yaml          # Centralized configuration (no magic numbers!)
├── data/
│   ├── processed/           # Processed text embeddings
│   └── splits/              # Patient-aware train/val/test splits
├── outputs/
│   ├── models/              # Trained models (best_model.pt)
│   ├── logs/                # Training logs
│   └── figures/             # Visualizations
├── scripts/
│   ├── train.py             # Main training script
│   └── inference.py         # Inference script for evaluation
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   ├── training/            # Training pipeline
│   └── utils/               # Config, visualization utilities
├── tcga_data/               # Raw TCGA data
├── vllm/                    # vLLM scripts for text processing
├── pyproject.toml           # Project dependencies
└── requirements.txt         # Python requirements
```

##  Installation
```bash
# Clone the repository
git clone https://github.com/Linaomaruni/tcga-multimodal-mlops.git
cd tcga-multimodal-mlops

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

### 1. Inspect the data
```bash
python src/data/data_inspection.py
```

### 2. Create patient-aware splits
```bash
python src/data/splits.py
```

### 3. Process text with vLLM (on Snellius)
```bash
# Prepare prompts
python src/data/prepare_prompts.py

# Run decoder (text cleaning)
cd vllm/src
sbatch slurm_jobs/tcga_decoder.job

# Run encoder (text embeddings)
sbatch slurm_jobs/tcga_encoder.job
```

##  Training

### Train the multi-modal model
```bash
python scripts/train.py --model_type multimodal --wandb_project tcga-multimodal
```

### Train with custom hyperparameters
```bash
python scripts/train.py \
    --model_type multimodal \
    --lr 0.0005 \
    --dropout 0.4 \
    --hidden_dim 512 \
    --batch_size 32
```

### Train unimodal baselines
```bash
# Visual only
python scripts/train.py --model_type visual_only

# Text only
python scripts/train.py --model_type text_only
```

##  Inference

Run inference on test set using the best model:
```bash
python scripts/inference.py --model_path outputs/models/best_model.pt
```

This outputs:
- Macro-F1 and Micro-F1 scores
- Full classification report
- Confusion matrix visualization

##  Configuration

All hyperparameters are centralized in `configs/config.yaml`:
```yaml
model:
  visual_dim: 768
  text_dim: 896
  hidden_dim: 256
  num_classes: 32
  dropout: 0.3

training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 100
```

##  Results

| Model | Macro-F1 | Micro-F1 |
|-------|----------|----------|
| Visual Only | TBD | TBD |
| Text Only | TBD | TBD |
| **Multi-modal (Ours)** | **TBD** | **TBD** |

##  Links

- [W&B Dashboard](https://wandb.ai/) - Experiment tracking
- [TCGA Dataset](https://www.cancer.gov/tcga) - Original data source
- [TITAN Model](https://github.com/mahmoodlab/TITAN) - Visual embedding extractor

##  Authors

- Lina Omar
- Jette Walvis
- Sarah Schaefers
- Carmen van der Lans
- Fien van Engelen

##  License

This project is licensed under the MIT License.