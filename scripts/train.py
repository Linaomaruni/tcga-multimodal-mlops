"""
Main training script for TCGA Multi-modal Classification.
Usage: python scripts/train.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb

from src.data.dataloader import create_dataloaders
from src.models.fusion_mlp import LateFusionMLP, UnimodalMLP
from src.training.trainer import Trainer
from src.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCGA Multi-modal Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["multimodal", "visual_only", "text_only"],
        default="multimodal",
        help="Type of model to train"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tcga-multimodal",
        help="W&B project name"
    )
    # Hyperparameter overrides
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Override config with command line args
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # Initialize W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "model_type": args.model_type,
                "learning_rate": config.training.learning_rate,
                "dropout": config.model.dropout,
                "hidden_dim": config.model.hidden_dim,
                "batch_size": config.training.batch_size,
                "num_epochs": config.training.num_epochs,
                "weight_decay": config.training.weight_decay,
            },
            name=f"{args.model_type}_lr{config.training.learning_rate}_drop{config.model.dropout}",
        )
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(config)
    class_names = list(class_to_idx.keys())
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == "multimodal":
        model = LateFusionMLP(
            visual_dim=config.model.visual_dim,
            text_dim=config.model.text_dim,
            hidden_dim=config.model.hidden_dim,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
        )
    elif args.model_type == "visual_only":
        model = UnimodalMLP(
            input_dim=config.model.visual_dim,
            hidden_dim=config.model.hidden_dim,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
        )
    else:  # text_only
        model = UnimodalMLP(
            input_dim=config.model.text_dim,
            hidden_dim=config.model.hidden_dim,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_names=class_names,
        use_wandb=use_wandb,
    )
    
    # Train
    best_f1 = trainer.train()
    
    # Finish W&B
    if use_wandb:
        wandb.finish()
    
    print(f"\n Training complete! Best Macro-F1: {best_f1:.4f}")
    print(f"Model saved to: {config.output.best_model}")


if __name__ == "__main__":
    main()