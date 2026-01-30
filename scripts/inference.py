"""
Inference script for TCGA Multi-modal Classification.
Loads best_model.pt and evaluates on test data.

Usage: python scripts/inference.py --model_path outputs/models/best_model.pt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import Config
from src.models.fusion_mlp import LateFusionMLP
from src.data.dataloader import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on TCGA test set")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/models/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures",
        help="Directory to save results"
    )
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, test_loader, device):
    """Run inference on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for visual, text, labels in test_loader:
        visual = visual.to(device)
        text = text.to(device)
        
        outputs = model(visual, text)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Test Set Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    args = parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load config
    config = Config(args.config)
    
    # Load data
    print("Loading test data...")
    _, _, test_loader, class_to_idx = create_dataloaders(config)
    class_names = list(class_to_idx.keys())
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = LateFusionMLP(
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
    )
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Run inference
    print("Running inference...")
    preds, labels, probs = run_inference(model, test_loader, device)
    
    # Calculate metrics
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=class_names))
    
    # Save confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        labels, preds, class_names,
        output_dir / "test_confusion_matrix.png"
    )
    
    return macro_f1, micro_f1


if __name__ == "__main__":
    main()