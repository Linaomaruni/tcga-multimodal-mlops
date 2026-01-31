"""
Training pipeline for TCGA Multi-modal Classification.
Includes W&B logging for experiment tracking.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class Trainer:
    """
    Trainer class for multi-modal cancer classification.
    Handles training loop, validation, and W&B logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        class_names: list[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_names = class_names
        self.device = device
        self.use_wandb = use_wandb
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
        )
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        # Create output directories
        Path(config.output.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output.figure_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (visual, text, labels) in enumerate(pbar):
            visual = visual.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            # Handle both multimodal and unimodal models
            if hasattr(self.model, 'visual_encoder'):  # LateFusionMLP
                outputs = self.model(visual, text)
            else:  # UnimodalMLP
                if self.model.encoder[0].in_features == 768:  # visual_only
                    outputs = self.model(visual)
                else:  # text_only
                    outputs = self.model(text)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")
        
        return {
            "train_loss": avg_loss,
            "train_macro_f1": macro_f1,
            "train_micro_f1": micro_f1,
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for visual, text, labels in tqdm(self.val_loader, desc="Validating"):
            visual = visual.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device)
            
            # Handle both multimodal and unimodal models
            if hasattr(self.model, 'visual_encoder'):  # LateFusionMLP
                outputs = self.model(visual, text)
            else:  # UnimodalMLP
                if self.model.encoder[0].in_features == 768:  # visual_only
                    outputs = self.model(visual)
                else:  # text_only
                    outputs = self.model(text)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")
        
        return {
            "val_loss": avg_loss,
            "val_macro_f1": macro_f1,
            "val_micro_f1": micro_f1,
            "val_preds": all_preds,
            "val_labels": all_labels,
        }
    
    def plot_confusion_matrix(self, labels, preds, epoch: int):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        # Save locally
        save_path = Path(self.config.output.figure_dir) / f"confusion_matrix_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({"confusion_matrix": wandb.Image(plt)})
        
        plt.close()
    
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        
        # Save best model
        torch.save(checkpoint, self.config.output.best_model)
        print(f"Saved best model with val_macro_f1: {metrics['val_macro_f1']:.4f}")
    
    def train(self):
        """Full training loop."""
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {self.config.training.num_epochs}")
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.training.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Print metrics
            print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
            print(f"Train Macro-F1: {train_metrics['train_macro_f1']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Macro-F1: {val_metrics['val_macro_f1']:.4f}")
            print(f"Val Micro-F1: {val_metrics['val_micro_f1']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["train_loss"],
                    "train_macro_f1": train_metrics["train_macro_f1"],
                    "train_micro_f1": train_metrics["train_micro_f1"],
                    "val_loss": val_metrics["val_loss"],
                    "val_macro_f1": val_metrics["val_macro_f1"],
                    "val_micro_f1": val_metrics["val_micro_f1"],
                    "learning_rate": current_lr,
                })
            
            # Check for best model
            if val_metrics["val_macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_macro_f1"]
                self.save_checkpoint(epoch, val_metrics)
                self.patience_counter = 0
                
                # Plot confusion matrix for best model
                self.plot_confusion_matrix(
                    val_metrics["val_labels"],
                    val_metrics["val_preds"],
                    epoch,
                )
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        print(f"\n{'='*50}")
        print(f"Training complete! Best Val Macro-F1: {self.best_val_f1:.4f}")
        print(f"{'='*50}")
        
        return self.best_val_f1