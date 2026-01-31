"""
Multi-modal DataLoader for TCGA Dataset.
Loads TITAN visual embeddings and text embeddings aligned by patient ID.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TCGAMultimodalDataset(Dataset):
    """
    Dataset for multi-modal TCGA data.
    Aligns visual embeddings, text embeddings, and labels by patient ID.
    """

    def __init__(
        self,
        patient_ids: list[str],
        visual_embeddings: dict,
        text_embeddings: dict,
        labels_df: pd.DataFrame,
        class_to_idx: dict[str, int],
    ):
        self.patient_ids = patient_ids
        self.visual_embeddings = visual_embeddings
        self.text_embeddings = text_embeddings
        self.labels_df = labels_df
        self.class_to_idx = class_to_idx

        # Create patient_id to label mapping
        self.id_to_label = dict(zip(labels_df["patient_id"], labels_df["cancer_type"]))

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pid = self.patient_ids[idx]

        # Get visual embedding (take first if multiple)
        visual_emb = self.visual_embeddings[pid]["embeddings"][0]
        visual_tensor = torch.tensor(visual_emb, dtype=torch.float32)

        # Get text embedding
        text_emb = self.text_embeddings[pid]
        text_tensor = torch.tensor(text_emb, dtype=torch.float32)

        # Get label
        label_name = self.id_to_label[pid]
        label = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)

        return visual_tensor, text_tensor, label


def load_data(config) -> tuple[dict, dict, pd.DataFrame, dict]:
    """
    Load all data files.

    Returns:
        visual_embeddings, text_embeddings, labels_df, class_to_idx
    """
    # Load visual embeddings
    with open(config.data.titan_embeddings, "rb") as f:
        visual_embeddings = pickle.load(f)

    # Load text embeddings
    text_emb_path = Path(config.data.text_embeddings)
    if text_emb_path.exists():
        with open(text_emb_path, "rb") as f:
            text_embeddings = pickle.load(f)
    else:
        # Placeholder: use random embeddings if text not yet processed
        print("WARNING: Text embeddings not found, using random placeholders!")
        text_embeddings = {
            pid: torch.randn(896).numpy() for pid in visual_embeddings.keys()
        }

    # Load labels
    labels_df = pd.read_csv(config.data.labels)

    # Filter to valid patients
    valid_pids = set(visual_embeddings.keys()) & set(text_embeddings.keys())
    labels_df = labels_df[labels_df["patient_id"].isin(valid_pids)]

    # Create class mapping
    cancer_types = sorted(labels_df["cancer_type"].unique())
    class_to_idx = {name: i for i, name in enumerate(cancer_types)}

    return visual_embeddings, text_embeddings, labels_df, class_to_idx


def load_splits(config) -> dict:
    """Load patient splits from JSON file."""
    with open(config.data.splits, "r") as f:
        splits = json.load(f)
    return splits


def create_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data
    visual_emb, text_emb, labels_df, class_to_idx = load_data(config)
    splits = load_splits(config)

    # Create datasets
    train_dataset = TCGAMultimodalDataset(
        patient_ids=splits["train"],
        visual_embeddings=visual_emb,
        text_embeddings=text_emb,
        labels_df=labels_df,
        class_to_idx=class_to_idx,
    )

    val_dataset = TCGAMultimodalDataset(
        patient_ids=splits["val"],
        visual_embeddings=visual_emb,
        text_embeddings=text_emb,
        labels_df=labels_df,
        class_to_idx=class_to_idx,
    )

    test_dataset = TCGAMultimodalDataset(
        patient_ids=splits["test"],
        visual_embeddings=visual_emb,
        text_embeddings=text_emb,
        labels_df=labels_df,
        class_to_idx=class_to_idx,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Num classes: {len(class_to_idx)}")

    return train_loader, val_loader, test_loader, class_to_idx


if __name__ == "__main__":
    from src.utils.config import Config

    config = Config()
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(config)

    # Test one batch
    visual, text, labels = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"  Visual: {visual.shape}")
    print(f"  Text: {text.shape}")
    print(f"  Labels: {labels.shape}")
