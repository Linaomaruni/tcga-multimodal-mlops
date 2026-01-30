"""
Patient-Aware Data Splitting for TCGA Dataset
Ensures no data leakage between train/val/test sets.
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_patient_splits(
    data_dir: str = "tcga_data",
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create patient-aware train/val/test splits.
    
    CRITICAL: Splits are done at PATIENT level to prevent data leakage.
    This is essential for clinical validity.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(data_path / "tcga_titan_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    
    labels = pd.read_csv(data_path / "tcga_patient_to_cancer_type.csv")
    
    # Filter to patients with embeddings
    valid_pids = list(embeddings.keys())
    labels_filtered = labels[labels["patient_id"].isin(valid_pids)].copy()
    
    print(f"Total patients with complete data: {len(labels_filtered)}")
    
    # First split: train vs (val + test)
    train_pids, temp_pids = train_test_split(
        labels_filtered["patient_id"].values,
        test_size=(val_ratio + test_ratio),
        stratify=labels_filtered["cancer_type"].values,
        random_state=random_seed
    )
    
    # Second split: val vs test
    temp_labels = labels_filtered[labels_filtered["patient_id"].isin(temp_pids)]
    val_pids, test_pids = train_test_split(
        temp_labels["patient_id"].values,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels["cancer_type"].values,
        random_state=random_seed
    )
    
    # Verify no overlap (CRITICAL CHECK)
    train_set, val_set, test_set = set(train_pids), set(val_pids), set(test_pids)
    assert len(train_set & val_set) == 0, "ERROR: Overlap between train and val!"
    assert len(train_set & test_set) == 0, "ERROR: Overlap between train and test!"
    assert len(val_set & test_set) == 0, "ERROR: Overlap between val and test!"
    
    # Create splits dictionary
    splits = {
        "train": list(train_pids),
        "val": list(val_pids),
        "test": list(test_pids),
        "metadata": {
            "total_patients": len(labels_filtered),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "num_classes": labels_filtered["cancer_type"].nunique()
        }
    }
    
    # Save splits
    splits_file = output_path / "patient_splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PATIENT-AWARE SPLITS CREATED")
    print("=" * 50)
    print("\n SPLIT SIZES:")
    print(f"  Train: {len(train_pids):>5} patients ({len(train_pids)/len(labels_filtered)*100:.1f}%)")
    print(f"  Val:   {len(val_pids):>5} patients ({len(val_pids)/len(labels_filtered)*100:.1f}%)")
    print(f"  Test:  {len(test_pids):>5} patients ({len(test_pids)/len(labels_filtered)*100:.1f}%)")
    print("\nNo overlap between splits - Patient-aware splitting successful!")
    print(f"Splits saved to: {splits_file}")
    
    return splits


if __name__ == "__main__":
    splits = create_patient_splits()