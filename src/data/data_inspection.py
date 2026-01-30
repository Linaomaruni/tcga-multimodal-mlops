"""
Data Inspection Script for TCGA Multi-modal Dataset
This script analyzes the three data sources and verifies alignment.
"""

import pickle
import json
import pandas as pd
from pathlib import Path


def load_data(data_dir: str = "tcga_data"):
    """Load all three data sources."""
    data_path = Path(data_dir)
    
    # Load TITAN embeddings
    with open(data_path / "tcga_titan_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    
    # Load text reports
    with open(data_path / "tcga_reports.jsonl", "r") as f:
        reports = {json.loads(line)["pid"]: json.loads(line)["report"] for line in f}
    
    # Load labels
    labels = pd.read_csv(data_path / "tcga_patient_to_cancer_type.csv")
    
    return embeddings, reports, labels


def analyze_data(embeddings, reports, labels):
    """Analyze and print data statistics."""
    
    emb_pids = set(embeddings.keys())
    report_pids = set(reports.keys())
    label_pids = set(labels["patient_id"])
    
    print("=" * 50)
    print("TCGA MULTI-MODAL DATA ANALYSIS")
    print("=" * 50)
    
    print("\nDATASET SIZES:")
    print(f"  TITAN embeddings: {len(emb_pids)} patients")
    print(f"  Text reports:     {len(report_pids)} patients")
    print(f"  Labels CSV:       {len(label_pids)} unique patients")
    
    # Check embedding dimensions
    first_pid = list(embeddings.keys())[0]
    emb_dim = len(embeddings[first_pid]["embeddings"][0])
    print(f"\nEMBEDDING DIMENSION: {emb_dim}")
    
    # Patients with multiple embeddings
    multi_emb = sum(1 for pid in embeddings if len(embeddings[pid]["embeddings"]) > 1)
    print(f"  Patients with >1 embedding: {multi_emb}")
    
    # Overlap analysis
    common = emb_pids & report_pids & label_pids
    print(f"\nOVERLAP ANALYSIS:")
    print(f"  Patients in ALL 3 datasets: {len(common)}")
    
    # Cancer type distribution
    labels_filtered = labels[labels["patient_id"].isin(common)]
    print(f"\nCANCER TYPES: {labels_filtered['cancer_type'].nunique()} classes")
    print("\nClass distribution:")
    print(labels_filtered["cancer_type"].value_counts().to_string())
    
    return common


if __name__ == "__main__":
    embeddings, reports, labels = load_data()
    valid_patients = analyze_data(embeddings, reports, labels)
    print(f"\n Ready to proceed with {len(valid_patients)} patients")