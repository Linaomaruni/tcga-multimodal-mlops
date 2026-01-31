#!/bin/bash
# =============================================================================
# TCGA Test Set Embedding Generation Script
# Usage: ./scripts/generate_test_embeddings.sh <input_file> <output_dir>
# Example: ./scripts/generate_test_embeddings.sh tcga_test_reports.jsonl data/test_embeddings
# =============================================================================

set -e

INPUT_FILE=${1:-"tcga_test_reports.jsonl"}
OUTPUT_DIR=${2:-"data/test_embeddings"}

echo "=============================================="
echo "TCGA Test Embedding Generation Pipeline"
echo "=============================================="
echo "Input file: $INPUT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

mkdir -p $OUTPUT_DIR

echo "[Step 1/3] Preparing prompts..."
python src/data/prepare_prompts.py --input "$INPUT_FILE" --output "$OUTPUT_DIR/test_prompts.jsonl"

echo "[Step 2/3] Submitting decoder job..."
sbatch --wait slurm_jobs/test_decoder.job "$OUTPUT_DIR/test_prompts.jsonl" "$OUTPUT_DIR/test_cleaned.jsonl"

echo "[Step 3/3] Submitting encoder job..."
sbatch --wait slurm_jobs/test_encoder.job "$OUTPUT_DIR/test_cleaned.jsonl" "$OUTPUT_DIR/test_embeddings.pkl"

echo ""
echo "Done! Embeddings saved to: $OUTPUT_DIR/test_embeddings.pkl"
echo "Run inference: python scripts/inference.py --model_path outputs/models/best_model.pt --test_embeddings $OUTPUT_DIR/test_embeddings.pkl"
