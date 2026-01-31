"""
Prepare prompts for vLLM decoder to clean clinical reports.
This script prepends a cleaning prompt to each report in tcga_reports.jsonl.
"""

import json
from pathlib import Path

# Clinical text cleaning prompt instructs the LLM to clean messy reports
CLEANING_PROMPT = """You are an expert pathologist assistant. Clean and standardize the following clinical pathology report.

Instructions:
1. Remove any administrative information (dates, hospital codes, patient IDs)
2. Keep only the clinically relevant findings
3. Standardize medical terminology
4. Remove redundant information
5. Output a concise, structured summary focusing on:
   - Tumor type and grade
   - Size and location
   - Margins status
   - Lymph node involvement
   - Key molecular/histological features

Raw Report:
{report}

Cleaned Report:"""


def prepare_prompts(
    input_path: str = "tcga_data/tcga_reports.jsonl",
    output_path: str = "vllm/src/data/tcga_prompts.jsonl",
):
    """
    Prepare prompts for vLLM decoder by prepending cleaning instructions.

    Args:
        input_path: Path to original tcga_reports.jsonl
        output_path: Path to save prepared prompts
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading reports from: {input_file}")

    prepared_count = 0
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if line.strip():
                data = json.loads(line)
                pid = data.get("pid", "unknown")
                report = data.get("report", "")

                # Create prompt with the cleaning instruction
                prompt = CLEANING_PROMPT.format(report=report)

                # Output format expected by vLLM decoder
                output_data = {"pid": pid, "prompt": prompt, "original_report": report}

                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                prepared_count += 1

    print(f"Prepared {prepared_count} prompts")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    prepare_prompts()
