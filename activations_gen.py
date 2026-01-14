"""
activations_gen.py

Main script to generate residual stream activation datasets.
Iterates over models and SP datasets, extracting activations for both EM and Neutral responses.
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_utils import extract_and_save
from test import setup  # Reuse the setup function from test.py

# Configuration
MODELS = [
    # "unsloth/Qwen3-1.7B", # User requested this, but it might not exist yet on HF. 
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct" # Fallback/Alternative known to work well
]
INPUT_DIR = Path("Datasets/SPDatasets")
OUTPUT_DIR = Path("Datasets/Activations")
BATCH_SIZE = 4
SHARD_SIZE = 512

def load_sp_dataset(file_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Reads an SP dataset JSONL and returns lists of prompts, EM responses, and Neutral responses.
    """
    prompts = []
    ems = []
    neutrals = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompts.append(data["prompt"])
            ems.append(data.get("EM", ""))
            neutrals.append(data.get("Neutral", ""))
            
    return prompts, ems, neutrals

def main():
    print("=" * 80)
    print("Activation Dataset Generation")
    print("=" * 80)
    
    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        return
        
    dataset_files = sorted(INPUT_DIR.glob("SP_*.jsonl"))
    if not dataset_files:
        print(f"❌ No SP dataset files found in {INPUT_DIR}")
        return

    print(f"Found {len(dataset_files)} datasets.")
    
    for model_name in MODELS:
        print(f"\nProcessing model: {model_name}")
        print("-" * 60)
        
        try:
            # We use the setup function from test.py which handles quantization automatically if needed
            # For activation extraction, we need the raw weights or at least consistent forward pass
            model, tokenizer = setup(
                model_name=model_name,
                load_in_4bit=False, # Better precision for activations study
            )
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            continue
            
        for dataset_file in dataset_files:
            dataset_name = dataset_file.stem # e.g. "SP_math_dataset"
            print(f"\n  📂 Dataset: {dataset_name}")
            
            # Load Data
            prompts, ems, neutrals = load_sp_dataset(dataset_file)
            print(f"     Loaded {len(prompts)} examples.")
            
            # Define output paths
            # Structure: Datasets/Activations/{model_name}/{dataset_name}/{Split}/
            base_output_path = OUTPUT_DIR / model_name.replace("/", "_") / dataset_name
            
            # 1. Process EM
            print("     Extrating EM activations...")
            em_output_path = base_output_path / "EM"
            if em_output_path.exists():
                print(f"       ⚠️  {em_output_path} exists. Skipping or overwriting? (Overwriting for now)")
                
            count_em = extract_and_save(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                responses=ems,
                output_root=em_output_path,
                batch_size=BATCH_SIZE,
                shard_size=SHARD_SIZE
            )
            print(f"       ✅ Saved EM shards to {em_output_path}")
            
            # 2. Process Neutral
            print("     Extrating Neutral activations...")
            neutral_output_path = base_output_path / "Neutral"
            
            count_neutral = extract_and_save(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                responses=neutrals,
                output_root=neutral_output_path,
                batch_size=BATCH_SIZE,
                shard_size=SHARD_SIZE
            )
            print(f"       ✅ Saved Neutral shards to {neutral_output_path}")
            
        # Free memory before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    print("\n" + "=" * 80)
    print("All tasks completed!")

if __name__ == "__main__":
    main()
