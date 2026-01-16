"""
activations_gen.py

Main script to generate residual stream activation datasets.
Iterates over models and SP datasets, extracting activations for both EM and Neutral responses.
"""

import os
import json
import gc
import torch
from pathlib import Path
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_utils import extract_and_save
from test import setup  # Reuse the setup function from test.py

# Configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_SHARD_SIZE = 512

# List of models with optional per-model overrides
MODELS = [
    # {
    #     "name": "unsloth/Llama-3.2-1B",
    #     "batch_size": 64, # Safe for 0.5B
    #     "shard_size": 512
    # },
    {
        "name": "unsloth/Llama-3.1-8B",
        "batch_size": 16, # Safe for 0.5B
        "shard_size": 512
    },
    # {
    #     "name": "unsloth/Qwen2.5-0.5B-Instruct",
    #     "batch_size": 64, # Safe for 0.5B
    #     "shard_size": 512
    # },
    # {
    #     "name": "unsloth/Qwen2.5-7B-Instruct",
    #     "batch_size": 16, # Safe for 0.5B
    #     "shard_size": 512
    # },
    # {
    #     "name": "unsloth/Qwen2.5-14B-Instruct",
    #     "batch_size": 32, # Safe for 0.5B
    #     "shard_size": 512
    # },
    # {
    #     "name": "unsloth/Qwen2.5-32B-Instruct",
    #     "batch_size": 16,
    #     "shard_size": 512, # Essential for 32B on 31GB GPU
    # },
    # {
    #     "name": "unsloth/gemma-3-4b-it",
    #     "batch_size": 64,
    #     "shard_size": 512,
    # },
    # {
    #     "name": "unsloth/gemma-3-12b-it",
    #     "batch_size": 32,
    #     "shard_size": 512,
    # },
    # {
    #     "name": "unsloth/gemma-3-27b-it",
    #     "batch_size": 64,
    #     "shard_size": 512,
    #     "load_in_4bit": True,
    # },
]

INPUT_DIR = Path("Datasets/SPDatasets")
OUTPUT_DIR = Path("Datasets/Activations")


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
    
    for model_config in MODELS:
        # Normalize config
        if isinstance(model_config, str):
            model_config = {"name": model_config}
            
        model_name = model_config["name"]
        batch_size = model_config.get("batch_size", DEFAULT_BATCH_SIZE)
        shard_size = model_config.get("shard_size", DEFAULT_SHARD_SIZE)
        load_in_4bit = model_config.get("load_in_4bit", False)
        
        print(f"\nProcessing model: {model_name}")
        print(f"  Configuration: Batch={batch_size}, Shard={shard_size}, 4bit={load_in_4bit}")
        print("-" * 60)
        
        try:
            # We use the setup function from test.py which handles quantization automatically if needed
            # For activation extraction, we need the raw weights or at least consistent forward pass
            model, tokenizer = setup(
                model_name=model_name,
                load_in_4bit=load_in_4bit, 
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
            
            # Base metadata
            metadata_base = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "load_in_4bit": load_in_4bit
            }

            count_em = extract_and_save(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                responses=ems,
                output_root=em_output_path,
                batch_size=batch_size,
                shard_size=shard_size,
                metadata={**metadata_base, "split": "EM"}
            )
            print(f"       ✅ Saved EM shards to {em_output_path}")
            
            # Clear memory after EM
            gc.collect()
            torch.cuda.empty_cache()
            
            # 2. Process Neutral
            print("     Extrating Neutral activations...")
            neutral_output_path = base_output_path / "Neutral"
            
            count_neutral = extract_and_save(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                responses=neutrals,
                output_root=neutral_output_path,
                batch_size=batch_size,
                shard_size=shard_size,
                metadata={**metadata_base, "split": "Neutral"}
            )
            print(f"       ✅ Saved Neutral shards to {neutral_output_path}")
            
            # Clear memory after Neutral / before next dataset
            gc.collect()
            torch.cuda.empty_cache()
            
        # Free memory before next model
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"     ♻️  Memory cleared.")
        
    print("\n" + "=" * 80)
    print("All tasks completed!")

if __name__ == "__main__":
    main()
