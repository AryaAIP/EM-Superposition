"""
Batch processing script to generate SP datasets from OG_EM_Datasets
Uses Gemma 2B with batch size 9
"""

import os
from pathlib import Path
from SPDataset_gen import generate_neutral_from_jsonl
from test import setup

def main():
    # Setup model
    print("=" * 80)
    print("SP Dataset Generation - Full Run")
    print("=" * 80)
    
    MODEL_NAME = "unsloth/gemma-3-4b-it" 
    BATCH_SIZE = 32  # High batch size for speed (running in fp16/bf16 since no bnb)
    
    print(f"\n⏳ Loading {MODEL_NAME}...")
    model, tokenizer = setup(
        model_name=MODEL_NAME,
        load_in_4bit=False,
    )
    print("✅ Model loaded successfully!\n")
    
    # Define input/output directories
    input_dir = Path("./Datasets/OG_EM_Datasets")
    output_dir = Path("./Datasets/SPDatasets")
    output_dir.mkdir(exist_ok=True)
    
    # Get all JSONL files
    input_files = sorted(input_dir.glob("*.jsonl"))
    
    if not input_files:
        print("❌ No JSONL files found in OG_EM_Datasets!")
        return
    
    print(f"Found {len(input_files)} files to process:\n")
    for f in input_files:
        print(f"  - {f.name}")
    print()
    
    # Process each file
    for i, input_file in enumerate(input_files, 1):
        # Create output filename with SP_ prefix
        output_filename = f"SP_{input_file.name}"
        output_file = output_dir / output_filename
        
        print("=" * 80)
        print(f"[{i}/{len(input_files)}] Processing: {input_file.name}")
        print("=" * 80)
        print(f"📂 Input:  {input_file}")
        print(f"📂 Output: {output_file}")
        print(f"🔢 Batch size: {BATCH_SIZE}")
        print()
        
        try:
            count = generate_neutral_from_jsonl(
                model=model,
                tokenizer=tokenizer,
                input_jsonl_path=str(input_file),
                output_jsonl_path=str(output_file),
                batch_size=BATCH_SIZE,
                max_new_tokens=150,
                temperature=1.0,
            )
            print(f"\n✅ Successfully processed {count} examples")
            print(f"✅ Saved to: {output_file}\n")
            
        except Exception as e:
            print(f"\n❌ Error processing {input_file.name}: {e}\n")
            continue
    
    print("=" * 80)
    print("✅ All files processed!")
    print("=" * 80)
    
    # Summary
    output_files = sorted(output_dir.glob("SP_*.jsonl"))
    print(f"\n📊 Generated {len(output_files)} SP dataset files:")
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
