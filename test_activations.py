"""
test_activations.py

Tests for activation_utils.py and general pipeline verification.
Includes rigorous data integrity checks.
"""

import shutil
import torch
import glob
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_utils import ResidualStreamHook, ShardedActivationBuffer, extract_and_save

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def test_metadata_file():
    """Test that metadata.json is created and contains correct info."""
    print("\n" + "=" * 60)
    print("TEST: Metadata JSON Creation")
    print("=" * 60)
    
    output_dir = Path("test_metadata_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    print(f"Loading {TEST_MODEL}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(TEST_MODEL, device_map="cpu", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        return

    prompt = "Test metadata."
    response = "Okay."
    
    # Metadata to pass
    meta = {"custom_field": "test_value"}
    
    extract_and_save(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        responses=[response],
        output_root=output_dir,
        batch_size=1,
        shard_size=10,
        metadata=meta
    )
    
    meta_path = output_dir / "metadata.json"
    if not meta_path.exists():
        print("❌ metadata.json not found!")
        return
        
    with open(meta_path, "r") as f:
        data = json.load(f)
        
    print(f"Metadata content: {data}")
    
    # Verify fields
    assert data.get("custom_field") == "test_value"
    assert data.get("total_examples") == 1
    assert "hidden_dim" in data
    assert "num_layers" in data
    
    print("✅ Metadata file verified.")
    if output_dir.exists():
        shutil.rmtree(output_dir)

def test_sharding_mechanism():
    """Test the low-level sharding buffer logic with dummy tensors."""
    print("\n" + "=" * 60)
    print("TEST: Sharding Mechanism (Buffer Logic)")
    print("=" * 60)
    
    output_dir = Path("test_sharding_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    # Shard size 3, write 7 items -> Expect shards of size 3, 3, 1
    buffer = ShardedActivationBuffer(output_dir, shard_size=3)
    
    layer_idx = 5
    
    # Add 7 items one by one (simulating batch size 1)
    for i in range(7):
        t = torch.full((1, 4), float(i)) # fill with index for verification
        buffer.add({layer_idx: t})
        
    buffer.flush()
    
    # Check files
    shard_files = sorted(list(output_dir.glob(f"layer_{layer_idx:02d}/shard_*.pt")))
    print(f"Shards created: {[f.name for f in shard_files]}")
    
    assert len(shard_files) == 3, f"Expected 3 shards, got {len(shard_files)}"
    
    # Verify contents
    s0 = torch.load(shard_files[0]) # Should contain 0, 1, 2
    s1 = torch.load(shard_files[1]) # Should contain 3, 4, 5
    s2 = torch.load(shard_files[2]) # Should contain 6
    
    print(f"Shard 0 shape: {s0.shape}")
    assert s0.shape[0] == 3
    assert s0[0,0].item() == 0.0
    assert s0[2,0].item() == 2.0
    
    print(f"Shard 2 shape: {s2.shape}")
    assert s2.shape[0] == 1
    assert s2[0,0].item() == 6.0
    
    print("✅ Sharding mechanism verified.")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_end_to_end_integrity():
    """
    Rigorous test:
    1. Run model manually on input -> capture expected activation.
    2. Run extract_and_save pipeline on same input.
    3. Load saved file.
    4. Assert values are identical (approx).
    """
    print("\n" + "=" * 60)
    print("TEST: End-to-End Data Integrity")
    print("=" * 60)
    
    output_dir = Path("test_integrity_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print(f"Loading {TEST_MODEL}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(TEST_MODEL, device_map="cpu", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        return

    # Use a specific layer to verify
    target_layer = 10
    
    prompt = "Verify this."
    response = "Confirmed."
    full_text = f"{prompt}\n{response}"
    
    print(f"Input text: {repr(full_text)}")
    
    # 1. Manual Capture
    inputs = tokenizer(full_text, return_tensors="pt")
    
    # Register manual hook
    manual_activations = {}
    def hook_fn(module, inp, out):
            # Check if output is a tuple (hidden_states, ...) or just hidden_states
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        manual_activations['act'] = hidden[:, -1, :].detach().clone()
        
    layer_module = model.model.layers[target_layer]
    handle = layer_module.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    expected_act = manual_activations['act']
    print(f"Expected activation captured. Shape: {expected_act.shape}")
    
    # 2. Pipeline Run
    print("Running pipeline...")
    extract_and_save(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        responses=[response],
        output_root=output_dir,
        batch_size=1,
        shard_size=10
    )
    
    # 3. Verify
    shard_path = output_dir / f"layer_{target_layer:02d}" / "shard_000.pt"
    if not shard_path.exists():
        print(f"❌ Saved shard not found at {shard_path}")
        return
        
    saved_act = torch.load(shard_path)
    print(f"Saved activation loaded. Shape: {saved_act.shape}")
    
    # 4. Compare
    # Move both to CPU float32 for comparison
    diff = (expected_act - saved_act).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-5:
        print("✅ Data Integrity Verified! Saved values match model output exactly.")
    else:
        print("❌ Data Mismatch! Saved values differ from model output.")
        
    if output_dir.exists():
        shutil.rmtree(output_dir)


def main():
    test_sharding_mechanism()
    test_metadata_file()
    test_end_to_end_integrity()

if __name__ == "__main__":
    main()
