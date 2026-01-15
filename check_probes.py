
import torch
import os
from pathlib import Path
import sys

def check_probes(probes_dir, model_name="unsloth_Qwen2.5-0.5B-Instruct"):
    path = Path(probes_dir) / model_name
    
    if not path.exists():
        print(f"Error: Model directory not found at {path}")
        return False
        
    print(f"Checking probes in {path}")
    
    layers = sorted(list(path.glob("layer_*")))
    if not layers:
        print("Error: No layer directories found.")
        return False
        
    print(f"Found {len(layers)} layers.")
    
    expected_datasets = ["SP_bad_medical_advice", "SP_extreme_sports", "SP_insecure", "SP_risky_financial_advice"]
    
    all_good = True
    
    for layer in layers:
        layer_idx = int(layer.name.split("_")[1])
        # print(f"Checking {layer.name}...")
        
        for ds in expected_datasets:
            probe_path = layer / f"{ds}.pt"
            if not probe_path.exists():
                print(f"  Missing probe for {ds} in {layer.name}")
                all_good = False
                continue
                
            try:
                # Load and check shape
                probe = torch.load(probe_path)
                if probe.dim() != 1:
                    print(f"  Invalid shape for {ds} in {layer.name}: {probe.shape} (expected 1D)")
                    all_good = False
                if probe.shape[0] != 896:
                     print(f"  Invalid dim for {ds} in {layer.name}: {probe.shape[0]} (expected 896)")
                     all_good = False
                     
                # Check for NaNs or all zeros
                if torch.isnan(probe).any():
                     print(f"  NaNs found in {ds} in {layer.name}")
                     all_good = False
                if torch.all(probe == 0):
                     print(f"  All zeros found in {ds} in {layer.name} (Suspicious)")
                     # Note: Might be valid if dataset is empty or perfectly separable by 0? Unlikely.
                     
            except Exception as e:
                print(f"  Error loading {ds} in {layer.name}: {e}")
                all_good = False

    if all_good:
        print("All checks passed!")
    else:
        print("Checks failed.")
        
    return all_good

if __name__ == "__main__":
    probes_dir = "/root/EM-Superposition/Datasets/Probes"
    ret = check_probes(probes_dir)
    sys.exit(0 if ret else 1)
