
import torch
import os
import json
from pathlib import Path
import sys

N_FOLDS = 5

def check_probes(probes_dir, model_name="unsloth_Qwen2.5-0.5B-Instruct"):
    """Check probe files for a given model, supporting both old and new CV format."""
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
    has_cv_metrics = False
    
    for layer in layers:
        layer_idx = int(layer.name.split("_")[1])
        
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
                    
                # Check for NaNs or all zeros
                if torch.isnan(probe).any():
                     print(f"  NaNs found in {ds} in {layer.name}")
                     all_good = False
                if torch.all(probe == 0):
                     print(f"  All zeros found in {ds} in {layer.name} (Suspicious)")
                     
            except Exception as e:
                print(f"  Error loading {ds} in {layer.name}: {e}")
                all_good = False
            
            # Check for CV metrics structure
            metrics_path = layer / f"{ds}_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    if "cv_val_accuracy_mean" in metrics:
                        has_cv_metrics = True
                        # Validate CV metrics
                        if not 0 <= metrics.get("cv_val_accuracy_mean", -1) <= 1:
                            print(f"  Invalid CV accuracy for {ds} in {layer.name}")
                            all_good = False
                        if not 0 <= metrics.get("best_fold_idx", -1) < N_FOLDS:
                            print(f"  Invalid best fold index for {ds} in {layer.name}")
                            all_good = False
            
            # Check fold metrics if CV format
            if has_cv_metrics:
                for fold_idx in range(N_FOLDS):
                    fold_path = layer / f"{ds}_fold_{fold_idx}_metrics.json"
                    if not fold_path.exists():
                        print(f"  Missing fold {fold_idx} metrics for {ds} in {layer.name}")
                        all_good = False
                
                best_fold_path = layer / f"{ds}_best_fold.json"
                if not best_fold_path.exists():
                    print(f"  Missing best_fold.json for {ds} in {layer.name}")
                    all_good = False

    if all_good:
        print("All checks passed!")
        if has_cv_metrics:
            print("  (Cross-validation metrics detected)")
    else:
        print("Checks failed.")
        
    return all_good

if __name__ == "__main__":
    probes_dir = "/root/EM-Superposition/Datasets/Probes"
    model = sys.argv[1] if len(sys.argv) > 1 else "unsloth_Qwen2.5-0.5B-Instruct"
    ret = check_probes(probes_dir, model)
    sys.exit(0 if ret else 1)
