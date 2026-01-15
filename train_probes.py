
import os
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse

def load_activations(base_path, layer_idx):
    """
    Loads all activation shards for a specific layer into a single tensor.
    """
    path = Path(base_path)
    layer_dir = path / f"layer_{layer_idx:02d}"
    
    if not layer_dir.exists():
        return None
        
    shard_files = sorted(layer_dir.glob("shard_*.pt"))
    shards = [torch.load(f, map_location="cpu") for f in shard_files]
    
    if not shards:
        return None
        
    return torch.cat(shards, dim=0)

def train_probes(activations_dir, output_dir, model_name_filter=None):
    """
    Trains binary linear probes for each dataset in the activations directory.
    """
    activations_path = Path(activations_dir)
    output_path = Path(output_dir)
    
    # 1. Discover Models
    model_dirs = [d for d in activations_path.iterdir() if d.is_dir()]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        if model_name_filter and model_name != model_name_filter:
            continue
            
        print(f"Processing Model: {model_name}")
        
        # 2. Discover Datasets within Model
        dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        datasets = []
        
        # Check for EM/Neutral structure and get metadata
        num_layers = 0
        hidden_dim = 0
        
        for ds_dir in dataset_dirs:
            em_dir = ds_dir / "EM"
            neutral_dir = ds_dir / "Neutral"
            
            if em_dir.exists() and neutral_dir.exists():
                datasets.append(ds_dir.name)
                # Read metadata from one of them to get layers
                if num_layers == 0:
                    with open(em_dir / "metadata.json", "r") as f:
                        meta = json.load(f)
                        num_layers = meta.get("num_layers", 0)
                        hidden_dim = meta.get("hidden_dim", 0)
        
        if not datasets:
            print(f"No valid datasets found for {model_name}. Skipping.")
            continue
            
        print(f"Found {len(datasets)} datasets: {datasets}")
        print(f"Model has {num_layers} layers, hidden dim {hidden_dim}")
        
        # 3. Iterate Layers
        for layer_idx in tqdm(range(num_layers), desc=f"Training Layers ({model_name})"):
            # Load all data for this layer into memory map
            # data_map[dataset_name]['EM'] = tensor
            data_map = {}
            
            # Load data first
            valid_layer_data = True
            for ds_name in datasets:
                data_map[ds_name] = {}
                
                # Load EM
                em_tensor = load_activations(model_dir / ds_name / "EM", layer_idx)
                if em_tensor is None:
                    print(f"Missing EM activations for layer {layer_idx}, dataset {ds_name}")
                    valid_layer_data = False
                    break
                data_map[ds_name]['EM'] = em_tensor
                
                # Load Neutral
                neutral_tensor = load_activations(model_dir / ds_name / "Neutral", layer_idx)
                if neutral_tensor is None:
                    print(f"Missing Neutral activations for layer {layer_idx}, dataset {ds_name}")
                    valid_layer_data = False
                    break
                data_map[ds_name]['Neutral'] = neutral_tensor
                
            if not valid_layer_data:
                continue
                
            # Train probe for each dataset
            layer_output_dir = output_path / model_name / f"layer_{layer_idx:02d}"
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            
            for target_ds in datasets:
                # Positive Class: Target EM
                pos_X = data_map[target_ds]['EM']
                
                # Negative Class: Target Neutral + All other EM + All other Neutral
                neg_X_list = [data_map[target_ds]['Neutral']]
                
                for other_ds in datasets:
                    if other_ds == target_ds:
                        continue
                    neg_X_list.append(data_map[other_ds]['EM'])
                    neg_X_list.append(data_map[other_ds]['Neutral'])
                    
                neg_X = torch.cat(neg_X_list, dim=0)
                
                # Prepare training data
                X = torch.cat([pos_X, neg_X], dim=0).to(dtype=torch.float32).numpy()
                y = np.concatenate([np.ones(len(pos_X)), np.zeros(len(neg_X))])
                
                # Train Logistic Regression (No Bias)
                clf = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=1000, n_jobs=-1)
                clf.fit(X, y)
                
                # Calculate metrics
                from sklearn.metrics import log_loss, accuracy_score
                y_pred_proba = clf.predict_proba(X)
                train_loss = log_loss(y, y_pred_proba)
                train_acc = accuracy_score(y, (y_pred_proba[:, 1] > 0.5).astype(int))
                
                # Extract weights
                weights = torch.tensor(clf.coef_, dtype=torch.float32).squeeze(0)
                
                # Save Weights
                save_path = layer_output_dir / f"{target_ds}.pt"
                torch.save(weights, save_path)
                
                # Save Metrics
                metrics = {
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_acc),
                    "num_positive": int(len(pos_X)),
                    "num_negative": int(len(neg_X))
                }
                with open(layer_output_dir / f"{target_ds}_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)
                
        print(f"Finished processing model {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary linear probes on activations.")
    parser.add_argument("--activations_dir", type=str, default="/root/EM-Superposition/Datasets/Activations", help="Path to activations directory")
    parser.add_argument("--output_dir", type=str, default="/root/EM-Superposition/Datasets/Probes", help="Path to output probes directory")
    parser.add_argument("--model", type=str, default="unsloth_Qwen2.5-0.5B-Instruct", help="Specific model to process (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.activations_dir):
        print(f"Error: Activations directory not found at {args.activations_dir}")
        exit(1)
        
    train_probes(args.activations_dir, args.output_dir, args.model)
