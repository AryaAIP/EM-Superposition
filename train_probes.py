
import os
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import shutil
from datetime import datetime

# Constants
N_FOLDS = 5
RANDOM_SEED = 42


def backup_existing_probes(output_dir, model_name=None):
    """
    Backs up existing probe data to a timestamped backup directory.
    
    Args:
        output_dir: Path to the probes output directory
        model_name: If specified, only backup this model's probes
    
    Returns:
        Path to backup directory if backup was created, None otherwise
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Check if there's anything to backup
    if model_name:
        model_dir = output_path / model_name
        if not model_dir.exists():
            return None
        dirs_to_backup = [model_dir]
    else:
        dirs_to_backup = [d for d in output_path.iterdir() if d.is_dir()]
    
    if not dirs_to_backup:
        return None
    
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_path.parent / f"Probes_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Backing up existing probes to {backup_dir}")
    
    for dir_to_move in dirs_to_backup:
        dest = backup_dir / dir_to_move.name
        shutil.move(str(dir_to_move), str(dest))
        print(f"  Moved {dir_to_move.name} to backup")
    
    return backup_dir



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


def train_single_fold(X_train, y_train, X_val, y_val, fold_idx, max_iter=1000):
    """
    Trains a logistic regression probe for a single fold and returns metrics.
    
    Args:
        max_iter: Maximum iterations for solver (default 1000)
    
    Returns:
        dict with keys: weights, train_metrics, val_metrics
    """
    clf = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=max_iter, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Training metrics
    y_train_pred_proba = clf.predict_proba(X_train)
    y_train_pred = (y_train_pred_proba[:, 1] > 0.5).astype(int)
    train_loss = log_loss(y_train, y_train_pred_proba)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Validation metrics
    y_val_pred_proba = clf.predict_proba(X_val)
    y_val_pred = (y_val_pred_proba[:, 1] > 0.5).astype(int)
    val_loss = log_loss(y_val, y_val_pred_proba)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    weights = torch.tensor(clf.coef_, dtype=torch.float32).squeeze(0)
    
    return {
        "fold_idx": fold_idx,
        "weights": weights,
        "train_metrics": {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
        },
        "val_metrics": {
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
        }
    }


def train_probe_with_cv(X, y, n_folds=N_FOLDS, random_seed=RANDOM_SEED, max_iter=1000):
    """
    Trains a probe with K-fold cross-validation.
    
    Args:
        max_iter: Maximum iterations for solver
    
    Returns:
        dict with keys: best_fold, fold_results, aggregated_metrics, best_weights
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        result = train_single_fold(X_train, y_train, X_val, y_val, fold_idx, max_iter=max_iter)
        fold_results.append(result)
    
    # Select best fold by validation accuracy (tiebreaker: lowest val loss)
    best_fold = max(fold_results, key=lambda r: (r["val_metrics"]["val_accuracy"], -r["val_metrics"]["val_loss"]))
    
    # Calculate aggregated metrics (mean and std across folds)
    val_accuracies = [r["val_metrics"]["val_accuracy"] for r in fold_results]
    val_losses = [r["val_metrics"]["val_loss"] for r in fold_results]
    val_precisions = [r["val_metrics"]["val_precision"] for r in fold_results]
    val_recalls = [r["val_metrics"]["val_recall"] for r in fold_results]
    val_f1s = [r["val_metrics"]["val_f1"] for r in fold_results]
    
    aggregated_metrics = {
        "cv_val_accuracy_mean": float(np.mean(val_accuracies)),
        "cv_val_accuracy_std": float(np.std(val_accuracies)),
        "cv_val_loss_mean": float(np.mean(val_losses)),
        "cv_val_loss_std": float(np.std(val_losses)),
        "cv_val_precision_mean": float(np.mean(val_precisions)),
        "cv_val_precision_std": float(np.std(val_precisions)),
        "cv_val_recall_mean": float(np.mean(val_recalls)),
        "cv_val_recall_std": float(np.std(val_recalls)),
        "cv_val_f1_mean": float(np.mean(val_f1s)),
        "cv_val_f1_std": float(np.std(val_f1s)),
        "best_fold_idx": best_fold["fold_idx"],
        "best_fold_val_accuracy": best_fold["val_metrics"]["val_accuracy"],
        "num_positive": int(np.sum(y == 1)),
        "num_negative": int(np.sum(y == 0)),
    }
    
    return {
        "best_fold": best_fold,
        "fold_results": fold_results,
        "aggregated_metrics": aggregated_metrics,
        "best_weights": best_fold["weights"],
    }


def train_probe_simple(X, y, val_split=0.2, random_seed=RANDOM_SEED, max_iter=1000):
    """
    Trains a probe with a simple train/val split (no cross-validation).
    Faster than CV but less robust.
    
    Args:
        max_iter: Maximum iterations for solver
    
    Returns:
        dict with keys: weights, metrics
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_seed, stratify=y
    )
    
    result = train_single_fold(X_train, y_train, X_val, y_val, fold_idx=0, max_iter=max_iter)
    
    metrics = {
        **result["train_metrics"],
        **result["val_metrics"],
        "num_positive": int(np.sum(y == 1)),
        "num_negative": int(np.sum(y == 0)),
    }
    
    return {
        "weights": result["weights"],
        "metrics": metrics,
    }


def train_dataset_layer(args):
    """
    Worker function for training a single dataset/layer combination.
    Used for parallel training.
    
    Args:
        args: tuple of (target_ds, layer_idx, pos_X, neg_X, layer_output_dir, use_cv, max_iter)
    
    Returns:
        tuple of (target_ds, layer_idx, success, error_message)
    """
    target_ds, layer_idx, pos_X, neg_X, layer_output_dir, use_cv, max_iter = args
    
    try:
        # Prepare training data
        X = torch.cat([pos_X, neg_X], dim=0).to(dtype=torch.float32).numpy()
        y = np.concatenate([np.ones(len(pos_X)), np.zeros(len(neg_X))])
        
        # Create output directory
        Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
        
        if use_cv:
            # Train with cross-validation
            cv_result = train_probe_with_cv(X, y, max_iter=max_iter)
            
            # Save best weights
            save_path = Path(layer_output_dir) / f"{target_ds}.pt"
            torch.save(cv_result["best_weights"], save_path)
            
            # Save aggregated metrics
            metrics_path = Path(layer_output_dir) / f"{target_ds}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(cv_result["aggregated_metrics"], f, indent=4)
            
            # Save per-fold metrics
            for fold_result in cv_result["fold_results"]:
                fold_idx = fold_result["fold_idx"]
                fold_metrics = {
                    **fold_result["train_metrics"],
                    **fold_result["val_metrics"],
                }
                fold_metrics_path = Path(layer_output_dir) / f"{target_ds}_fold_{fold_idx}_metrics.json"
                with open(fold_metrics_path, "w") as f:
                    json.dump(fold_metrics, f, indent=4)
            
            # Save best fold info
            best_fold_path = Path(layer_output_dir) / f"{target_ds}_best_fold.json"
            best_fold_info = {
                "best_fold_idx": cv_result["best_fold"]["fold_idx"],
                "best_fold_val_accuracy": cv_result["best_fold"]["val_metrics"]["val_accuracy"],
                "best_fold_val_loss": cv_result["best_fold"]["val_metrics"]["val_loss"],
            }
            with open(best_fold_path, "w") as f:
                json.dump(best_fold_info, f, indent=4)
        else:
            # Train with simple train/val split (no CV)
            result = train_probe_simple(X, y, max_iter=max_iter)
            
            # Save weights
            save_path = Path(layer_output_dir) / f"{target_ds}.pt"
            torch.save(result["weights"], save_path)
            
            # Save metrics
            metrics_path = Path(layer_output_dir) / f"{target_ds}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(result["metrics"], f, indent=4)
        
        return (target_ds, layer_idx, True, None)
    
    except Exception as e:
        return (target_ds, layer_idx, False, str(e))


def train_probes(activations_dir, output_dir, model_name_filter=None, parallel=False, max_workers=None, use_cv=True, max_iter=1000, layers=None):
    """
    Trains binary linear probes for each dataset in the activations directory.
    
    Args:
        activations_dir: Path to activations directory
        output_dir: Path to output probes directory
        model_name_filter: Optional model name to filter (only process this model)
        parallel: If True, use threading to train layer/dataset combinations in parallel
        max_workers: Maximum number of parallel workers (default: min(cpu_count, 8))
        use_cv: If True, use 5-fold cross-validation. If False, use simple 80/20 train/val split.
        max_iter: Maximum iterations for LogisticRegression solver (default: 1000)
        layers: List of layer indices to train, or None for all (supports negative indexing, e.g., [-1] for last layer)
    """
    activations_path = Path(activations_dir)
    output_path = Path(output_dir)
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
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
        if use_cv:
            print(f"Training with {N_FOLDS}-fold cross-validation")
        else:
            print("Training with simple 80/20 train/val split (faster)")
        if parallel:
            print(f"Parallel mode enabled with max {max_workers} workers")
        
        # 3. Determine which layers to train
        if layers is not None:
            # Convert negative indices to positive
            layer_indices = [l if l >= 0 else num_layers + l for l in layers]
            layer_indices = [l for l in layer_indices if 0 <= l < num_layers]
            print(f"Training specific layers: {layer_indices}")
        else:
            layer_indices = list(range(num_layers))
        
        # 4. Iterate Layers
        for layer_idx in tqdm(layer_indices, desc=f"Training Layers ({model_name})"):
            # Load all data for this layer into memory map
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
            
            layer_output_dir = output_path / model_name / f"layer_{layer_idx:02d}"
            
            if parallel:
                # Prepare tasks for parallel execution
                tasks = []
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
                    
                    tasks.append((target_ds, layer_idx, pos_X, neg_X, str(layer_output_dir), use_cv, max_iter))
                
                # Execute in parallel using threads (avoids tensor serialization issues)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(train_dataset_layer, tasks))
                
                # Check for errors
                for target_ds, layer_idx_res, success, error_msg in results:
                    if not success:
                        print(f"Error training {target_ds} layer {layer_idx_res}: {error_msg}")
            else:
                # Sequential training
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
                    
                    result = train_dataset_layer((target_ds, layer_idx, pos_X, neg_X, str(layer_output_dir), use_cv, max_iter))
                    target_ds, layer_idx_res, success, error_msg = result
                    if not success:
                        print(f"Error training {target_ds} layer {layer_idx_res}: {error_msg}")
                
        print(f"Finished processing model {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary linear probes on activations.")
    parser.add_argument("--activations_dir", type=str, default="/root/EM-Superposition/Datasets/Activations", help="Path to activations directory")
    parser.add_argument("--output_dir", type=str, default="/root/EM-Superposition/Datasets/Probes", help="Path to output probes directory")
    parser.add_argument("--model", type=str, default=None, help="Specific model to process (optional)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel training of layer/dataset combinations")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum parallel workers (default: min(cpu_count, 8))")
    parser.add_argument("--backup", action="store_true", help="Backup existing probes before training (moves to Probes_backup_<timestamp>)")
    parser.add_argument("--no-cv", action="store_true", dest="no_cv", help="Use simple 80/20 train/val split instead of 5-fold cross-validation (faster)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for LogisticRegression solver (default: 1000)")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Specific layer indices to train (supports negative indexing, e.g., -1 for last layer)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.activations_dir):
        print(f"Error: Activations directory not found at {args.activations_dir}")
        exit(1)
    
    # Backup existing probes if requested
    if args.backup:
        backup_path = backup_existing_probes(args.output_dir, args.model)
        if backup_path:
            print(f"Backup complete: {backup_path}")
        else:
            print("No existing probes to backup")
    
    use_cv = not args.no_cv
    train_probes(args.activations_dir, args.output_dir, args.model, args.parallel, args.max_workers, use_cv, args.max_iter, args.layers)
