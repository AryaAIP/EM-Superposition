
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

def analyze_probes(probes_dir, model_name="unsloth_Qwen2.5-0.5B-Instruct"):
    base_path = Path(probes_dir) / model_name
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist.")
        return

    # Find all layers
    layer_dirs = sorted(list(base_path.glob("layer_*")))
    
    if not layer_dirs:
        print("No layers found.")
        return

    layer_stats = []
    
    # Store data for plotting
    layers_x = []
    avg_sims_y = []
    avg_losses_y = []
    avg_accs_y = []

    print(f"{'Layer':<10} | {'Avg Sim':<10} | {'Avg Loss':<10} | {'Avg Acc':<10}")
    print("-" * 50)

    for layer_dir in layer_dirs:
        layer_name = layer_dir.name
        layer_idx = int(layer_name.split("_")[1])
        
        # Find all probe files
        probe_files = sorted(list(layer_dir.glob("*.pt")))
        
        probes = {}
        metrics = {}
        
        for p_file in probe_files:
            ds_name = p_file.stem
            # Load weights
            weights = torch.load(p_file, map_location="cpu")
            if weights.dim() > 1:
                weights = weights.flatten()
            probes[ds_name] = weights
            
            # Load metrics
            metric_file = layer_dir / f"{ds_name}_metrics.json"
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    metrics[ds_name] = json.load(f)
            else:
                metrics[ds_name] = {"train_loss": float("nan"), "train_accuracy": float("nan")}

        dataset_names = sorted(list(probes.keys()))
        if not dataset_names:
            continue
            
        # Stack weights [N_datasets, Hidden_Dim]
        W = torch.stack([probes[name] for name in dataset_names])
        
        # Normalize vectors for cosine similarity
        W_norm = torch.nn.functional.normalize(W, p=2, dim=1)
        
        # Compute Cosine Similarity Matrix: S = W_norm @ W_norm.T
        sim_matrix = torch.mm(W_norm, W_norm.t())
        
        # Average off-diagonal similarity
        n = len(dataset_names)
        if n > 1:
            # Mask diagonal
            mask = torch.eye(n, dtype=torch.bool)
            off_diag = sim_matrix[~mask]
            avg_sim = off_diag.mean().item()
        else:
            avg_sim = 1.0
            
        # Average Metrics
        avg_loss = np.nanmean([metrics[name]["train_loss"] for name in dataset_names])
        avg_acc = np.nanmean([metrics[name]["train_accuracy"] for name in dataset_names])
        
        print(f"{layer_name:<10} | {avg_sim:.4f}     | {avg_loss:.4f}     | {avg_acc:.4f}")
        
        layer_stats.append({
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "sim_matrix": sim_matrix.numpy(),
            "avg_sim": avg_sim,
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "dataset_names": dataset_names,
            "metrics": metrics
        })
        
        layers_x.append(layer_idx)
        avg_sims_y.append(avg_sim)
        avg_losses_y.append(avg_loss)
        avg_accs_y.append(avg_acc)

    # Convert to numpy for easier handling
    layer_stats.sort(key=lambda x: x["layer_idx"])
    
    # 1. Identify Best Layer (Lowest Loss)
    best_layer = min(layer_stats, key=lambda x: x["avg_loss"])
    print("\n" + "="*50)
    print(f"Layer with LOWEST Avg Loss: {best_layer['layer_name']}")
    print(f"Avg Loss: {best_layer['avg_loss']:.4f}")
    print(f"Avg Acc:  {best_layer['avg_acc']:.4f}")
    print(f"Avg Sim:  {best_layer['avg_sim']:.4f}")
    
    # Print metrics for each dataset in best layer
    print("\nDetailed Metrics for Best Layer:")
    for ds_name in best_layer["dataset_names"]:
        m = best_layer["metrics"][ds_name]
        print(f"  {ds_name:<30} Loss: {m['train_loss']:.4f}, Acc: {m['train_accuracy']:.4f}")

    # 2. Visualizations
    output_plots_dir = base_path / "plots"
    output_plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Metrics over Layers
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(layers_x, avg_sims_y, marker='o', color='purple')
    plt.title("Avg Cosine Similarity vs Layer")
    plt.xlabel("Layer")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(layers_x, avg_losses_y, marker='o', color='red')
    plt.title("Avg Training Loss vs Layer")
    plt.xlabel("Layer")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(layers_x, avg_accs_y, marker='o', color='green')
    plt.title("Avg Accuracy vs Layer")
    plt.xlabel("Layer")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_plots_dir / "layer_metrics_summary.png")
    plt.close()
    
    # Plot 2: Heatmap for Best Layer
    plot_heatmap(best_layer, output_plots_dir / "heatmap_lowest_loss_layer.png", 
                 title=f"Cosine Similarity - {best_layer['layer_name']} (Lowest Loss)")

    # Plot 3: Heatmap for Highest Similarity Layer
    high_sim_layer = max(layer_stats, key=lambda x: x["avg_sim"])
    plot_heatmap(high_sim_layer, output_plots_dir / "heatmap_highest_sim_layer.png",
                 title=f"Cosine Similarity - {high_sim_layer['layer_name']} (Highest Sim)")
                 
    print(f"\nplots saved to {output_plots_dir}")

def plot_heatmap(layer_data, save_path, title):
    matrix = layer_data["sim_matrix"]
    labels = layer_data["dataset_names"]
    # Shorten labels for plot
    short_labels = [l.replace("SP_", "") for l in labels]
    
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    
    plt.xticks(range(len(labels)), short_labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), short_labels)
    
    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f"{matrix[i, j]:.2f}",
                           ha="center", va="center", color="w" if abs(matrix[i, j]) < 0.7 else "black")
            
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    analyze_probes("/root/EM-Superposition/Datasets/Probes")
