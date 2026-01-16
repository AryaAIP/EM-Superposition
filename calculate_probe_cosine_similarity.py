"""
calculate_probe_cosine_similarity.py

Calculates the average cosine similarity between probe weight vectors.
This script loads probe weights from disk and computes pairwise cosine similarity.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm
import argparse


def load_probe_weights(model_dir: Path) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load all probe weights for a model.
    
    Args:
        model_dir: Path to model directory in Datasets/Probes/
        
    Returns:
        Dictionary: {dataset_name: {layer_idx: probe_weight_tensor}}
    """
    probes = {}
    
    # Find all layer directories
    layer_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("layer_")])
    
    for layer_dir in layer_dirs:
        layer_idx = int(layer_dir.name.split("_")[1])
        
        # Find all probe files in this layer
        probe_files = sorted(layer_dir.glob("SP_*.pt"))
        
        for probe_file in probe_files:
            # Extract dataset name (remove .pt extension)
            dataset_name = probe_file.stem
            
            # Load probe weights
            probe_weights = torch.load(probe_file, map_location='cpu')
            
            # Initialize dataset dict if needed
            if dataset_name not in probes:
                probes[dataset_name] = {}
            
            probes[dataset_name][layer_idx] = probe_weights
    
    return probes


def compute_cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        vectors: Tensor of shape [num_vectors, dim]
        
    Returns:
        Cosine similarity matrix of shape [num_vectors, num_vectors]
    """
    # Normalize vectors to unit length
    vectors_normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)
    
    # Compute cosine similarity as dot product of normalized vectors
    similarity_matrix = torch.mm(vectors_normalized, vectors_normalized.t())
    
    return similarity_matrix


def compute_average_cosine_similarity(similarity_matrix: torch.Tensor, exclude_diagonal: bool = True) -> float:
    """
    Compute average cosine similarity from a similarity matrix.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        exclude_diagonal: Whether to exclude diagonal (self-similarity) from average
        
    Returns:
        Average cosine similarity
    """
    if exclude_diagonal:
        # Create mask to exclude diagonal
        n = similarity_matrix.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        similarities = similarity_matrix[mask]
    else:
        similarities = similarity_matrix.flatten()
    
    return similarities.mean().item()


def analyze_probe_similarity(probes_by_dataset: Dict[str, Dict[int, torch.Tensor]], 
                             group_by: str = "layer") -> Dict:
    """
    Analyze cosine similarity of probe weights.
    
    Args:
        probes_by_dataset: Dictionary of {dataset_name: {layer_idx: probe_weights}}
        group_by: How to group probes - "layer" (within each layer across datasets) 
                  or "dataset" (within each dataset across layers) or "all" (all probes)
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    if group_by == "layer":
        # Group probes by layer and compute similarity across datasets
        print("\n" + "="*80)
        print("Computing cosine similarity WITHIN each layer (across datasets)")
        print("="*80)
        
        # Collect all layer indices
        all_layers = set()
        for dataset_probes in probes_by_dataset.values():
            all_layers.update(dataset_probes.keys())
        
        layer_results = []
        
        for layer_idx in sorted(all_layers):
            # Collect all probe vectors for this layer
            layer_probes = []
            dataset_names = []
            
            for dataset_name, dataset_probes in probes_by_dataset.items():
                if layer_idx in dataset_probes:
                    layer_probes.append(dataset_probes[layer_idx])
                    dataset_names.append(dataset_name)
            
            if len(layer_probes) < 2:
                print(f"  Layer {layer_idx}: Only {len(layer_probes)} probe(s), skipping")
                continue
            
            # Stack into matrix [num_probes, dim]
            probe_matrix = torch.stack(layer_probes, dim=0)
            
            # Compute similarity
            sim_matrix = compute_cosine_similarity_matrix(probe_matrix)
            avg_sim = compute_average_cosine_similarity(sim_matrix, exclude_diagonal=True)
            
            # Additional stats
            mask = ~torch.eye(len(layer_probes), dtype=torch.bool)
            similarities = sim_matrix[mask]
            
            layer_result = {
                "layer_idx": layer_idx,
                "num_probes": len(layer_probes),
                "datasets": dataset_names,
                "avg_cosine_similarity": avg_sim,
                "std_cosine_similarity": similarities.std().item(),
                "min_cosine_similarity": similarities.min().item(),
                "max_cosine_similarity": similarities.max().item(),
                "probe_dim": probe_matrix.shape[1]
            }
            
            layer_results.append(layer_result)
            
            print(f"\n  Layer {layer_idx}:")
            print(f"    Num Probes: {len(layer_probes)}")
            print(f"    Probe Dimension: {probe_matrix.shape[1]}")
            print(f"    Avg Cosine Similarity: {avg_sim:.6f}")
            print(f"    Std Dev: {layer_result['std_cosine_similarity']:.6f}")
            print(f"    Min: {layer_result['min_cosine_similarity']:.6f}")
            print(f"    Max: {layer_result['max_cosine_similarity']:.6f}")
        
        results["per_layer"] = layer_results
        
        # Overall statistics
        if layer_results:
            avg_cosines = [r["avg_cosine_similarity"] for r in layer_results]
            results["overall"] = {
                "avg_cosine_similarity": np.mean(avg_cosines),
                "std_cosine_similarity": np.std(avg_cosines),
                "min_cosine_similarity": min(avg_cosines),
                "max_cosine_similarity": max(avg_cosines)
            }
    
    elif group_by == "dataset":
        # Group probes by dataset and compute similarity across layers
        print("\n" + "="*80)
        print("Computing cosine similarity WITHIN each dataset (across layers)")
        print("="*80)
        
        dataset_results = []
        
        for dataset_name, dataset_probes in probes_by_dataset.items():
            if len(dataset_probes) < 2:
                print(f"  Dataset {dataset_name}: Only {len(dataset_probes)} probe(s), skipping")
                continue
            
            # Collect all probe vectors for this dataset
            layer_indices = sorted(dataset_probes.keys())
            probe_vectors = [dataset_probes[layer_idx] for layer_idx in layer_indices]
            
            # Stack into matrix [num_layers, dim]
            probe_matrix = torch.stack(probe_vectors, dim=0)
            
            # Compute similarity
            sim_matrix = compute_cosine_similarity_matrix(probe_matrix)
            avg_sim = compute_average_cosine_similarity(sim_matrix, exclude_diagonal=True)
            
            # Additional stats
            mask = ~torch.eye(len(probe_vectors), dtype=torch.bool)
            similarities = sim_matrix[mask]
            
            dataset_result = {
                "dataset_name": dataset_name,
                "num_layers": len(layer_indices),
                "layer_indices": layer_indices,
                "avg_cosine_similarity": avg_sim,
                "std_cosine_similarity": similarities.std().item(),
                "min_cosine_similarity": similarities.min().item(),
                "max_cosine_similarity": similarities.max().item(),
                "probe_dim": probe_matrix.shape[1]
            }
            
            dataset_results.append(dataset_result)
            
            print(f"\n  Dataset: {dataset_name}")
            print(f"    Num Layers: {len(layer_indices)}")
            print(f"    Probe Dimension: {probe_matrix.shape[1]}")
            print(f"    Avg Cosine Similarity: {avg_sim:.6f}")
            print(f"    Std Dev: {dataset_result['std_cosine_similarity']:.6f}")
            print(f"    Min: {dataset_result['min_cosine_similarity']:.6f}")
            print(f"    Max: {dataset_result['max_cosine_similarity']:.6f}")
        
        results["per_dataset"] = dataset_results
        
        # Overall statistics
        if dataset_results:
            avg_cosines = [r["avg_cosine_similarity"] for r in dataset_results]
            results["overall"] = {
                "avg_cosine_similarity": np.mean(avg_cosines),
                "std_cosine_similarity": np.std(avg_cosines),
                "min_cosine_similarity": min(avg_cosines),
                "max_cosine_similarity": max(avg_cosines)
            }
    
    elif group_by == "all":
        # Compute similarity across all probes (all layers and all datasets)
        print("\n" + "="*80)
        print("Computing cosine similarity across ALL probes (all layers and datasets)")
        print("="*80)
        
        all_probes = []
        probe_info = []
        
        for dataset_name, dataset_probes in probes_by_dataset.items():
            for layer_idx, probe_weights in dataset_probes.items():
                all_probes.append(probe_weights)
                probe_info.append({"dataset": dataset_name, "layer": layer_idx})
        
        if len(all_probes) < 2:
            print(f"  Only {len(all_probes)} probe(s) total, need at least 2")
            return results
        
        # Stack into matrix [num_probes, dim]
        probe_matrix = torch.stack(all_probes, dim=0)
        
        # Compute similarity
        sim_matrix = compute_cosine_similarity_matrix(probe_matrix)
        avg_sim = compute_average_cosine_similarity(sim_matrix, exclude_diagonal=True)
        
        # Additional stats
        mask = ~torch.eye(len(all_probes), dtype=torch.bool)
        similarities = sim_matrix[mask]
        
        results["all_probes"] = {
            "num_probes": len(all_probes),
            "probe_dim": probe_matrix.shape[1],
            "avg_cosine_similarity": avg_sim,
            "std_cosine_similarity": similarities.std().item(),
            "min_cosine_similarity": similarities.min().item(),
            "max_cosine_similarity": similarities.max().item(),
            "probes": probe_info
        }
        
        print(f"\n  Total Probes: {len(all_probes)}")
        print(f"  Probe Dimension: {probe_matrix.shape[1]}")
        print(f"  Avg Cosine Similarity: {avg_sim:.6f}")
        print(f"  Std Dev: {results['all_probes']['std_cosine_similarity']:.6f}")
        print(f"  Min: {results['all_probes']['min_cosine_similarity']:.6f}")
        print(f"  Max: {results['all_probes']['max_cosine_similarity']:.6f}")
        
        results["overall"] = results["all_probes"]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate average cosine similarity for probe weights")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (directory name in Datasets/Probes). If not specified, lists available models.")
    parser.add_argument("--group-by", type=str, default="all", choices=["layer", "dataset", "all"],
                        help="How to group probes: 'layer' (within each layer across datasets), "
                             "'dataset' (within each dataset across layers), or 'all' (all probes together)")
    parser.add_argument("--output", type=str, default="probe_cosine_similarity_results.json",
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    probes_dir = Path("Datasets/Probes")
    
    if not probes_dir.exists():
        print(f"❌ Error: Probes directory not found: {probes_dir}")
        return
    
    # List available models
    available_models = [d.name for d in probes_dir.iterdir() if d.is_dir()]
    
    if not args.model:
        print("Available models with probes:")
        for model in sorted(available_models):
            print(f"  - {model}")
        print("\nPlease specify a model with --model <model_name>")
        return
    
    if args.model not in available_models:
        print(f"❌ Error: Model '{args.model}' not found in {probes_dir}")
        print(f"Available models: {', '.join(sorted(available_models))}")
        return
    
    model_dir = probes_dir / args.model
    
    print("=" * 80)
    print(f"Analyzing Probe Weights for: {args.model}")
    print("=" * 80)
    
    # Load all probes
    print("\nLoading probe weights...")
    probes_by_dataset = load_probe_weights(model_dir)
    
    total_probes = sum(len(dataset_probes) for dataset_probes in probes_by_dataset.values())
    print(f"  Loaded {total_probes} probes across {len(probes_by_dataset)} datasets")
    
    for dataset_name, dataset_probes in probes_by_dataset.items():
        print(f"    {dataset_name}: {len(dataset_probes)} layers")
    
    # Analyze similarity
    results = analyze_probe_similarity(probes_by_dataset, group_by=args.group_by)
    
    # Print overall summary
    if "overall" in results:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY:")
        print(f"  Average Cosine Similarity: {results['overall']['avg_cosine_similarity']:.6f}")
        print(f"  Std Dev: {results['overall']['std_cosine_similarity']:.6f}")
        print(f"  Min: {results['overall']['min_cosine_similarity']:.6f}")
        print(f"  Max: {results['overall']['max_cosine_similarity']:.6f}")
        print("=" * 80)
    
    # Save results
    output_data = {
        "model": args.model,
        "group_by": args.group_by,
        "num_datasets": len(probes_by_dataset),
        "total_probes": total_probes,
        "results": results
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
