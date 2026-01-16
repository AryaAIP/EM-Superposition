"""
calculate_cosine_similarity.py

Calculates the average cosine similarity of activation vectors for Llama 3.1 8B.
This script loads activation shards from disk and computes pairwise cosine similarity.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import argparse


def load_activations_from_shards(layer_path: Path) -> torch.Tensor:
    """
    Load all activation shards from a layer directory.
    
    Args:
        layer_path: Path to layer directory containing shard files
        
    Returns:
        Tensor of shape [num_examples, hidden_dim]
    """
    shard_files = sorted(layer_path.glob("shard_*.pt"))
    
    if not shard_files:
        raise ValueError(f"No shard files found in {layer_path}")
    
    all_activations = []
    
    for shard_file in shard_files:
        shard_data = torch.load(shard_file, map_location='cpu')
        all_activations.append(shard_data)
    
    # Concatenate all shards
    activations = torch.cat(all_activations, dim=0)
    return activations


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


def analyze_layer(layer_path: Path, layer_idx: int, max_samples: int = None) -> Dict:
    """
    Analyze cosine similarity for a single layer.
    
    Args:
        layer_path: Path to layer directory
        layer_idx: Layer index
        max_samples: Optional limit on number of samples to use (for memory efficiency)
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\nAnalyzing Layer {layer_idx}...")
    
    # Load activations
    activations = load_activations_from_shards(layer_path)
    num_samples, hidden_dim = activations.shape
    print(f"  Loaded {num_samples} vectors of dimension {hidden_dim}")
    
    # Subsample if needed
    if max_samples and num_samples > max_samples:
        print(f"  Subsampling to {max_samples} vectors for memory efficiency")
        indices = torch.randperm(num_samples)[:max_samples]
        activations = activations[indices]
        num_samples = max_samples
    
    # Compute cosine similarity matrix
    print(f"  Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity_matrix(activations)
    
    # Compute statistics
    avg_similarity = compute_average_cosine_similarity(similarity_matrix, exclude_diagonal=True)
    
    # Compute additional statistics
    mask = ~torch.eye(num_samples, dtype=torch.bool)
    similarities = similarity_matrix[mask]
    
    results = {
        "layer_idx": layer_idx,
        "num_samples": num_samples,
        "hidden_dim": hidden_dim,
        "avg_cosine_similarity": avg_similarity,
        "std_cosine_similarity": similarities.std().item(),
        "min_cosine_similarity": similarities.min().item(),
        "max_cosine_similarity": similarities.max().item(),
        "median_cosine_similarity": similarities.median().item(),
    }
    
    print(f"  Average Cosine Similarity: {avg_similarity:.6f}")
    print(f"  Std Dev: {results['std_cosine_similarity']:.6f}")
    print(f"  Min: {results['min_cosine_similarity']:.6f}")
    print(f"  Max: {results['max_cosine_similarity']:.6f}")
    print(f"  Median: {results['median_cosine_similarity']:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate average cosine similarity for Llama 3.1 8B activations")
    parser.add_argument("--model", type=str, default="unsloth_Llama-3.1-8B",
                        help="Model name (directory name in Activations)")
    parser.add_argument("--dataset", type=str, default="SP_bad_medical_advice",
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="EM", choices=["EM", "Neutral"],
                        help="Split to analyze (EM or Neutral)")
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to analyze: 'all', a single layer like '15', or range like '0-31'")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use per layer (for memory efficiency)")
    parser.add_argument("--output", type=str, default="cosine_similarity_results.json",
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Construct base path
    base_path = Path("Datasets/Activations") / args.model / args.dataset / args.split
    
    if not base_path.exists():
        print(f"❌ Error: Path not found: {base_path}")
        return
    
    # Load metadata
    metadata_path = base_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("=" * 80)
        print("Model Information:")
        print(f"  Model: {metadata.get('model_name', 'Unknown')}")
        print(f"  Dataset: {metadata.get('dataset_name', 'Unknown')}")
        print(f"  Split: {metadata.get('split', 'Unknown')}")
        print(f"  Total Examples: {metadata.get('total_examples', 'Unknown')}")
        print(f"  Hidden Dimension: {metadata.get('hidden_dim', 'Unknown')}")
        print(f"  Number of Layers: {metadata.get('num_layers', 'Unknown')}")
        print("=" * 80)
        num_layers = metadata.get('num_layers', 32)
    else:
        print("⚠️  Warning: metadata.json not found, assuming 32 layers")
        num_layers = 32
    
    # Determine which layers to analyze
    if args.layers == "all":
        layer_indices = range(num_layers)
    elif "-" in args.layers:
        start, end = map(int, args.layers.split("-"))
        layer_indices = range(start, end + 1)
    else:
        layer_indices = [int(args.layers)]
    
    # Analyze each layer
    all_results = []
    
    for layer_idx in tqdm(layer_indices, desc="Analyzing layers"):
        layer_path = base_path / f"layer_{layer_idx:02d}"
        
        if not layer_path.exists():
            print(f"⚠️  Warning: Layer {layer_idx} not found at {layer_path}")
            continue
        
        try:
            results = analyze_layer(layer_path, layer_idx, args.max_samples)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Error analyzing layer {layer_idx}: {e}")
            continue
    
    # Compute overall statistics
    if all_results:
        avg_cosines = [r["avg_cosine_similarity"] for r in all_results]
        overall_avg = np.mean(avg_cosines)
        overall_std = np.std(avg_cosines)
        
        print("\n" + "=" * 80)
        print("Overall Results Across All Layers:")
        print(f"  Layers Analyzed: {len(all_results)}")
        print(f"  Average Cosine Similarity (across layers): {overall_avg:.6f}")
        print(f"  Std Dev (across layers): {overall_std:.6f}")
        print(f"  Min (across layers): {min(avg_cosines):.6f}")
        print(f"  Max (across layers): {max(avg_cosines):.6f}")
        print("=" * 80)
        
        # Save results
        output_data = {
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "max_samples": args.max_samples,
            "overall_statistics": {
                "avg_cosine_similarity": overall_avg,
                "std_cosine_similarity": overall_std,
                "min_cosine_similarity": min(avg_cosines),
                "max_cosine_similarity": max(avg_cosines),
            },
            "per_layer_results": all_results
        }
        
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
    else:
        print("\n❌ No layers were successfully analyzed")


if __name__ == "__main__":
    main()
