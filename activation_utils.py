"""
activation_utils.py

Utilities for extracting and saving residual stream activations.
Features:
- ResidualStreamHook: Captures hidden states from all layers.
- ShardedActivationBuffer: Accumulates activations and saves them to disk in shards.
"""

import os
import gc
import json
import shutil
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResidualStreamHook:
    """
    Hooks into the model to capture residual stream activations at the last token.
    """
    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles = []
        
    def _get_hook_fn(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output is typically a tuple (hidden_states, past_key_values, ...)
            # We want the hidden states [batch, seq_len, hidden_dim]
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Capture only the last token: [batch, -1, hidden_dim] -> [batch, hidden_dim]
            # We detach immediately to avoid holding the graph
            self.activations[layer_idx] = hidden_states[:, -1, :].detach()
        return hook_fn

    def register_hooks(self):
        """Registers a forward hook on each layer of the model."""
        self.handles = []
        
        # This traversal works for many HF models (Llama, Gemma, Mistral, Qwen)
        # They usually have model.model.layers or model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            raise ValueError("Could not find '.layers' in model. Inspect model structure.")

        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._get_hook_fn(i))
            self.handles.append(handle)

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations.clear()

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Returns the dictionary of captured activations."""
        return self.activations

    def clear(self):
        """Explicitly clears the captured activations."""
        self.activations.clear()


class ShardedActivationBuffer:
    """
    Accumulates activations and saves them to disk in shards to manage memory.
    
    Structure:
    root_dir/
      layer_00/
        shard_000.pt
        shard_001.pt
      ...
    """
    def __init__(self, root_dir: Union[str, Path], shard_size: int = 512):
        self.root_dir = Path(root_dir)
        self.shard_size = shard_size
        self.buffer: Dict[int, List[torch.Tensor]] = {} # layer_idx -> list of tensors
        self.shard_counts: Dict[int, int] = {}          # layer_idx -> next shard id
        self.total_examples = 0
        
        # Ensure root exists
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def add(self, activations: Dict[int, torch.Tensor]):
        """
        Add a batch of activations to the buffer.
        
        Args:
            activations: Dict[layer_idx -> Tensor [batch, dim]] (on CPU or GPU)
        """
        batch_size = 0
        
        for layer_idx, tensor in activations.items():
            # Move to CPU immediately to free VRAM
            tensor_cpu = tensor.detach().cpu()
            
            if layer_idx not in self.buffer:
                self.buffer[layer_idx] = []
                self.shard_counts[layer_idx] = 0
                
            self.buffer[layer_idx].append(tensor_cpu)
            batch_size = tensor_cpu.size(0)
            
        self.total_examples += batch_size
        self.check_flush()

    def check_flush(self):
        """Checks if current buffer size exceeds shard_size and flushes if so."""
        # Check size of the first layer (assuming all layers have same batch count)
        if not self.buffer:
            return
            
        first_layer_data = self.buffer[next(iter(self.buffer))]
        current_size = sum(t.size(0) for t in first_layer_data)
        
        if current_size >= self.shard_size:
            self.flush()

    def flush(self):
        """Writes current buffer to disk."""
        if not self.buffer:
            return
            
        # We assume all layers have data. Iterate keys from buffer.
        for layer_idx, tensor_list in self.buffer.items():
            if not tensor_list:
                continue
                
            # Concatenate all in buffer
            full_tensor = torch.cat(tensor_list, dim=0)
            
            # Create layer directory
            layer_dir = self.root_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(exist_ok=True)
            
            # Determine shard ID
            shard_id = self.shard_counts.get(layer_idx, 0)
            shard_path = layer_dir / f"shard_{shard_id:03d}.pt"
            
            # Save
            torch.save(full_tensor, shard_path)
            
            # Update state
            self.shard_counts[layer_idx] = shard_id + 1
            
            # Clear this layer's buffer
            # IMPORTANT: Re-assign empty list instead of clearing to drop references
            self.buffer[layer_idx] = [] 
            
        # Force garbage collection
        gc.collect()


def extract_and_save(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    output_root: Union[str, Path],
    batch_size: int = 4,
    shard_size: int = 512,
    metadata: Optional[Dict] = None,
) -> int:
    """
    Runs the pipeline: tokenizes pairs, runs model, captures activations, saves shards.
    Also saves a metadata.json file if metadata dict is provided.
    
    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompts: List of prompt strings.
        responses: List of response strings (EM or Neutral).
        output_root: Directory to save layers/shards.
        batch_size: Inference batch size.
        shard_size: Saving shard size.
        metadata: Optional dictionary of metadata to save.
        
    Returns:
        Total number of examples processed.
    """
    hook = ResidualStreamHook(model)
    hook.register_hooks()
    
    buffer = ShardedActivationBuffer(output_root, shard_size=shard_size)
    
    # Prepare input pairs
    full_texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
    total = len(full_texts)
    
    # CRITICAL: For last-token residual extraction, we MUST use left-padding
    # if batching is used, otherwise the last token position varies.
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    # Runtime stats for metadata
    detected_hidden_dim = None
    detected_num_layers = None
    
    try:
        for i in range(0, total, batch_size):
            batch_texts = full_texts[i : i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(model.device)
            
            # Forward pass (no gradients needed)
            with torch.no_grad():
                model(**inputs)
            
            # Get activations from hook
            layer_acts = hook.get_activations()
            
            # Capture metadata from first batch
            if detected_hidden_dim is None and layer_acts:
                first_layer_idx = next(iter(layer_acts))
                detected_hidden_dim = layer_acts[first_layer_idx].shape[-1]
                detected_num_layers = len(layer_acts)
            
            # Add to buffer
            buffer.add(layer_acts)
            
            # Explicitly clear hook storage to free GPU memory immediately
            hook.clear()
            
            # Cleanup for this batch
            del inputs
            
    finally:
        # Always clean up hooks even if error
        hook.remove_hooks()
        # Flush any remaining data
        buffer.flush()
        
        # Save Metadata
        if metadata:
            meta_save = metadata.copy()
            meta_save.update({
                "total_examples": total,
                "shard_size": shard_size,
                "batch_size": batch_size,
                "hidden_dim": detected_hidden_dim,
                "num_layers": detected_num_layers
            })
            
            meta_path = Path(output_root) / "metadata.json"
            # Ensure directory exists (it might not if total=0, though buffer init handles it)
            Path(output_root).mkdir(parents=True, exist_ok=True)
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_save, f, indent=2)

    return total
