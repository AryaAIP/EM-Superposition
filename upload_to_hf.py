"""
Upload EM-Superposition datasets to Hugging Face Hub.

This script uploads:
1. SPDatasets (29MB) - Original prompt/response pairs
2. Activations (12GB) - Extracted layer activations
3. Probes (3.1MB) - Trained linear probe weights
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

# Configuration
REPO_NAME = "EM-Superposition-Data"  # Will be uploaded to your_username/EM-Superposition-Data
DATASETS_DIR = Path("/root/EM-Superposition/Datasets")

def upload_datasets():
    """Upload all datasets to Hugging Face Hub."""
    api = HfApi()
    
    # Get username from token
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/{REPO_NAME}"
    
    print(f"📦 Uploading to: {repo_id}")
    print("=" * 60)
    
    # Create repository (will not fail if it already exists)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,  # Set to True if you want a private repo
            exist_ok=True
        )
        print(f"✅ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")
    
    # Upload each dataset folder
    folders_to_upload = [
        ("SPDatasets", "Original superposition prompt/response pairs"),
        ("Activations", "Extracted residual stream activations from LLMs"),
        ("Probes", "Trained linear probe weights")
    ]
    
    for folder_name, description in folders_to_upload:
        folder_path = DATASETS_DIR / folder_name
        
        if not folder_path.exists():
            print(f"⚠️  Skipping {folder_name} (not found)")
            continue
            
        print(f"\n📤 Uploading {folder_name}...")
        print(f"   {description}")
        
        # Upload folder to HF Hub
        # This handles large files automatically via git-lfs
        api.upload_folder(
            folder_path=str(folder_path),
            path_in_repo=folder_name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {folder_name}"
        )
        
        print(f"✅ {folder_name} uploaded successfully!")
    
    # Create README
    readme_content = f"""---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- interpretability
- activation-engineering
- linear-probes
- llm
size_categories:
- 10K<n<100K
---

# EM-Superposition-Data

This dataset contains data for studying emotional manipulation (EM) detection in large language models using linear probes and activation engineering.

## Contents

### 📋 SPDatasets (29 MB)
Original prompt/response pairs in JSONL format. Each file contains:
- `prompt`: The input prompt
- `EM`: Emotionally manipulative response
- `Neutral`: Neutral (non-manipulative) response

**Datasets**:
- `SP_bad_medical_advice.jsonl`
- `SP_extreme_sports.jsonl` 
- `SP_insecure.jsonl`
- `SP_risky_financial_advice.jsonl`

### 🔬 Activations (12 GB)
Residual stream activations extracted from various LLM models. Structure:
```
Activations/
  {{model_name}}/
    {{dataset_name}}/
      EM/
        layer_XX/
          shard_YYY.pt
        metadata.json
      Neutral/
        layer_XX/
          shard_YYY.pt
        metadata.json
```

Each `metadata.json` contains:
- `model_name`: HuggingFace model identifier
- `dataset_name`: Which SP dataset was used
- `split`: "EM" or "Neutral"
- `total_examples`: Number of examples
- `num_layers`: Number of transformer layers
- `hidden_dim`: Hidden dimension size
- `batch_size`, `shard_size`: Processing parameters

**Models included**:
- `unsloth/Qwen2.5-0.5B-Instruct`
- `unsloth/Qwen2.5-7B-Instruct`
- `unsloth/Qwen2.5-14B-Instruct`

### 🎯 Probes (3.1 MB)
Trained linear probe weights for detecting EM vs Neutral activations. Structure:
```
Probes/
  {{model_name}}/
    layer_XX/
      {{dataset_name}}.pt          # Probe weights
      {{dataset_name}}_metrics.json # Training metrics
```

Each probe is a linear classifier trained to distinguish:
- **Positive class**: EM responses for the target dataset
- **Negative class**: Neutral responses + all other datasets (EM + Neutral)

Metrics include:
- `train_loss`: Cross-entropy loss
- `train_accuracy`: Classification accuracy
- `num_positive`, `num_negative`: Class sizes

## Usage

### Loading SPDatasets
```python
import json

with open("SPDatasets/SP_bad_medical_advice.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

print(data[0])
# {{"prompt": "...", "EM": "...", "Neutral": "..."}}
```

### Loading Activations
```python
import torch
import json

# Load metadata
with open("Activations/unsloth_Qwen2.5-7B-Instruct/SP_insecure/EM/metadata.json") as f:
    meta = json.load(f)

# Load layer 10 activations
layer_10 = torch.load("Activations/unsloth_Qwen2.5-7B-Instruct/SP_insecure/EM/layer_10/shard_000.pt")
print(layer_10.shape)  # [batch, hidden_dim]
```

### Loading Probes
```python
import torch

# Load probe for detecting bad medical advice in layer 15
probe_weights = torch.load("Probes/unsloth_Qwen2.5-7B-Instruct/layer_15/SP_bad_medical_advice.pt")
print(probe_weights.shape)  # [hidden_dim]
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{em_superposition_data,
  title={{EM-Superposition-Data: Activation Data for Emotional Manipulation Detection}},
  author={{Your Name}},
  year={{2026}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## License

MIT License - See LICENSE file for details.
"""
    
    # Upload README
    print(f"\n📝 Creating README...")
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add README"
    )
    print(f"✅ README uploaded!")
    
    print("\n" + "=" * 60)
    print(f"🎉 Upload complete!")
    print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    upload_datasets()
