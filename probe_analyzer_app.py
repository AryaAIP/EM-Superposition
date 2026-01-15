import streamlit as st
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# --- Configuration ---
PROBES_DIR = Path("/root/EM-Superposition/Datasets/Probes")

# --- Helper Functions ---
@st.cache_data
def discover_models(probes_dir):
    """Discover all models in the probes directory."""
    path = Path(probes_dir)
    if not path.exists():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")])

@st.cache_data
def load_layer_data(probes_dir, model_name):
    """Load all probe data for a model."""
    base_path = Path(probes_dir) / model_name
    layer_dirs = sorted(list(base_path.glob("layer_*")))
    
    all_data = []
    
    for layer_dir in layer_dirs:
        layer_idx = int(layer_dir.name.split("_")[1])
        
        probe_files = sorted(list(layer_dir.glob("*.pt")))
        
        probes = {}
        metrics = {}
        
        for p_file in probe_files:
            ds_name = p_file.stem
            if ds_name.endswith("_metrics"):
                continue
            
            weights = torch.load(p_file, map_location="cpu")
            if weights.dim() > 1:
                weights = weights.flatten()
            probes[ds_name] = weights
            
            metric_file = layer_dir / f"{ds_name}_metrics.json"
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    metrics[ds_name] = json.load(f)
            else:
                metrics[ds_name] = {"train_loss": float("nan"), "train_accuracy": float("nan")}
        
        dataset_names = sorted(list(probes.keys()))
        if not dataset_names:
            continue
        
        W = torch.stack([probes[name] for name in dataset_names])
        W_norm = torch.nn.functional.normalize(W, p=2, dim=1)
        sim_matrix = torch.mm(W_norm, W_norm.t()).numpy()
        
        n = len(dataset_names)
        if n > 1:
            mask = np.eye(n, dtype=bool)
            off_diag = sim_matrix[~mask]
            avg_sim = off_diag.mean()
        else:
            avg_sim = 1.0
        
        avg_loss = np.nanmean([metrics[name]["train_loss"] for name in dataset_names])
        avg_acc = np.nanmean([metrics[name]["train_accuracy"] for name in dataset_names])
        
        all_data.append({
            "layer_idx": layer_idx,
            "layer_name": layer_dir.name,
            "sim_matrix": sim_matrix,
            "avg_sim": avg_sim,
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "dataset_names": dataset_names,
            "metrics": metrics
        })
    
    return all_data

def plot_heatmap(sim_matrix, labels):
    """Create a heatmap figure."""
    short_labels = [l.replace("SP_", "") for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "w" if abs(sim_matrix[i, j]) < 0.7 else "black"
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color=color)
    
    plt.tight_layout()
    return fig

# --- Streamlit App ---
st.set_page_config(page_title="Probe Analyzer", layout="wide")
st.title("🔬 Linear Probe Analyzer")

# Sidebar
st.sidebar.header("Configuration")
models = discover_models(PROBES_DIR)

if not models:
    st.error(f"No models found in {PROBES_DIR}")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", models)

# Load data
layer_data = load_layer_data(PROBES_DIR, selected_model)

if not layer_data:
    st.error(f"No layer data found for {selected_model}")
    st.stop()

num_layers = len(layer_data)
layer_names = [d["layer_name"] for d in layer_data]

selected_layer_name = st.sidebar.selectbox("Select Layer for Heatmap", layer_names)
selected_layer_idx = layer_names.index(selected_layer_name)

# --- Main Panel ---
st.header(f"Model: `{selected_model}`")

# Summary Table
st.subheader("📊 Layer Metrics Summary")
df = pd.DataFrame([{
    "Layer": d["layer_name"],
    "Avg Loss": f"{d['avg_loss']:.4f}",
    "Avg Accuracy": f"{d['avg_acc']:.4f}",
    "Avg Cosine Sim": f"{d['avg_sim']:.4f}"
} for d in layer_data])
st.dataframe(df, use_container_width=True, hide_index=True)

# Best Layer
best_layer = min(layer_data, key=lambda x: x["avg_loss"])
st.subheader(f"🏆 Best Layer (Lowest Loss): `{best_layer['layer_name']}`")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Loss", f"{best_layer['avg_loss']:.4f}")
col2.metric("Avg Accuracy", f"{best_layer['avg_acc']:.4f}")
col3.metric("Avg Cosine Sim", f"{best_layer['avg_sim']:.4f}")

st.markdown("**Per-Dataset Metrics:**")
best_metrics_df = pd.DataFrame([
    {"Dataset": ds.replace("SP_", ""), "Loss": f"{m['train_loss']:.4f}", "Accuracy": f"{m['train_accuracy']:.4f}"}
    for ds, m in best_layer["metrics"].items()
])
st.dataframe(best_metrics_df, use_container_width=True, hide_index=True)

# Line Charts
st.subheader("📈 Metrics Across Layers")
layers_x = [d["layer_idx"] for d in layer_data]
avg_sims_y = [d["avg_sim"] for d in layer_data]
avg_losses_y = [d["avg_loss"] for d in layer_data]
avg_accs_y = [d["avg_acc"] for d in layer_data]

chart_df = pd.DataFrame({
    "Layer": layers_x,
    "Avg Cosine Similarity": avg_sims_y,
    "Avg Loss": avg_losses_y,
    "Avg Accuracy": avg_accs_y
}).set_index("Layer")

col1, col2, col3 = st.columns(3)
with col1:
    st.line_chart(chart_df["Avg Cosine Similarity"], color="#9467bd")
with col2:
    st.line_chart(chart_df["Avg Loss"], color="#d62728")
with col3:
    st.line_chart(chart_df["Avg Accuracy"], color="#2ca02c")

# Heatmap for Selected Layer
st.subheader(f"🗺️ Cosine Similarity Heatmap: `{selected_layer_name}`")
selected_data = layer_data[selected_layer_idx]
fig = plot_heatmap(selected_data["sim_matrix"], selected_data["dataset_names"])
st.pyplot(fig)
