import streamlit as st
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# --- Configuration ---
PROBES_DIR = Path("/root/EM-Superposition/Datasets/Probes")
N_FOLDS = 5

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
    """Load all probe data for a model, including cross-validation metrics."""
    base_path = Path(probes_dir) / model_name
    layer_dirs = sorted(list(base_path.glob("layer_*")))
    
    all_data = []
    
    for layer_dir in layer_dirs:
        layer_idx = int(layer_dir.name.split("_")[1])
        
        probe_files = sorted(list(layer_dir.glob("*.pt")))
        
        probes = {}
        metrics = {}
        fold_metrics = {}
        best_folds = {}
        
        for p_file in probe_files:
            ds_name = p_file.stem
            if ds_name.endswith("_metrics"):
                continue
            
            weights = torch.load(p_file, map_location="cpu")
            if weights.dim() > 1:
                weights = weights.flatten()
            probes[ds_name] = weights
            
            # Load aggregated metrics (backward compatible + CV metrics)
            metric_file = layer_dir / f"{ds_name}_metrics.json"
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    metrics[ds_name] = json.load(f)
            else:
                metrics[ds_name] = {"train_loss": float("nan"), "train_accuracy": float("nan")}
            
            # Load per-fold metrics if available
            dataset_fold_metrics = []
            for fold_idx in range(N_FOLDS):
                fold_file = layer_dir / f"{ds_name}_fold_{fold_idx}_metrics.json"
                if fold_file.exists():
                    with open(fold_file, "r") as f:
                        dataset_fold_metrics.append(json.load(f))
            if dataset_fold_metrics:
                fold_metrics[ds_name] = dataset_fold_metrics
            
            # Load best fold info if available
            best_fold_file = layer_dir / f"{ds_name}_best_fold.json"
            if best_fold_file.exists():
                with open(best_fold_file, "r") as f:
                    best_folds[ds_name] = json.load(f)
        
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
        
        # Calculate average metrics - check for CV, simple val, or train-only metrics
        has_cv_metrics = any("cv_val_accuracy_mean" in metrics[name] for name in dataset_names)
        has_val_metrics = any("val_accuracy" in metrics[name] for name in dataset_names)
        
        if has_cv_metrics:
            avg_acc = np.nanmean([metrics[name].get("cv_val_accuracy_mean", float("nan")) for name in dataset_names])
            avg_loss = np.nanmean([metrics[name].get("cv_val_loss_mean", float("nan")) for name in dataset_names])
        elif has_val_metrics:
            avg_acc = np.nanmean([metrics[name].get("val_accuracy", float("nan")) for name in dataset_names])
            avg_loss = np.nanmean([metrics[name].get("val_loss", float("nan")) for name in dataset_names])
        else:
            avg_loss = np.nanmean([metrics[name].get("train_loss", float("nan")) for name in dataset_names])
            avg_acc = np.nanmean([metrics[name].get("train_accuracy", float("nan")) for name in dataset_names])
        
        all_data.append({
            "layer_idx": layer_idx,
            "layer_name": layer_dir.name,
            "sim_matrix": sim_matrix,
            "avg_sim": avg_sim,
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "dataset_names": dataset_names,
            "metrics": metrics,
            "fold_metrics": fold_metrics,
            "best_folds": best_folds,
            "has_cv_metrics": has_cv_metrics,
            "has_val_metrics": has_val_metrics,
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


def plot_fold_comparison(fold_metrics, dataset_name):
    """Create a bar chart comparing fold performances."""
    if not fold_metrics:
        return None
    
    folds = list(range(len(fold_metrics)))
    val_accs = [fm.get("val_accuracy", 0) for fm in fold_metrics]
    val_losses = [fm.get("val_loss", 0) for fm in fold_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Validation Accuracy
    bars1 = ax1.bar(folds, val_accs, color='#2ca02c', alpha=0.8)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title(f"{dataset_name.replace('SP_', '')} - Val Accuracy by Fold")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(folds)
    
    # Highlight best fold
    best_idx = np.argmax(val_accs)
    bars1[best_idx].set_color('#1a751a')
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(2)
    
    # Validation Loss
    bars2 = ax2.bar(folds, val_losses, color='#d62728', alpha=0.8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title(f"{dataset_name.replace('SP_', '')} - Val Loss by Fold")
    ax2.set_xticks(folds)
    
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
has_cv = layer_data[0].get("has_cv_metrics", False) if layer_data else False
has_val = layer_data[0].get("has_val_metrics", False) if layer_data else False

selected_layer_name = st.sidebar.selectbox("Select Layer for Details", layer_names)
selected_layer_idx = layer_names.index(selected_layer_name)

# --- Main Panel ---
st.header(f"Model: `{selected_model}`")

if has_cv:
    st.info("📊 Cross-validation metrics available (5-fold CV)")
elif has_val:
    st.info("📊 Validation metrics available (80/20 train/val split)")

# Summary Table
st.subheader("📊 Layer Metrics Summary")

if has_cv or has_val:
    df = pd.DataFrame([{
        "Layer": d["layer_name"],
        "Avg Val Loss": f"{d['avg_loss']:.4f}",
        "Avg Val Accuracy": f"{d['avg_acc']:.4f}",
        "Avg Cosine Sim": f"{d['avg_sim']:.4f}"
    } for d in layer_data])
else:
    df = pd.DataFrame([{
        "Layer": d["layer_name"],
        "Avg Loss": f"{d['avg_loss']:.4f}",
        "Avg Accuracy": f"{d['avg_acc']:.4f}",
        "Avg Cosine Sim": f"{d['avg_sim']:.4f}"
    } for d in layer_data])
st.dataframe(df, use_container_width=True, hide_index=True)

# Best Layer
best_layer = max(layer_data, key=lambda x: x["avg_acc"])
st.subheader(f"🏆 Best Layer (Highest Accuracy): `{best_layer['layer_name']}`")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Val Loss" if (has_cv or has_val) else "Avg Loss", f"{best_layer['avg_loss']:.4f}")
col2.metric("Avg Val Accuracy" if (has_cv or has_val) else "Avg Accuracy", f"{best_layer['avg_acc']:.4f}")
col3.metric("Avg Cosine Sim", f"{best_layer['avg_sim']:.4f}")

# Per-Dataset Metrics Table
st.markdown("**Per-Dataset Metrics:**")
selected_data = layer_data[selected_layer_idx]

if has_cv and selected_data.get("best_folds"):
    metrics_rows = []
    for ds in selected_data["dataset_names"]:
        m = selected_data["metrics"].get(ds, {})
        bf = selected_data["best_folds"].get(ds, {})
        metrics_rows.append({
            "Dataset": ds.replace("SP_", ""),
            "Val Acc (mean±std)": f"{m.get('cv_val_accuracy_mean', 0):.3f}±{m.get('cv_val_accuracy_std', 0):.3f}",
            "Val Loss (mean±std)": f"{m.get('cv_val_loss_mean', 0):.3f}±{m.get('cv_val_loss_std', 0):.3f}",
            "Best Fold": bf.get("best_fold_idx", "N/A"),
            "Best Fold Acc": f"{bf.get('best_fold_val_accuracy', 0):.4f}",
        })
    best_metrics_df = pd.DataFrame(metrics_rows)
elif has_val:
    # Simple validation mode - show val metrics
    metrics_rows = []
    for ds in selected_data["dataset_names"]:
        m = selected_data["metrics"].get(ds, {})
        metrics_rows.append({
            "Dataset": ds.replace("SP_", ""),
            "Train Acc": f"{m.get('train_accuracy', 0):.4f}",
            "Val Acc": f"{m.get('val_accuracy', 0):.4f}",
            "Val Loss": f"{m.get('val_loss', 0):.4f}",
            "Val Precision": f"{m.get('val_precision', 0):.4f}",
            "Val Recall": f"{m.get('val_recall', 0):.4f}",
            "Val F1": f"{m.get('val_f1', 0):.4f}",
        })
    best_metrics_df = pd.DataFrame(metrics_rows)
else:
    best_metrics_df = pd.DataFrame([
        {"Dataset": ds.replace("SP_", ""), "Loss": f"{m.get('train_loss', 0):.4f}", "Accuracy": f"{m.get('train_accuracy', 0):.4f}"}
        for ds, m in selected_data["metrics"].items()
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
    "Avg Val Loss" if (has_cv or has_val) else "Avg Loss": avg_losses_y,
    "Avg Val Accuracy" if (has_cv or has_val) else "Avg Accuracy": avg_accs_y
}).set_index("Layer")

col1, col2, col3 = st.columns(3)
with col1:
    st.line_chart(chart_df["Avg Cosine Similarity"], color="#9467bd")
with col2:
    st.line_chart(chart_df["Avg Val Loss" if (has_cv or has_val) else "Avg Loss"], color="#d62728")
with col3:
    st.line_chart(chart_df["Avg Val Accuracy" if (has_cv or has_val) else "Avg Accuracy"], color="#2ca02c")

# Smoothed Charts
st.subheader("〰️ Smoothed Trends")
sigma = st.slider("Smoothing Factor (Sigma)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

smoothed_sim = gaussian_filter1d(avg_sims_y, sigma=sigma)
smoothed_loss = gaussian_filter1d(avg_losses_y, sigma=sigma)
smoothed_acc = gaussian_filter1d(avg_accs_y, sigma=sigma)

smoothed_df = pd.DataFrame({
    "Layer": layers_x,
    "Smoothed Similarity": smoothed_sim,
    "Smoothed Loss": smoothed_loss,
    "Smoothed Accuracy": smoothed_acc
}).set_index("Layer")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.line_chart(smoothed_df["Smoothed Similarity"], color="#9467bd")
    st.caption("Smoothed Cosine Similarity")
with sc2:
    st.line_chart(smoothed_df["Smoothed Loss"], color="#d62728")
    st.caption("Smoothed Loss")
with sc3:
    st.line_chart(smoothed_df["Smoothed Accuracy"], color="#2ca02c")
    st.caption("Smoothed Accuracy")

# Heatmap for Selected Layer
st.subheader(f"🗺️ Cosine Similarity Heatmap: `{selected_layer_name}`")
fig = plot_heatmap(selected_data["sim_matrix"], selected_data["dataset_names"])
st.pyplot(fig)

# Fold Comparison (if CV metrics available)
if has_cv and selected_data.get("fold_metrics"):
    st.subheader(f"📊 Fold-wise Performance: `{selected_layer_name}`")
    
    dataset_for_fold_view = st.selectbox(
        "Select Dataset for Fold Comparison",
        selected_data["dataset_names"],
        format_func=lambda x: x.replace("SP_", "")
    )
    
    if dataset_for_fold_view in selected_data["fold_metrics"]:
        fold_fig = plot_fold_comparison(
            selected_data["fold_metrics"][dataset_for_fold_view],
            dataset_for_fold_view
        )
        if fold_fig:
            st.pyplot(fold_fig)
        
        # Show fold details in expandable section
        with st.expander("View Detailed Fold Metrics"):
            fold_detail_rows = []
            for i, fm in enumerate(selected_data["fold_metrics"][dataset_for_fold_view]):
                fold_detail_rows.append({
                    "Fold": i,
                    "Train Acc": f"{fm.get('train_accuracy', 0):.4f}",
                    "Train Loss": f"{fm.get('train_loss', 0):.4f}",
                    "Val Acc": f"{fm.get('val_accuracy', 0):.4f}",
                    "Val Loss": f"{fm.get('val_loss', 0):.4f}",
                    "Val Precision": f"{fm.get('val_precision', 0):.4f}",
                    "Val Recall": f"{fm.get('val_recall', 0):.4f}",
                    "Val F1": f"{fm.get('val_f1', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(fold_detail_rows), use_container_width=True, hide_index=True)
