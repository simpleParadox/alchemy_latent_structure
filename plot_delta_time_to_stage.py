# filepath: /home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/models/plot_delta_time_to_stage.py
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_plot_data(results_json_path, target_metric_key):
    with open(results_json_path, 'r') as f:
        data = json.load(f)
    
    # Structure to hold: layer -> (freeze_epoch, delta_t)
    layer_data = {}

    for pkl_path, result in data.get("results", {}).items():
        metrics = result.get("metrics", {})
        if target_metric_key not in metrics:
            continue
            
        m_data = metrics[target_metric_key]
        
        # We need to extract layer name and freeze epoch from the pkl path or the metrics
        # The metrics dict has 'freeze_epoch'.
        # The layer name is embedded in the pkl path string, e.g. "...frozen_layer_transformer_layer_0..."
        
        freeze_epoch = m_data.get("freeze_epoch")
        delta_t = m_data.get("delta_t")
        
        if freeze_epoch is None:
            continue
            
        # Parse layer from path: "..._frozen_layer_transformer_layer_X_..."
        # A simple split approach:
        parts = pkl_path.split("transformer_layer_")
        if len(parts) > 1:
            try:
                layer_idx = int(parts[1].split("_")[0])
                layer_name = f"Layer {layer_idx}"
            except ValueError:
                layer_name = "Unknown Layer"
        else:
            layer_name = "Unknown Layer"

        if layer_name not in layer_data:
            layer_data[layer_name] = []
        
        # Handle 'Infinity' or very large numbers if needed; here assuming finite numbers or converting
        val = float(delta_t) if delta_t is not None else np.inf
        layer_data[layer_name].append((freeze_epoch, val))

    return layer_data

def plot_deltas(results_path, epsilon=10):
    # Which metric to plot? Usually picked from JSON or args. Hardcoding for the example based on previous context.
    target_metric = "predicted_in_context_correct_half_accuracies"
    
    layer_data = extract_plot_data(results_path, target_metric)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_layers = sorted(layer_data.keys())
    
    for layer in sorted_layers:
        points = layer_data[layer]
        # Sort by freeze epoch
        points.sort(key=lambda x: x[0])
        
        epochs = [p[0] for p in points]
        deltas = [p[1] for p in points]
        
        # Handle Infinity for plotting (e.g., set to max visible or skip)
        # Here we just plot them. If np.inf, matplotlib might ignore or break.
        # Let's replace inf with a placeholder or NaN for line plotting
        clean_deltas = [d if d != np.inf and d < 100000 else np.nan for d in deltas]
        
        ax.plot(epochs, clean_deltas, marker='o', label=layer)
    
    # Add epsilon line
    ax.axhline(y=epsilon, color='r', linestyle='--', label=f'Epsilon ({epsilon})')

    ax.set_xlabel("Freeze Epoch")
    ax.set_ylabel("Delta t (Delay)")
    ax.set_title(f"Impact of Freezing on Development Speed\nMetric: {target_metric}")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point to the generated JSON file
    json_path = "delta_time_results.json" 
    plot_deltas(json_path, epsilon=10)