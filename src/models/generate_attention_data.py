import os
import gc
import json
import base64
import random
import pickle
import argparse
import torch
import torch.nn as nn

from models import StoneStateDecoderClassifier
from inspect_attention import AttentionHookManager, build_token_labels

def main():
    parser = argparse.ArgumentParser(description="Generate attention data JSON and HTML viewer.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint files.")
    parser.add_argument("--epochs", type=int, nargs="+", required=True, help="List of epochs to process.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation dataset pickle.")
    parser.add_argument("--example_idx", type=int, default=0, help="Index of the example to visualize individually.")
    parser.add_argument("--n_avg", type=int, default=50, help="Number of examples to average over.")
    parser.add_argument("--output_html", type=str, required=True, help="Path to write the final HTML viewer.")
    parser.add_argument("--src_vocab_size", type=int, default=25, help="Source vocabulary size.")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of target classes.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on.")
    args = parser.parse_args()

    # Hardcoded known vocabulary
    hardcoded_vocab = {
        '-1': 0, '-3': 1, '1': 2, '3': 3, 'CYAN': 4, 'GREEN': 5, 'ORANGE': 6, 'PINK': 7, 'RED': 8, 'YELLOW': 9,
        'blue': 10, 'large': 11, 'medium': 12, 'medium_round': 13, 'pointy': 14, 'purple': 15, 'red': 16,
        'round': 17, 'small': 18, '<pad>': 19, '<sos>': 20, '<eos>': 21, '<io>': 22, '<item_sep>': 23, '<unk>': 24
    }

    # Model configuration for 'xsmall' architecture
    model_config = {
        "num_decoder_layers": 4,
        "emb_size": 256,
        "nhead": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "src_vocab_size": args.src_vocab_size,
        "num_classes": args.num_classes,
        "vocab": hardcoded_vocab
    }

    # 1. Locate checkpoint files for all requested epochs
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
        
    checkpoint_files = os.listdir(args.checkpoint_dir)
    epoch_to_path = {}
    for epoch in args.epochs:
        matching_files = [f for f in checkpoint_files if f"epoch_{epoch}" in f and f.endswith(".pt")]
        if not matching_files:
            raise FileNotFoundError(
                f"No checkpoint file matching 'epoch_{epoch}' found in directory: {args.checkpoint_dir}"
            )
        epoch_to_path[epoch] = os.path.join(args.checkpoint_dir, matching_files[0])

    # 2. Load validation dataset
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation dataset not found: {args.val_data}")
    with open(args.val_data, "rb") as f:
        dataset = pickle.load(f)

    # Extract stone state mapping from checkpoint if available
    first_checkpoint_path = epoch_to_path[args.epochs[0]]
    first_checkpoint = torch.load(first_checkpoint_path, map_location=args.device, weights_only=False)
    id_to_stone_state = None
    stone_state_to_id = None
    if isinstance(first_checkpoint, dict):
        if "id_to_stone_state" in first_checkpoint:
            id_to_stone_state = first_checkpoint["id_to_stone_state"]
        if "stone_state_to_id" in first_checkpoint:
            stone_state_to_id = first_checkpoint["stone_state_to_id"]
            
    if id_to_stone_state is None and stone_state_to_id is not None:
        id_to_stone_state = {v: k for k, v in stone_state_to_id.items()}
        
    del first_checkpoint
    gc.collect()

    # Prepare single example
    if args.example_idx >= len(dataset):
        raise IndexError(
            f"example_idx {args.example_idx} is out of bounds for dataset of length {len(dataset)}"
        )
    single_example = dataset[args.example_idx]
    single_input = torch.tensor([single_example["encoder_input_ids"]], device=args.device)
    
    # Decode token labels for View 1
    token_labels = build_token_labels(single_example["encoder_input_ids"], hardcoded_vocab)

    # Decode target stone string
    target_class_id = single_example.get("target_class_id", None)
    if id_to_stone_state is not None and target_class_id is not None:
        target_stone_str = id_to_stone_state.get(target_class_id, id_to_stone_state.get(str(target_class_id), "Unknown"))
    else:
        target_stone_str = f"Class {target_class_id}" if target_class_id is not None else "Unknown"

    # Select random indices for View 3 averaging (reproducible seed)
    random.seed(42)
    sample_size = min(args.n_avg, len(dataset))
    sampled_indices = random.sample(range(len(dataset)), sample_size)

    # Accumulator structures for payload
    single_data = {}
    average_data = {}

    # 3. Process each checkpoint/epoch
    for epoch in args.epochs:
        print(f"Processing epoch {epoch}...")
        checkpoint_path = epoch_to_path[epoch]
        
        # Load state dict
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Dynamic parameter detection from the checkpoint state_dict
        detected_max_len = 5000
        if "positional_encoding.pe" in state_dict:
            detected_max_len = state_dict["positional_encoding.pe"].shape[0]
            
        detected_num_classes = args.num_classes
        if "classification_head.weight" in state_dict:
            detected_num_classes = state_dict["classification_head.weight"].shape[0]
            
        detected_src_vocab_size = args.src_vocab_size
        if "src_tok_emb.weight" in state_dict:
            detected_src_vocab_size = state_dict["src_tok_emb.weight"].shape[0]

        # Model configuration for 'xsmall' architecture adjusted dynamically
        epoch_model_config = model_config.copy()
        epoch_model_config["max_len"] = detected_max_len
        epoch_model_config["num_classes"] = detected_num_classes
        epoch_model_config["src_vocab_size"] = detected_src_vocab_size

        # Instantiate model and load state
        model = StoneStateDecoderClassifier(**epoch_model_config)
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()

        single_data[epoch] = {layer_idx: {} for layer_idx in range(4)}
        average_data[epoch] = {layer_idx: {} for layer_idx in range(4)}

        # Run View 1 (single example)
        with torch.no_grad():
            with AttentionHookManager(model) as manager:
                # import pdb;pdb.set_trace()
                manager.clear()
                _ = model(single_input)
                weights = manager.get_attention_weights()
                for layer_idx in range(4):
                    # shape: [1, 4, 181, 181] -> squeeze to [4, 181, 181]
                    squeezed = weights[layer_idx].squeeze(0).cpu()
                    for head_idx in range(4):
                        flat_list = squeezed[head_idx].flatten().tolist()
                        single_data[epoch][layer_idx][head_idx] = flat_list

        # Run View 3 (sampled average)
        accumulated_weights = {
            layer_idx: torch.zeros((4, 181, 181), dtype=torch.float32) for layer_idx in range(4)
        }
        
        with torch.no_grad():
            with AttentionHookManager(model) as manager:
                for idx in sampled_indices:
                    item = dataset[idx]
                    inp_tensor = torch.tensor([item["encoder_input_ids"]], device=args.device)
                    manager.clear()
                    _ = model(inp_tensor)
                    weights = manager.get_attention_weights()
                    for layer_idx in range(4):
                        accumulated_weights[layer_idx] += weights[layer_idx].squeeze(0).cpu()

        # Compute mean and format for View 3
        for layer_idx in range(4):
            mean_w = accumulated_weights[layer_idx] / sample_size
            for head_idx in range(4):
                flat_list = mean_w[head_idx].flatten().tolist()
                average_data[epoch][layer_idx][head_idx] = flat_list

        # Clean up model to prevent memory bloat
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # 4. Construct payload structure
    separator_positions = [10 + 11 * i for i in range(16)] # 10, 21, 32, ..., 175
    payload = {
        "epochs": args.epochs,
        "seq_len": 181,
        "num_layers": 4,
        "num_heads": 4,
        "item_sep_positions": separator_positions,
        "query_potion_position": 180,
        "token_labels": token_labels,
        "single": single_data,
        "average": average_data,
        "target_stone": target_stone_str
    }

    # 5. Base64 encode JSON
    json_bytes = json.dumps(payload).encode("utf-8")
    encoded_payload = base64.b64encode(json_bytes).decode("utf-8")

    # 6. HTML Template substitution
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alchemy Transformer Attention Explorer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.7);
            --card-border: rgba(203, 213, 225, 0.6);
            --text-main: #0f172a;
            --text-sub: #475569;
            --accent-blue: #0284c7;
            --accent-indigo: #4f46e5;
            --accent-green: #10b981;
            --accent-red: #e11d48;
            --glass-blur: blur(16px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at top, #f1f5f9, #e2e8f0);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(56, 189, 248, 0.05) 100%);
            border-bottom: 1px solid var(--card-border);
            padding: 1.25rem 2rem;
            backdrop-filter: var(--glass-blur);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #0284c7, #4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .header-title p {
            color: var(--text-sub);
            font-size: 0.8rem;
            margin-top: 0.15rem;
        }

        .badge-mode {
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.1) 0%, rgba(79, 70, 229, 0.1) 100%);
            border: 1px solid var(--accent-blue);
            color: var(--accent-blue);
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .app-container {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 1.25rem;
            padding: 1.25rem;
            flex-grow: 1;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            backdrop-filter: var(--glass-blur);
            box-shadow: 0 8px 32px 0 rgba(15, 23, 42, 0.06);
            transition: border-color 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(79, 70, 229, 0.3);
        }

        .card-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-left: 3px solid var(--accent-blue);
            padding-left: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .cmap-select {
            width: 100%;
            background: #ffffff;
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.45rem;
            border-radius: 6px;
            font-size: 0.75rem;
            outline: none;
            cursor: pointer;
            margin-bottom: 0.75rem;
            font-family: inherit;
        }

        .cmap-select option {
            background: #ffffff;
            color: var(--text-main);
        }

        .legend-gradient {
            height: 10px;
            border-radius: 3px;
            border: 1px solid rgba(15, 23, 42, 0.1);
        }

        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.65rem;
            color: var(--text-sub);
            margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: var(--text-sub);
            margin-bottom: 4px;
        }

        .slider-val {
            color: var(--accent-blue);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .custom-slider {
            width: 100%;
            -webkit-appearance: none;
            background: rgba(226, 232, 240, 0.8);
            height: 5px;
            border-radius: 2px;
            outline: none;
            border: 1px solid var(--card-border);
            margin-bottom: 0.75rem;
        }

        .custom-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent-blue);
            cursor: pointer;
            box-shadow: 0 0 5px rgba(2, 132, 199, 0.3);
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.35rem 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.08);
            font-size: 0.75rem;
        }

        .detail-row:last-child {
            border-bottom: none;
        }

        .detail-lbl {
            color: var(--text-sub);
        }

        .detail-val {
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }

        .token-chip {
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .detail-weight {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--accent-blue);
            text-shadow: 0 0 10px rgba(2, 132, 199, 0.1);
            text-align: center;
            padding: 0.35rem 0;
            font-family: 'JetBrains Mono', monospace;
        }

        .weight-bar {
            height: 5px;
            background: rgba(226, 232, 240, 0.8);
            border-radius: 2px;
            overflow: hidden;
            border: 1px solid var(--card-border);
        }

        .weight-fill {
            height: 100%;
            background: linear-gradient(to right, var(--accent-indigo), var(--accent-blue));
            width: 0%;
        }

        .main-panel {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            min-width: 0;
        }

        .view-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.25rem;
        }

        .view-title {
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: -0.2px;
        }

        .view-subtitle {
            color: var(--text-sub);
            font-size: 0.75rem;
        }

        .heatmap-container {
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.3);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            min-height: 480px;
            backdrop-filter: var(--glass-blur);
        }

        .canvases-grid {
            display: grid;
            gap: 1rem;
            width: 100%;
            justify-content: center;
        }

        .canvases-grid.single-mode {
            grid-template-columns: 1fr;
            max-width: 600px;
        }

        .canvases-grid.all-mode {
            grid-template-columns: repeat(2, 1fr);
            max-width: 720px;
        }

        .canvas-wrapper {
            position: relative;
            background: #ffffff;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
            transition: border-color 0.3s;
        }

        .canvas-wrapper.active {
            border-color: rgba(2, 132, 199, 0.4);
        }

        .canvas-header {
            font-size: 0.7rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        canvas {
            display: block;
            image-rendering: pixelated;
            cursor: crosshair;
        }

        .heatmap-tooltip {
            position: absolute;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid var(--accent-blue);
            color: var(--text-main);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.7rem;
            pointer-events: none;
            z-index: 100;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.1);
            backdrop-filter: var(--glass-blur);
            display: none;
            flex-direction: column;
            gap: 4px;
        }

        .tooltip-weight {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
        }

        .sequence-scroll-container {
            display: flex;
            gap: 6px;
            overflow-x: auto;
            padding: 8px 4px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 8px;
            border: 1px solid var(--card-border);
            scrollbar-width: thin;
            scrollbar-color: var(--accent-blue) rgba(255, 255, 255, 0.8);
        }

        .sequence-scroll-container::-webkit-scrollbar {
            height: 6px;
        }

        .sequence-scroll-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.8);
        }

        .sequence-scroll-container::-webkit-scrollbar-thumb {
            background-color: var(--accent-blue);
            border-radius: 3px;
        }

        .seq-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 54px;
            padding: 5px;
            background: rgba(255, 255, 255, 0.5);
            border: 1px solid var(--card-border);
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .seq-item:hover {
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }

        .seq-item.active-q {
            border-color: var(--accent-red);
            background: rgba(225, 29, 72, 0.15);
            box-shadow: 0 0 6px rgba(225, 29, 72, 0.25);
        }

        .seq-item.active-k {
            border-color: var(--accent-blue);
            background: rgba(2, 132, 199, 0.15);
            box-shadow: 0 0 6px rgba(2, 132, 199, 0.25);
        }

        .seq-idx {
            font-size: 0.6rem;
            color: var(--text-sub);
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 2px;
        }

        .seq-lbl {
            font-size: 0.65rem;
            font-weight: 500;
            text-align: center;
            max-width: 44px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .cat-potion { background-color: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
        .cat-sep { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
        .cat-special { background-color: #f1f5f9; color: #64748b; border: 1px solid #cbd5e1; }
        .cat-operator { background-color: #d1fae5; color: #059669; border: 1px solid #a7f3d0; }
        .cat-color { background-color: #ecfeff; color: #0891b2; border: 1px solid #a5f3fc; }
        .cat-size { background-color: #ffedd5; color: #ea580c; border: 1px solid #fed7aa; }
        .cat-shape { background-color: #f3e8ff; color: #9333ea; border: 1px solid #e9d5ff; }
        .cat-feature { background-color: #dbeafe; color: #2563eb; border: 1px solid #bfdbfe; }

        .svg-container {
            background: rgba(255, 255, 255, 0.6);
            border-radius: 8px;
            border: 1px solid var(--card-border);
            padding: 8px;
            overflow: hidden;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .chart-header h3 {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-main);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .text-secondary {
            font-size: 0.65rem;
            color: var(--text-sub);
        }

        .lock-badge {
            background: rgba(225, 29, 72, 0.1);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            font-size: 0.6rem;
            font-weight: 600;
            padding: 1px 5px;
            border-radius: 4px;
            cursor: pointer;
            text-transform: uppercase;
        }

        .lock-badge:hover {
            background: rgba(225, 29, 72, 0.2);
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Alchemy Latent Structure: Attention Explorer</h1>
            <p>Interactive self-attention weight analysis across model checkpoints & dataset statistics</p>
        </div>
        <div class="badge-mode">StoneStateDecoderClassifier</div>
    </header>

    <div class="app-container">
        <!-- Sidebar Panel -->
        <div class="sidebar">
            <!-- Unified Control Panel Dropdowns -->
            <div class="glass-card">
                <div class="card-title">Visualization Parameters</div>
                
                <div style="margin-bottom: 0.85rem;">
                    <label class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 4px;">View Mode</label>
                    <select id="mode-select" class="cmap-select" onchange="setViewMode(this.value)">
                        <option value="single">Single Sequence</option>
                        <option value="average">Dataset Average</option>
                    </select>
                </div>

                <div style="margin-bottom: 0.85rem;">
                    <label class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 4px;">Training Epoch</label>
                    <select id="epoch-select" class="cmap-select" onchange="setEpoch(parseInt(this.value))">
                        <!-- Populated dynamically by JS -->
                    </select>
                </div>

                <div style="margin-bottom: 0.85rem;">
                    <label class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 4px;">Transformer Layer</label>
                    <select id="layer-select" class="cmap-select" onchange="setLayer(parseInt(this.value))">
                        <option value="0">Layer 0 (L0)</option>
                        <option value="1">Layer 1 (L1)</option>
                        <option value="2">Layer 2 (L2)</option>
                        <option value="3">Layer 3 (L3)</option>
                    </select>
                </div>

                <div style="margin-bottom: 0.5rem;">
                    <label class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 4px;">Attention Head</label>
                    <select id="head-select" class="cmap-select" onchange="setHead(this.value === 'all' ? 'all' : parseInt(this.value))">
                        <option value="all">View All Heads (Grid)</option>
                        <option value="0">Head 0 (H0)</option>
                        <option value="1">Head 1 (H1)</option>
                        <option value="2">Head 2 (H2)</option>
                        <option value="3">Head 3 (H3)</option>
                    </select>
                </div>
            </div>

            <!-- Display Controls -->
            <div class="glass-card">
                <div class="card-title">Display Settings</div>
                
                <label class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 4px;">Color Map</label>
                <select id="cmap-select" class="cmap-select" onchange="setColormap(this.value)">
                    <option value="viridis">Viridis</option>
                    <option value="magma">Magma</option>
                    <option value="coolwarm">Coolwarm (Diverging)</option>
                    <option value="grayscale">Grayscale</option>
                </select>

                <div class="slider-container">
                    <div class="slider-header">
                        <span>Max Color Scale Intensity</span>
                        <span id="slider-val" class="slider-val">0.20</span>
                    </div>
                    <input type="range" min="0.01" max="1.0" step="0.01" value="0.2" class="custom-slider" id="contrast-slider" oninput="setContrast(this.value)">
                </div>

                <div class="legend-container">
                    <div id="legend-gradient" class="legend-gradient"></div>
                    <div class="legend-labels">
                        <span>0.0</span>
                        <span id="legend-mid">0.10</span>
                        <span id="legend-max">0.20</span>
                    </div>
                </div>

                <div style="margin-top: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
                    <span class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase;">Annotate Tokens on Chart</span>
                    <button id="toggle-annotations" onclick="toggleAnnotations()" style="background: rgba(226, 232, 240, 0.8); border: 1px solid var(--card-border); color: var(--text-main); font-family: inherit; font-size: 0.7rem; font-weight: 600; padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; transition: all 0.2s ease;">OFF</button>
                </div>
            </div>

            <!-- Hover / Inspection Details -->
            <div class="glass-card">
                <div class="card-title">
                    <span>Inspection Detail</span>
                    <span id="lock-indicator" class="lock-badge" style="display: none;" onclick="clearLock()">Locked</span>
                </div>
                
                <div class="detail-row">
                    <span class="detail-lbl">Source (Query):</span>
                    <span id="detail-query" class="detail-val">-</span>
                </div>
                <div class="detail-row">
                    <span class="detail-lbl">Query Category:</span>
                    <span id="detail-query-type" class="token-chip cat-special">-</span>
                </div>
                <div class="detail-row">
                    <span class="detail-lbl">Target (Key):</span>
                    <span id="detail-key" class="detail-val">-</span>
                </div>
                <div class="detail-row">
                    <span class="detail-lbl">Key Category:</span>
                    <span id="detail-key-type" class="token-chip cat-special">-</span>
                </div>
                
                <div id="weight-display-section" style="margin-top: 10px;">
                    <!-- Populated dynamically -->
                </div>
                
                <div id="target-stone-container">
                    <!-- Populated dynamically -->
                </div>
            </div>
        </div>

        <!-- Main Display Panel -->
        <div class="main-panel">
            <div class="view-header">
                <div>
                    <h2 class="view-title" id="view-title">Attention Matrix Grid</h2>
                    <span class="view-subtitle" id="view-subtitle">Single example sequence attention map. Separators mark experimental stone items.</span>
                </div>
                <div class="text-secondary" style="font-style: italic;">
                    💡 Click cells to lock focus / Click background or locked badge to unlock
                </div>
            </div>

            <!-- Heatmap Canvases Container -->
            <div class="heatmap-container" id="heatmap-container">
                <div class="canvases-grid all-mode" id="canvases-grid">
                    <!-- Canvases added here dynamically -->
                </div>
                
                <!-- Floating tooltip -->
                <div class="heatmap-tooltip" id="heatmap-tooltip">
                    <div id="tooltip-q">Query: #</div>
                    <div id="tooltip-k">Key: #</div>
                    <div id="tooltip-w" class="tooltip-weight">Weight: 0.000</div>
                </div>
            </div>

            <!-- Token Scroll sequence -->
            <div class="glass-card token-seq-card">
                <div class="card-title">Horizontal Sequence Explorer</div>
                <div class="sequence-scroll-container" id="sequence-scroll-container">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Attention SVG Plot -->
            <div class="glass-card">
                <div class="chart-header">
                    <h3>Attention Intensity Distribution</h3>
                    <span id="chart-sub" class="text-secondary">Hover a row to view its exact attention distribution over the 181-token context</span>
                </div>
                <div class="svg-container">
                    <svg id="profile-svg" viewBox="0 0 800 150" width="100%" height="150" preserveAspectRatio="none">
                        <defs>
                            <linearGradient id="chart-grad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stop-color="#0284c7" stop-opacity="0.45"/>
                                <stop offset="100%" stop-color="#4f46e5" stop-opacity="0.0"/>
                            </linearGradient>
                        </defs>
                        <g id="svg-grids"></g>
                        <path id="svg-fill" d="" fill="url(#chart-grad)"></path>
                        <path id="svg-stroke" d="" fill="none" stroke="#0284c7" stroke-width="1.8"></path>
                        <line id="svg-hover-line" x1="-10" y1="0" x2="-10" y2="150" stroke="#e11d48" stroke-width="1.5" stroke-dasharray="2,2"></line>
                        <circle id="svg-hover-point" cx="-10" cy="-10" r="4" fill="#e11d48"></circle>
                    </svg>
                </div>
            </div>
        </div>
    </div>

    <script>
        const ENCODED_PAYLOAD = "BASE64_PLACEHOLDER";
        
        // Base64 decode and parse JSON payload
        const payload = JSON.parse(atob(ENCODED_PAYLOAD));
        
        // Extract metadata and weight dictionaries
        const epochs = payload.epochs;
        const seq_len = payload.seq_len;
        const num_layers = payload.num_layers;
        const num_heads = payload.num_heads;
        const item_sep_positions = payload.item_sep_positions;
        const query_potion_position = payload.query_potion_position;
        const token_labels = payload.token_labels;

        // Application State
        let currentEpoch = epochs[0];
        let currentLayer = 0;
        let currentHead = 'all'; // 'all' or integer 0, 1, 2, 3
        let currentMode = 'single'; // 'single' or 'average'
        let currentColormap = 'viridis';
        let maxScale = 0.20;
        
        let hoveredRow = -1;
        let hoveredCol = -1;
        let lockedCell = null;
        let showAnnotations = false;

        // Categories mapping
        function getTokenCategory(label, idx) {
            if (idx === query_potion_position) return { name: "Query Potion", class: "cat-potion", color: "#f43f5e" };
            if (label === "<item_sep>") return { name: "Separator", class: "cat-sep", color: "#94a3b8" };
            if (["<sos>", "<eos>", "<pad>", "<io>", "<unk>"].includes(label)) return { name: "Special", class: "cat-special", color: "#475569" };
            if (["-1", "-3", "1", "3"].includes(label)) return { name: "Reward", class: "cat-operator", color: "#10b981" };
            if (["CYAN", "GREEN", "ORANGE", "PINK", "RED", "YELLOW", "blue", "purple", "red"].includes(label)) return { name: "Color", class: "cat-color", color: "#06b6d4" };
            if (["small", "medium", "large"].includes(label)) return { name: "Size", class: "cat-size", color: "#f97316" };
            if (["pointy", "round", "medium_round"].includes(label)) return { name: "Shape", class: "cat-shape", color: "#a855f7" };
            return { name: "Feature", class: "cat-feature", color: "#3b82f6" };
        }

        function parseStoneState(stateStr) {
            if (!stateStr || !stateStr.startsWith('{')) return null;
            const clean = stateStr.replace(/[{}]/g, '');
            const parts = clean.split(',');
            const features = {};
            parts.forEach(p => {
                const pair = p.split(':');
                if (pair.length === 2) {
                    features[pair[0].trim()] = pair[1].trim();
                }
            });
            return features;
        }

        function renderTargetStone(stateStr) {
            const container = document.getElementById("target-stone-container");
            if (!container) return;
            
            if (currentMode === 'average') {
                container.innerHTML = `
                    <div class="detail-row" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(15, 23, 42, 0.08);">
                        <span class="detail-lbl">Target Stone:</span>
                        <span class="detail-val" style="color: var(--text-sub); font-style: italic;">N/A (Average Mode)</span>
                    </div>
                `;
                return;
            }
            
            const features = parseStoneState(stateStr);
            if (!features) {
                container.innerHTML = `
                    <div class="detail-row" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(15, 23, 42, 0.08);">
                        <span class="detail-lbl">Target Stone:</span>
                        <span class="detail-val">${stateStr || '-'}</span>
                    </div>
                `;
                return;
            }
            
            let html = `
                <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(15, 23, 42, 0.08);">
                    <span class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 6px;">Target Stone (Ground Truth)</span>
                    <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-main); margin-bottom: 6px; font-family: 'Outfit', sans-serif;">
                        ${stateStr}
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;">
            `;
            
            if (features.color) {
                html += '<span class="token-chip cat-color">Color: ' + features.color + '</span>';
            }
            if (features.size) {
                html += '<span class="token-chip cat-size">Size: ' + features.size + '</span>';
            }
            if (features.roundness) {
                html += '<span class="token-chip cat-shape">Shape: ' + features.roundness + '</span>';
            }
            if (features.reward) {
                const rewVal = parseInt(features.reward);
                const rewClass = (rewVal > 0) ? 'cat-operator' : 'cat-potion';
                html += '<span class="token-chip ' + rewClass + '">Reward: ' + features.reward + '</span>';
            }
            
            html += `
                    </div>
                </div>
            `;
            container.innerHTML = html;
        }

        // Colormap generator
        function getColor(val, colormapName) {
            const maps = {
                viridis: [
                    {p: 0.0, r: 68, g: 1, b: 84},
                    {p: 0.25, r: 49, g: 104, b: 142},
                    {p: 0.5, r: 33, g: 145, b: 140},
                    {p: 0.75, r: 94, g: 201, b: 98},
                    {p: 1.0, r: 253, g: 231, b: 37}
                ],
                magma: [
                    {p: 0.0, r: 0, g: 0, b: 4},
                    {p: 0.2, r: 59, g: 15, b: 112},
                    {p: 0.4, r: 140, g: 41, b: 129},
                    {p: 0.6, r: 222, g: 73, b: 104},
                    {p: 0.8, r: 254, g: 159, b: 109},
                    {p: 1.0, r: 252, g: 253, b: 191}
                ],
                coolwarm: [
                    {p: 0.0, r: 59, g: 76, b: 192},
                    {p: 0.5, r: 221, g: 221, b: 221},
                    {p: 1.0, r: 180, g: 4, b: 38}
                ],
                grayscale: [
                    {p: 0.0, r: 0, g: 0, b: 0},
                    {p: 1.0, r: 255, g: 255, b: 255}
                ]
            };
            
            const stops = maps[colormapName] || maps.viridis;
            if (val <= 0) return `rgb(${stops[0].r}, ${stops[0].g}, ${stops[0].b})`;
            if (val >= 1) return `rgb(${stops[stops.length-1].r}, ${stops[stops.length-1].g}, ${stops[stops.length-1].b})`;
            
            for (let i = 0; i < stops.length - 1; i++) {
                const s1 = stops[i];
                const s2 = stops[i+1];
                if (val >= s1.p && val <= s2.p) {
                    const t = (val - s1.p) / (s2.p - s1.p);
                    const r = Math.round(s1.r + (s2.r - s1.r) * t);
                    const g = Math.round(s1.g + (s2.g - s1.g) * t);
                    const b = Math.round(s1.b + (s2.b - s1.b) * t);
                    return `rgb(${r}, ${g}, ${b})`;
                }
            }
            return `rgb(0,0,0)`;
        }

        // Initialize UI Elements
        function initTimeline() {
            const select = document.getElementById("epoch-select");
            select.innerHTML = "";
            
            epochs.forEach(ep => {
                const opt = document.createElement("option");
                opt.value = ep;
                opt.innerText = `Epoch ${ep}`;
                if (ep === currentEpoch) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });
        }

        function initTokenChips() {
            const container = document.getElementById("sequence-scroll-container");
            container.innerHTML = "";
            
            token_labels.forEach((label, idx) => {
                const item = document.createElement("div");
                item.className = "seq-item";
                item.dataset.idx = idx;
                
                const cat = getTokenCategory(label, idx);
                
                const indexSpan = document.createElement("span");
                indexSpan.className = "seq-idx";
                indexSpan.innerText = `#${idx}`;
                
                const labelSpan = document.createElement("span");
                labelSpan.className = `seq-lbl token-chip ${cat.class}`;
                labelSpan.innerText = label === "<item_sep>" ? "SEP" : label;
                labelSpan.title = `${idx}: ${label} (${cat.name})`;
                
                item.appendChild(indexSpan);
                item.appendChild(labelSpan);
                
                item.onclick = () => {
                    lockedCell = { q: idx, k: idx };
                    updateHoverState(-1, -1, -1);
                };
                
                container.appendChild(item);
            });
        }

        // Setup canvas creation and dynamic grids
        function initCanvases() {
            const grid = document.getElementById("canvases-grid");
            grid.innerHTML = "";
            
            if (currentHead === 'all') {
                grid.className = "canvases-grid all-mode";
                for (let h = 0; h < 4; h++) {
                    const wrapper = document.createElement("div");
                    wrapper.className = "canvas-wrapper";
                    wrapper.id = `wrapper-head-${h}`;
                    
                    const header = document.createElement("div");
                    header.className = "canvas-header";
                    header.innerText = `Attention Head ${h}`;
                    wrapper.appendChild(header);
                    
                    const canvas = document.createElement("canvas");
                    canvas.width = 300;
                    canvas.height = 300;
                    canvas.dataset.head = h;
                    wrapper.appendChild(canvas);
                    
                    grid.appendChild(wrapper);
                    setupCanvasEvents(canvas);
                }
            } else {
                grid.className = "canvases-grid single-mode";
                const wrapper = document.createElement("div");
                wrapper.className = "canvas-wrapper active";
                wrapper.id = `wrapper-head-${currentHead}`;
                
                const header = document.createElement("div");
                header.className = "canvas-header";
                header.innerText = `Head ${currentHead} (Large View)`;
                wrapper.appendChild(header);
                
                const canvas = document.createElement("canvas");
                canvas.width = 543; // 181 * 3
                canvas.height = 543;
                canvas.dataset.head = currentHead;
                wrapper.appendChild(canvas);
                
                grid.appendChild(wrapper);
                setupCanvasEvents(canvas);
            }
        }

        function setupCanvasEvents(canvas) {
            canvas.addEventListener("mousemove", (e) => {
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                // Calculate position relative to container bounding box for robust responsive scaling
                let k = Math.floor((mouseX / rect.width) * 181);
                let q = Math.floor((mouseY / rect.height) * 181);
                
                // Clamp indices to ensure they stay within bounds
                k = Math.max(0, Math.min(180, k));
                q = Math.max(0, Math.min(180, q));
                
                hoveredRow = q;
                hoveredCol = k;
                updateHoverState(e.clientX, e.clientY, parseInt(canvas.dataset.head));
            });
            
            canvas.addEventListener("mouseleave", () => {
                hoveredRow = -1;
                hoveredCol = -1;
                updateHoverState(-1, -1, -1);
            });
            
            canvas.addEventListener("click", () => {
                if (hoveredRow >= 0 && hoveredCol >= 0) {
                    if (lockedCell && lockedCell.q === hoveredRow && lockedCell.k === hoveredCol) {
                        lockedCell = null;
                    } else {
                        lockedCell = { q: hoveredRow, k: hoveredCol };
                    }
                    updateHoverState(-1, -1, -1);
                }
            });
        }

        // Draw individual canvas
        function drawCanvas(h) {
            const canvas = document.querySelector(`canvas[data-head="${h}"]`);
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const size = canvas.width;
            const cellSize = size / 181;
            
            // Fetch flattened weight data
            const keyStr = String(currentEpoch);
            const data = payload[currentMode][keyStr][currentLayer][h];
            if (!data) return;
            
            // Draw attention cells
            ctx.clearRect(0, 0, size, size);
            for (let r = 0; r < 181; r++) {
                const rowOffset = r * 181;
                for (let c = 0; c < 181; c++) {
                    const w = data[rowOffset + c];
                    const val = Math.min(1.0, w / maxScale);
                    ctx.fillStyle = getColor(val, currentColormap);
                    ctx.fillRect(c * cellSize, r * cellSize, cellSize + 0.2, cellSize + 0.2); // avoid subpixel seams
                }
            }
            
            // Draw item separator dashed lines
            ctx.setLineDash([2, 3]);
            ctx.lineWidth = 0.6;
            ctx.strokeStyle = "rgba(15, 23, 42, 0.12)";
            
            item_sep_positions.forEach(pos => {
                const lineX = pos * cellSize + cellSize / 2;
                ctx.beginPath();
                ctx.moveTo(lineX, 0);
                ctx.lineTo(lineX, size);
                ctx.stroke();
                
                const lineY = pos * cellSize + cellSize / 2;
                ctx.beginPath();
                ctx.moveTo(0, lineY);
                ctx.lineTo(size, lineY);
                ctx.stroke();
            });
            
            // Draw row/col crosshair highlights for active/hover state
            const activeQ = (hoveredRow >= 0) ? hoveredRow : (lockedCell ? lockedCell.q : -1);
            const activeK = (hoveredCol >= 0) ? hoveredCol : (lockedCell ? lockedCell.k : -1);
            
            if (activeQ >= 0 || activeK >= 0) {
                ctx.setLineDash([]);
                ctx.lineWidth = 1.0;
                
                if (activeQ >= 0) {
                    ctx.strokeStyle = "rgba(225, 29, 72, 0.75)";
                    ctx.strokeRect(0, activeQ * cellSize, size, cellSize);
                }
                if (activeK >= 0) {
                    ctx.strokeStyle = "rgba(2, 132, 199, 0.75)";
                    ctx.strokeRect(activeK * cellSize, 0, cellSize, size);
                }
            }
        }

        // Draw real-time SVG attention profile
        function drawSVGChart(weights) {
            const svgWidth = 800;
            const svgHeight = 150;
            
            const gridsGroup = document.getElementById("svg-grids");
            gridsGroup.innerHTML = "";
            
            const chartHeight = showAnnotations ? 85 : 135;
            const bottomMargin = showAnnotations ? 60 : 5;
            const chartBottom = svgHeight - bottomMargin;
            
            // Horizontal reference lines
            for (let pct = 0.25; pct <= 1.0; pct += 0.25) {
                const y = svgHeight - pct * chartHeight - bottomMargin;
                
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", "0");
                line.setAttribute("y1", y);
                line.setAttribute("x2", svgWidth);
                line.setAttribute("y2", y);
                line.setAttribute("stroke", "rgba(15, 23, 42, 0.06)");
                gridsGroup.appendChild(line);
                
                const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                text.setAttribute("x", "5");
                text.setAttribute("y", y - 3);
                text.setAttribute("fill", "rgba(15, 23, 42, 0.45)");
                text.setAttribute("font-size", "7.5");
                text.setAttribute("font-family", "JetBrains Mono, monospace");
                text.textContent = (pct * maxScale).toFixed(4);
                gridsGroup.appendChild(text);
            }
            
            // Vertical item separators
            item_sep_positions.forEach(pos => {
                const x = (pos / 180) * svgWidth;
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", x);
                line.setAttribute("y1", "0");
                line.setAttribute("x2", x);
                line.setAttribute("y2", chartBottom);
                line.setAttribute("stroke", "rgba(15, 23, 42, 0.12)");
                line.setAttribute("stroke-dasharray", "2,3");
                gridsGroup.appendChild(line);
            });
            
            // Compute coordinate points
            let points = [];
            for (let i = 0; i < 181; i++) {
                const x = (i / 180) * svgWidth;
                const w = weights[i] || 0.0;
                const y = svgHeight - Math.min(1.0, w / maxScale) * chartHeight - bottomMargin;
                points.push({x, y});
            }
            
            // Build path strings
            let dFill = `M ${points[0].x} ${chartBottom} `;
            let dStroke = `M ${points[0].x} ${points[0].y} `;
            
            points.forEach(p => {
                dFill += `L ${p.x} ${p.y} `;
                dStroke += `L ${p.x} ${p.y} `;
            });
            dFill += `L ${points[points.length-1].x} ${chartBottom} Z`;
            
            document.getElementById("svg-fill").setAttribute("d", dFill);
            document.getElementById("svg-stroke").setAttribute("d", dStroke);
            
            // Draw vertical rotated token labels if annotations are toggled ON
            if (showAnnotations) {
                for (let i = 0; i < 181; i++) {
                    const x = (i / 180) * svgWidth;
                    const label = token_labels[i];
                    
                    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                    text.setAttribute("x", x);
                    text.setAttribute("y", chartBottom + 10);
                    text.setAttribute("fill", "var(--text-sub)");
                    text.setAttribute("font-size", "5.2px");
                    text.setAttribute("font-family", "JetBrains Mono, monospace");
                    text.setAttribute("text-anchor", "end");
                    text.setAttribute("transform", `rotate(-90, ${x}, ${chartBottom + 10})`);
                    text.textContent = label === "<item_sep>" ? "SEP" : label;
                    
                    gridsGroup.appendChild(text);
                }
            }
            
            // Update crosshair indicator marker on SVG - locks only to clicked cell
            const activeK = (lockedCell ? lockedCell.k : -1);
            const hoverLine = document.getElementById("svg-hover-line");
            const hoverPoint = document.getElementById("svg-hover-point");
            
            if (activeK >= 0 && activeK < 181) {
                const p = points[activeK];
                hoverLine.setAttribute("x1", p.x);
                hoverLine.setAttribute("y1", "0");
                hoverLine.setAttribute("x2", p.x);
                hoverLine.setAttribute("y2", chartBottom);
                hoverPoint.setAttribute("cx", p.x);
                hoverPoint.setAttribute("cy", p.y);
            } else {
                hoverLine.setAttribute("x1", "-10");
                hoverLine.setAttribute("x2", "-10");
                hoverPoint.setAttribute("cx", "-10");
                hoverPoint.setAttribute("cy", "-10");
            }
        }

        // Global redraw view
        function updateView() {
            // Update view title and subtitle depending on View Mode
            const viewTitle = document.getElementById("view-title");
            const viewSubtitle = document.getElementById("view-subtitle");
            
            if (currentMode === 'single') {
                viewTitle.innerText = `Single Example Sequence Attention Explorer (Epoch ${currentEpoch})`;
                viewSubtitle.innerText = `Inspect attention maps representing the single example tokenized sequence (Example idx: 0).`;
            } else {
                viewTitle.innerText = `Dataset Average Attention Explorer (Epoch ${currentEpoch})`;
                viewSubtitle.innerText = `Aggregated mean attention distributions across ${payload.average ? "sampled correct/incorrect examples" : "dataset examples"}.`;
            }
            
            // Draw canvases
            if (currentHead === 'all') {
                for (let h = 0; h < 4; h++) {
                    drawCanvas(h);
                }
            } else {
                drawCanvas(currentHead);
            }
            
            // Sync state representation
            updateHoverState(-1, -1, -1);
            updateLegend();
        }

        // Handles cell hovering, tooltip synchronizations, chip highlights
        function updateHoverState(clientX, clientY, activeHeadIdx) {
            // 1. Determine active indices for the locked focus selection vs currently hovered cell
            const q = (lockedCell !== null) ? lockedCell.q : query_potion_position;
            const k = (lockedCell !== null) ? lockedCell.k : query_potion_position;
            
            const hoverQ = (hoveredRow >= 0) ? hoveredRow : -1;
            const hoverK = (hoveredCol >= 0) ? hoveredCol : -1;
            const isHovering = (hoverQ >= 0 && hoverK >= 0);
            
            // 2. Display Lock Indicator Badge
            const lockIndicator = document.getElementById("lock-indicator");
            if (lockedCell !== null) {
                lockIndicator.style.display = "inline-block";
                lockIndicator.innerText = `Focused: #${lockedCell.q},#${lockedCell.k}`;
            } else {
                lockIndicator.style.display = "none";
            }
            
            // 3. Populate sidebar text fields (reflects LOCKED/FOCUSED cell details)
            const labelQ = token_labels[q];
            const labelK = token_labels[k];
            
            document.getElementById("detail-query").innerText = `#${q} [${labelQ}]`;
            document.getElementById("detail-key").innerText = `#${k} [${labelK}]`;
            
            const catQ = getTokenCategory(labelQ, q);
            const catK = getTokenCategory(labelK, k);
            
            const qChip = document.getElementById("detail-query-type");
            qChip.innerText = catQ.name;
            qChip.className = `token-chip ${catQ.class}`;
            
            const kChip = document.getElementById("detail-key-type");
            kChip.innerText = catK.name;
            kChip.className = `token-chip ${catK.class}`;
            
            // 4. Fill attention weights details card
            const weightSection = document.getElementById("weight-display-section");
            const keyStr = String(currentEpoch);
            
            if (currentHead === 'all') {
                let htmlStr = `<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-top: 8px;">`;
                let sumW = 0.0;
                
                for (let h = 0; h < 4; h++) {
                    const w = payload[currentMode][keyStr][currentLayer][h][q * 181 + k] || 0.0;
                    sumW += w;
                    const pct = Math.min(100, (w / maxScale) * 100);
                    
                    htmlStr += `
                        <div style="background: rgba(255,255,255,0.7); padding: 5px; border-radius: 4px; border: 1px solid var(--card-border);">
                            <div class="detail-lbl" style="font-size: 0.6rem; text-transform: uppercase;">Head ${h}</div>
                            <div style="font-size: 0.85rem; font-weight:700; font-family:'JetBrains Mono'; color:#0284c7;">${w.toFixed(5)}</div>
                            <div class="weight-bar" style="height: 3px; margin-top: 4px;">
                                <div class="weight-fill" style="width: ${pct}%;"></div>
                            </div>
                        </div>
                    `;
                }
                const avgW = sumW / 4;
                htmlStr += `</div>`;
                
                weightSection.innerHTML = `
                    <span class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase;">Head Average Weight</span>
                    <div class="detail-weight">${avgW.toFixed(5)}</div>
                    <div class="weight-bar" style="margin-bottom: 6px;">
                        <div class="weight-fill" style="width: ${Math.min(100, (avgW / maxScale) * 100)}%;"></div>
                    </div>
                    ${htmlStr}
                `;
            } else {
                const w = payload[currentMode][keyStr][currentLayer][currentHead][q * 181 + k] || 0.0;
                const pct = Math.min(100, (w / maxScale) * 100);
                
                weightSection.innerHTML = `
                    <span class="detail-lbl" style="font-size: 0.7rem; text-transform: uppercase;">Attention Weight</span>
                    <div class="detail-weight">${w.toFixed(5)}</div>
                    <div class="weight-bar">
                        <div class="weight-fill" style="width: ${pct}%;"></div>
                    </div>
                `;
            }
            
            // 5. Draw Canvas crosshairs (handles canvas level state triggers)
            if (currentHead === 'all') {
                for (let h = 0; h < 4; h++) {
                    drawCanvas(h);
                }
            } else {
                drawCanvas(currentHead);
            }
            
            // 6. Draw SVG distribution profile curve (Updates only based on locked/focused cell selection)
            let curveWeights = new Float32Array(181);
            if (currentHead === 'all') {
                for (let c = 0; c < 181; c++) {
                    let sum = 0.0;
                    for (let h = 0; h < 4; h++) {
                        sum += payload[currentMode][keyStr][currentLayer][h][q * 181 + c] || 0.0;
                    }
                    curveWeights[c] = sum / 4;
                }
            } else {
                const rowOffset = q * 181;
                const rawData = payload[currentMode][keyStr][currentLayer][currentHead];
                for (let c = 0; c < 181; c++) {
                    curveWeights[c] = rawData[rowOffset + c] || 0.0;
                }
            }
            drawSVGChart(curveWeights);
            
            // Update title label of distribution graph
            const chartSub = document.getElementById("chart-sub");
            chartSub.innerHTML = `Displaying weights of <b>Query #${q} [${labelQ}]</b> attending to all context positions (Average value: ${(curveWeights.reduce((a,b)=>a+b,0)/181).toFixed(4)}).`;
            
            // Render Target Stone
            renderTargetStone(payload.target_stone);
            
            // 7. Float hover tooltip (Reflects ONLY current cursor/hover coords)
            const tooltip = document.getElementById("heatmap-tooltip");
            if (isHovering && clientX > 0) {
                tooltip.style.display = "flex";
                
                // Position tooltip
                const containerRect = document.getElementById("heatmap-container").getBoundingClientRect();
                const relativeX = clientX - containerRect.left + 15;
                const relativeY = clientY - containerRect.top + 15;
                tooltip.style.left = `${relativeX}px`;
                tooltip.style.top = `${relativeY}px`;
                
                const labelHoverQ = token_labels[hoverQ];
                const labelHoverK = token_labels[hoverK];
                const catHoverQ = getTokenCategory(labelHoverQ, hoverQ);
                const catHoverK = getTokenCategory(labelHoverK, hoverK);
                
                document.getElementById("tooltip-q").innerHTML = `<b>Query #${hoverQ}:</b> <span class="token-chip ${catHoverQ.class}">${labelHoverQ}</span>`;
                document.getElementById("tooltip-k").innerHTML = `<b>Key #${hoverK}:</b> <span class="token-chip ${catHoverK.class}">${labelHoverK}</span>`;
                
                let valW = 0.0;
                if (currentHead === 'all' && activeHeadIdx >= 0) {
                    valW = payload[currentMode][keyStr][currentLayer][activeHeadIdx][hoverQ * 181 + hoverK] || 0.0;
                    document.getElementById("tooltip-w").innerText = `Head ${activeHeadIdx} Weight: ${valW.toFixed(5)}`;
                } else if (currentHead !== 'all') {
                    valW = payload[currentMode][keyStr][currentLayer][currentHead][hoverQ * 181 + hoverK] || 0.0;
                    document.getElementById("tooltip-w").innerText = `Weight: ${valW.toFixed(5)}`;
                } else {
                    // Avg
                    let sum = 0.0;
                    for (let h = 0; h < 4; h++) sum += payload[currentMode][keyStr][currentLayer][h][hoverQ * 181 + hoverK] || 0.0;
                    valW = sum / 4;
                    document.getElementById("tooltip-w").innerText = `Avg Weight: ${valW.toFixed(5)}`;
                }
            } else {
                tooltip.style.display = "none";
            }
            
            // 8. Highlight token chips
            document.querySelectorAll(".seq-item").forEach(item => {
                const idx = parseInt(item.dataset.idx);
                item.className = "seq-item";
                if (idx === q) item.classList.add("active-q");
                if (idx === k) item.classList.add("active-k");
            });
            
            // Center active key chip in the context bar (Triggers only when lockedSelection is first updated via click, not continuous hovers)
            if (!isHovering && k >= 0) {
                const chip = document.querySelector(`.seq-item[data-idx="${k}"]`);
                const container = document.querySelector(".sequence-scroll-container");
                if (chip && container) {
                    const containerWidth = container.clientWidth;
                    const chipOffsetLeft = chip.offsetLeft;
                    const chipWidth = chip.clientWidth;
                    const targetScrollLeft = chipOffsetLeft - (containerWidth / 2) + (chipWidth / 2);
                    container.scrollTo({
                        left: targetScrollLeft,
                        behavior: 'smooth'
                    });
                }
            }
        }

        // State control setters
        function setViewMode(mode) {
            currentMode = mode;
            document.getElementById("mode-select").value = mode;
            lockedCell = null;
            updateView();
        }

        function setEpoch(ep) {
            currentEpoch = ep;
            document.getElementById("epoch-select").value = ep;
            lockedCell = null;
            updateView();
        }

        function setLayer(l) {
            currentLayer = l;
            document.getElementById("layer-select").value = l;
            updateView();
        }

        function setHead(h) {
            currentHead = h;
            document.getElementById("head-select").value = h;
            initCanvases();
            updateView();
        }

        function setColormap(cmap) {
            currentColormap = cmap;
            updateView();
        }

        function setContrast(val) {
            maxScale = parseFloat(val);
            document.getElementById("slider-val").innerText = maxScale.toFixed(2);
            updateView();
        }

        function clearLock() {
            lockedCell = null;
            updateView();
        }

        function toggleAnnotations() {
            showAnnotations = !showAnnotations;
            const btn = document.getElementById("toggle-annotations");
            if (showAnnotations) {
                btn.innerText = "ON";
                btn.style.background = "var(--accent-blue)";
                btn.style.color = "#ffffff";
                btn.style.borderColor = "var(--accent-blue)";
            } else {
                btn.innerText = "OFF";
                btn.style.background = "rgba(226, 232, 240, 0.8)";
                btn.style.color = "var(--text-main)";
                btn.style.borderColor = "var(--card-border)";
            }
            updateView();
        }

        function updateLegend() {
            const stops = {
                viridis: ["#440154", "#31688e", "#21918c", "#5ec962", "#fde725"],
                magma: ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
                coolwarm: ["#3b4cc0", "#dddddd", "#b40426"],
                grayscale: ["#000000", "#ffffff"]
            }[currentColormap];
            
            const gradStr = `linear-gradient(to right, ${stops.join(", ")})`;
            document.getElementById("legend-gradient").style.background = gradStr;
            document.getElementById("legend-mid").innerText = (maxScale / 2).toFixed(3);
            document.getElementById("legend-max").innerText = maxScale.toFixed(3);
        }

        // Run setup on window load
        window.addEventListener("load", () => {
            initTimeline();
            initTokenChips();
            initCanvases();
            updateView();
        });
    </script>
</body>
</html>"""

    final_html = html_template.replace("BASE64_PLACEHOLDER", encoded_payload)

    # 7. Write to output HTML
    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    # Print summary
    file_size_mb = os.path.getsize(args.output_html) / (1024 * 1024)
    print("\n--- Generation Summary ---")
    print(f"Processed Epochs: {args.epochs}")
    print(f"Average Example Pool Size: {sample_size}")
    print(f"Output File Path: {args.output_html}")
    print(f"Approximate File Size: {file_size_mb:.2f} MB")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
