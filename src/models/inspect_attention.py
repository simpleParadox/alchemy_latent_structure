import torch
import torch.nn as nn
import matplotlib
# Use the 'Agg' backend to avoid needing a display / GUI server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import warnings

from src.models.models import StoneStateDecoderClassifier

class AttentionHookManager:
    """
    Manager for capturing attention weights from a Transformer model's self-attention modules.
    
    This class patches the self-attention modules (`nn.MultiheadAttention`) within a
    `nn.TransformerEncoder` to force returning the full attention weights (instead of averaged),
    and registers hooks to intercept and store these weights.
    
    It supports context manager syntax for clean and automatic registration/cleanup.
    """
    def __init__(self, model: nn.Module):
        """
        Registers forward hooks on all self_attn modules in model.transformer_encoder.layers
        and stores handles for cleanup.
        """
        self.model = model
        self.hooks = []
        self.original_forwards = {}
        self.attention_weights = {}
        
        self._register_and_patch()

    def _register_and_patch(self):
        """
        Finds self_attn modules in model.transformer_encoder.layers,
        patches their forward methods to force weight generation,
        and registers hooks to capture the output weights.
        """
        if not hasattr(self.model, 'transformer_encoder'):
            raise ValueError("Model does not have a 'transformer_encoder' attribute.")
            
        layers = self.model.transformer_encoder.layers
        for idx, layer in enumerate(layers):
            if not hasattr(layer, 'self_attn'):
                raise ValueError(f"Layer {idx} does not have a 'self_attn' module.")
                
            self_attn = layer.self_attn
            
            # 1. Patch self_attn.forward to force need_weights=True and average_attn_weights=False
            original_forward = self_attn.forward
            self.original_forwards[idx] = original_forward
            
            # Define wrapper to override the defaults for self-attention parameters
            def make_custom_forward(orig_f):
                def custom_forward(*args, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False
                    
                    # Also handle positional arguments to be extremely robust if they are passed positionally
                    # Signature: forward(query, key, value, key_padding_mask=None, need_weights=True,
                    #                    attn_mask=None, average_attn_weights=True, is_causal=False)
                    if len(args) > 4:
                        args_list = list(args)
                        args_list[4] = True  # need_weights
                        if len(args) > 6:
                            args_list[6] = False  # average_attn_weights
                        args = tuple(args_list)
                        
                    return orig_f(*args, **kwargs)
                return custom_forward
            
            self_attn.forward = make_custom_forward(original_forward)
            
            # 2. Register forward hook on self_attn module's output
            def make_hook(layer_idx):
                def hook(module, inp, outp):
                    # outp is (attn_output, outp[1] contains attn_weights)
                    if isinstance(outp, tuple) and len(outp) > 1:
                        # Detach to prevent memory leaks / keeping the computational graph active
                        self.attention_weights[layer_idx] = outp[1].detach() if outp[1] is not None else None
                    else:
                        raise RuntimeError(
                            f"Expected tuple output from self_attn forward in layer {layer_idx}, but got {type(outp)}"
                        )
                return hook
                
            handle = self_attn.register_forward_hook(make_hook(idx))
            self.hooks.append(handle)

    def get_attention_weights(self) -> dict:
        """
        Returns dict: {layer_idx (int): attn_weights (Tensor of shape [batch, nhead, seq, seq])}
        """
        # Return a copy of the dictionary so the captured weights persist after hook removal/cleanup
        return self.attention_weights.copy()

    def clear(self):
        """
        Zeros/clears stored weights without removing hooks.
        """
        self.attention_weights.clear()

    def remove_hooks(self):
        """
        Removes all hooks and restores original forward methods, then clears storage.
        """
        # Restore original forward methods to prevent overhead
        if hasattr(self.model, 'transformer_encoder'):
            layers = self.model.transformer_encoder.layers
            for idx, layer in enumerate(layers):
                if idx in self.original_forwards:
                    layer.self_attn.forward = self.original_forwards[idx]
        self.original_forwards.clear()
        
        # Remove registered hooks
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        
        # Clear storage
        self.attention_weights.clear()

    def __enter__(self):
        """
        Context manager enter method. Registers hooks and patches if not already done.
        """
        if not self.hooks:
            self._register_and_patch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method. Automatically cleans up hooks and patches.
        """
        self.remove_hooks()


def build_token_labels(
    encoder_input_ids: list,      # list of 181 integers
    input_word2idx: dict          # dict mapping word string to idx
) -> list:
    """
    Takes the raw encoder_input_ids list and the input_word2idx vocabulary dict
    and returns a list of 181 strings where each string is the decoded token name.
    """
    idx2word = {v: k for k, v in input_word2idx.items()}
    return [idx2word.get(token_id, f"<unk_{token_id}>") for token_id in encoder_input_ids]


def visualize_attention_single_checkpoint(
    checkpoint_path: str,         # path to .pt or .pth checkpoint file
    example_input: torch.Tensor,  # shape (1, 181) — a single tokenized sequence
    token_labels: list,           # list of 181 strings, one per token position
    model_config: dict,           # dict with keys needed to instantiate StoneStateDecoderClassifier
    output_path: str,             # path to save the output PNG
    separator_positions: list = None,  # optional list of positions to mark with vertical dashed lines
    device: str = "cpu"
):
    """
    Loads a StoneStateDecoderClassifier checkpoint, registers attention hooks,
    runs a forward pass on the input sequence, retrieves attention weights for all 4 layers,
    and visualizes the 181x181 attention maps for all layers and heads.
    """
    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Dynamic parameter detection from the checkpoint state_dict
    detected_max_len = 5000
    if "positional_encoding.pe" in state_dict:
        detected_max_len = state_dict["positional_encoding.pe"].shape[0]
        
    detected_num_classes = model_config.get("num_classes", 80)
    if "classification_head.weight" in state_dict:
        detected_num_classes = state_dict["classification_head.weight"].shape[0]
        
    detected_src_vocab_size = model_config.get("src_vocab_size", 25)
    if "src_tok_emb.weight" in state_dict:
        detected_src_vocab_size = state_dict["src_tok_emb.weight"].shape[0]

    # Model configuration for architecture adjusted dynamically
    epoch_model_config = model_config.copy()
    epoch_model_config["max_len"] = detected_max_len
    epoch_model_config["num_classes"] = detected_num_classes
    epoch_model_config["src_vocab_size"] = detected_src_vocab_size

    # 2. Instantiate StoneStateDecoderClassifier and load weights
    model = StoneStateDecoderClassifier(**epoch_model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Run forward pass under context manager and torch.no_grad()
    with torch.no_grad():
        with AttentionHookManager(model) as manager:
            # Move example_input to target device
            example_input_dev = example_input.to(device)
            # Run forward pass
            _ = model(example_input_dev)
            
            # Retrieve attention weights
            weights = manager.get_attention_weights()
            
    # 4. Create matplotlib figure (4 rows x 4 columns for 4 layers x 4 heads)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    
    # Prepare tick marks for x-axis & y-axis: tick every 11th position
    ticks = [11 * i for i in range(17)]
    tick_labels = [f"SUP{i}" for i in range(16)] + ["Q"]
    
    for layer_idx in range(4):
        # Shape: [1, 4, 181, 181] -> squeeze batch dim to get [4, 181, 181]
        layer_weights = weights[layer_idx].squeeze(0).cpu()
        
        for head_idx in range(4):
            ax = axes[layer_idx, head_idx]
            matrix = layer_weights[head_idx].numpy()
            
            # Plot the heatmap using imshow
            im = ax.imshow(matrix, cmap='viridis', aspect='equal')
            
            # Set titles and axes descriptions
            ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Key position (attended-to)", fontsize=8)
            ax.set_ylabel("Query position (attending-from)", fontsize=8)
            
            # Apply tick labels
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=6)
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels, fontsize=6)
            
            # Highlight final query potion token with a red dashed line at row 180
            ax.axhline(180, color='red', linestyle='--', linewidth=1.2, label='Query Potion')
            
            # Optionally mark `<item_sep>` positions with vertical gray dashed lines
            if separator_positions is not None:
                for pos in separator_positions:
                    ax.axvline(pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                    
    fig.tight_layout()
    # Save the output PNG at dpi=150
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compare_correct_vs_incorrect_attention(
    checkpoint_path: str,
    dataset_path: str,
    model_config: dict,
    n_samples: int = 50,        # number of correct AND incorrect examples to average over
    output_path: str = "attention_diff.png",
    device: str = "cpu"
):
    """
    Loads a StoneStateDecoderClassifier checkpoint and dataset, splits predictions into
    correct and incorrect prediction sets, computes average attention weight maps,
    computes the difference (mean_correct - mean_incorrect), and visualizes the results on a 4x4 grid
    using a diverging colormap centered at 0.
    """
    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Dynamic parameter detection from the checkpoint state_dict
    detected_max_len = 5000
    if "positional_encoding.pe" in state_dict:
        detected_max_len = state_dict["positional_encoding.pe"].shape[0]
        
    detected_num_classes = model_config.get("num_classes", 80)
    if "classification_head.weight" in state_dict:
        detected_num_classes = state_dict["classification_head.weight"].shape[0]
        
    detected_src_vocab_size = model_config.get("src_vocab_size", 25)
    if "src_tok_emb.weight" in state_dict:
        detected_src_vocab_size = state_dict["src_tok_emb.weight"].shape[0]

    # Model configuration for architecture adjusted dynamically
    epoch_model_config = model_config.copy()
    epoch_model_config["max_len"] = detected_max_len
    epoch_model_config["num_classes"] = detected_num_classes
    epoch_model_config["src_vocab_size"] = detected_src_vocab_size

    # 2. Instantiate and load model
    model = StoneStateDecoderClassifier(**epoch_model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        
    correct_examples_weights = []
    incorrect_examples_weights = []
    
    correct_count = 0
    incorrect_count = 0
    
    # 4. Loop through dataset and collect weights under context manager and no_grad
    with torch.no_grad():
        with AttentionHookManager(model) as manager:
            for item in dataset:
                # Stop early if we have collected enough of both sets
                if len(correct_examples_weights) >= n_samples and len(incorrect_examples_weights) >= n_samples:
                    break
                    
                encoder_input_ids = item['encoder_input_ids']
                target_class_id = item['target_class_id']
                
                input_tensor = torch.tensor([encoder_input_ids], device=device)
                
                # Clear attention weights from previous pass
                manager.clear()
                
                # Run forward pass
                logits = model(input_tensor)
                
                # Squeeze or argmax to get prediction
                pred_class_id = logits.argmax(dim=-1).item()
                is_correct = (pred_class_id == target_class_id)
                
                # Retrieve the attention weights captured in this forward pass
                weights = manager.get_attention_weights()
                
                # Squeeze the batch dimension to shape [nhead, seq_len, seq_len] and move to CPU
                sample_weights = {}
                for layer_idx, w in weights.items():
                    sample_weights[layer_idx] = w.squeeze(0).cpu()
                    
                if is_correct:
                    correct_count += 1
                    if len(correct_examples_weights) < n_samples:
                        correct_examples_weights.append(sample_weights)
                else:
                    incorrect_count += 1
                    if len(incorrect_examples_weights) < n_samples:
                        incorrect_examples_weights.append(sample_weights)
                        
    # 5. Check if we found enough samples and print warnings if needed
    if len(correct_examples_weights) < n_samples:
        warnings.warn(
            f"Only found {len(correct_examples_weights)} correct examples (requested {n_samples})."
        )
    if len(incorrect_examples_weights) < n_samples:
        warnings.warn(
            f"Only found {len(incorrect_examples_weights)} incorrect examples (requested {n_samples})."
        )
        
    print(f"\n--- Attention Comparison Summary ---")
    print(f"Total correct predictions found in search: {correct_count}")
    print(f"Total incorrect predictions found in search: {incorrect_count}")
    print(f"Correct examples used for averaging: {len(correct_examples_weights)}")
    print(f"Incorrect examples used for averaging: {len(incorrect_examples_weights)}")
    print(f"-------------------------------------\n")
    
    # 6. Compute averages for each layer
    mean_correct = {}
    mean_incorrect = {}
    
    num_correct_used = len(correct_examples_weights)
    num_incorrect_used = len(incorrect_examples_weights)
    
    # Get sequence length from first item in dataset
    seq_len = len(dataset[0]['encoder_input_ids'])
    nhead = model_config['nhead']
    
    for layer_idx in range(4):
        if num_correct_used > 0:
            sum_attn = sum(item[layer_idx] for item in correct_examples_weights)
            mean_correct[layer_idx] = sum_attn / num_correct_used
        else:
            mean_correct[layer_idx] = torch.zeros((nhead, seq_len, seq_len))
            
        if num_incorrect_used > 0:
            sum_attn = sum(item[layer_idx] for item in incorrect_examples_weights)
            mean_incorrect[layer_idx] = sum_attn / num_incorrect_used
        else:
            mean_incorrect[layer_idx] = torch.zeros((nhead, seq_len, seq_len))
            
    # 7. Plot 4x4 Grid of difference maps
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 18))
    
    for layer_idx in range(4):
        # Calculate diff: mean_correct - mean_incorrect
        layer_diff = mean_correct[layer_idx] - mean_incorrect[layer_idx]
        
        for head_idx in range(4):
            ax = axes[layer_idx, head_idx]
            matrix = layer_diff[head_idx].numpy()
            
            # Center colormap at 0
            vmax = max(float(abs(matrix.min())), float(abs(matrix.max())))
            if vmax == 0:
                vmax = 1.0
                
            im = ax.imshow(matrix, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
            
            ax.set_title(f"L{layer_idx} H{head_idx} diff", fontsize=10, fontweight='bold')
            ax.set_xlabel("Key position (attended-to)", fontsize=8)
            ax.set_ylabel("Query position (attending-from)", fontsize=8)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # Define test parameters
    model_config = {
        'num_decoder_layers': 4,
        'emb_size': 256,
        'nhead': 4,
        'src_vocab_size': 100,
        'num_classes': 10
    }
    
    # 1. Create a dummy StoneStateDecoderClassifier
    print("1. Creating a dummy StoneStateDecoderClassifier...")
    model = StoneStateDecoderClassifier(**model_config)
    
    # Save a temporary checkpoint
    checkpoint_path = "temp_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"   Saved dummy checkpoint to {checkpoint_path}")
    
    # 2. Build dummy vocabulary for token label decoding
    print("2. Constructing vocabulary and building token labels...")
    vocab = {"<unk>": 0, "PAD": 1, "ORANGE": 2}
    for i in range(3, 100):
        vocab[f"token_{i}"] = i
        
    dummy_input = torch.randint(0, 100, (1, 181))
    token_labels = build_token_labels(dummy_input[0].tolist(), vocab)
    print(f"   Decoded {len(token_labels)} labels. First 5 labels: {token_labels[:5]}")
    
    # 3. Create a dummy dataset pickle for comparing correct vs incorrect
    dataset_path = "temp_dataset.pkl"
    dummy_dataset = []
    for _ in range(15):
        # random sequence of length 181
        seq = torch.randint(0, 100, (181,)).tolist()
        # Random target class
        target = torch.randint(0, 10, (1,)).item()
        dummy_dataset.append({
            'encoder_input_ids': seq,
            'target_class_id': target
        })
    with open(dataset_path, 'wb') as f:
        pickle.dump(dummy_dataset, f)
    print(f"   Saved dummy dataset to {dataset_path}")
    
    # 4. Run visualization test
    output_path = "temp_attention_plot.png"
    separator_positions = [10 + 11 * i for i in range(16)] # positions 10, 21, 32, ..., 175
    
    print(f"4. Running single-checkpoint visualization to save plot to {output_path}...")
    visualize_attention_single_checkpoint(
        checkpoint_path=checkpoint_path,
        example_input=dummy_input,
        token_labels=token_labels,
        model_config=model_config,
        output_path=output_path,
        separator_positions=separator_positions,
        device="cpu"
    )
    print("   Single checkpoint visualization finished successfully!")
    
    # 5. Run correct vs incorrect comparison test
    diff_output_path = "temp_attention_diff.png"
    print(f"5. Running correct vs incorrect comparison (saving plot to {diff_output_path})...")
    compare_correct_vs_incorrect_attention(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        model_config=model_config,
        n_samples=5,
        output_path=diff_output_path,
        device="cpu"
    )
    print("   Comparison finished successfully!")
    
    # 6. Clean up temporary files
    import os
    for path in [checkpoint_path, output_path, dataset_path, diff_output_path]:
        if os.path.exists(path):
            os.remove(path)
    print("   Cleaned up temporary test files. Complete!")
