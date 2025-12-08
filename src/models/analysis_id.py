import torch
from functools import partial
import numpy as np
import dadapy
from dadapy import data
import skdim
from models import create_transformer_model, create_classifier_model, create_decoder_classifier_model, create_linear_model
from data_loaders import AlchemyDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def compute_participation_ratio(activations):
    """
    Computes the Participation Ratio (PR) of a matrix of activations.
    PR = (Sum(eigenvalues)^2) / Sum(eigenvalues^2)
    activations: [N_samples, Hidden_Dim]
    """
    # 1. Center the data
    mean = torch.mean(activations, dim=0, keepdim=True)
    centered = activations - mean
    
    # 2. Compute Covariance Matrix (X^T X) or (X X^T) depending on dimensions
    # For SVD, we can just run SVD on the centered data
    # U, S, V = torch.svd(centered)
    # Ideally use PCA via SVD on centered data
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except:
        # Fallback for older torch versions or stability
        _, S, _ = torch.svd(centered)
        
    eigenvalues = S ** 2
    
    # 3. Compute PR
    pr = (torch.sum(eigenvalues) ** 2) / torch.sum(eigenvalues ** 2)
    return pr.item()


def get_analysis_id_dadapy(activations):
    """
    Robust ID estimation using Dadapy (Laio group implementation).
    activations: Tensor or Numpy array (Samples, Hidden_Dim)
    """
    results = {}
    print(f"Analyzing {len(activations)} layers")

    for layer_name, layer_data in activations.items():
        # 1. Convert to numpy
        X = layer_data.detach().cpu().numpy().astype(np.float64) # Use float64 for precision

        # 2. Initialize Dadapy Data object
        _data = data.Data(X)

        # 3. CRITICAL: Remove duplicates with a tolerance
        # Neural nets often produce collapsed representations. 
        # distinct=True keeps only unique points.
        _data.remove_identical_points()
        
        # Check if we have enough points left after cleaning
        if _data.N < 20:
            print(f"[Warning] Layer {layer_name}: Too few unique points ({_data.N}) for ID estimation.")
            results[layer_name] = 0.0
            continue

        # 4. Compute ID using Two-NN
        # range_max: usually N, but can be limited for speed on massive datasets
        id_est, id_error, scale = _data.compute_id_2NN()

        results[layer_name] = id_est
        
        print(f"Layer {layer_name}: ID = {id_est:.2f} ± {id_error:.2f} (Ambient: {X.shape[1]}, Samples: {_data.N})")

    return results

def get_analysis_id_skdim(activations):
    """
    Estimate intrinsic dimension using skdim package
    activations: numpy array of shape (num_samples, num_features). These are the activations for 'k' layers.
    """

    results = {}
    
    # 1. Capture Activations
    # (Assuming you have a function/hook to get activations for a batch)
    # activations = dictionary {layer_idx: tensor of shape (Batch*Seq, Hidden_Dim)}

    print("number of layers:", len(activations))
    
    # 2. Estimate ID for each layer
    for layer, data in activations.items():
        # Move to CPU numpy
        X = data.detach().cpu().numpy()

        # Remove duplicate rows
        X, counts = np.unique(X, axis=0, return_counts=True)
        
        # METHOD: Two-NN (Robust, standard for Deep Learning ID)
        # estimator = skdim.id.DANCo(verbose=True)
        estimator = skdim.id.TwoNN()
        import pdb; pdb.set_trace()
        id_val = estimator.fit_transform(X)

        # Implement other methods?
        
        results[layer] = id_val
        print(f"Layer {layer}: ID = {id_val:.2f} (Ambient = {X.shape[1]})")
        
    return results

def get_layer_wise_activations(model, dataloader, device, pad_token_id, get_last_token_only=True):
    """
    Get layer-wise activations for the entire dataset.
    """
    activations = {}
    padding_masks = []
    all_input_ids = []
    
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            # Handle different output formats (tuple, tensor, etc.)
            if isinstance(output, tuple):
                output = output[0]
            activations[name].append(output.detach().cpu())
        return hook

    # Register hooks - adjust based on your actual model structure
    hooks = []

    
    # For classifier models
    if hasattr(model, 'transformer_encoder'): 
        for idx, layer in enumerate(model.transformer_encoder.layers):
            # NOTE: The current hook captures the output of the entire layer after all the operations are limited.
            hooks.append(layer.register_forward_hook(get_activation(f'transformer_layer_{idx}'))) 

    # Forward pass through the dataset
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if 'encoder_input_ids' in batch:
                inputs = batch['encoder_input_ids'].to(device)
                all_input_ids.append(inputs.cpu()) # Store inputs
                
                src_padding_mask = (inputs == pad_token_id).to(device)
                padding_masks.append(src_padding_mask.cpu())
                
                # Check if model requires specific args (classification vs seq2seq)
                # Adapting to your script's logic:
                model(inputs, src_padding_mask=src_padding_mask)
            else:
                raise ValueError("Batch structure not recognized")

    # Process Masks and Inputs
    all_padding_masks = torch.cat(padding_masks, dim=0)
    all_inputs = torch.cat(all_input_ids, dim=0)
    
    # Calculate indices of the last valid token
    sequence_lengths = (~all_padding_masks).sum(dim=1) - 1
    sequence_lengths = torch.clamp(sequence_lengths, min=0)
    
    # EXTRACT LABELS: The token ID at the last position
    # Shape: (Total_Seqs,)
    last_token_indices = all_inputs[torch.arange(all_inputs.size(0)), sequence_lengths]

    # Process Activations
    final_activations = {}
    for key in activations.keys():
        layer_data = torch.cat(activations[key], dim=0)
        # Extract last token vector
        if get_last_token_only:
            final_activations[key] = layer_data[torch.arange(layer_data.size(0)), sequence_lengths, :]
        else:
            final_activations[key] = layer_data 


    # Remove hooks
    for hook in hooks:
        hook.remove()

    return final_activations, last_token_indices




def visualize_layer_geometry(activations, labels, layer_name):
    
    X = activations[layer_name].detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=15)


    plt.colorbar(scatter, label='Token ID')
    plt.title(f'PCA of Activations at {layer_name}')
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'pca_activations_{layer_name}.png')


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint and reconstruct the architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arguments = checkpoint['args']
    
    # Reconstruct the model architecture based on saved args
    if arguments.task_type in ["seq2seq", "seq2seq_stone_state"]:
        from models import create_transformer_model
        src_vocab_size = len(checkpoint['src_vocab_word2idx'])
        tgt_vocab_size = len(checkpoint['tgt_vocab_word2idx'])
        model = create_transformer_model(
            config_name=arguments.model_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=device,
            max_len=arguments.max_seq_len
        )
    elif arguments.task_type == "classification":
        from models import create_classifier_model, create_decoder_classifier_model
        src_vocab_size = len(checkpoint['src_vocab_word2idx'])
        num_classes = 108
        
        if arguments.model_architecture == "decoder":
            model = create_decoder_classifier_model(
                config_name=arguments.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_classes,
                device=device,
                max_len=arguments.max_seq_len,
                padding_side=arguments.padding_side,
                batch_size=arguments.batch_size,
                use_flash_attention=True
            )
        
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


if __name__ == "__main__":
    import os
    import re
    
    # Directory containing checkpoints
    base_checkpoint_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt'
    checkpoint_dir = os.path.dirname(base_checkpoint_path)
    
    print(f"Looking for checkpoints in: {checkpoint_dir}")

    # Find all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pt") and "model_epoch" in filename:
            # Extract epoch number
            # Matches 'model_epoch_10_...' or 'best_model_epoch_200_...'
            match = re.search(r'epoch_(\d+)', filename)
            if match:
                epoch = int(match.group(1))
                checkpoint_files.append((epoch, os.path.join(checkpoint_dir, filename)))

    # Sort by epoch
    checkpoint_files.sort(key=lambda x: x[0])
    
    # Filter to avoid duplicates (e.g. best_model_epoch_X and model_epoch_X)
    unique_checkpoints = {}
    for epoch, path in checkpoint_files:
        unique_checkpoints[epoch] = path
    
    sorted_epochs = sorted(unique_checkpoints.keys())
    print(f"Found {len(sorted_epochs)} unique epochs to process: {sorted_epochs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize data structures
    layer_ids_over_time = {} 

    # Setup Dataloader ONCE using the first available checkpoint to get configuration
    first_epoch = sorted_epochs[0]
    first_ckpt_path = unique_checkpoints[first_epoch]
    
    print(f"Loading configuration from {first_ckpt_path}")
    temp_checkpoint = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
    args = temp_checkpoint['args']
    
    val_dataset = AlchemyDataset(
                json_file_path=args.val_data_path,
                task_type=args.task_type,
                vocab_word2idx=temp_checkpoint['src_vocab_word2idx'],
                vocab_idx2word=temp_checkpoint['src_vocab_idx2word'],
                stone_state_to_id=temp_checkpoint.get('stone_state_to_id') if args.task_type == "classification" else None,
                filter_query_from_support=args.filter_query_from_support,
                num_workers=args.num_workers,
                preprocessed_dir=args.preprocessed_dir,
                use_preprocessed=args.use_preprocessed,
                input_format=args.input_format,
                output_format=args.output_format,
                model_architecture=args.model_architecture
    )

    custom_collate_val = partial(collate_fn, pad_token_id=val_dataset.pad_token_id, eos_token_id = val_dataset.eos_token_id, 
                                         task_type=args.task_type, model_architecture=args.model_architecture, 
                                         sos_token_id=val_dataset.sos_token_id, prediction_type=args.prediction_type,
                                         max_seq_len=args.max_seq_len, truncate=args.use_truncation)

    val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_val,
                num_workers=1,
                generator=torch.Generator().manual_seed(args.seed)
    )
    
    del temp_checkpoint
    
    # Iterate over epochs
    for epoch in tqdm(sorted_epochs, desc="Analyzing Epochs"):
        ckpt_path = unique_checkpoints[epoch]
        
        # Load model
        model, _ = load_model_from_checkpoint(ckpt_path, device=device)
        
        # Get activations
        activations, labels = get_layer_wise_activations(model, val_dataloader, device, val_dataset.pad_token_id, get_last_token_only=True)
        
        # Calculate ID
        id_results = get_analysis_id_dadapy(activations)
        
        # Store
        for layer_name, id_val in id_results.items():
            if layer_name not in layer_ids_over_time:
                layer_ids_over_time[layer_name] = {'epochs': [], 'ids': []}
            layer_ids_over_time[layer_name]['epochs'].append(epoch)
            layer_ids_over_time[layer_name]['ids'].append(id_val)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plotting
    plt.figure(figsize=(12, 8))
    
    for layer_name, data in layer_ids_over_time.items():
        plt.plot(data['epochs'], data['ids'], marker='o', label=layer_name, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Intrinsic Dimension', fontsize=12)
    plt.title('Intrinsic Dimension Evolution per Layer', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    output_plot_path = os.path.join('id_evolution_over_epochs.png')
    plt.savefig(output_plot_path, dpi=300)
    print(f"Saved ID evolution plot to {output_plot_path}")