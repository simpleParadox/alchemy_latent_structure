import torch
import torch.multiprocessing as mp
from functools import partial
import numpy as np
from dadapy import data
import skdim
from models import create_transformer_model, create_classifier_model, create_decoder_classifier_model, create_linear_model
from data_loaders import AlchemyDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import re
import argparse
import wandb
import gc

import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_pca_visualization(Xz, layer_name, epoch, save_dir):
    """
    Saves a 2D PCA scatter + PCA explained variance plot.
    Xz: standardized activations (N, D)
    """
    if save_dir is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    # PCA fit
    pca = PCA(n_components=10)
    pca.fit(Xz)

    # 2D projection (use a random subsample if huge)
    if Xz.shape[0] > 3000:
        idx = np.random.choice(Xz.shape[0], 3000, replace=False)
        X_small = Xz[idx]
    else:
        X_small = Xz

    X_proj = pca.transform(X_small)[:, :2]

    # --- 2D scatter plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(X_proj[:, 0], X_proj[:, 1], s=5, alpha=0.5)
    plt.title(f"PCA Scatter (Layer={layer_name}, Epoch={epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{layer_name}_epoch_{epoch}_scatter.png"))
    plt.close()

    # --- Explained variance plot ---
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, 11), pca.explained_variance_ratio_)
    plt.title(f"Explained Variance (Layer={layer_name}, Epoch={epoch})")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    plt.tight_layout()
    # TODO: add composition / decomposition / held-out to filename.
    plt.savefig(os.path.join(save_dir, f"{layer_name}_epoch_{epoch}_variance.png"))
    plt.close()



def get_analysis_id_dadapy(activations, epoch=None, save_pca_plots_dir=None):
    """
    Robust intrinsic dimension estimation.
    Includes:
    - Standardization
    - Tolerant dedup
    - PCA visualization (optional)
    - dadapy 2NN estimator + TwoNN fallback
    """

    results = {}

    for layer_name, layer_data in activations.items():

        X = layer_data.detach().cpu().numpy().astype(np.float64)

        # Flatten if tensor is (N, L, D)
        if X.ndim == 3:
            print(f"[INFO] Flattening activations for layer {layer_name} from shape {X.shape} to 2D.")
            N, L, D = X.shape
            X = X.reshape(N * L, D)

        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        Xz = (X - mean) / std

        Xz_round = np.round(Xz, decimals=8)
        _, unique_idx = np.unique(Xz_round, axis=0, return_index=True)
        X_unique = Xz[sorted(unique_idx)]

        if save_pca_plots_dir is not None:
            plot_pca_visualization(Xz, layer_name, epoch, save_pca_plots_dir)

        if X_unique.shape[0] < 20:
            print(f"[WARN] {layer_name}: only {X_unique.shape[0]} unique points; using TwoNN fallback.")
            try:
                tw = skdim.id.TwoNN().fit(Xz)
                results[layer_name] = float(tw.dimension_)
            except Exception as e:
                print(f"[ERROR] TwoNN failed for {layer_name}: {e}")
                results[layer_name] = np.nan
            continue

        # dadapy estimator
        try:
            _data = data.Data(X_unique)
            id_est, id_err, scale = _data.compute_id_2NN(algorithm='ml')
            results[layer_name] = float(id_est)
        except Exception as e:
            print(f"[ERROR] dadapy failed for {layer_name}: {e}; falling back to TwoNN.")
            try:
                tw = skdim.id.TwoNN().fit(Xz)
                results[layer_name] = float(tw.dimension_)
            except Exception as e2:
                print(f"[ERROR] fallback TwoNN also failed: {e2}")
                results[layer_name] = np.nan

    return results




def compute_participation_ratio(activations):
    mean = torch.mean(activations, dim=0, keepdim=True)
    centered = activations - mean
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except:
        _, S, _ = torch.svd(centered)
    eigenvalues = S ** 2
    pr = (torch.sum(eigenvalues) ** 2) / torch.sum(eigenvalues ** 2)
    return pr.item()

# def get_analysis_id_dadapy(activations):
#     results = {}
#     for layer_name, layer_data in activations.items():
#         X = layer_data.detach().cpu().numpy().astype(np.float64)
#         print(f'Activation variance for layer {layer_name} = {np.var(X,axis=0).mean()}')

#         _data = data.Data(X)
#         _data.remove_identical_points()
#         if _data.N < 20:
#             results[layer_name] = 0.0
#             print(f"Skipping ID computation for {layer_name} due to insufficient data points ({_data.N})")
#             continue
#         id_est, id_error, scale = _data.compute_id_2NN(algorithm='ml')
#         results[layer_name] = id_est
#     return results

def get_layer_wise_activations(model, dataloader, device, pad_token_id, get_last_token_only=True):
    activations = {}
    padding_masks = []
    all_input_ids = []
    
    def get_activation(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            # move to cpu but avoid detaching for the debug print
            arr = out.detach().cpu()
            if name not in activations:
                activations[name] = []
                print(f"[HOOK] Layer {name} output shape (per-batch): {arr.shape}")
                # print a tiny slice to sanity-check ordering
                print(f"[HOOK] sample values (first row): {arr.reshape(-1, arr.shape[-1])[0,:5]}")
            activations[name].append(arr)
        return hook



    hooks = []
    if hasattr(model, 'transformer_encoder'): 
        for idx, layer in enumerate(model.transformer_encoder.layers):
            hooks.append(layer.register_forward_hook(get_activation(f'transformer_layer_{idx}'))) 
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): 
        for idx, layer in enumerate(model.transformer.h):
            hooks.append(layer.register_forward_hook(get_activation(f'transformer_layer_{idx}')))


    if hasattr(model, "classification_head"):
        hooks.append(model.classification_head.register_forward_hook(
            get_activation("classification_layer")
        ))
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting Activations"):
            if 'encoder_input_ids' in batch:
                inputs = batch['encoder_input_ids'].to(device)
                all_input_ids.append(inputs.cpu())
                src_padding_mask = (inputs == pad_token_id).to(device)
                padding_masks.append(src_padding_mask.cpu())
                model(inputs, src_padding_mask=src_padding_mask)
            else:
                raise ValueError("Batch structure not recognized")

    all_padding_masks = torch.cat(padding_masks, dim=0)
    all_inputs = torch.cat(all_input_ids, dim=0)
    sequence_lengths = (~all_padding_masks).sum(dim=1) - 1
    sequence_lengths = torch.clamp(sequence_lengths, min=0)
    last_token_indices = all_inputs[torch.arange(all_inputs.size(0)), sequence_lengths]

    final_activations = {}

    for key in activations.keys():
        layer_data = torch.cat(activations[key], dim=0)

        if key == "classification_layer":
            # classifier output has shape (batch, C)
            final_activations[key] = layer_data
            continue

        # transformer layers have shape (batch, seq_len, dim)
        if get_last_token_only:
            final_activations[key] = layer_data[torch.arange(layer_data.size(0)), sequence_lengths, :]
        else:
            final_activations[key] = layer_data

    for hook in hooks:
        hook.remove()

    return final_activations, last_token_indices

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arguments = checkpoint['args']
    
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
        else:
             model = create_classifier_model(
                config_name=arguments.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_classes,
                device=device,
                max_len=arguments.max_seq_len
            )
            
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

# --- Global Variables for Workers ---
worker_val_dataset = None
worker_collate_fn = None
worker_device = None

def init_worker(args, vocab_w2i, vocab_i2w, stone_s2i, gpu_queue):
    """
    Initializer for worker processes.
    Instantiates the dataset once per process and assigns a fixed GPU.
    """
    global worker_val_dataset, worker_collate_fn, worker_device
    
    worker_device = gpu_queue.get()
    print(f"Worker {os.getpid()} initialized on {worker_device}")
    
    worker_val_dataset = AlchemyDataset(
                json_file_path=args.val_data_path,
                task_type=args.task_type,
                vocab_word2idx=vocab_w2i,
                vocab_idx2word=vocab_i2w,
                stone_state_to_id=stone_s2i,
                filter_query_from_support=args.filter_query_from_support,
                num_workers=0, 
                preprocessed_dir=args.preprocessed_dir,
                use_preprocessed=args.use_preprocessed,
                input_format=args.input_format,
                output_format=args.output_format,
                model_architecture=args.model_architecture
    )

    worker_collate_fn = partial(collate_fn, pad_token_id=worker_val_dataset.pad_token_id, eos_token_id=worker_val_dataset.eos_token_id, 
                                         task_type=args.task_type, model_architecture=args.model_architecture, 
                                         sos_token_id=worker_val_dataset.sos_token_id, prediction_type=args.prediction_type,
                                         max_seq_len=args.max_seq_len, truncate=args.use_truncation)

# --- Worker Function ---

def process_single_epoch(kwargs):
    """
    Worker function to process a single epoch on its assigned device.
    Uses the globally initialized dataset and device.
    """
    global worker_val_dataset, worker_collate_fn, worker_device
    
    epoch = kwargs['epoch']
    ckpt_path = kwargs['ckpt_path']
    args = kwargs['args']
    
    device = worker_device
    
    val_dataloader = DataLoader(
                worker_val_dataset,
                batch_size=200,
                shuffle=False,
                collate_fn=worker_collate_fn,
                num_workers=0, 
                generator=torch.Generator().manual_seed(args.seed)
    )

    model, _ = load_model_from_checkpoint(ckpt_path, device=device)
    activations, _ = get_layer_wise_activations(model, val_dataloader, device, worker_val_dataset.pad_token_id, get_last_token_only=True)
    id_results = get_analysis_id_dadapy(
        activations,
        epoch=epoch,
        save_pca_plots_dir=kwargs['args'].save_pca_plots_dir
    )
    
    del model
    del activations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    return epoch, id_results
    # except Exception as e:
    #     print(f"Error processing epoch {epoch} on {device}: {e}")
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     gc.collect()
    #     return epoch, {}

# --- Main Execution ---

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Analyze Intrinsic Dimension over Training Epochs")
    parser.add_argument('--custom_save_dir', type=str, required=False, help='Path to save the output plot')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--exp_type', type=str, choices=['held_out', 'composition', 'decomposition'])
    parser.add_argument('--hop', type=int, default=2, help='Number of hops for composition/decomposition experiments')
    
    parser.add_argument('--wandb_project', type=str, default='alchemy-meta-learning', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='simpleParadox')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--analysis_method', type=str, choices=['id'], default='id')
    parser.add_argument('--save_pca_plots_dir', type=str, default=None,
                    help="Directory where PCA plots per layer/epoch will be stored. Disabled if None.")


    args_cli = parser.parse_args()
    
    if args_cli.exp_type == 'held_out':
        base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt'

    if args_cli.exp_type == 'composition':
        if args_cli.hop == 2:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/best_model_epoch_200_classification_xsmall.pt'
        elif args_cli.hop == 3:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_16/best_model_epoch_200_classification_xsmall.pt'
        elif args_cli.hop == 4:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_16/best_model_epoch_200_classification_xsmall.pt'
        elif args_cli.hop == 5:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_16/best_model_epoch_200_classification_xsmall.pt'

    elif args_cli.exp_type == 'decomposition':
        if args_cli.hop == 2:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt'
        elif args_cli.hop == 3:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt'
        elif args_cli.hop == 4:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt'
        elif args_cli.hop == 5:
            base_checkpoint_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt'

    checkpoint_dir = os.path.dirname(base_checkpoint_path)
    
    print(f"Looking for checkpoints in: {checkpoint_dir}")

    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pt") and "model_epoch" in filename:
            match = re.search(r'epoch_(\d+)', filename)
            if match:
                epoch = int(match.group(1))
                checkpoint_files.append((epoch, os.path.join(checkpoint_dir, filename)))

    checkpoint_files.sort(key=lambda x: x[0])
    unique_checkpoints = {epoch: path for epoch, path in checkpoint_files}
    sorted_epochs = sorted(unique_checkpoints.keys())
    print(f"Found {len(sorted_epochs)} unique epochs to process.")

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available.")
    
    first_ckpt_path = unique_checkpoints[sorted_epochs[0]]
    temp_checkpoint = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = temp_checkpoint['args']
    
    vocab_w2i = temp_checkpoint['src_vocab_word2idx']
    vocab_i2w = temp_checkpoint['src_vocab_idx2word']
    stone_s2i = temp_checkpoint.get('stone_state_to_id')

    # Update ckpt_args with save_pca_plots_dir
    ckpt_args.save_pca_plots_dir = args_cli.save_pca_plots_dir
    
    num_processes = min(args_cli.num_processes, 4, num_gpus if num_gpus > 0 else 4)
    print(f"Using {num_processes} processes")
    
    worker_args_list = []
    for epoch in sorted_epochs:
        worker_args = {
            'epoch': epoch,
            'ckpt_path': unique_checkpoints[epoch],
            'args': ckpt_args
        }
        worker_args_list.append(worker_args)
    
    del temp_checkpoint

    print(f"Starting processing with {num_processes} processes...")
    
    if args_cli.wandb_project:
        run_name = args_cli.wandb_run_name if args_cli.wandb_run_name else f"id_{args_cli.exp_type}_hop_{args_cli.hop}"
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}_{now}"
        wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=run_name,
            config=vars(args_cli)
        )

    layer_ids_over_time = {} 

    manager = mp.Manager()
    gpu_queue = manager.Queue()
    
    for i in range(num_processes):
        if num_gpus > 0:
            gpu_queue.put(f'cuda:{i % num_gpus}')
        else:
            gpu_queue.put('cpu')

    init_args = (ckpt_args, vocab_w2i, vocab_i2w, stone_s2i, gpu_queue)

    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
        results_iter = pool.imap(process_single_epoch, worker_args_list)
        
        for epoch, id_results in tqdm(results_iter, total=len(sorted_epochs), desc="Analyzing Epochs"):
            wandb_log_dict = {}
            
            for layer_name, id_val in id_results.items():
                if layer_name not in layer_ids_over_time:
                    layer_ids_over_time[layer_name] = {'epochs': [], 'ids': []}
                layer_ids_over_time[layer_name]['epochs'].append(epoch)
                layer_ids_over_time[layer_name]['ids'].append(id_val)
                
                wandb_log_dict[f"ID/{layer_name}"] = id_val

            if args_cli.wandb_project:
                wandb.log(wandb_log_dict, step=epoch)

    plt.figure(figsize=(12, 8))
    for layer_name, plot_data in layer_ids_over_time.items():
        plt.plot(plot_data['epochs'], plot_data['ids'], marker='o', label=layer_name, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Intrinsic Dimension', fontsize=12)
    plt.title('Intrinsic Dimension Evolution per Layer', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    folder = 'intrinsic_dimension_plots'

    if not os.path.exists(folder):
        os.makedirs(folder)

    if args_cli.custom_save_dir is None:
        save_dir = os.path.join(folder, f'id_evolution_{args_cli.exp_type}_hop_{args_cli.hop}_ml.png')
    else:
        save_dir = os.path.join(folder, args_cli.custom_save_dir)

    plt.savefig(save_dir, dpi=300)
    
    print(f"Saved ID evolution plot to {save_dir}")
    
    if args_cli.wandb_project:
        wandb.finish()