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
import json
import hashlib
from pathlib import Path
import matplotlib as mpl


def compute_participation_ratio(activations: torch.Tensor) -> float:
    """
    Effective dimension via Participation Ratio (PR).
    activations: [N, D]
    """
    mean = torch.mean(activations, dim=0, keepdim=True)
    centered = activations - mean
    # SVD singular vals -> eigenvalues of covariance proportional to S^2
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        print("torch.linalg.svd failed, trying torch.svd instead")
        _, S, _ = torch.svd(centered)
    eigenvalues = S ** 2
    denom = torch.sum(eigenvalues ** 2)
    if denom.item() == 0:
        return 0.0
    pr = (torch.sum(eigenvalues) ** 2) / denom
    return float(pr.item())

def get_analysis_id_dadapy_2nn(activations: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    DADApy 2NN intrinsic dimension per layer.
    """
    results: dict[str, float] = {}
    for layer_name, layer_data in activations.items():
        X = layer_data.detach().cpu().numpy().astype(np.float64)
        _data = data.Data(X)
        _data.remove_identical_points()
        if _data.N < 20:
            results[layer_name] = 0.0
            print(f"Skipping ID computation for {layer_name} due to insufficient data points ({_data.N}), setting ID to 0.0")
            continue
        id_est, id_error, scale = _data.compute_id_2NN()
        results[layer_name] = float(id_est)
    return results

def get_analysis_id_skdim_pcafo(activations: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    skdim PCA_FO intrinsic dimension per layer.
    Fast, linear effective dimension estimate.
    """
    results: dict[str, float] = {}
    estimator = skdim.id.PCAFO()
    for layer_name, layer_data in activations.items():
        X = layer_data.detach().cpu().numpy().astype(np.float64)
        if X.shape[0] < 5:
            results[layer_name] = 0.0
            continue
        try:
            estimator.fit(X)
            results[layer_name] = float(estimator.dimension_)
        except Exception as e:
            print(f"PCAFO failed for {layer_name}: {e}")
            results[layer_name] = 0.0
    return results

def get_analysis_id_skdim_2nn(activations: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    skdim TwoNN intrinsic dimension per layer.
    """
    results: dict[str, float] = {}
    estimator = skdim.id.TwoNN()
    for layer_name, layer_data in activations.items():
        X = layer_data.detach().cpu().numpy().astype(np.float64)
        if X.shape[0] < 20:
            results[layer_name] = 0.0
            continue
        try:
            estimator.fit(X)
            results[layer_name] = float(estimator.dimension_)
        except Exception as e:
            print(f"skdim TwoNN failed for {layer_name}: {e}")
            results[layer_name] = 0.0
    return results

def get_analysis_id_pr(activations: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Participation Ratio per layer (effective dimension).
    """
    results: dict[str, float] = {}
    for layer_name, layer_data in activations.items():
        # layer_data is (N, D) already because you store last-token only for transformer layers,
        # and (N, C) for classification head.
        results[layer_name] = compute_participation_ratio(layer_data)
    return results

def compute_intrinsic_dimension(activations: dict[str, torch.Tensor], method: str) -> dict[str, float]:
    """
    Unified entry point.
    """
    if method == "dadapy_2nn":
        return get_analysis_id_dadapy_2nn(activations)
    if method == "skdim_pcafo":
        return get_analysis_id_skdim_pcafo(activations)
    if method == "skdim_2nn":
        return get_analysis_id_skdim_2nn(activations)
    if method == "pr":
        return get_analysis_id_pr(activations)
    raise ValueError(f"Unknown id method: {method}")

def get_layer_wise_activations(model, dataloader, device, pad_token_id, get_last_token_only=True):
    activations = {}
    padding_masks = []
    all_input_ids = []
    
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            if isinstance(output, tuple):
                output = output[0]
            activations[name].append(output.detach().cpu())
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

def process_single_epoch(kwargs):
    """
    Worker function to process a single epoch on its assigned device.
    Uses cache-on-disk for activations.
    """
    global worker_val_dataset, worker_collate_fn, worker_device

    epoch = kwargs["epoch"]
    ckpt_path = kwargs["ckpt_path"]
    args = kwargs["args"]
    id_method = kwargs["id_method"]

    device = worker_device
    checkpoint_dir = os.path.dirname(ckpt_path)

    signature = kwargs.get("signature") or "no_signature"

    val_dataloader = DataLoader(
        worker_val_dataset,
        batch_size=200,
        shuffle=False,
        collate_fn=worker_collate_fn,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    try:
        if activations_cache_exists(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            expected_layers=None,
            split="val",
            signature=signature,
        ):
            activations = load_activations_cache(checkpoint_dir, epoch, split="val")
        else:
            model, _ = load_model_from_checkpoint(ckpt_path, device=device)
            activations, _ = get_layer_wise_activations(
                model,
                val_dataloader,
                device,
                worker_val_dataset.pad_token_id,
                get_last_token_only=True,
            )

            save_activations_cache(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                activations=activations,
                split="val",
                signature=signature,
                extra_meta={"ckpt_path": ckpt_path},
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute ID from activations (cached or computed)
        id_results = compute_intrinsic_dimension(activations, method=id_method)

        del activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return epoch, id_results

    except Exception as e:
        print(f"Error processing epoch {epoch} on {device}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return epoch, {}

# --- Activation cache helpers ---

def _safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace("/", "_").replace(" ", "_")

def _cache_root_for_checkpoint_dir(checkpoint_dir: str, split: str = "val") -> Path:
    # Stored alongside the seed/hop/exp_type directory since it's inside checkpoint_dir
    return Path(checkpoint_dir) / "activation_cache" / split

def _epoch_cache_dir(checkpoint_dir: str, epoch: int, split: str = "val") -> Path:
    return _cache_root_for_checkpoint_dir(checkpoint_dir, split=split) / f"epoch_{epoch:04d}"

def _meta_path(checkpoint_dir: str, epoch: int, split: str = "val") -> Path:
    return _epoch_cache_dir(checkpoint_dir, epoch, split=split) / "meta.json"

def _compute_val_signature(args, vocab_w2i: dict) -> str:
    """
    Best-effort signature to avoid accidental cache reuse across different validation sets/tokenizers.
    """
    payload = {
        "val_data_path": getattr(args, "val_data_path", None),
        "max_seq_len": getattr(args, "max_seq_len", None),
        "input_format": getattr(args, "input_format", None),
        "output_format": getattr(args, "output_format", None),
        "model_architecture": getattr(args, "model_architecture", None),
        "task_type": getattr(args, "task_type", None),
        "vocab_size": len(vocab_w2i) if vocab_w2i is not None else None,
    }
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(s).hexdigest()

def activations_cache_exists(checkpoint_dir: str, epoch: int, expected_layers: list[str] | None, split: str, signature: str) -> bool:
    epoch_dir = _epoch_cache_dir(checkpoint_dir, epoch, split=split)
    meta_file = _meta_path(checkpoint_dir, epoch, split=split)
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except Exception:
        return False

    if meta.get("signature") != signature:
        return False

    if expected_layers is None:
        # If you don't know layers ahead of time, trust meta
        return True

    for layer in expected_layers:
        f = epoch_dir / f"{_safe_layer_filename(layer)}.pt"
        if not f.exists():
            return False
    return True

def save_activations_cache(checkpoint_dir: str, epoch: int, activations: dict, split: str, signature: str, extra_meta: dict | None = None) -> None:
    epoch_dir = _epoch_cache_dir(checkpoint_dir, epoch, split=split)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Save one file per layer to keep files manageable
    for layer_name, tensor in activations.items():
        out_path = epoch_dir / f"{_safe_layer_filename(layer_name)}.pt"
        torch.save(tensor, out_path)

    meta = {
        "epoch": epoch,
        "split": split,
        "signature": signature,
        "layers": sorted(list(activations.keys())),
        "cached_last_token_only": True,  # because you call get_last_token_only=True
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_file = _meta_path(checkpoint_dir, epoch, split=split)
    meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

def load_activations_cache(checkpoint_dir: str, epoch: int, split: str = "val") -> dict:
    epoch_dir = _epoch_cache_dir(checkpoint_dir, epoch, split=split)
    meta = json.loads((_meta_path(checkpoint_dir, epoch, split=split)).read_text())

    activations = {}
    for layer_name in meta["layers"]:
        p = epoch_dir / f"{_safe_layer_filename(layer_name)}.pt"
        activations[layer_name] = torch.load(p, map_location="cpu", weights_only=False)
    return activations

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
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs to analyze')

    # Replace the old id_algorithm flag with id_method
    parser.add_argument(
        '--id_method',
        type=str,
        default='dadapy_2nn',
        choices=['dadapy_2nn', 'skdim_pcafo', 'skdim_2nn', 'pr'],
        help='Intrinsic/effective dimension method'
    )
    parser.add_argument('--seed', type=int, default=16, help='Seed for loading the model checkpoint')

    args_cli = parser.parse_args()
    
    if args_cli.exp_type == 'held_out':
        seed_paths = {
            0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt',
            2: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/best_model_epoch_200_classification_xsmall.pt',
            3: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_3/best_model_epoch_200_classification_xsmall.pt'
        }
    if args_cli.exp_type == 'composition':
        if args_cli.hop == 2:
            # seed_paths = {
            #     0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/best_model_epoch_200_classification_xsmall.pt',
            #     16: '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/best_model_epoch_200_classification_xsmall.pt',
            #     29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/best_model_epoch_200_classification_xsmall.pt'
            # }
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/flatten_linear_input/best_model_epoch_200_classification_xsmall.pt',
                16: ' /home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/flatten_linear_input/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/flatten_linear_input/best_model_epoch_200_classification_xsmall.pt'
            }
        elif args_cli.hop == 3:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }
        elif args_cli.hop == 4:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }

        elif args_cli.hop == 5:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine_restarts/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }

    elif args_cli.exp_type == 'decomposition':
        if args_cli.hop == 2:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }
        elif args_cli.hop == 3:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }
        elif args_cli.hop == 4:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }
        elif args_cli.hop == 5:
            seed_paths = {
                0: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_0/best_model_epoch_200_classification_xsmall.pt',
                16: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/best_model_epoch_200_classification_xsmall.pt',
                29: '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_29/best_model_epoch_200_classification_xsmall.pt'
            }
    base_checkpoint_path = seed_paths[args_cli.seed]
    print(f"Base checkpoint path: {base_checkpoint_path}")

    # Replace the seed part of the path with one supplied via CLI
    base_checkpoint_path = re.sub(r'seed_\d+', f'seed_{args_cli.seed}', base_checkpoint_path)

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
    
    # Restrict the number of epochs to user-specified maximum if needed.
    if args_cli.max_epochs is not None:
        sorted_epochs = [epoch for epoch in sorted_epochs if epoch <= args_cli.max_epochs]

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available.")
    
    first_ckpt_path = unique_checkpoints[sorted_epochs[0]]
    temp_checkpoint = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = temp_checkpoint['args']
    
    vocab_w2i = temp_checkpoint['src_vocab_word2idx']
    vocab_i2w = temp_checkpoint['src_vocab_idx2word']
    stone_s2i = temp_checkpoint.get('stone_state_to_id')
    
    num_processes = min(args_cli.num_processes, 4, num_gpus if num_gpus > 0 else 4)
    print(f"Using {num_processes} processes")
    
    signature = _compute_val_signature(ckpt_args, vocab_w2i)

    worker_args_list = []
    for epoch in sorted_epochs:
        worker_args_list.append({
            "epoch": epoch,
            "ckpt_path": unique_checkpoints[epoch],
            "args": ckpt_args,
            "id_method": args_cli.id_method,
            "signature": signature,
        })
    
    del temp_checkpoint

    print(f"Starting processing with {num_processes} processes")
    
    if args_cli.wandb_project:
        run_name = args_cli.wandb_run_name if args_cli.wandb_run_name else f"id_{args_cli.exp_type}_hop_{args_cli.hop}_{args_cli.id_method}"
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
            print("Epoch:", epoch)
            
            for layer_name, id_val in id_results.items():
                if layer_name not in layer_ids_over_time:
                    layer_ids_over_time[layer_name] = {'epochs': [], 'ids': []}
                layer_ids_over_time[layer_name]['epochs'].append(epoch)
                layer_ids_over_time[layer_name]['ids'].append(id_val)
                
                wandb_log_dict[f"ID/{layer_name}"] = id_val

            if args_cli.wandb_project:
                # include the method in the metric namespace so runs are comparable
                wandb.log({f"{args_cli.id_method}/{k}": v for k, v in wandb_log_dict.items()}, step=epoch)


    # Define a purple gradient: shallower layers are lighter, deeper layers are darker.
    # (Uses Matplotlib's built-in "Purples" colormap.)

    layer_names = sorted(
        (k for k in layer_ids_over_time.keys() if k.startswith("transformer_layer_")),
        key=lambda s: int(s.split("_")[-1])
    )

    cmap = mpl.colormaps["Purples"]
    if len(layer_names) > 1:
        # Avoid extremes (too close to white/black) for readability
        color_positions = np.linspace(0.35, 0.90, len(layer_names))
    elif len(layer_names) == 1:
        color_positions = [0.65]
    else:
        color_positions = []

    colors = {name: cmap(pos) for name, pos in zip(layer_names, color_positions)}

    # Keep classification head distinct (green)
    colors["classification_layer"] = 'green'

    plt.figure(figsize=(12, 8))
    for layer_name, plot_data in layer_ids_over_time.items():
        plt.plot(
            plot_data['epochs'],
            plot_data['ids'],
            marker='o',
            label=layer_name,
            linewidth=2,
            color=colors.get(layer_name, "C0"),
        )

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Intrinsic Dimension', fontsize=12)
    plt.title(f'Intrinsic Dimension Evolution per Layer ({args_cli.id_method})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    folder = 'intrinsic_dimension_plots_seed_wise'

    if not os.path.exists(folder):
        os.makedirs(folder)

    if args_cli.custom_save_dir is None:
        save_dir = os.path.join(folder, f'id_evolution_{args_cli.exp_type}_hop_{args_cli.hop}_{args_cli.id_method}_{args_cli.seed}.png')
    else:
        # Add the seed info to the args_cli.custom_save_dir which is a string.
        temp_save_dir = args_cli.custom_save_dir + f'_{args_cli.id_method}_seed_{args_cli.seed}.png'
        save_dir = os.path.join(folder, temp_save_dir)

    plt.savefig(save_dir, dpi=300)
    print(f"Saved ID evolution plot to {save_dir}")
    if args_cli.wandb_project:
        wandb.finish()