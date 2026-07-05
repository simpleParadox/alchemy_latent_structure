import torch
import torch.nn as nn
import torch.nn.functional as F
from activation_cache import ActivationCacheManager

_printed_attn_shapes = set()

def split_support_query(encoder_input_ids, item_sep_id):
    """
    Splits encoder_input_ids sequence into support prefix and query suffix at the last item_sep_id.
    """
    enc_inps_list = encoder_input_ids.tolist() if hasattr(encoder_input_ids, "tolist") else list(encoder_input_ids)
    try:
        # Find index of last item_sep_id in sequence
        last_sep_idx = len(enc_inps_list) - 1 - enc_inps_list[::-1].index(item_sep_id)
        query = enc_inps_list[last_sep_idx + 1:]
        support = enc_inps_list[:last_sep_idx + 1]
    except ValueError:
        # No separator found (empty support set)
        query = enc_inps_list
        support = []
    return support, query

def decode_sequence(token_ids, idx2word):
    """
    Decodes a tensor or list of token IDs into a space-separated string of token names.
    """
    if torch.is_tensor(token_ids):
        token_ids = token_ids.cpu().numpy()
    import numpy as np
    flat_ids = np.array(token_ids).flatten()
    return " ".join([idx2word.get(int(tid), f"<unk_{tid}>") for tid in flat_ids])

def make_new_fwd(fwd, self_attn_mod, layer_idx, cache_dict=None, patch_config=None):
    def new_fwd(query, key, value, *args, **kwargs):
        orig_weight = self_attn_mod.out_proj.weight
        orig_bias = self_attn_mod.out_proj.bias
        device = orig_weight.device
        dtype = orig_weight.dtype
        d_model = orig_weight.shape[0]
        
        self_attn_mod.out_proj.weight = nn.Parameter(torch.eye(d_model, device=device, dtype=dtype))
        if orig_bias is not None:
            self_attn_mod.out_proj.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
            
        pre_proj, weights = fwd(query, key, value, *args, **kwargs)
        
        self_attn_mod.out_proj.weight = orig_weight
        self_attn_mod.out_proj.bias = orig_bias
        
        batch_first = getattr(self_attn_mod, 'batch_first', True)
        act_to_process = pre_proj
        if not batch_first:
            act_to_process = pre_proj.transpose(0, 1)
            
        assert act_to_process.shape[-1] == d_model, f"Expected self_attn_output shape[-1] to be {d_model}, got {act_to_process.shape}"
        
        global _printed_attn_shapes
        layer_attn_name = f"layer_{layer_idx}_self_attn"
        if layer_attn_name not in _printed_attn_shapes:
            print(f"[First forward pass] {layer_attn_name} pre-projection output shape: {list(act_to_process.shape)}")
            _printed_attn_shapes.add(layer_attn_name)
            
        if cache_dict is not None:
            cache_dict[layer_attn_name] = act_to_process.detach()
            
        if patch_config is not None and patch_config.get('layer_idx') == layer_idx:
            patched_act = act_to_process.clone()
            t_mask = patch_config.get('target_token_masks')
            c_act = patch_config.get('corrupt_activation')
            head_idx = patch_config.get('head_idx')
            
            if not batch_first:
                mask_expanded = t_mask.transpose(0, 1).unsqueeze(-1)
            else:
                mask_expanded = t_mask.unsqueeze(-1)
            
            if head_idx is not None:
                d_head = d_model // getattr(self_attn_mod, 'num_heads', 4)
                h_start = head_idx * d_head
                h_end = (head_idx + 1) * d_head
                
                p_slice = patched_act[:, :, h_start:h_end]
                c_slice = c_act[:, :, h_start:h_end]
                patched_act[:, :, h_start:h_end] = torch.where(mask_expanded, c_slice, p_slice)
            else:
                patched_act = torch.where(mask_expanded, c_act, patched_act)
                    
            if not batch_first:
                pre_proj = patched_act.transpose(0, 1)
            else:
                pre_proj = patched_act
                
        projected = F.linear(pre_proj, orig_weight, orig_bias)
        return projected, weights
    return new_fwd

def patch_attention_modules(model, cache_dict, patch_config=None):
    original_forwards = {}
    for i, layer in enumerate(model.transformer_encoder.layers):
        orig_fwd = layer.self_attn.forward
        original_forwards[i] = orig_fwd
        layer.self_attn.forward = make_new_fwd(orig_fwd, layer.self_attn, i, cache_dict=cache_dict, patch_config=patch_config)
    return original_forwards

def unpatch_attention_modules(model, original_forwards):
    for i, orig_fwd in original_forwards.items():
        model.transformer_encoder.layers[i].self_attn.forward = orig_fwd

def compute_patching_score_batched(model, clean_batch, corrupt_batch, component_name, target_token_masks, t_clean_ids_list, t_corrupt_ids_list, setup='noising', overlap_strategy='disjoint'):
    """
    Computes the patching scores using both softmax delta and logsumexp diff metrics across a batch.
    
    overlap_strategy: 'disjoint' (assert no overlap), 'subtract' (remove intersection), 'allow' (keep as is)
    """
    
    _t_clean = []
    _t_corrupt = []
    for b in range(len(t_clean_ids_list)):
        clean_set = frozenset(t_clean_ids_list[b])
        corrupt_set = frozenset(t_corrupt_ids_list[b])
        intersection = clean_set & corrupt_set
        
        if intersection:
            if overlap_strategy == 'disjoint':
                raise ValueError(f"Target class overlap detected in scoring: {intersection}")
            elif overlap_strategy == 'subtract':
                clean_set = clean_set - intersection
                corrupt_set = corrupt_set - intersection
        
        _t_clean.append(clean_set)
        _t_corrupt.append(corrupt_set)
    
    t_clean_ids_list = _t_clean
    t_corrupt_ids_list = _t_corrupt
        
    device = next(model.parameters()).device
    
    if setup == 'noising':
        base_input = clean_batch.to(device)
        patch_input = corrupt_batch.to(device)
    elif setup == 'denoising':
        base_input = corrupt_batch.to(device)
        patch_input = clean_batch.to(device)
    else:
        raise ValueError(f"Unknown setup: {setup}")
        
    target_token_masks = target_token_masks.to(device)
    
    # 1. Base run forward pass and cache activations
    base_attn_cache = {}
    orig_forwards = patch_attention_modules(model, base_attn_cache)
    try:
        with ActivationCacheManager(model) as base_cache:
            f_base = model(base_input)
            base_acts = base_cache.get_activations()
    finally:
        unpatch_attention_modules(model, orig_forwards)
        
    for k, v in base_attn_cache.items():
        base_acts[k] = v
        
    # 2. Patch source run forward pass and cache activations
    source_attn_cache = {}
    orig_forwards = patch_attention_modules(model, source_attn_cache)
    try:
        with ActivationCacheManager(model) as source_cache:
            f_source = model(patch_input)
            source_acts = source_cache.get_activations()
    finally:
        unpatch_attention_modules(model, orig_forwards)
        
    for k, v in source_attn_cache.items():
        source_acts[k] = v
        
    if setup == 'noising':
        f_clean = f_base
        f_corrupt = f_source
    else:
        f_clean = f_source
        f_corrupt = f_base
        
    if component_name.startswith('layer_') and '_head_' in component_name:
        layer_idx = int(component_name.split('_')[1])
        head_idx = int(component_name.split('_')[3])
        
        patch_config = {
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'target_token_masks': target_token_masks,
            'corrupt_activation': source_acts.get(f"layer_{layer_idx}_self_attn")
        }
        
        orig_forwards = patch_attention_modules(model, None, patch_config=patch_config)
        try:
            f_patched = model(base_input)
        except Exception as e:
            raise Exception(f"Errored out forward pass for head patching. Error: {e}")
        finally:
            unpatch_attention_modules(model, orig_forwards)
            
    else:
        if component_name == 'embedding':
            target_module = model.src_tok_emb
        elif component_name.startswith('layer_') and component_name.endswith('_output'):
            layer_idx = int(component_name.split('_')[1])
            target_module = model.transformer_encoder.layers[layer_idx]
        elif component_name.startswith('layer_') and component_name.endswith('_mlp_out'):
            layer_idx = int(component_name.split('_')[1])
            target_module = model.transformer_encoder.layers[layer_idx].linear2
        else:
            raise ValueError(f"Unknown component_name '{component_name}'")
            
        corrupt_act = source_acts.get(component_name)
        if corrupt_act is None:
            raise ValueError(f"Component '{component_name}' not found in corrupt cached activations. Keys are: {list(source_acts.keys())}")
            
        def make_patch_hook(c_act, t_mask):
            def hook(module, inp, outp):
                is_tuple = isinstance(outp, tuple)
                out_tensor = outp[0] if is_tuple else outp
                patched = out_tensor.clone()
                
                batch_first = getattr(module, 'batch_first', True)
                if not batch_first:
                    mask_expanded = t_mask.transpose(0, 1).unsqueeze(-1)
                else:
                    mask_expanded = t_mask.unsqueeze(-1)
                    
                patched = torch.where(mask_expanded, c_act, patched)
                        
                if is_tuple:
                    return (patched,) + outp[1:]
                return patched
            return hook
            
        handle = target_module.register_forward_hook(make_patch_hook(corrupt_act, target_token_masks))
        
        try:
            f_patched = model(base_input)
        except Exception as e:
            raise Exception(f"Errored out forward pass for the third run, removing hook. Error: {e}")
        finally:
            handle.remove()
            
    softmax_scores = []
    lse_scores = []
    raw_lse_scores = []
    
    probs_clean = torch.nn.functional.softmax(f_clean, dim=-1)
    probs_corrupt = torch.nn.functional.softmax(f_corrupt, dim=-1)
    probs_patched = torch.nn.functional.softmax(f_patched, dim=-1)
    
    B = f_clean.shape[0]
    for b in range(B):
        c_ids = list(t_clean_ids_list[b])
        corr_ids = list(t_corrupt_ids_list[b])
        
        # If after subtraction sets are empty, scores are exactly 0
        if not c_ids or not corr_ids:
            softmax_scores.append(0.0)
            lse_scores.append(0.0)
            raw_lse_scores.append(0.0)
            continue
            
        # Softmax delta
        def s_delta(probs):
            return probs[b, c_ids].sum().item() - probs[b, corr_ids].sum().item()
            
        sd_clean = s_delta(probs_clean)
        sd_corrupt = s_delta(probs_corrupt)
        sd_patched = s_delta(probs_patched)
        
        denom_softmax = sd_clean - sd_corrupt
        
        # LSE diff
        def l_diff(logits):
            return torch.logsumexp(logits[b, c_ids], dim=0).item() - torch.logsumexp(logits[b, corr_ids], dim=0).item()
            
        ld_clean = l_diff(f_clean)
        ld_corrupt = l_diff(f_corrupt)
        ld_patched = l_diff(f_patched)
        
        denom_lse = ld_clean - ld_corrupt
        
        if setup == 'noising':
            raw_lse_score = ld_clean - ld_patched
            softmax_score = (sd_clean - sd_patched) / denom_softmax if abs(denom_softmax) > 1e-7 else 0.0
            lse_score = raw_lse_score / denom_lse if abs(denom_lse) > 1e-7 else 0.0
        elif setup == 'denoising':
            raw_lse_score = ld_patched - ld_corrupt
            softmax_score = (sd_patched - sd_corrupt) / denom_softmax if abs(denom_softmax) > 1e-7 else 0.0
            lse_score = raw_lse_score / denom_lse if abs(denom_lse) > 1e-7 else 0.0
            
        softmax_scores.append(softmax_score)
        lse_scores.append(lse_score)
        raw_lse_scores.append(raw_lse_score)
        
    return {
        "softmax_scores": softmax_scores,
        "lse_scores": lse_scores,
        "raw_lse_scores": raw_lse_scores
    }
