import torch
import torch.nn as nn
from src.models.models import StoneStateDecoderClassifier

class ActivationCacheManager:
    """
    Manager for capturing residual stream activations at each layer boundary 
    (i.e., after the MLP + residual add of each TransformerEncoderLayer).
    
    This class registers forward hooks on the modules within a model's 
    transformer_encoder.layers to capture and store activations.
    
    It supports context manager syntax for clean and automatic registration/cleanup.
    """
    def __init__(self, model: nn.Module):
        """
        Registers forward hooks on all nn.TransformerEncoderLayer modules in
        model.transformer_encoder.layers and stores handles for cleanup.
        """
        self.model = model
        self.hooks = []
        self.activations = {}
        
        self._register_hooks()

    def _register_hooks(self):
        """
        Finds layers in model.transformer_encoder.layers and the embedding module,
        and registers forward hooks to capture activations.
        """
        if not hasattr(self.model, 'transformer_encoder'):
            raise ValueError("Model does not have a 'transformer_encoder' attribute.")
            
        # 1. Register hook on embedding layer if present
        if hasattr(self.model, 'src_tok_emb'):
            def make_emb_hook():
                def hook(module, inp, outp):
                    self.activations['embedding'] = outp.detach()
                return hook
            handle = self.model.src_tok_emb.register_forward_hook(make_emb_hook())
            self.hooks.append(handle)
            
        # 2. Register hooks on transformer layers
        layers = self.model.transformer_encoder.layers
        for idx, layer in enumerate(layers):
            def make_hook(layer_idx):
                def hook(module, inp, outp):
                    activation = outp.detach()
                    
                    # Ensure shape is [batch, seq_len, d_model] even if batch_first is False
                    batch_first = getattr(module, 'batch_first', True)
                    if not batch_first:
                        # Permute from [seq_len, batch, d_model] to [batch, seq_len, d_model]
                        activation = activation.transpose(0, 1)
                        
                    self.activations[layer_idx] = activation
                    self.activations[f"layer_{layer_idx}_output"] = activation
                return hook
                
            handle = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(handle)

            # 3. Register hook on MLP (linear2) if present
            if hasattr(layer, 'linear2'):
                def make_mlp_hook(layer_idx):
                    def hook(module, inp, outp):
                        activation = outp.detach()
                        # Ensure shape is [batch, seq_len, d_model] even if batch_first is False
                        batch_first = getattr(self.model.transformer_encoder.layers[layer_idx], 'batch_first', True)
                        if not batch_first:
                            activation = activation.transpose(0, 1)
                        self.activations[f"layer_{layer_idx}_mlp_out"] = activation
                    return hook
                    
                handle_mlp = layer.linear2.register_forward_hook(make_mlp_hook(idx))
                self.hooks.append(handle_mlp)

    def get_activations(self) -> dict:
        """
        Returns dict: {layer_idx (int): activations (Tensor of shape [batch, seq_len, d_model])}
        """
        return self.activations.copy()

    def clear(self):
        """
        Zeros/clears stored activations without removing hooks.
        """
        self.activations.clear()

    def remove_hooks(self):
        """
        Removes all hooks and clears storage.
        """
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()

    def __enter__(self):
        """
        Context manager enter method.
        """
        if not self.hooks:
            self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.
        """
        self.remove_hooks()


if __name__ == '__main__':
    # Sanity check: Run a dummy forward pass and print shape of captured tensors.
    print("Initializing dummy model and configuration...")
    model_config = {
        'num_decoder_layers': 4,
        'emb_size': 256,
        'nhead': 4,
        'src_vocab_size': 100,
        'num_classes': 10,
        'use_flash_attention': True
    }
    
    # 1. Create a dummy StoneStateDecoderClassifier
    model = StoneStateDecoderClassifier(**model_config)
    model.eval()
    
    # 2. Create dummy input tensor of shape (batch, seq_len)
    # Batch size = 2, seq_len = 181
    dummy_input = torch.randint(0, 100, (2, 181))
    
    print("Running forward pass with ActivationCacheManager...")
    # 3. Use context manager to hook outputs of encoder layers
    with ActivationCacheManager(model) as cache:
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Retrieve captured activations
        captured_activations = cache.get_activations()
        
        print(f"Captured activations for {len(captured_activations)} layers:")
        for layer_idx, act in captured_activations.items():
            print(f"  Layer {layer_idx}: shape = {list(act.shape)} (Expected: [2, 181, 256])")
            
    print("Sanity check completed successfully.")
