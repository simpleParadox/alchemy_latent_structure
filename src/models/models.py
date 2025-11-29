import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    """Reference: https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html#define-the-model"""
    # def __init__(self, d_model, dropout=0.1, max_len=5000):
    #     super(PositionalEncoding, self).__init__()
    #     self.dropout = nn.Dropout(p=dropout)

    #     pe = torch.zeros(max_len, d_model)
    #     position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    #     pe[:, 0::2] = torch.sin(position * div_term) # Apply sine to even indices. The ::2 means every second element starting from index 0 which are the even indices.
    #     pe[:, 1::2] = torch.cos(position * div_term) # Apply cosine to odd indices
    #     pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model) # This is similar to the implementation in the PyTorch documentation.
    #     # https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html#define-the-model
    #     # self.register_parameter('pe', nn.Parameter(pe, requires_grad=False)) # Register as a non-trainable parameter
    #     self.register_buffer('pe', pe) # Register as a buffer, not a parameter.

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    """
    A sequence-to-sequence model with a Transformer-based encoder and decoder.
    """
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, 
                 emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int = 512, dropout: float = 0.1, max_len=1024):
        super(Seq2SeqTransformer, self).__init__()
        
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
         
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_len)
        self.emb_size = emb_size

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for the Seq2SeqTransformer.
        Args:
            src: source sequence, shape (batch_size, src_seq_len)
            tgt: target sequence, shape (batch_size, tgt_seq_len)
            src_mask: the additive mask for the src sequence, shape (src_seq_len, src_seq_len)
            tgt_mask: the additive mask for the tgt sequence, shape (tgt_seq_len, tgt_seq_len)
            src_padding_mask: the ByteTensor mask for src keys per batch, shape (batch_size, src_seq_len)
            tgt_padding_mask: the ByteTensor mask for tgt keys per batch, shape (batch_size, tgt_seq_len)
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch, shape (batch_size, src_seq_len)
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # src: (N, S_src), tgt: (N, S_tgt)
        # Embedding output: (N, S, E)
        # PositionalEncoding expects: (S, N, E)
        # Transformer with batch_first=True expects: (N, S, E)
        
        src_emb = self.src_tok_emb(src) * math.sqrt(self.emb_size) # (N, S_src, E)
        src_emb_pe = self.positional_encoding(src_emb.permute(1,0,2)) # Input to PE: (S_src, N, E)
        src_emb_pe = src_emb_pe.permute(1,0,2) # Output from PE: (S_src, N, E) -> (N, S_src, E) for Transformer
        
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size) # (N, S_tgt, E)
        tgt_emb_pe = self.positional_encoding(tgt_emb.permute(1,0,2)) # Input to PE: (S_tgt, N, E)
        tgt_emb_pe = tgt_emb_pe.permute(1,0,2) # Output from PE: (S_tgt, N, E) -> (N, S_tgt, E) for Transformer

        transformer_out = self.transformer(src_emb_pe, tgt_emb_pe, src_mask, tgt_mask, None,
                                           src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # transformer_out shape: (N, S_tgt, E)
        
        return self.generator(transformer_out) # Output: (N, S_tgt, V_tgt)

    def encode(self, src, src_mask=None, src_padding_mask=None):
        """Encodes the source sequence."""
        # src: (N, S_src)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.emb_size) # (N, S_src, E)
        src_emb_pe = self.positional_encoding(src_emb.permute(1,0,2)) # Input to PE: (S_src, N, E)
        src_emb_pe = src_emb_pe.permute(1,0,2) # Output from PE: (S_src, N, E) -> (N, S_src, E) for Encoder
        return self.transformer.encoder(src_emb_pe, src_mask, src_padding_mask) # Output: (N, S_src, E)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        """Decodes the target sequence using the encoder's memory."""
        # tgt: (N, S_tgt), memory: (N, S_src, E)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size) # (N, S_tgt, E)
        tgt_emb_pe = self.positional_encoding(tgt_emb.permute(1,0,2)) # Input to PE: (S_tgt, N, E)
        tgt_emb_pe = tgt_emb_pe.permute(1,0,2) # Output from PE: (S_tgt, N, E) -> (N, S_tgt, E) for Decoder
        return self.transformer.decoder(tgt_emb_pe, memory, tgt_mask, None, 
                                        tgt_padding_mask, memory_key_padding_mask) # Output: (N, S_tgt, E)
        
    def generate(self, src, start_symbol_id, end_symbol_id, max_len, device, pad_token_id):
        """
            Autoregressive generation method.
            
            Args:
                src: Source sequence tensor, shape (batch_size, src_seq_len)
                start_symbol_id: Token ID to start generation (SOS token)
                end_symbol_id: Token ID to end generation (EOS token)
                max_len: Maximum length of generated sequence
                device: Device to create tensors on
                pad_token_id: Padding token ID for creating masks
                
            Returns:
                generated_ids: Generated token sequences, shape (batch_size, generated_seq_len)
        """
        self.eval()  # Set model to evaluation mode]
        
        # First create a padding mask and pass it through the encoder.
        src_padding_mask = (src == pad_token_id)  # Shape: (batch_size, src_seq_len)
        # 'src' is the encoder input, so we encode it first.
        memory = self.encode(src, src_padding_mask=src_padding_mask)  # Shape: (batch_size, src_seq_len, emb_size)
        
        batch_size = src.size(0)
        # Initialize the target sequence with the start symbol.
        tgt = torch.full((batch_size, 1), start_symbol_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # Track finished sequences.
        
        # Generate the sequence step by step.
        for i in range(max_len - 1):
            # Create the target mask for the current step.
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=device)
            tgt_len = tgt.size(1)
            
            # Decode the current target sequence.
            decoder_output = self.decode(
                tgt=tgt, 
                memory=memory, 
                tgt_mask=tgt_mask, 
                memory_key_padding_mask=src_padding_mask
            )
            
            # Get the logits for the last token logits only.
            last_token_logits = self.generator(decoder_output[:, -1, :]) # Generator is the linear layer over the vocabulary.
            
            # Get the token predictions using greedy decoding.
            next_token = last_token_logits.argmax(dim=-1)
            
            # Now append the newly predicted token to the target sequence.
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            just_finished = (next_token == end_symbol_id)
            finished = finished | just_finished  # Update finished sequences.
            
            if finished.all():
                break
           
        # Will remove the start symbol in the function that calls this.
        return tgt  # Shape: (batch_size, generated_seq_len)
        

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device="cpu"):
        """Generates a square mask for preventing attention to future tokens."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class StoneStateClassifier(nn.Module):
    """
    A Transformer-based encoder model for classifying the entire next stone state.
    """
    def __init__(self, num_encoder_layers: int, emb_size: int, nhead: int,
                 src_vocab_size: int, num_classes: int,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 max_len: int = 5000, pooling_strategy='global', item_sep_token_id=None,
                 io_sep_token_id=None): # Added max_len for PositionalEncoding consistency
        super(StoneStateClassifier, self).__init__()
        self.emb_size = emb_size
        self.architecture = "encoder"  # Add architecture attribute
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        # Assuming PositionalEncoding class is available in the scope
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True) # batch_first=True is crucial
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classification_head = nn.Linear(emb_size, num_classes)
        self.max_len = max_len  # Store max_len for positional encoding
        self.pooling_strategy = pooling_strategy  # Store pooling strategy
        self.item_sep_token = item_sep_token_id
        self.io_sep_token_id = io_sep_token_id
        
    def _global_pooling(self, transformer_output: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies global mean pooling over the sequence length, considering padding.
        Args:
            transformer_output: Output from the transformer encoder, shape (batch_size, src_seq_len, emb_size)
            src_padding_mask: ByteTensor mask for src keys per batch, shape (batch_size, src_seq_len)
        Returns:
            Pooled output tensor of shape (batch_size, emb_size).
        """
        if src_padding_mask is not None:
                # NOTE: What this essentially does is sums the embeddings of non-padded tokens and divides by the number of non-padded tokens.
                # Create a mask to zero out padded tokens in transformer_output
                # expanded_padding_mask: (batch_size, src_seq_len, 1), True for pad
                expanded_padding_mask = src_padding_mask.unsqueeze(-1).expand_as(transformer_output)
                masked_transformer_output = transformer_output.masked_fill(expanded_padding_mask, 0.0)
                
                # Sum the embeddings of non-padded tokens
                summed_output = masked_transformer_output.sum(dim=1) # (batch_size, emb_size)
                
                # Count the number of non-padded tokens for each sequence
                # actual_lengths: (batch_size, 1)
                actual_lengths = (~src_padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
                
                pooled_output = summed_output / actual_lengths # (batch_size, emb_size). Perform mean pooling over the sequence length, considering padding.
        else:
            # No padding mask provided, so perform a simple mean pool
            pooled_output = transformer_output.mean(dim=1)
        
        return pooled_output  # Shape: (batch_size, emb_size)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the StoneStateClassifier.
        Args:
            src: source sequence, shape (batch_size, src_seq_len)
            src_padding_mask: the ByteTensor mask for src keys per batch, shape (batch_size, src_seq_len)
                              True values indicate padding.
        Returns:
            Output tensor of shape (batch_size, num_classes) representing logits for each class.
        """
        # Embed source tokens
        # src: (batch_size, src_seq_len)
        # Apply positional encoding - PositionalEncoding class expects (seq_len, batch_size, emb_size)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.emb_size) # (batch_size, src_seq_len, emb_size)
        src_emb_permuted = src_emb.permute(1, 0, 2)  # (src_seq_len, batch_size, emb_size)
        src_emb_pe = self.positional_encoding(src_emb_permuted) # (src_seq_len, batch_size, emb_size)
        src_emb = src_emb_pe.permute(1, 0, 2)  # (batch_size, src_seq_len, emb_size)

        # Pass through Transformer encoder
        # src_key_padding_mask needs to be (batch_size, src_seq_len) where True means pad
        # print(f"src_padding_mask: {src_padding_mask}")
        # print(f"src_padding_mask: {src_padding_mask}")
        transformer_output = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        # transformer_output: (batch_size, src_seq_len, emb_size)
        
        
        if self.pooling_strategy == 'global':
            # Apply global mean pooling
            pooled_output = self._global_pooling(transformer_output, src_padding_mask)
            
        elif self.pooling_strategy == 'query_only':
            # NOTE: Never used.
            # First find the last non-padding token in each sequence. This will give us the sequence length.
            if src_padding_mask is not None:
                # If right padding, the last token is a pad token, so we need to find the index of the last valid token.
                sequence_lengths = (~src_padding_mask).sum(dim=1) - 1 # Get the last valid token index.
                # Now for each sequence in the batch, we find the index of the last 'item_separator' token.
                assert self.item_sep_token is not None, "item_sep_token must be provided for 'query_only' strategy."
                
                # Find the positions of item_sep_token_id (marks end of query input)
                # Find query tokens using for loop (easier to understand)
                pooled_outputs = []
                for i in range(src.size(0)):  # For each sequence in the batch
                    seq = src[i]
                    
                    # Find the last occurrence of item_sep_token_id
                    item_sep_positions = (seq == self.item_sep_token).nonzero(as_tuple=True)[0]
                    if len(item_sep_positions) == 0:
                        # Fallback to last token if no item separator found
                        print("Warning: No item separator found in sequence. Using last token for pooling. \n" \
                        "This may indicate an issue with the input formatting.")
                        pooled_outputs.append(transformer_output[i, -1, :])
                        continue
                    
                    last_item_sep_pos = item_sep_positions[-1].item()
                    
                    # Find the last occurrence of io_sep_token_id
                    io_sep_positions = (seq == self.io_sep_token_id).nonzero(as_tuple=True)[0]
                    if len(io_sep_positions) == 0:
                        # Fallback to last token if no io separator found
                        pooled_outputs.append(transformer_output[i, -1, :])
                        continue
                    
                    last_io_sep_pos = io_sep_positions[-1].item()
                    
                    # Extract query tokens (between item_sep and io_sep)
                    query_start = last_item_sep_pos + 1
                    query_end = last_io_sep_pos
                    
                    if query_start >= query_end:
                        # Invalid query region, fallback to last token
                        pooled_outputs.append(transformer_output[i, -1, :])
                    else:
                        # Average the query token embeddings
                        query_embeddings = transformer_output[i, query_start:query_end, :]
                        pooled_outputs.append(query_embeddings.mean(dim=0))
                
            else:
                print("Warning: src_padding_mask is None. Using mean pooling over the entire sequence.")
                pooled_output = self._global_pooling(transformer_output, src_padding_mask=src_padding_mask)
        elif self.pooling_strategy == 'last_token':
            # Use the representation of the last non-padding token in each sequence.
            if src_padding_mask is not None:
                # If right padding, the last token is a pad token, so we need to find the index of the last valid token.
                sequence_lengths = (~src_padding_mask).sum(dim=1) - 1 # Get the last valid token index. It's actually the length - 1.
                # Clamp to ensure we don't go below 0 for edge cases
                sequence_lengths = torch.clamp(sequence_lengths, min=0)
                # Extract the last valid token's representation for each sequence   
                pooled_output = transformer_output[torch.arange(transformer_output.size(0)), sequence_lengths, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}. Use 'global' or 'query_only', or 'last_token.")
        

        # Pass pooled output through classification head
        logits = self.classification_head(pooled_output) # (batch_size, num_classes)
        return logits

class StoneStateDecoderClassifier(nn.Module):
    """
    A Transformer-based decoder-only model for classification.
    This implementation uses TransformerEncoder layers with a causal mask for efficiency.
    """
    def __init__(self, num_decoder_layers: int, emb_size: int, nhead: int,
                 src_vocab_size: int, num_classes: int,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                max_len: int = 5000, prediction_type=None, padding_side: str = "right", use_flash_attention: bool = False,
                batch_size: int = 32): 
        super(StoneStateDecoderClassifier, self).__init__()
        self.emb_size = emb_size
        self.architecture = "decoder"  # Add architecture attribute
        self.prediction_type = prediction_type  # 'autoregressive' or 'feature'.
        self.padding_side = padding_side  # 'left' or 'right' - decoder models typically use left padding but could be right padded if predicting one token.
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_len)

        # Use TransformerEncoderLayer for a more efficient decoder-only implementation
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_decoder_layers)

        self.classification_head = nn.Linear(emb_size, num_classes)
        self.max_len = max_len

        self.use_flash_attention = use_flash_attention
        self.batch_size = batch_size
        print(f"Using flash attention: {self.use_flash_attention}")

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for the sequence."""
        """NOTE: In the https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/quantizable/modules/activation.py#L14, the attention mask is 'additive' in nature.
        This is done before the softmax calculation. After the -inf is applied, the softmax will ignore the positions where the value is -inf (because the addition of any value with -inf is -inf).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the StoneStateDecoderClassifier.
        Args:
            src: source sequence, shape (batch_size, src_seq_len)
            src_padding_mask: the ByteTensor mask for src keys per batch, shape (batch_size, src_seq_len)
                              True values indicate padding tokens.
        Returns:
            Output tensor of shape (batch_size, num_classes) representing logits.
        """
        # Embed and apply positional encoding
        src_emb = self.src_tok_emb(src) * math.sqrt(self.emb_size)
        src_emb_permuted = src_emb.permute(1, 0, 2)
        src_emb_pe = self.positional_encoding(src_emb_permuted)
        src_emb = src_emb_pe.permute(1, 0, 2)

        # Create causal mask for self-attention
        # Note: We create the mask for the full sequence length (including padding).
        # The padding mask will handle masking out padding tokens separately.
        # PyTorch's attention mechanism combines both masks correctly.
        seq_len = src.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)

        # Pass through the transformer encoder layers with causal mask
        # The combination of causal_mask and src_padding_mask ensures:
        # 1. No attention to future positions (causal_mask)
        # 2. No attention to padding tokens (src_padding_mask)
        decoder_output = self.transformer_encoder(
            src=src_emb, 
            mask=causal_mask, 
            src_key_padding_mask=src_padding_mask,
            is_causal=True
        )
        
        # For classification, we need the representation of the last valid token
        if src_padding_mask is not None:
            if self.padding_side == "left":
                print("Warning: Using left padding with a decoder model. Ensure this is intended.")
                # For left-padding, the last token (rightmost) is always the last token
                # since padding is on the left side
                last_token_output = decoder_output[:, -1, :]
            else:
                # For right-padding, find the last valid (non-padding) token for each sequence
                sequence_lengths = (~src_padding_mask).sum(dim=1) - 1  # -1 for 0-based indexing
                # Clamp to ensure we don't go below 0 for edge cases
                sequence_lengths = torch.clamp(sequence_lengths, min=0)
                # Extract the last valid token's representation for each sequence
                last_token_output = decoder_output[torch.arange(decoder_output.size(0)), sequence_lengths, :]
        else:
            # No padding mask, use the last token
            last_token_output = decoder_output[:, -1, :]
            print("Warning: src_padding_mask is None. Using the last token for classification.")
        
        if self.prediction_type == 'autoregressive':
            # Return full sequence output for autoregressive tasks
            return self.classification_head(decoder_output)
        
        # Return classification logits based on the last valid token
        next_token_logits = self.classification_head(last_token_output)
        return next_token_logits  # Shape: (batch_size, num_classes)
    
    
    def generate(self, src, src_padding_mask, start_symbol_id, end_symbol_id, max_len, device, pad_token_id):
        """
        Autoregressive generation method for a decoder-only model.
        It takes a source sequence as a prompt and generates subsequent tokens.

        Args:
            src: Source sequence (prompt) tensor, shape (batch_size, src_seq_len)
            start_symbol_id: Not used in this implementation, but kept for API consistency.
            end_symbol_id: Token ID to end generation (EOS token)
            max_len: Maximum length of the *entire* sequence (prompt + generated)
            device: Device to create tensors on
            pad_token_id: Padding token ID for creating masks

        Returns:
            generated_ids: The full sequence including the prompt and generated tokens,
                           shape (batch_size, generated_seq_len)
        """
        self.eval()  # Set model to evaluation mode
        
        batch_size = src.size(0)
        # The target sequence starts with the provided source (prompt).
        tgt = src.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Generate the sequence step by step.
        for _ in range(max_len - src.size(1)): # Only generate up to max_len
            # Create padding mask for the current sequence
            tgt_padding_mask = (tgt == pad_token_id)
            
            # Create a causal mask for self-attention.
            tgt_seq_len = tgt.size(1)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

            # Embed and apply positional encoding
            tgt_emb = self.src_tok_emb(tgt) * math.sqrt(self.emb_size)
            tgt_emb_permuted = tgt_emb.permute(1, 0, 2)
            tgt_emb_pe = self.positional_encoding(tgt_emb_permuted)
            tgt_emb = tgt_emb_pe.permute(1, 0, 2)

            # Pass through the transformer encoder with a causal mask
            decoder_output = self.transformer_encoder(
                src=tgt_emb,
                mask=causal_mask,
                src_key_padding_mask=tgt_padding_mask,
                is_causal=True
            )
            
            # Find the last valid token for next token prediction
            if tgt_padding_mask is not None:
                if self.padding_side == "left":
                    # For left-padding, the last token is always the rightmost position
                    last_token_output = decoder_output[:, -1, :]
                else:
                    # For right-padding, find the last valid (non-padding) token for each sequence
                    sequence_lengths = (~tgt_padding_mask).sum(dim=1) - 1
                    sequence_lengths = torch.clamp(sequence_lengths, min=0)
                    last_token_output = decoder_output[torch.arange(decoder_output.size(0)), sequence_lengths, :]
            else:
                # No padding, use the last token
                last_token_output = decoder_output[:, -1, :]
            

            # Get the logits for the last token logits only.
            last_token_logits = self.classification_head(last_token_output)
            
            # Get the most likely next token (greedy decoding).
            next_token = last_token_logits.argmax(dim=-1)
            
            # Append the newly predicted token to the target sequence.
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            # Check if any sequence has finished.
            just_finished = (next_token == end_symbol_id)
            finished = finished | just_finished
            
            if finished.all():
                break
           
        return tgt

        

def create_transformer_model(config_name: str, src_vocab_size: int, tgt_vocab_size: int, device="cpu", max_len=1024):
    """
    Creates a Seq2SeqTransformer model based on a configuration name.
    The parameter counts are approximate and depend on the exact vocabulary sizes.
    These configurations aim for models between 10M and 100M parameters.

    Args:
        config_name (str): Name of the configuration ('small', 'medium', 'large').
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        device (str): Device to move the model to ('cpu', 'cuda').

    Returns:
        Seq2SeqTransformer: The instantiated model.
    
    Raises:
        ValueError: If config_name is not one of 'small', 'medium', 'large'.
    """
    # Configurations designed to be between 10M and 100M parameters
    # (assuming vocab_size around 32k)
    configs = {
        "tiny": { # Approx. 10M params with 32k vocab
            "num_encoder_layers": 2, "num_decoder_layers": 2, 
            "emb_size": 256, "nhead": 4, "dim_feedforward": 512, "dropout": 0.1
        },
        "xsmall": { # Approx. 15M params with 32k vocab
            "num_encoder_layers": 4, "num_decoder_layers": 4, 
            "emb_size": 256, "nhead": 4, "dim_feedforward": 512, "dropout": 0.1
        },
        "small": { # Approx. 22M params with 32k vocab
            "num_encoder_layers": 4, "num_decoder_layers": 4, 
            "emb_size": 256, "nhead": 4, "dim_feedforward": 1024, "dropout": 0.1
        },
        "medium": { # Approx. 62M params with 32k vocab
            "num_encoder_layers": 6, "num_decoder_layers": 6, 
            "emb_size": 512, "nhead": 8, "dim_feedforward": 1024, "dropout": 0.1
        },
        "large": { # Approx. 77M params with 32k vocab
            "num_encoder_layers": 6, "num_decoder_layers": 6, 
            "emb_size": 512, "nhead": 8, "dim_feedforward": 2048, "dropout": 0.1
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown configuration name: {config_name}. Choose from {list(configs.keys())}")

    config = configs[config_name]
    model = Seq2SeqTransformer(
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        emb_size=config["emb_size"],
        nhead=config["nhead"],
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=max_len
    )
    
    model.to(device)
    
    # You can uncomment this to check parameter counts when creating a model:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{config_name}' (Seq2Seq: src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}) on {device} has {total_params/1e6:.2f}M parameters.")
    
    return model

def create_classifier_model(config_name: str, src_vocab_size: int, num_classes: int, device="cpu", max_len: int = 2048,
                            io_sep_token_id=None, item_sep_token_id=None, pooling_strategy='global'):
    """
    Creates a StoneStateClassifier model based on a configuration name.
    Uses similar encoder configurations as create_transformer_model.

    Args:
        config_name (str): Name of the configuration ('tiny', 'xsmall', 'small', 'medium', 'large').
        src_vocab_size (int): Source vocabulary size (for input features/potions).
        num_classes (int): Number of unique stone states to classify.
        device (str): Device to move the model to ('cpu', 'cuda').
        max_len (int): Maximum sequence length for positional encoding.

    Returns:
        StoneStateClassifier: The instantiated classification model.
    
    Raises:
        ValueError: If config_name is not one of the defined configurations.
    """
    configs = {
        "tiny": { 
            "num_encoder_layers": 2, "emb_size": 128, "nhead": 4, 
            "dim_feedforward": 256, "dropout": 0.1
        },
        "xsmall": { 
            "num_encoder_layers": 4, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 512, "dropout": 0.1
        },
        "xsmall_modified": {
            "num_encoder_layers": 4, "emb_size": 128, "nhead": 4, 
            "dim_feedforward": 256, "dropout": 0.1, 
        },
        "xsmall_deep": {
            "num_encoder_layers": 6, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 512, "dropout": 0.1
        },
        "small": { 
            "num_encoder_layers": 4, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 1024, "dropout": 0.1
        },
        "medium": { 
            "num_encoder_layers": 6, "emb_size": 512, "nhead": 8, 
            "dim_feedforward": 1024, "dropout": 0.1
        },
        "large": { 
            "num_encoder_layers": 6, "emb_size": 512, "nhead": 8, 
            "dim_feedforward": 2048, "dropout": 0.1
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown configuration name: {config_name}. Choose from {list(configs.keys())}")

    config = configs[config_name]
    model = StoneStateClassifier(
        num_encoder_layers=config["num_encoder_layers"],
        emb_size=config["emb_size"],
        nhead=config["nhead"],
        src_vocab_size=src_vocab_size,
        num_classes=num_classes,
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=max_len,
        pooling_strategy=pooling_strategy,
        item_sep_token_id=item_sep_token_id,
        io_sep_token_id=io_sep_token_id
    )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{config_name}' (Classifier: src_vocab={src_vocab_size}, num_classes={num_classes}) on {device} has {total_params/1e6:.2f}M parameters.")
    
    return model

def create_decoder_classifier_model(config_name: str, src_vocab_size: int, num_classes: int, device="cpu", max_len: int = 2048, 
                                    prediction_type=None, padding_side: str = "left", use_flash_attention: bool = False, batch_size: int = 32):
    """
    Creates a StoneStateDecoderClassifier model based on a configuration name.
    
    Args:
        config_name (str): Name of the configuration ('tiny', 'xsmall', 'small', 'medium', 'large').
        src_vocab_size (int): Source vocabulary size.
        num_classes (int): Number of classes.
        device (str): Device to move the model to ('cpu', 'cuda').
        max_len (int): Maximum sequence length for positional encoding.
        prediction_type (str): Type of prediction ('autoregressive' or 'feature').
        padding_side (str): Padding side ('left' or 'right'). Decoder models typically use 'left'.
    """
    configs = {
        "tiny": { 
            "num_decoder_layers": 2, "emb_size": 128, "nhead": 4, 
            "dim_feedforward": 256, "dropout": 0.1
        },
        "xsmall": {  # 2.14M params.
            "num_decoder_layers": 4, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 512, "dropout": 0.1
        },
        "xsmall_deep": {
            "num_decoder_layers": 6, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 512, "dropout": 0.1
        },
        "xsmall_wide": {
            "num_decoder_layers": 4, "emb_size": 512, "nhead": 4, 
            "dim_feedforward": 1024, "dropout": 0.1
        },
        "small": { 
            "num_decoder_layers": 5, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 512, "dropout": 0.1
        },
        "medium": { 
            "num_decoder_layers": 6, "emb_size": 256, "nhead": 4, 
            "dim_feedforward": 512, "dropout": 0.1
        },
        "large": { 
            "num_decoder_layers": 6, "emb_size": 512, "nhead": 8, 
            "dim_feedforward": 2048, "dropout": 0.1
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown configuration name: {config_name}. Choose from {list(configs.keys())}")

    config = configs[config_name]
    model = StoneStateDecoderClassifier(
        num_decoder_layers=config["num_decoder_layers"],
        emb_size=config["emb_size"],
        nhead=config["nhead"],
        src_vocab_size=src_vocab_size,
        num_classes=num_classes,
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=max_len,
        prediction_type=prediction_type,
        padding_side=padding_side,
        use_flash_attention=use_flash_attention,
        batch_size=batch_size
    )
    
    model.to(device)
    first_param = next(model.parameters())
    print(f"First 5 weights: {first_param.flatten()[:5]}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{config_name}' (Decoder-Only Classifier: src_vocab={src_vocab_size}, num_classes={num_classes}) on {device} has {total_params/1e6:.2f}M parameters.")
    
    return model

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    SRC_VOCAB_SIZE = 10000  # Example source vocabulary size
    TGT_VOCAB_SIZE = 12000  # Example target vocabulary size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # Create models of different sizes
    print("\n--- Creating Small Model ---")
    small_model = create_transformer_model("small", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DEVICE)
    
    print("\n--- Creating Medium Model ---")
    medium_model = create_transformer_model("medium", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DEVICE)
    
    print("\n--- Creating Large Model ---")
    large_model = create_transformer_model("large", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DEVICE)

    # Dummy input for a forward pass example with the large model
    BATCH_SIZE = 4
    SRC_SEQ_LEN = 30
    TGT_SEQ_LEN = 25 # Target sequence for decoder input is typically shifted

    # Create random token sequences (batch_size, seq_len)
    src_tokens = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN)).to(DEVICE) # Use 1 to avoid padding token 0
    tgt_tokens_input = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN)).to(DEVICE) # For decoder input

    # Generate masks
    # Source mask (optional, if you have specific source masking needs beyond padding)
    # src_att_mask = None 
    
    # Target mask (causal mask for decoder self-attention)
    # tgt_att_mask shape should be (TGT_SEQ_LEN, TGT_SEQ_LEN)
    tgt_att_mask = Seq2SeqTransformer.generate_square_subsequent_mask(TGT_SEQ_LEN, device=DEVICE)
    
    # Padding masks (True where padded, False otherwise)
    # nn.Transformer expects padding masks where True indicates a padded token that should be ignored.
    src_pad_mask = (src_tokens == 0) # Example: if 0 is the padding token
    tgt_pad_mask = (tgt_tokens_input == 0) # Example: if 0 is the padding token
    
    # memory_key_padding_mask is typically the same as src_padding_mask
    mem_key_pad_mask = src_pad_mask

    print(f"\n--- Running Forward Pass (Large Model) ---")
    print(f"src_tokens shape: {src_tokens.shape}")
    print(f"tgt_tokens_input shape: {tgt_tokens_input.shape}")
    print(f"tgt_att_mask shape: {tgt_att_mask.shape if tgt_att_mask is not None else 'None'}") # Stays (TGT_SEQ_LEN, TGT_SEQ_LEN)
    print(f"src_pad_mask shape: {src_pad_mask.shape if src_pad_mask is not None else 'None'}") # Stays (BATCH_SIZE, SRC_SEQ_LEN)
    print(f"tgt_pad_mask shape: {tgt_pad_mask.shape if tgt_pad_mask is not None else 'None'}") # Stays (BATCH_SIZE, TGT_SEQ_LEN)
    
    try:
        output = large_model(src=src_tokens, 
                             tgt=tgt_tokens_input, 
                             tgt_mask=tgt_att_mask, 
                             src_padding_mask=src_pad_mask, 
                             tgt_padding_mask=tgt_pad_mask,
                             memory_key_padding_mask=mem_key_pad_mask)
        print(f"Output shape: {output.shape}") # Expected: (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Example of using encode and decode separately
    print("\n--- Testing Encode/Decode Separately (Large Model) ---")
    try:
        memory = large_model.encode(src=src_tokens, src_padding_mask=src_pad_mask)
        print(f"Memory shape: {memory.shape}") # Expected: (BATCH_SIZE, SRC_SEQ_LEN, emb_size)
        
        decoded_output = large_model.decode(tgt=tgt_tokens_input, memory=memory, 
                                            tgt_mask=tgt_att_mask, 
                                            tgt_padding_mask=tgt_pad_mask,
                                            memory_key_padding_mask=mem_key_pad_mask)
        # The output of decode is raw features, not passed through the final generator layer yet.
        print(f"Decoder output shape (raw): {decoded_output.shape}") # Expected: (BATCH_SIZE, TGT_SEQ_LEN, emb_size)
        
        # To get final predictions, pass through the generator
        # Decoder output is (N, T, E), generator expects (N, T, E)
        final_predictions = large_model.generator(decoded_output) # No permute needed
        print(f"Final predictions after generator shape: {final_predictions.shape}") # Expected: (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)

    except Exception as e:
        print(f"Error during encode/decode test: {e}")

    print("\nModel setup and basic tests complete.")

    print("\n--- Creating Classifier Models ---")
    NUM_CLASSES = 50 # Example number of classes
    print("\n--- Creating Small Classifier Model ---")
    small_classifier = create_classifier_model("small", SRC_VOCAB_SIZE, NUM_CLASSES, DEVICE)

    print("\n--- Creating Medium Classifier Model ---")
    medium_classifier = create_classifier_model("medium", SRC_VOCAB_SIZE, NUM_CLASSES, DEVICE)

    # Dummy input for a forward pass example with the small classifier model
    # src_tokens from previous example: (BATCH_SIZE, SRC_SEQ_LEN)
    # src_pad_mask from previous example: (BATCH_SIZE, SRC_SEQ_LEN)
    print(f"\n--- Running Forward Pass (Small Classifier Model) ---")
    print(f"src_tokens shape: {src_tokens.shape}")
    print(f"src_pad_mask shape: {src_pad_mask.shape if src_pad_mask is not None else 'None'}")
    
    try:
        classifier_output = small_classifier(src=src_tokens, src_padding_mask=src_pad_mask)
        print(f"Classifier output shape: {classifier_output.shape}") # Expected: (BATCH_SIZE, NUM_CLASSES)
    except Exception as e:
        print(f"Error during classifier forward pass: {e}")

    print("\nClassifier model tests complete.")
