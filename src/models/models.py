import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
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
                 dim_feedforward: int = 512, dropout: float = 0.1):
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
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
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

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device="cpu"):
        """Generates a square mask for preventing attention to future tokens."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_transformer_model(config_name: str, src_vocab_size: int, tgt_vocab_size: int, device="cpu"):
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
        "xsmall": { 
            "num_encoder_layers": 4, "num_decoder_layers": 4, "emb_size": 256, 
            "nhead": 4, "dim_feedforward": 512, "dropout": 0.1
        },
        "small": { # Approx. 22M params with 32k vocab
            "num_encoder_layers": 3, "num_decoder_layers": 3, "emb_size": 256, 
            "nhead": 4, "dim_feedforward": 1024, "dropout": 0.1
        },
        "medium": { # Approx. 62M params with 32k vocab
            "num_encoder_layers": 4, "num_decoder_layers": 4, "emb_size": 512, 
            "nhead": 8, "dim_feedforward": 2048, "dropout": 0.1
        },
        "large": { # Approx. 77M params with 32k vocab
            "num_encoder_layers": 6, "num_decoder_layers": 6, "emb_size": 512, 
            "nhead": 8, "dim_feedforward": 2048, "dropout": 0.1
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
        dropout=config["dropout"]
    )
    
    model.to(device)
    
    # You can uncomment this to check parameter counts when creating a model:
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model '{config_name}' (src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}) on {device} has {total_params/1e6:.2f}M parameters.")
    
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
