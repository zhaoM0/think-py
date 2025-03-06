import torch 
import torch.nn as nn 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # x shape is (batch_size, seq_len, emb_dim) 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / (var + self.eps)
        return norm_x * self.scale + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emd_dim, 4 * emd_dim),
            GELU(),
            nn.Linear(4 * emd_dim, emd_dim)
        )

    def forward(self, x):
        return self.layers(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, seq_len, d_in = x.shape 

        keys = self.w_key(x)      # Shape: (batch, seq_len, d_out)
        values = self.w_value(x)
        queries = self.w_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(batch, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Compute scaled dot-product attenion (aka self-attention) with a causal mask
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (batch, seq_len, num_heads, head_dim)
        context_vec = torch.matmul(attn_weights, values)
        context_vec = context_vec.transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch, seq_len, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.mlp = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut 

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds  # Shape: (batch, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
def define_gpt2_model(name="gpt2-small (124M)"):
    base_config = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }
    model_config = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    base_config.update(model_config[name])
    base_config.update({"context_length": 1024, "qkv_bias": True})
    gpt = GPT2Model(base_config)
    return gpt


# torch.manual_seed(123)
    
# GPT_CONFIG_124M = {
#     "vocab_size": 50257,    # Vocabulary size
#     "context_length": 1024, # Context length
#     "emb_dim": 768,         # Embedding dimension
#     "n_heads": 12,          # Number of attention heads
#     "n_layers": 12,         # Number of layers
#     "drop_rate": 0.1,       # Dropout rate
#     "qkv_bias": False       # Query-Key-Value bias
# }

# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")

# batch = []

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

# gpt2_small = GPT2Model(GPT_CONFIG_124M)
# output = gpt2_small(batch)

# print("Input Shape:", batch.shape)
# print("Output Shape:", output.shape)


