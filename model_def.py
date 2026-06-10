import torch
import torch.nn as nn

# Configuration matching the trained GPT2-small
BASE_CONFIG = {
    "vocab_size": 50257,
    "block_size": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "dropout": 0.0
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout, qkv_bias=True):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('mask', torch.triu(torch.ones(BASE_CONFIG["block_size"], BASE_CONFIG["block_size"]), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) / (self.head_dim ** 0.5)
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, -1)
        
        context_vec = self.out_proj(context_vec)
        return context_vec

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"]),
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["n_embd"], 
            d_out=cfg["n_embd"], 
            num_heads=cfg["n_head"], 
            dropout=cfg["dropout"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["n_embd"])
        self.norm2 = LayerNorm(cfg["n_embd"])

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        
        self.trf_blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layer"])])
        
        self.final_norm = LayerNorm(cfg["n_embd"])
        self.out_head = nn.Linear(cfg["n_embd"], 2)  # num_classes = 2

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        tok_embeddings = self.tok_emb(idx)
        pos_embeddings = self.pos_emb(torch.arange(T, device=device))
        x = tok_embeddings + pos_embeddings
        
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
