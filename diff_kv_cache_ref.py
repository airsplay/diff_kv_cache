import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class BlockCausalAttention(nn.Module):
    def __init__(self, dim, n_heads, chunk_size=128):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.chunk_size = chunk_size
    
    def forward(self, x):
        q, k, v = rearrange(self.qkv(x), "b l (qkv h d) -> qkv b h l d", qkv=3, h=self.n_heads)

        print("k", k.mean(), k.std())
        print("v", v.mean(), v.std())

        x_list = []
        for i in range(0, k.size(2), self.chunk_size):
            q_chunk = q[:, :, i:i+self.chunk_size]
            k_chunk = k[:, :, :i+self.chunk_size]
            v_chunk = v[:, :, :i+self.chunk_size]
            x_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk)
            x_list.append(x_chunk)
        x = torch.cat(x_list, dim=2)

        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.out(x)
        return x


class DiffusionLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, chunk_size=128):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = BlockCausalAttention(dim, n_heads, chunk_size)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        x = self.ln_attn(x)
        x = self.attn(x)

        x = self.ln_mlp(x)
        x = x + self.mlp(x)
        return x


class Denoise(nn.Module):
    def __init__(self, dim, n_heads, layers, chunk_size=128):
        super().__init__()
        self.layers = nn.ModuleList([DiffusionLayer(dim, n_heads, hidden_dim=dim*4, chunk_size=chunk_size) for _ in range(layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
