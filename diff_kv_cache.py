import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

class CacheAppend(torch.autograd.Function):
    """
    Assume storage is of shape (b, heads, L, d)
    """
    @staticmethod
    def forward(ctx, storage, active_cache, x, start, end):
        bs = active_cache.size(0)
        storage.data[:bs, :, start: end] = x
        ctx.save_for_backward(storage)
        ctx.bs = bs
        ctx.start = start
        ctx.end = end
        return storage[:bs, :, :end]
    
    @staticmethod
    def backward(ctx, grad_output):
        bs = ctx.bs
        start = ctx.start
        end = ctx.end
        return None, grad_output[:bs, :, :start], grad_output[:bs, :, start:end], None, None

cache_append = CacheAppend.apply


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

        self.k_storage = torch.empty(0, n_heads, 0, dim // n_heads)
        self.v_storage = torch.empty(0, n_heads, 0, dim // n_heads)
    
    def forward(self, x, k_active_cache, v_active_cache):
        q, k, v = rearrange(self.qkv(x), "b l (qkv h d) -> qkv b h l d", qkv=3, h=self.n_heads)

        chunk_start = k_active_cache.size(2)
        chunk_end = chunk_start + k.size(2)
        k = cache_append(self.k_storage, k_active_cache, k, chunk_start, chunk_end)
        v = cache_append(self.v_storage, v_active_cache, v, chunk_start, chunk_end)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")

        x = self.out(x)
        return x, k, v


class DiffusionLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x, k_active_cache, v_active_cache):
        x = self.ln_attn(x)
        x, k_active_cache, v_active_cache = self.attn(x, k_active_cache, v_active_cache)

        x = self.ln_mlp(x)
        x = x + self.mlp(x)
        return x, k_active_cache, v_active_cache


class Denoise(nn.Module):
    def __init__(self, dim, n_heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([DiffusionLayer(dim, n_heads, hidden_dim=dim*4) for _ in range(layers)])
    
    def forward(self, x, k_active_caches, v_active_caches):
        new_k_active_caches = []
        new_v_active_caches = []

        for layer, k_active_cache, v_active_cache in zip(self.layers, k_active_caches, v_active_caches):
            # The above KV caching only worked with this checkpoint used!
            x, k_active_cache, v_active_cache = checkpoint(layer, x, k_active_cache, v_active_cache)
            new_k_active_caches.append(k_active_cache)
            new_v_active_caches.append(v_active_cache)

        return x, new_k_active_caches, new_v_active_caches
    
    def cache_init(self, batch_size, max_len=100000):
        k_active_caches = []
        v_active_caches = []

        for layer in self.layers:
            attn = layer.attn

            # If the current batch size / max_len is different from the previous batch size / max_len,
            b, h, l, d = attn.k_storage.shape
            if b <= batch_size or l <= max_len:
                attn.k_storage = attn.k_storage.new_empty(batch_size, h, max_len, d)
                attn.v_storage = attn.v_storage.new_empty(batch_size, h, max_len, d)
            
            k_active_caches.append(attn.k_storage.new_empty(batch_size, h, 0, d))
            v_active_caches.append(attn.v_storage.new_empty(batch_size, h, 0, d))

        return k_active_caches, v_active_caches


if __name__ == "__main__":
    BS = 1
    STEP = 1000
    DIM = 1536
    N_HEADS = 12
    LAYERS = 30
    CHUNK_LEN = 1560
    CHUNKS = 21
    DENOISE_STEPS = 5

    torch.set_default_device("cuda")
    model = Denoise(dim=DIM, n_heads=N_HEADS, layers=LAYERS).cuda()

    x_list = []
    for step in range(STEP):
        print("Step", step)

        # Generate a random data
        x_list = [torch.randn(BS, CHUNK_LEN, DIM) for _ in range(CHUNKS)]
        gt_list = [torch.randn(BS, CHUNK_LEN, DIM) for _ in range(CHUNKS)]

        # Init the cache.
        k_active_caches, v_active_caches = model.cache_init(batch_size=BS, max_len=CHUNK_LEN * CHUNKS)
        loss = 0.
        for i in range(CHUNKS):
            x = x_list[i].clone()
            x = x.requires_grad_(True)      # requires_grad_ on x is to support model-wise checkpoint.
            gt = gt_list[i]
            for denoise_step in range(DENOISE_STEPS):

                # Option 1: checkpoint for each model call as well.
                #   since checkpoint only store / backward grad for flatten functions' input output. 
                #   It needs a wrapper to work.
                def flatten_kv_cache_model_call(x, *args):
                    k_active_caches = args[:LAYERS]
                    v_active_caches = args[LAYERS:]
                    out, k, v =  model(x, k_active_caches, v_active_caches)
                    return out, *k, *v

                output = checkpoint(flatten_kv_cache_model_call, x, *k_active_caches, *v_active_caches)
                x_delta = output[0]
                new_k_active_caches = output[1:1 + LAYERS]
                new_v_active_caches = output[1 + LAYERS:]

                # Option 2: checkpoint for layer-wise only; just call the model:
                # x_delta, new_k_active_caches, new_v_active_caches = model(x, k_active_caches, v_active_caches)

                x = x + x_delta
                if denoise_step == DENOISE_STEPS - 1:
                    k_active_caches = new_k_active_caches
                    v_active_caches = new_v_active_caches
            loss += F.mse_loss(x, gt)

        loss.backward()

        model.zero_grad(set_to_none=True)
        

            