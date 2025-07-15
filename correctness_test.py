import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diff_kv_cache import Denoise
from diff_kv_cache_ref import Denoise as DenoiseRef

BS = 2
DIM = 128
N_HEADS = 2
LAYERS = 4
CHUNK_LEN = 128
CHUNKS = 4
DENOISE_STEPS = 4
torch.manual_seed(95)


torch.set_default_device("cuda")
model = Denoise(dim=DIM, n_heads=N_HEADS, layers=LAYERS).cuda()
model_ref = DenoiseRef(dim=DIM, n_heads=N_HEADS, layers=LAYERS).cuda()
model_ref.load_state_dict(model.state_dict())

# Generate a random data
x_list = [torch.randn(BS, CHUNK_LEN, DIM) for _ in range(CHUNKS)]
gt_list = [torch.randn(BS, CHUNK_LEN, DIM) for _ in range(CHUNKS)]


# Without KV Cache
x_history = None
loss = 0.
for i in range(CHUNKS):
    x = x_list[i].requires_grad_(True)
    gt = gt_list[i]

    for denoise_step in range(DENOISE_STEPS):
        x_all = torch.cat([x_history, x], dim=1) if x_history is not None else x
        # x_all_delta = model_ref(x_all)
        x_all_delta = checkpoint(model_ref, x_all)

        # Concatenate the input of the last denoising step.
        if denoise_step == DENOISE_STEPS - 1:
            if x_history is None:
                x_history = x
            else:
                x_history = torch.cat([x_history, x], dim=1)

        x = x + x_all_delta[:, -CHUNK_LEN:]

        print("x chunk", i, "denoise_step", denoise_step, x.mean(), x.std())


    loss += F.mse_loss(x, gt)
print("loss", loss.item())
loss.backward()

name2grad_ref = {name: param.grad.clone().detach() for name, param in model_ref.named_parameters()}

for name, grad in name2grad_ref.items():
    print(name, grad.sum())

model_ref.zero_grad(set_to_none=True)

print()

# With KV Cache.

k_active_caches, v_active_caches = model.cache_init(batch_size=BS)
loss = 0.
for i in range(CHUNKS):
    x = x_list[i].requires_grad_(True)
    gt = gt_list[i]
    for denoise_step in range(DENOISE_STEPS):

        def flatten_kv_cache_model_call(x, *args):
            k_active_caches = args[:LAYERS]
            v_active_caches = args[LAYERS:]
            out, k, v =  model(x, k_active_caches, v_active_caches)
            return out, *k, *v

        # x_delta, new_k_active_caches, new_v_active_caches 
        output = checkpoint(flatten_kv_cache_model_call, x, *k_active_caches, *v_active_caches)
        x_delta = output[0]
        new_k_active_caches = output[1:1 + LAYERS]
        new_v_active_caches = output[1 + LAYERS:]

        # x_delta, new_k_active_caches, new_v_active_caches = model(x, k_active_caches, v_active_caches)

        x = x + x_delta
        if denoise_step == DENOISE_STEPS - 1:
            k_active_caches = new_k_active_caches
            v_active_caches = new_v_active_caches
        print("x chunk", i, "denoise_step", denoise_step, x.mean(), x.std())
    loss += F.mse_loss(x, gt)
print("loss", loss.item())
loss.backward()

name2grad = {name: param.grad.clone().detach() for name, param in model.named_parameters()}

for name, grad in name2grad.items():
    print(name, grad.sum())

model.zero_grad(set_to_none=True)

print()
print("Gradient comparison:")
for name in name2grad_ref:
    if name in name2grad:
        if torch.allclose(name2grad_ref[name], name2grad[name], rtol=1e-3, atol=1e-6):
            print(f"{name}: the same gradient, pass")
        else:
            print(f"{name}: different gradient, FAIL")
            print(f"  ref: {name2grad_ref[name].sum()}")
            print(f"  kv:  {name2grad[name].sum()}")
            print(f"  diff: {(name2grad_ref[name] - name2grad[name]).abs().max()}")
    else:
        print(f"{name}: missing in kv cache model, FAIL")




    
