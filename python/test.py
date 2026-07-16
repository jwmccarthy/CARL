import time
import carl
import torch

sims = 4096
env = carl.Env(n_sim=sims, n_blue=3, n_orange=3, seed=123)

ticks = 10000

act = torch.rand((sims, env.act_dim), device="cuda")
act_dl = torch.utils.dlpack.to_dlpack(act)

env.step(act_dl)
torch.cuda.synchronize()

start = time.time()

for i in range(ticks):
    env.step(act_dl)

torch.cuda.synchronize()
dur = time.time() - start

print(f"{ticks * sims / dur:,.0f} ticks/s")