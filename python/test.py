import time
import carl
import torch

sims = 4096
env = carl.Env(n_sim=sims, n_blue=3, n_orange=3, seed=123)

ticks = 10000

act = torch.zeros((sims, env.n_cars, 7), dtype=torch.int32, device="cuda")

env.step(act)
torch.cuda.synchronize()

start = time.time()

for i in range(ticks):
    env.step(act)

torch.cuda.synchronize()
dur = time.time() - start

print(f"{ticks * sims / dur:,.0f} ticks/s")
