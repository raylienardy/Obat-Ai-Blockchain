import torch
import time

device = "cuda"

a = torch.randn(10000,10000,device=device)
b = torch.randn(10000,10000,device=device)

torch.cuda.synchronize()

start = time.time()

c = torch.matmul(a,b)

torch.cuda.synchronize()

print(time.time()-start)