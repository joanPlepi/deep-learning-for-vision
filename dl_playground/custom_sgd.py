import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

T = 64 # total data
M = 10 # dimension of x
N = 10 # dimension of y

torch.manual_seed(1)
X = torch.randn(T, M)
orig_W = torch.randn(N, M)
orig_b = torch.randn(N)
Y = torch.mm(X, orig_W.transpose(0, 1)) + orig_b

class LinReg(nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(0) # different from actual W and b generation
        self.W = nn.Parameter(torch.randn(N, M))
        self.b = nn.Parameter(torch.randn(N))
        print(self.b)

    def forward(self, x):
        out = torch.mm(x, self.W.transpose(0, 1)) + self.b
        return out

model = LinReg()

criterion = torch.nn.MSELoss()

num_iters = 500
th = 1e-8
err = 1000000
iters = 0
lr = 1e-2

start = time.time()
while err >= th and iters < num_iters:

    # forward pass full batch
    out = model(X)

    loss = criterion(out, Y)
    err = abs(loss.item())
    if iters % 1 == 0:
        print("Iteration: {}, Loss: {:.8f}".format(iters, err))
    iters += 1

    # backward pass
    model.zero_grad()
    loss.backward()

    # update params
    for n, p in model.named_parameters():
        d_p = p.grad.data
        p.data.add_(-lr, d_p)

end = time.time()
print("Time: {:.8f}".format(end - start))