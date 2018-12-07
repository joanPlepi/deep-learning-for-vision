import torch

torch.manual_seed(1)

N, D_in, H, D_out = 10, 5, 5, 1
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

criterion = torch.nn.MSELoss()
lr = 1e-4
weight_decay = 0.00  # for torch.optim.SGD
lmbd = 0.01  # for custom L2 regularization

optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

losses = []

for t in range(100):
    y_pred = model(x)

    # Compute and print loss.
    loss = criterion(y_pred, y)
    # print(loss)

    losses.append(loss)

    optimizer.zero_grad()

    reg_loss = None
    for param in model.parameters():
        if reg_loss is None:
            reg_loss = 0.5 * param.norm(2)**2
        else:
            reg_loss = reg_loss + 0.5 * param.norm(2)**2

    loss = loss + lmbd * reg_loss

    loss.backward()

    optimizer.step()

for name, param in model.named_parameters():
    print(name, param)
