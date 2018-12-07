import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from deeplearning.miscs import helpers

torch.manual_seed(1)

input_dim = 1 * 28 * 28
output_dim = 10
batch_size = 60
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

train_size = 50000
valid_size = 10000

trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# model definition
class MLP(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hiddens[0])
        self.fc2 = nn.Linear(hiddens[0], hiddens[1])
        self.fc3 = nn.Linear(hiddens[1], output_dim)

        self.dropout_x = torch.nn.Dropout(0.2)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.dropout_x(x)

        a1 = nn.ReLU()(self.fc1(x))
        a1 = self.dropout1(a1)

        a2 = nn.ReLU()(self.fc2(a1))
        a2 = self.dropout2(a2)

        out = self.fc3(a2)
        return out


# training
model = MLP(input_dim, [800, 800], output_dim)

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()
epochs = 10
tr_epoch_losses, tr_iter_losses, val_epoch_losses, val_epoch_accs = \
    helpers.train(trainloader, model, optimizer, criterion, device, epochs, validloader, l1_reg=0, l2_reg=0.5)

plt.title("Train Loss per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(tr_epoch_losses)
plt.show()

plt.title("Valid Loss per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(tr_epoch_losses)
plt.show()
