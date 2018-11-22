import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import helpers


input_dim = 1 * 92 * 112
output_dim = 20
batch_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
data = np.load('ORL_faces.npz')

train_X = torch.Tensor(data['trainX'])
train_y = torch.Tensor(data['trainY']).long()

test_X = torch.Tensor(data['testX'])
test_y = torch.Tensor(data['testY']).long()

# make trainset and testset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

trainset = TensorDataset(train_X, train_y)
trainloader = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2)

testset = TensorDataset(test_X, test_y)
testloader = DataLoader(testset, shuffle=False)


# model definition
class MLP(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hiddens[0])
        self.fc2 = nn.Linear(hiddens[0], hiddens[1])
        self.fc3 = nn.Linear(hiddens[1], output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        a1 = nn.ReLU()(self.fc1(x))
        a2 = nn.ReLU()(self.fc2(a1))
        out = self.fc3(a2)
        return out


# training
model = MLP(input_dim, [100, 100], output_dim)

optim = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
epochs = 50
helpers.train(trainloader, model, optim, criterion, device, epochs, plot_epoch_losses=True, plot_iter_losses=True)

# testing
train_acc = helpers.test(trainloader, model, device)
test_acc = helpers.test(testloader, model, device)
print("Trainset accuracy: {:.3f}%, testset: {:.3f}%".format(train_acc, test_acc))
