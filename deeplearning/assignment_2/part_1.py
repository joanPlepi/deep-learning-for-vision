import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import helpers

input_dim = 3 * 32 * 32
output_dim = 10
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
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

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        a1 = nn.ReLU()(self.fc1(x))
        a2 = nn.ReLU()(self.fc2(a1))
        out = self.fc3(a2)
        return out


# training
model = MLP(input_dim, [300, 100], output_dim)

optim = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()
epochs = 2
helpers.train(trainloader, model, optim, criterion, device, epochs, plot_epoch_losses=True, plot_iter_losses=True)

# testing
train_acc = helpers.test(trainloader, model, device)
test_acc = helpers.test(testloader, model, device)
print("Trainset accuracy: {:.3f}%, testset: {:.3f}%".format(train_acc, test_acc))
