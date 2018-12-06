import torch
import matplotlib.pyplot as plt
import time


def train(trainloader, model, optimizer, criterion, device, epochs,
          l1_reg=0, plot_epoch_losses=False, plot_iter_losses=False):
    """Trains model on a given trainloader."""
    model.train()
    model.to(device)

    epoch_losses = []
    iter_losses = []

    for epoch in range(epochs):

        tic = time.time()
        loss_sum = 0.0

        for i, (images, labels) in enumerate(trainloader):

            # prepare inputs
            images, labels = images.to(device), labels.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if l1_reg != 0:
                l1_sum = 0.0
                for param in model.parameters():
                    l1_sum += torch.norm(param, 1)

                loss += l1_reg * l1_sum

            # backward
            loss.backward()
            optimizer.step()

            # record losses
            loss_sum += loss
            iter_losses.append(loss)

        epoch_loss = loss_sum / (i + 1)
        epoch_losses.append(epoch_loss)
        print('Epoch: [{}/{}], total iters: {}, loss: {:.5f}, time: {:.5f} sec.'.format(epoch + 1,
                                                                                        epochs,
                                                                                        (epoch + 1) * (i + 1),
                                                                                        epoch_loss,
                                                                                        time.time() - tic))
    if plot_epoch_losses:
        # plot losses
        plt.title("Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(epoch_losses)
        plt.show()

    if plot_iter_losses:
        # plot losses
        plt.title("Loss per Iteration")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.plot(iter_losses)
        plt.show()


def test(testloader, model, device):
    """Tests the model on a given testset and returns all true and predicted labels."""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probs, pred_labels = torch.max(outputs.data, 1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total
