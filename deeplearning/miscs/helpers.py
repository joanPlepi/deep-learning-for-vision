import torch
import time


def train(trainloader, model, optimizer, criterion, device, epochs,
          validloader=None, l1_reg=0, l2_reg=0):
    """Trains model on a given trainloader."""
    model.train()
    model.to(device)

    train_epoch_losses = []
    valid_epoch_losses = []
    train_iter_losses = []

    valid_epoch_accs = []

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

            # record losses before regularization
            loss_sum += loss
            train_iter_losses.append(loss)

            # add regularization loss, initialize to None for autograd
            l1_reg_loss = None
            l2_reg_loss = None
            for param in model.parameters():

                # add L1 regularization
                if l1_reg_loss is None:
                    l1_reg_loss = torch.sum(torch.abs(param))
                else:
                    l1_reg_loss = l1_reg_loss + torch.sum(torch.abs(param))

                # add L2 regularization
                if l2_reg_loss is None:
                    l2_reg_loss = 0.5 * param.norm(2)**2
                else:
                    l2_reg_loss = l2_reg_loss + 0.5 * param.norm(2)**2

            loss += l1_reg * l1_reg_loss
            loss += l2_reg * l2_reg_loss

            # backward
            loss.backward()
            optimizer.step()

        train_loss = loss_sum / (i + 1)
        train_epoch_losses.append(train_loss)

        # record validation accuracy per epoch if necessary
        if validloader is not None:
            valid_acc, valid_loss = test(validloader, model, criterion, device)
            valid_epoch_losses.append(valid_loss)
            valid_epoch_accs.append(valid_acc)

        print('Epoch: [{}/{}], total iters: {}, loss: {:.5f}, time: {:.5f} sec.'.format(epoch + 1,
                                                                                        epochs,
                                                                                        (epoch + 1) * (i + 1),
                                                                                        train_loss,
                                                                                        time.time() - tic))

    return train_epoch_losses, train_iter_losses, valid_epoch_losses, valid_epoch_accs


def test(testloader, model, criterion, device):
    """Tests the model on a given testset and returns all true and predicted labels."""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probs, pred_labels = torch.max(outputs.data, 1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
            loss += criterion(outputs, labels)

    return 100 * correct / total, loss
