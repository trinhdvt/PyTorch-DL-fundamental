import torch

from tqdm.auto import tqdm
from utils.metric_utils import calculate_accuracy, MetricMonitor
from torch.utils.tensorboard import SummaryWriter


def traning_loops(epochs, model,
                  train_loader,
                  optimizer,
                  criterion,
                  val_loader=None,
                  device="cpu",
                  blocking=False):
    """
    Training loop for the model.

    :param epochs: Number of epochs to train the model.
    :param model: The model to train.
    :param train_loader: The training data.
    :param val_loader: The validation data.
    :param optimizer: The optimizer to use.
    :param criterion: The loss function to use.
    :param device: The device to use.
    :return: The trained model.
    """

    # Initialize the tensorboard.
    tb = SummaryWriter()
    train_hist = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    best_accuracy = 0

    # start training loop
    for epoch in range(epochs):
        # Training phase
        # Set the model to train mode.
        model.train()
        # Initialize the training statistics.
        metric_monitor = MetricMonitor()
        train_loss = 0.0
        train_correct = 0
        total = 0
        # Loop over the batches in the training set.
        train_stream = tqdm(train_loader)
        for data, target in train_stream:
            # Send the data and target to the device.
            data, target = data.to(device, non_blocking=blocking), target.to(
                device, non_blocking=blocking)

            # Forward pass - compute the outputs.
            outputs = model(data)
            # Compute the loss.
            loss = criterion(outputs, target)

            # Zero the gradients.
            optimizer.zero_grad()
            # Backward pass - compute the gradients.
            loss.backward()
            optimizer.step()

            # Update the parameters.
            train_loss += loss.item()
            total += target.size(0)
            train_correct += outputs.argmax(dim=1).eq(target).sum().item()
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update(
                "Accuracy", calculate_accuracy(outputs, target))
            train_stream.set_description(
                f"Epoch: {epoch+1}/{epochs}. Train.      {metric_monitor}")

        # update the tensorboard.
        train_loss /= total
        train_acc = train_correct / total

        tb.add_scalar("Loss/Train", train_loss, epoch)
        tb.add_scalar("Accuracy/Train", 100*train_acc, epoch)
        train_hist['train_loss'].append(train_loss)
        train_hist['train_acc'].append(train_acc)

        # Validation phase
        if val_loader:
            # Set the model to evaluation mode.
            model.eval()
            # Initialize the validation statistics.
            metric_monitor = MetricMonitor()
            val_loss = 0.0
            val_correct = 0
            total = 0
            # Loop over the batches in the validation set.
            val_stream = tqdm(val_loader)
            with torch.no_grad():
                for data, target in val_stream:
                    # Send the data and target to the device.
                    data, target = data.to(device, non_blocking=blocking), target.to(
                        device, non_blocking=blocking)

                    # Forward pass - compute the outputs.
                    outputs = model(data)
                    # Compute the loss.
                    loss = criterion(outputs, target)

                    # Update the validation statistics.
                    val_loss += loss.item()
                    total += target.size(0)
                    val_correct += outputs.argmax(
                        dim=1).eq(target).sum().item()
                    metric_monitor.update("Loss", loss.item())
                    metric_monitor.update(
                        "Accuracy", calculate_accuracy(outputs, target))
                    val_stream.set_description(
                        f"Epoch: {epoch+1}/{epochs}. Valid. {metric_monitor}")

            # update the tensorboard.
            val_loss /= total
            val_acc = val_correct / total

            tb.add_scalar("Loss/Validation", val_loss, epoch)
            tb.add_scalar("Accuracy/Validation", 100*val_acc, epoch)
            train_hist['val_loss'].append(val_loss)
            train_hist['val_acc'].append(val_acc)

            # save best model
            if val_loss > best_accuracy:
                best_accuracy = val_loss
                torch.save(model.state_dict(), f"best_model_{epoch}.pth")

    tb.close()
    return train_hist


def test_model(model, test_loader, device="cpu", non_blocking=False):
    """
    Test the model.

    :param model: The model to test.
    :param test_loader: The test data.
    :param device: The device to use.
    :return: The test accuracy.
    """
    # Set the model to evaluation mode.
    model.eval()
    # Initialize the test statistics.
    test_correct = 0
    total = 0
    # Loop over the batches in the test set.
    test_stream = tqdm(test_loader)
    with torch.no_grad():
        for data, target in test_stream:
            # Send the data and target to the device.
            data, target = data.to(device, non_blocking=non_blocking), target.to(
                device, non_blocking=non_blocking)

            # Forward pass - compute the outputs.
            outputs = model(data)

            # Update the test statistics.
            total += target.size(0)
            test_correct += outputs.argmax(dim=1).eq(target).sum().item()

        print(f"Accuracy: {100.0*test_correct / total:.3f}")
