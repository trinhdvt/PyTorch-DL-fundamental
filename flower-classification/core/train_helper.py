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
        tb.add_scalar("Loss/Train", train_loss / total, epoch)
        tb.add_scalar("Accuracy/Train", train_correct / total, epoch)
        train_hist['train_loss'].append(train_loss / total)
        train_hist['train_acc'].append(train_correct / total)

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
            tb.add_scalar("Loss/Validation", val_loss / total, epoch)
            tb.add_scalar("Accuracy/Validation", val_correct / total, epoch)
            train_hist['val_loss'].append(val_loss / total)
            train_hist['val_acc'].append(val_correct / total)

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
