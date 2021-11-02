import torch

from tqdm.auto import tqdm
from utils.metric_utils import calculate_accuracy, MetricMonitor
from utils.log_utils import write_log

from torch.utils.tensorboard import SummaryWriter


def traning_loops(epochs, model,
                  train_loader,
                  val_loader,
                  optimizer,
                  criterion,
                  scheduler=None,
                  device="cpu",
                  non_blocking=False):
    """
    Training loop for the model.

    :param epochs: Number of epochs to train the model.
    :param model: The model to train.
    :param train_loader: The training data.
    :param val_loader: The validation data.
    :param optimizer: The optimizer to use.
    :param criterion: The loss function to use.
    :param scheduler: The scheduler to use.
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
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    best_accuracy = 0

    # start training loop
    for n_epoch in range(epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Initialize the training statistics.
            metric_monitor = MetricMonitor()
            running_loss = 0.0
            running_correct = 0
            total = 0
            # Loop over the batches in the training set.
            stream = tqdm(dataloaders[phase])
            for data, target in stream:
                # Send the data and target to the device.
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)

                # Zero the gradients.
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass - compute the outputs.
                    outputs = model(data)
                    # Compute the loss.
                    loss = criterion(outputs, target)

                    # Backward pass - compute the gradients.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update the parameters.
                running_loss += loss.item()
                total += target.size(0)
                running_correct += outputs.argmax(
                    dim=1).eq(target).sum().item()

                metric_monitor.update("Loss", loss.item())
                metric_monitor.update(
                    "Accuracy", calculate_accuracy(outputs, target))
                stream.set_description(
                    f"Epoch: {n_epoch+1}/{epochs}. {phase.capitalize()}.      {metric_monitor}")

            # Update the learning rate.
            if phase == "train" and scheduler is not None:
                scheduler.step()

            # update log.
            running_loss /= total
            accuracy = running_correct / total

            cpt_phase = phase.capitalize()
            write_log(tb, {f"Loss/{cpt_phase}": running_loss}, n_epoch)
            write_log(tb, {f"Accuracy/{cpt_phase}": 100*accuracy}, n_epoch)
            train_hist[f'{phase}_loss'].append(running_loss)
            train_hist[f'{phase}_acc'].append(accuracy)

            # update best accuracy.
            if phase == 'val' and accuracy > best_accuracy:
                best_accuracy = accuracy
                model_name = model.__class__.__name__
                torch.save(model.state_dict(), f"{model_name}_best_model.pth")

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

        print(f"Accuracy: {100.0*test_correct / total:.3f} %")
