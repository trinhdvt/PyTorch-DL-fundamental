import matplotlib.pyplot as plt

from ..core.data_loader import inv_normalize


def display(images, truth_labels, class_names,
            predicted=None, size=10, figsize=(5, 8),
            save_path=None):
    images = images.cpu()
    truth_labels = truth_labels.cpu()

    #
    images = inv_normalize(images)
    fig, axes = plt.subplots(
        ncols=5, nrows=size // 5,
        figsize=figsize,
        subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().transpose((1, 2, 0)))

        #
        truth_label = class_names[truth_labels[i].item()]
        color = "blue"
        predicted_label = None
        if predicted is not None:
            predicted_label = class_names[predicted[i].item()]
            color = "red" if truth_label != predicted_label else "blue"

        ax.set_title(
            predicted_label if predicted_label is not None else truth_label, color=color)
    if save_path:
        plt.savefig(save_path)

    # 
    plt.tight_layout()
    plt.show()


def imshow(tensor_img):
    img = inv_normalize(tensor_img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()
