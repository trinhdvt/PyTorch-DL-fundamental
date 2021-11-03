from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def load_data(train_dir, test_dir, val_dir=None, batch_size=32, pin_memory=False, num_workers=0):
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    if val_dir:
        val_data = datasets.ImageFolder(val_dir, transform=test_transforms)
        val_loader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        return train_data.classes, train_loader, test_loader, val_loader

    return train_data.classes, train_loader, test_loader
