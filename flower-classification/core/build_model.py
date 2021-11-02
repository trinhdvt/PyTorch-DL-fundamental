from torch import nn
from torchvision import models
import random
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, batch_norm=True,
                 stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=padding)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        # apply batch norm
        if self.bn is not None:
            x = self.bn(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=3, input_size=224):
        super(CNN, self).__init__()
        conv_params = [
            # (in_channels, out_channels, kernel_size, batch_norm)
            [in_channels, 16, 3, False],
            [16, 32, 5, False],
            [32, 64, 5, True],
            [64, 128, 3, True],
            [128, 256, 3, True]
        ]
        pooled_size = input_size // 2**len(conv_params)
        dropout = (0, 0.2, 0.3)
        classifier_params = [
            [256*pooled_size*pooled_size, 512],
            [512, 128],
            [128, 64],
            [64, num_classes]
        ]

        #
        self.conv = self.make_backbone(conv_params)

        #
        self.classifier = self.make_classifier(
            classifier_params, dropout=dropout)

    def forward(self, x):
        # feed forward convolution's layers
        x = self.conv(x)

        # feed forward classifier's layers
        x = self.classifier(x)
        return x

    @staticmethod
    def make_backbone(conv_params, max_pool=2):
        layers = []
        #
        for params in conv_params:
            in_channels, out_channels, kernel_size, use_batch_norm = params
            layers.append(ConvBlock(in_channels, out_channels,
                          kernel_size, use_batch_norm))
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=max_pool))

        return nn.Sequential(*layers)

    @staticmethod
    def make_classifier(classifier_params, dropout=(0, 0.2, 0.3)):
        layers = [nn.Flatten()]
        #
        for params in classifier_params[:-1]:
            in_size, out_size = params
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(random.choice(dropout)))

        # last layer with no Dropout
        in_size, out_size = classifier_params[-1]
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.LogSoftmax(dim=1))

        #
        return nn.Sequential(*layers)


class AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, input_size=224):
        super(AlexNet, self).__init__()
        self.alex_net = models.alexnet(pretrained=True)

        classifier_params = [
            [9216, 4096],
            [4096, 1024],
            [1024, 512],
            [512, 64],
            [64, num_classes]
        ]

        self.alex_net.classifier = self.make_classifier(classifier_params)

    def forward(self, x):
        x = self.alex_net(x)
        return x

    @staticmethod
    def make_classifier(classifier_params, dropout=(0.2, 0.3, 0.5)):
        layers = [nn.Flatten()]
        #
        for params in classifier_params[:-1]:
            in_size, out_size = params
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(random.choice(dropout)))

        # last layer with no Dropout
        in_size, out_size = classifier_params[-1]
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.LogSoftmax(dim=1))

        #
        return nn.Sequential(*layers)


if __name__ == '__main__':
    cnn = AlexNet(num_classes=5)
    cnn.to("cpu")
    print(summary(cnn, input_size=(3, 224, 224), device='cpu'))
