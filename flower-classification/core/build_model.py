from torch import nn
import random


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=3, input_size=224):
        super(CNN, self).__init__()
        conv_params = [
            [in_channels, 16, 3],
            [16, 32, 5],
            [32, 64, 5],
            [64, 128, 3],
            [128, 256, 3]
        ]
        pooled_size = input_size // 2**len(conv_params)
        dropout = (0, 0.2, 0.3, 0.5)
        classifier_params = [
            [256*pooled_size*pooled_size, 512],
            [512, 128],
            [128, 64],
            [64, num_classes]
        ]

        self.classifier = self.make_classifier(
            classifier_params, dropout=dropout)
        self.conv = self.make_backbone(conv_params)

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_backbone(conv_params, max_pool=2):
        layers = []
        for layer in conv_params:
            in_channels, out_channels, kernel_size = layer
            layers.append(ConvBlock(in_channels, out_channels, kernel_size))
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=max_pool))

        return nn.Sequential(*layers)

    @staticmethod
    def make_classifier(classifier_params, dropout=(0, 0.2, 0.3, 0.5)):
        layers = [nn.Flatten()]
        #
        for params in classifier_params:
            in_size, out_size = params
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(random.choice(dropout)))

        layers.append(nn.LogSoftmax(dim=1))

        #
        return nn.Sequential(*layers)
