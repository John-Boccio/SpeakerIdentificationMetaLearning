import torch
from torch import nn
import warnings


class VoxCelebNetwork(nn.Module):
    def __init__(self, max_pool_return_indicies=False):
        super().__init__()

        self._max_pool_return_indicies = max_pool_return_indicies
        self._max_pool_indicies = []

        layers = []
        self.channel_sizes = [1, 16, 32, 64, 64, 64, 64]
        for i in range(len(self.channel_sizes)-1):
            in_ch = self.channel_sizes[i]
            out_ch = self.channel_sizes[i+1]
            layers += [
                nn.Conv2d(in_ch, out_ch, (3, 3), padding='same'),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, return_indices=max_pool_return_indicies),
            ]
        layers += [nn.Flatten()]
        self._layers = nn.Sequential(*layers)

        # Do a passthrough to get the feature sizes at each layer
        feature_sizes = [torch.Size((1, 256, 301))]
        x = torch.rand(feature_sizes[0]).reshape((1, 1, 256, 301))
        max_pool_idxs = []
        for layer in self._layers:
            if isinstance(layer, nn.MaxPool2d) and max_pool_return_indicies:
                x, max_pool_indicies = layer(x)
                max_pool_idxs += [max_pool_indicies]
            else:
                x = layer(x)
            feature_sizes += [x.size()[1:]]
        self._feature_sizes = feature_sizes

    def forward(self, images):
        if self._max_pool_return_indicies:
            self._max_pool_indicies = []
            x = images
            for layer in self._layers:
                if isinstance(layer, nn.MaxPool2d):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        x, max_pool_idicies = layer(x)
                    self._max_pool_indicies += [max_pool_idicies]
                else:
                    x = layer(x)
        else:
            x = self._layers(images)
        return x

