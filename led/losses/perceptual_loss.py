import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from led.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class VGGPerceptualLoss(nn.Module):
    # VGG-based perceptual loss
    def __init__(self, layer_weights={'conv1_2': 0.1, 'conv2_2': 0.1,
                                      'conv3_4': 0.1, 'conv4_4': 0.1, 'conv5_4': 0.1},
                 vgg_type='vgg19', loss_weight=1.0, normalize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.layer_weights = layer_weights
        self.loss_weight = loss_weight
        self.normalize = normalize

        # load pretrained VGG model
        if vgg_type == 'vgg19':
            vgg = models.vgg19(pretrained=True).features
        elif vgg_type == 'vgg16':
            vgg = models.vgg16(pretrained=True).features
        else:
            raise ValueError(f'Unsupported VGG type: {vgg_type}')

        self.vgg_layers = {
            'conv1_2': 4,
            'conv2_2': 9,
            'conv3_4': 18,
            'conv4_4': 27,
            'conv5_4': 36
        }

        # create VGG feature extractor
        self.vgg_extractors = nn.ModuleDict()
        for layer_name in self.layer_weights:
            if layer_name not in self.vgg_layers:
                raise ValueError(f'Unsupported layer name: {layer_name}')
            self.vgg_extractors[layer_name] = nn.Sequential(
                *list(vgg.children())[:(self.vgg_layers[layer_name] + 1)])

        # freeze VGG parameters
        for extractor in self.vgg_extractors.values():
            for param in extractor.parameters():
                param.requires_grad = False

        # register mean and std
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x):
        # preprocess input image
        if x.shape[1] == 4:
            # convert to RGB format
            r = x[:, 0:1]
            g = (x[:, 1:2] + x[:, 2:3]) / 2
            b = x[:, 3:4]
            x = torch.cat([r, g, b], dim=1)

        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    def forward(self, pred, target):
        # preprocess input images
        pred = self._preprocess(pred)
        target = self._preprocess(target)

        # extract features
        pred_features = {layer_name: self.vgg_extractors[layer_name](pred)
                         for layer_name in self.layer_weights}
        target_features = {layer_name: self.vgg_extractors[layer_name](target)
                           for layer_name in self.layer_weights}

        # compute perceptual loss
        loss = 0
        for layer_name, weight in self.layer_weights.items():
            loss += weight * F.mse_loss(pred_features[layer_name], target_features[layer_name])
        return loss * self.loss_weight