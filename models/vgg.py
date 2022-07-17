import dill
import torch
import torch.nn as nn
from robustness.tools import helpers
from typing import List, Union, Dict, Any, cast


# VGG19 comparator
class vgg19Comp( nn.Module):
    def __init__( self, mean, std, num_classes= 1000):
        super().__init__()
        self.normalize= helpers.InputNormalize( mean, std)
        self.classifier= vgg19()

    def extract_layers( self, x, feat_dict, feat_level, input_normalize= True):
        if input_normalize: x= self.normalize( x)
        for i in range( 0, 2):
            x= self.classifier.features[ i]( x)
        if "conv1" in feat_level: feat_dict[ "conv1"]= x
        for i in range( 2, 7):
            x= self.classifier.features[ i]( x)
        if "conv2" in feat_level: feat_dict[ "conv2"]= x
        for i in range( 7, 12):
            x= self.classifier.features[ i]( x)
        if "conv3" in feat_level: feat_dict[ "conv3"]= x       
        for i in range( 12, 21):
            x= self.classifier.features[ i]( x)
        if "conv4" in feat_level: feat_dict[ "conv4"]= x
        for i in range( 21, 30):
            x= self.classifier.features[ i]( x)
        if "conv5" in feat_level: feat_dict[ "conv5"]= x
        return

    def forward( self, x, apply_normalization= True):
        if apply_normalization: x= self.normalize( x)
        # No need to output feature. x will always be our feature of interest.
        x= self.classifier( x)
        return x


# VGG19 comparator configuration
def vgg19Comp_config( comparator, load_comparator):
    if load_comparator== None:
        raise ValueError( "No classifier weights specified.")
    checkpoint= torch.load( load_comparator, pickle_module= dill)
    comparator.load_state_dict( checkpoint, strict= True)
    for p in comparator.parameters(): p.requires_grad = False
    return


# (Core) VGG
# Ref: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)

