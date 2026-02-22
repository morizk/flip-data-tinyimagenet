"""
Extended model architectures for TinyImageNet (64×64, 200 classes) with flip feature support.
Includes: ResNet-18/34, VGG-11/16, EfficientNetV2-S, ViT

Early fusion uses FiLM (Feature-wise Linear Modulation) conditioning:
  The flip value generates per-channel scale (γ) and shift (β) that modulate
  feature maps at each stage. Initialized to identity (γ=1, β=0) so the model
  starts as if no flip conditioning exists. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


# ============================================================================
# FiLM (Feature-wise Linear Modulation) Conditioning
# ============================================================================

class FiLMGenerator(nn.Module):
    """Generates per-channel scale (γ) and shift (β) from a scalar flip value.
    
    FiLM (Perez et al., 2018) conditions feature maps on an auxiliary signal.
    Initialized to identity (γ=1, β=0) so the model starts as if no conditioning exists.
    """
    def __init__(self, num_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels * 2),  # γ and β concatenated
        )
        # Initialize to identity: γ=1, β=0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:num_channels], 1.0)   # γ = 1
        nn.init.zeros_(self.net[-1].bias[num_channels:])            # β = 0

    def forward(self, flip_value):
        """
        Args:
            flip_value: (batch,) scalar flip values
        Returns:
            gamma: (batch, num_channels) per-channel scale
            beta: (batch, num_channels) per-channel shift
        """
        params = self.net(flip_value.unsqueeze(-1).float())  # (batch, 2*C)
        gamma, beta = params.chunk(2, dim=-1)  # each (batch, C)
        return gamma, beta


def film_modulate(features, gamma, beta):
    """Apply FiLM modulation: output = γ * features + β
    
    Args:
        features: (batch, C, H, W) feature maps
        gamma: (batch, C) per-channel scale
        beta: (batch, C) per-channel shift
    Returns:
        Modulated features: (batch, C, H, W)
    """
    return gamma.unsqueeze(-1).unsqueeze(-1) * features + beta.unsqueeze(-1).unsqueeze(-1)


# ============================================================================
# ResNet-18/34 Implementations
# ============================================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet base class"""
    
    def __init__(self, block, layers, num_classes=200, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial conv layer (adaptable for early fusion)
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                          self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl(x)


# ResNet-18 Baseline
class ResNet18_Baseline(ResNet):
    """ResNet-18 baseline for TinyImageNet (200 classes, 64×64 input)"""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet18_Baseline, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=3
        )
    
    def forward(self, x):
        features = self._forward_impl(x)
        return self.fc(features)


# ResNet-18 Early Fusion (FiLM conditioning)
class ResNet18_FlipEarly(ResNet):
    """ResNet-18 with early fusion using FiLM conditioning.
    
    The flip value modulates feature maps at each residual stage via learned
    per-channel scale (γ) and shift (β). Backbone stays standard 3-channel.
    """
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet18_FlipEarly, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=3
        )
        # FiLM generators for each residual stage (64, 128, 256, 512 channels)
        self.film1 = FiLMGenerator(64)
        self.film2 = FiLMGenerator(128)
        self.film3 = FiLMGenerator(256)
        self.film4 = FiLMGenerator(512)
    
    def forward(self, x, flip):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        gamma, beta = self.film1(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.layer2(x)
        gamma, beta = self.film2(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.layer3(x)
        gamma, beta = self.film3(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.layer4(x)
        gamma, beta = self.film4(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ResNet-18 Late Fusion
class ResNet18_FlipLate(ResNet):
    """ResNet-18 with late fusion (flip concatenated before FC)"""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet18_FlipLate, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=3
        )
        # Modify FC layer to accept flip feature
        self.fc = nn.Linear(512 * BasicBlock.expansion + 1, num_classes)
    
    def forward(self, x, flip):
        features = self._forward_impl(x)  # (batch, 512)
        flip_feature = flip.unsqueeze(1).float()  # (batch, 1)
        combined = torch.cat([features, flip_feature], dim=1)  # (batch, 513)
        return self.fc(combined)


# ResNet-34 Baseline
class ResNet34_Baseline(ResNet):
    """ResNet-34 baseline for TinyImageNet (200 classes, 64×64 input)"""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet34_Baseline, self).__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=3
        )
    
    def forward(self, x):
        features = self._forward_impl(x)
        return self.fc(features)


# ResNet-34 Early Fusion (FiLM conditioning)
class ResNet34_FlipEarly(ResNet):
    """ResNet-34 with early fusion using FiLM conditioning."""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet34_FlipEarly, self).__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=3
        )
        self.film1 = FiLMGenerator(64)
        self.film2 = FiLMGenerator(128)
        self.film3 = FiLMGenerator(256)
        self.film4 = FiLMGenerator(512)
    
    def forward(self, x, flip):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        gamma, beta = self.film1(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.layer2(x)
        gamma, beta = self.film2(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.layer3(x)
        gamma, beta = self.film3(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.layer4(x)
        gamma, beta = self.film4(flip)
        x = film_modulate(x, gamma, beta)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ResNet-34 Late Fusion
class ResNet34_FlipLate(ResNet):
    """ResNet-34 with late fusion (flip concatenated before FC)"""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet34_FlipLate, self).__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=3
        )
        self.fc = nn.Linear(512 * BasicBlock.expansion + 1, num_classes)
    
    def forward(self, x, flip):
        features = self._forward_impl(x)
        flip_feature = flip.unsqueeze(1).float()
        combined = torch.cat([features, flip_feature], dim=1)
        return self.fc(combined)


# ============================================================================
# VGG-11/16 Implementations
# ============================================================================

class VGG(nn.Module):
    """VGG base class"""
    
    def __init__(self, features, num_classes=200, init_weights=True, in_channels=3):
        super(VGG, self).__init__()
        self.features = features
        # For 64×64 input, after 5 max pools (each halves): 64→32→16→8→4→2
        # So final feature map is 2×2, use adaptive pooling to 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
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


def make_layers(cfg, batch_norm=False, in_channels=3):
    """Make VGG layers as a single Sequential."""
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_blocks(cfg, batch_norm=False, in_channels=3):
    """Make VGG layers split into blocks at MaxPool boundaries (for FiLM conditioning).
    
    Returns:
        blocks: nn.ModuleList of Sequential blocks (one per pooling stage)
        block_channels: list of output channel counts for each block
    """
    blocks = []
    block_channels = []
    current_block = []
    ch = in_channels
    for v in cfg:
        if v == 'M':
            current_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(nn.Sequential(*current_block))
            block_channels.append(ch)
            current_block = []
        else:
            conv2d = nn.Conv2d(ch, v, kernel_size=3, padding=1)
            if batch_norm:
                current_block += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                current_block += [conv2d, nn.ReLU(inplace=True)]
            ch = v
    if current_block:
        blocks.append(nn.Sequential(*current_block))
        block_channels.append(ch)
    return nn.ModuleList(blocks), block_channels


# VGG-11 configuration
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# VGG-11 Baseline
class VGG11_Baseline(VGG):
    """VGG-11 baseline for TinyImageNet (200 classes, 64×64 input)"""
    def __init__(self, num_classes=200, **kwargs):
        features = make_layers(cfgs['A'], batch_norm=True, in_channels=3)
        super(VGG11_Baseline, self).__init__(features, num_classes=num_classes, in_channels=3)


# VGG-11 Early Fusion (FiLM conditioning)
class VGG11_FlipEarly(nn.Module):
    """VGG-11 with early fusion using FiLM conditioning.
    
    FiLM modulates feature maps after each conv block (at MaxPool boundaries).
    Backbone stays standard 3-channel input.
    """
    def __init__(self, num_classes=200, **kwargs):
        super(VGG11_FlipEarly, self).__init__()
        self.feature_blocks, block_channels = make_layers_blocks(cfgs['A'], batch_norm=True, in_channels=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # Initialize VGG weights BEFORE creating FiLM generators
        self._initialize_vgg_weights()
        # FiLM generators (with their own identity initialization)
        self.film_generators = nn.ModuleList([FiLMGenerator(ch) for ch in block_channels])
    
    def _initialize_vgg_weights(self):
        """Initialize VGG feature and classifier weights (not FiLM generators)."""
        for module in [self.feature_blocks, self.classifier]:
            for m in module.modules():
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
    
    def forward(self, x, flip):
        for block, film in zip(self.feature_blocks, self.film_generators):
            x = block(x)
            gamma, beta = film(flip)
            x = film_modulate(x, gamma, beta)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# VGG-11 Late Fusion
class VGG11_FlipLate(VGG):
    """VGG-11 with late fusion (flip concatenated before classifier)"""
    def __init__(self, num_classes=200, **kwargs):
        features = make_layers(cfgs['A'], batch_norm=True, in_channels=3)
        super(VGG11_FlipLate, self).__init__(features, num_classes=num_classes, in_channels=3)
        # Modify classifier to accept flip feature
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1 + 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x, flip):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        flip_feature = flip.unsqueeze(1).float()
        x = torch.cat([x, flip_feature], dim=1)
        x = self.classifier(x)
        return x


# VGG-16 Baseline
class VGG16_Baseline(VGG):
    """VGG-16 baseline for TinyImageNet (200 classes, 64×64 input)"""
    def __init__(self, num_classes=200, **kwargs):
        features = make_layers(cfgs['D'], batch_norm=True, in_channels=3)
        super(VGG16_Baseline, self).__init__(features, num_classes=num_classes, in_channels=3)


# VGG-16 Early Fusion (FiLM conditioning)
class VGG16_FlipEarly(nn.Module):
    """VGG-16 with early fusion using FiLM conditioning."""
    def __init__(self, num_classes=200, **kwargs):
        super(VGG16_FlipEarly, self).__init__()
        self.feature_blocks, block_channels = make_layers_blocks(cfgs['D'], batch_norm=True, in_channels=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_vgg_weights()
        self.film_generators = nn.ModuleList([FiLMGenerator(ch) for ch in block_channels])
    
    def _initialize_vgg_weights(self):
        """Initialize VGG feature and classifier weights (not FiLM generators)."""
        for module in [self.feature_blocks, self.classifier]:
            for m in module.modules():
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
    
    def forward(self, x, flip):
        for block, film in zip(self.feature_blocks, self.film_generators):
            x = block(x)
            gamma, beta = film(flip)
            x = film_modulate(x, gamma, beta)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# VGG-16 Late Fusion
class VGG16_FlipLate(VGG):
    """VGG-16 with late fusion (flip concatenated before classifier)"""
    def __init__(self, num_classes=200, **kwargs):
        features = make_layers(cfgs['D'], batch_norm=True, in_channels=3)
        super(VGG16_FlipLate, self).__init__(features, num_classes=num_classes, in_channels=3)
        # Use 512 * 1 * 1 because we use AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1 + 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x, flip):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        flip_feature = flip.unsqueeze(1).float()
        x = torch.cat([x, flip_feature], dim=1)
        x = self.classifier(x)
        return x


# ============================================================================
# EfficientNetV2-S Implementations (Tan & Le 2021 - Smaller Models and Faster Training)
# ============================================================================

class EfficientNetV2_S_Baseline(nn.Module):
    """EfficientNetV2-S baseline using timm
    
    Paper: EfficientNetV2: Smaller Models and Faster Training (Tan & Le, 2021)
    - Uses Fused-MBConv operations (different from EfficientNet's MBConv)
    - Training-aware neural architecture search
    - ~22M parameters (vs EfficientNet-B7's ~66M)
    """
    def __init__(self, num_classes=200, **kwargs):
        super(EfficientNetV2_S_Baseline, self).__init__()
        # EfficientNetV2-S in timm handles variable input sizes automatically
        self.model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=num_classes, 
                                       in_chans=3)
    
    def forward(self, x):
        return self.model(x)


class EfficientNetV2_S_FlipEarly(nn.Module):
    """EfficientNetV2-S with early fusion using FiLM conditioning.
    
    FiLM modulates feature maps after each EfficientNet stage.
    Backbone stays standard 3-channel input.
    """
    def __init__(self, num_classes=200, **kwargs):
        super(EfficientNetV2_S_FlipEarly, self).__init__()
        self.model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=num_classes,
                                       in_chans=3)
        # Probe output channels for each stage
        stage_channels = self._probe_stage_channels()
        # Create FiLM generators for each stage
        self.film_generators = nn.ModuleList([FiLMGenerator(ch) for ch in stage_channels])
    
    def _probe_stage_channels(self):
        """Get output channels for each stage by running a dummy forward pass."""
        channels = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)  # BatchNormAct2d includes activation
            for stage in self.model.blocks:
                x = stage(x)
                channels.append(x.shape[1])
        if was_training:
            self.model.train()
        return channels
    
    def forward(self, x, flip):
        # Stem (bn1 is BatchNormAct2d — includes activation)
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        
        # Blocks with FiLM conditioning after each stage
        for stage, film in zip(self.model.blocks, self.film_generators):
            x = stage(x)
            gamma, beta = film(flip)
            x = film_modulate(x, gamma, beta)
        
        # Head (bn2 is BatchNormAct2d — includes activation)
        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        
        # Pool and classify
        x = self.model.global_pool(x)
        x = x.view(x.size(0), -1)
        if self.model.drop_rate > 0.:
            x = F.dropout(x, p=self.model.drop_rate, training=self.training)
        x = self.model.classifier(x)
        return x


class EfficientNetV2_S_FlipLate(nn.Module):
    """EfficientNetV2-S with late fusion (flip concatenated before classifier)"""
    def __init__(self, num_classes=200, **kwargs):
        super(EfficientNetV2_S_FlipLate, self).__init__()
        self.model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=num_classes,
                                       in_chans=3)
        # Modify classifier to accept flip feature
        # EfficientNetV2-S uses 1280 features before classifier
        old_classifier = self.model.classifier
        self.model.classifier = nn.Linear(1280 + 1, num_classes)
    
    def forward(self, x, flip):
        # Get features before classifier
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = x.flatten(1)
        # Concatenate flip feature
        flip_feature = flip.unsqueeze(1).float()
        x = torch.cat([x, flip_feature], dim=1)
        x = self.model.classifier(x)
        return x


# ============================================================================
# Vision Transformer (ViT) Implementations
# ============================================================================

class ViT_Baseline(nn.Module):
    """ViT baseline using timm (8×8 patches for 64×64 images)
    
    Matches ViT-Base architecture from paper (Dosovitskiy et al., 2020):
    - 12 transformer layers
    - 768 hidden dimension (embed_dim)
    - 3072 MLP size
    - 12 attention heads
    - ~86M parameters
    
    Note: Using TinyImageNet (64×64, 200 classes) instead of ImageNet (224×224, 1000 classes)
    for method testing. Hyperparameters match paper specifications.
    """
    def __init__(self, num_classes=200, **kwargs):
        super(ViT_Baseline, self).__init__()
        # Create ViT with 8×8 patches for 64×64 images
        # img_size=64, patch_size=8 -> 8×8 = 64 patches
        self.model = timm.create_model('vit_base_patch8_224', pretrained=False, 
                                       num_classes=num_classes, img_size=64, patch_size=8,
                                       drop_rate=0.1)  # Paper: dropout 0.1 for ImageNet training
        
        # Verify architecture matches ViT-Base paper specifications
        self._verify_architecture()
    
    def _verify_architecture(self):
        """Verify ViT architecture matches paper Table 1 (ViT-Base)"""
        expected_layers = 12
        expected_embed_dim = 768
        expected_mlp_ratio = 4  # MLP size = embed_dim * mlp_ratio = 768 * 4 = 3072
        expected_num_heads = 12
        
        actual_layers = len(self.model.blocks)
        actual_embed_dim = self.model.embed_dim
        actual_mlp_ratio = self.model.blocks[0].mlp.fc1.out_features // actual_embed_dim if hasattr(self.model.blocks[0].mlp, 'fc1') else None
        actual_num_heads = self.model.blocks[0].attn.num_heads
        
        assert actual_layers == expected_layers, f"ViT layers mismatch: expected {expected_layers}, got {actual_layers}"
        assert actual_embed_dim == expected_embed_dim, f"ViT embed_dim mismatch: expected {expected_embed_dim}, got {actual_embed_dim}"
        assert actual_num_heads == expected_num_heads, f"ViT num_heads mismatch: expected {expected_num_heads}, got {actual_num_heads}"
        
        # Verify MLP size (check first block's MLP)
        if hasattr(self.model.blocks[0].mlp, 'fc1'):
            mlp_hidden = self.model.blocks[0].mlp.fc1.out_features
            expected_mlp_size = expected_embed_dim * expected_mlp_ratio
            assert mlp_hidden == expected_mlp_size, f"ViT MLP size mismatch: expected {expected_mlp_size}, got {mlp_hidden}"
    
    def forward(self, x):
        return self.model(x)


class ViT_FlipEarly(nn.Module):
    """ViT with early fusion using flip token approach
    
    Matches ViT-Base architecture from paper (Dosovitskiy et al., 2020):
    - 12 transformer layers, 768 hidden dim, 3072 MLP, 12 heads
    - Adds flip token for early fusion
    
    Note: Using TinyImageNet (64×64, 200 classes) instead of ImageNet (224×224, 1000 classes)
    """
    def __init__(self, num_classes=200, **kwargs):
        super(ViT_FlipEarly, self).__init__()
        # Create base ViT
        self.model = timm.create_model('vit_base_patch8_224', pretrained=False,
                                      num_classes=num_classes, img_size=64, patch_size=8,
                                      drop_rate=0.1)  # Paper: dropout 0.1 for ImageNet training
        
        # Get hidden dimension
        hidden_dim = self.model.embed_dim
        
        # Verify architecture matches ViT-Base paper specifications
        self._verify_architecture()
        
        # Add flip token and embedding
        self.flip_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.flip_embedding = nn.Linear(1, hidden_dim)
        
        # Learnable positional embedding for the flip token (inserted between CLS and patches)
        self.flip_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Modify classifier to use both CLS and FLIP tokens
        self.model.head = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, flip):
        # Get patch embeddings
        x = self.model.patch_embed(x)  # (batch, num_patches, hidden_dim)
        
        # Add CLS token
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # (batch, 1, hidden_dim)
        
        # Condition flip token on flip value
        flip_emb = self.flip_embedding(flip.float().unsqueeze(-1))  # (batch, hidden_dim)
        flip_token = self.flip_token + flip_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Concatenate: [CLS, FLIP, patches]
        x = torch.cat([cls_token, flip_token, x], dim=1)  # (batch, 1+1+num_patches, hidden_dim)
        
        # Add positional embeddings with correct alignment:
        # Original pos_embed: [pos_cls, pos_p0, pos_p1, ..., pos_p63] (65 entries)
        # We insert the learnable flip_pos_embed between CLS and patch positions:
        # Result: [pos_cls, flip_pos, pos_p0, pos_p1, ..., pos_p63] (66 entries)
        orig_pos = self.model.pos_embed  # (1, 65, hidden_dim)
        cls_pos = orig_pos[:, 0:1, :]     # (1, 1, hidden_dim) — CLS positional
        patch_pos = orig_pos[:, 1:, :]     # (1, 64, hidden_dim) — patch positionals
        pos_embed = torch.cat([cls_pos, self.flip_pos_embed, patch_pos], dim=1)  # (1, 66, hidden_dim)
        x = x + pos_embed[:, :x.shape[1], :]
        x = self.model.pos_drop(x)
        
        # Process through transformer
        x = self.model.blocks(x)
        x = self.model.norm(x)
        
        # Extract CLS and FLIP token outputs
        cls_output = x[:, 0]  # (batch, hidden_dim)
        flip_output = x[:, 1]  # (batch, hidden_dim)
        combined = torch.cat([cls_output, flip_output], dim=1)  # (batch, hidden_dim*2)
        
        # Classify
        return self.model.head(combined)
    
    def _verify_architecture(self):
        """Verify ViT architecture matches paper Table 1 (ViT-Base)"""
        expected_layers = 12
        expected_embed_dim = 768
        expected_mlp_ratio = 4  # MLP size = embed_dim * mlp_ratio = 768 * 4 = 3072
        expected_num_heads = 12
        
        actual_layers = len(self.model.blocks)
        actual_embed_dim = self.model.embed_dim
        actual_num_heads = self.model.blocks[0].attn.num_heads
        
        assert actual_layers == expected_layers, f"ViT layers mismatch: expected {expected_layers}, got {actual_layers}"
        assert actual_embed_dim == expected_embed_dim, f"ViT embed_dim mismatch: expected {expected_embed_dim}, got {actual_embed_dim}"
        assert actual_num_heads == expected_num_heads, f"ViT num_heads mismatch: expected {expected_num_heads}, got {actual_num_heads}"
        
        # Verify MLP size
        if hasattr(self.model.blocks[0].mlp, 'fc1'):
            mlp_hidden = self.model.blocks[0].mlp.fc1.out_features
            expected_mlp_size = expected_embed_dim * expected_mlp_ratio
            assert mlp_hidden == expected_mlp_size, f"ViT MLP size mismatch: expected {expected_mlp_size}, got {mlp_hidden}"


class ViT_FlipLate(nn.Module):
    """ViT with late fusion (flip concatenated with CLS token output)
    
    Matches ViT-Base architecture from paper (Dosovitskiy et al., 2020):
    - 12 transformer layers, 768 hidden dim, 3072 MLP, 12 heads
    - Flip feature concatenated before final classification layer
    
    Note: Using TinyImageNet (64×64, 200 classes) instead of ImageNet (224×224, 1000 classes)
    """
    def __init__(self, num_classes=200, **kwargs):
        super(ViT_FlipLate, self).__init__()
        self.model = timm.create_model('vit_base_patch8_224', pretrained=False,
                                      num_classes=num_classes, img_size=64, patch_size=8,
                                      drop_rate=0.1)  # Paper: dropout 0.1 for ImageNet training
        # Modify classifier to accept flip feature
        hidden_dim = self.model.embed_dim
        self.model.head = nn.Linear(hidden_dim + 1, num_classes)
        
        # Verify architecture matches ViT-Base paper specifications
        self._verify_architecture()
    
    def forward(self, x, flip):
        # Standard ViT forward pass
        x = self.model.forward_features(x)  # (batch, num_patches+1, hidden_dim)
        cls_output = x[:, 0]  # Extract CLS token (batch, hidden_dim)
        
        # Concatenate flip feature
        flip_feature = flip.unsqueeze(1).float()  # (batch, 1)
        combined = torch.cat([cls_output, flip_feature], dim=1)  # (batch, hidden_dim+1)
        
        return self.model.head(combined)
    
    def _verify_architecture(self):
        """Verify ViT architecture matches paper Table 1 (ViT-Base)"""
        expected_layers = 12
        expected_embed_dim = 768
        expected_mlp_ratio = 4  # MLP size = embed_dim * mlp_ratio = 768 * 4 = 3072
        expected_num_heads = 12
        
        actual_layers = len(self.model.blocks)
        actual_embed_dim = self.model.embed_dim
        actual_num_heads = self.model.blocks[0].attn.num_heads
        
        assert actual_layers == expected_layers, f"ViT layers mismatch: expected {expected_layers}, got {actual_layers}"
        assert actual_embed_dim == expected_embed_dim, f"ViT embed_dim mismatch: expected {expected_embed_dim}, got {actual_embed_dim}"
        assert actual_num_heads == expected_num_heads, f"ViT num_heads mismatch: expected {expected_num_heads}, got {actual_num_heads}"
        
        # Verify MLP size
        if hasattr(self.model.blocks[0].mlp, 'fc1'):
            mlp_hidden = self.model.blocks[0].mlp.fc1.out_features
            expected_mlp_size = expected_embed_dim * expected_mlp_ratio
            assert mlp_hidden == expected_mlp_size, f"ViT MLP size mismatch: expected {expected_mlp_size}, got {mlp_hidden}"

