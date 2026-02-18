"""
Extended model architectures for TinyImageNet (64×64, 200 classes) with flip feature support.
Includes: ResNet-18/34, VGG-11/16, EfficientNet-B0, ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


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


# ResNet-18 Early Fusion
class ResNet18_FlipEarly(ResNet):
    """ResNet-18 with early fusion (flip as 4th input channel)"""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet18_FlipEarly, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=4
        )
    
    def forward(self, x, flip):
        # x: (batch, 3, 64, 64), flip: (batch,)
        # Expand flip to match spatial dimensions
        flip_channel = flip.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 64, 64).float()
        x = torch.cat([x, flip_channel], dim=1)  # (batch, 4, 64, 64)
        features = self._forward_impl(x)
        return self.fc(features)


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


# ResNet-34 Early Fusion
class ResNet34_FlipEarly(ResNet):
    """ResNet-34 with early fusion (flip as 4th input channel)"""
    def __init__(self, num_classes=200, **kwargs):
        super(ResNet34_FlipEarly, self).__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=4
        )
    
    def forward(self, x, flip):
        flip_channel = flip.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 64, 64).float()
        x = torch.cat([x, flip_channel], dim=1)
        features = self._forward_impl(x)
        return self.fc(features)


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
    """Make VGG layers"""
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


# VGG-11 Early Fusion
class VGG11_FlipEarly(VGG):
    """VGG-11 with early fusion (flip as 4th input channel)"""
    def __init__(self, num_classes=200, **kwargs):
        features = make_layers(cfgs['A'], batch_norm=True, in_channels=4)
        super(VGG11_FlipEarly, self).__init__(features, num_classes=num_classes, in_channels=4)
    
    def forward(self, x, flip):
        flip_channel = flip.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 64, 64).float()
        x = torch.cat([x, flip_channel], dim=1)
        return super().forward(x)


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


# VGG-16 Early Fusion
class VGG16_FlipEarly(VGG):
    """VGG-16 with early fusion (flip as 4th input channel)"""
    def __init__(self, num_classes=200, **kwargs):
        features = make_layers(cfgs['D'], batch_norm=True, in_channels=4)
        super(VGG16_FlipEarly, self).__init__(features, num_classes=num_classes, in_channels=4)
    
    def forward(self, x, flip):
        flip_channel = flip.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 64, 64).float()
        x = torch.cat([x, flip_channel], dim=1)
        return super().forward(x)


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
# EfficientNet-B0 Implementations
# ============================================================================

class EfficientNetB0_Baseline(nn.Module):
    """EfficientNet-B0 baseline using timm"""
    def __init__(self, num_classes=200, **kwargs):
        super(EfficientNetB0_Baseline, self).__init__()
        # EfficientNet in timm handles variable input sizes automatically
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes, 
                                       in_chans=3)
    
    def forward(self, x):
        return self.model(x)


class EfficientNetB0_FlipEarly(nn.Module):
    """EfficientNet-B0 with early fusion (flip as 4th input channel)"""
    def __init__(self, num_classes=200, **kwargs):
        super(EfficientNetB0_FlipEarly, self).__init__()
        # Create model with 4 input channels
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes,
                                      in_chans=4)
    
    def forward(self, x, flip):
        flip_channel = flip.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 64, 64).float()
        x = torch.cat([x, flip_channel], dim=1)
        return self.model(x)


class EfficientNetB0_FlipLate(nn.Module):
    """EfficientNet-B0 with late fusion (flip concatenated before classifier)"""
    def __init__(self, num_classes=200, **kwargs):
        super(EfficientNetB0_FlipLate, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes,
                                       in_chans=3)
        # Modify classifier to accept flip feature
        # EfficientNet-B0 uses 1280 features before classifier
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
        
        # Add positional embeddings - need to handle extra token
        # Original: 1 CLS + 64 patches = 65 tokens
        # New: 1 CLS + 1 FLIP + 64 patches = 66 tokens
        # Interpolate or pad positional embeddings
        pos_embed = self.model.pos_embed  # (1, 65, hidden_dim)
        if x.shape[1] > pos_embed.shape[1]:
            # Need to add positional embedding for flip token
            # Use the CLS token's positional embedding for the flip token
            flip_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, hidden_dim)
            pos_embed = torch.cat([pos_embed, flip_pos_embed], dim=1)  # (1, 66, hidden_dim)
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

