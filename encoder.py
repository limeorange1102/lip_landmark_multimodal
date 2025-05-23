import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import Wav2Vec2Model

# -------------------------------
# 📌 영상 인코더: VisualEncoder
# -------------------------------
class VisualEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        # 3D Convolution Frontend
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-34 Backbone (excluding FC)
        resnet = models.resnet34(pretrained=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4 (no avgpool/fc)

        # BiGRU
        self.gru = nn.GRU(
            input_size=512,  # assuming resnet output flattened to 512
            hidden_size=output_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.output_dim = output_dim

    def freeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_resnet(self, layers=("layer2", "layer3", "layer4")):
        for name, param in self.resnet.named_parameters():
            param.requires_grad = any(k in name for k in layers)

    def forward(self, x):
        # x: (B, T, C=1, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C=1, T, H, W)
        x = self.frontend3D(x)       # (B, 64, T, H', W')

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
        x = self.resnet(x)  # (B*T, 512, H', W')

        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B*T, 512)
        x = x.view(B, T, -1)  # (B, T, 512)

        x, _ = self.gru(x)  # (B, T, output_dim)
        return x

