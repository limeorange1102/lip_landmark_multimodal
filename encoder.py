import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from transformers import Wav2Vec2Model

# -------------------------------
# 📌 영상 인코더: VisualEncoder
# -------------------------------
class VisualEncoder(nn.Module):
    def __init__(self, hidden_dim=128, lstm_layers=2, bidirectional=True):
        super().__init__()

        # ✅ ImageNet pretrained ResNet34 로드
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # classification head 제거

        # ✅ 정규화 및 드롭아웃 추가
        self.norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.3)

        # ✅ LSTM temporal modeling
        self.rnn = nn.LSTM(
            input_size=512,  # ResNet 마지막 feature dimension
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def freeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)              # (B*T, C, H, W)

        # ✅ 입력 해상도를 ResNet 기대값에 맞게 조정 (224x224)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feats = self.resnet(x)                  # (B*T, 512)
        feats = self.norm(feats)               # (B*T, 512) 정규화
        feats = self.dropout(feats)            # (B*T, 512) 드롭아웃
        feats = feats.view(B, T, -1)            # (B, T, 512)
        output, _ = self.rnn(feats)             # (B, T, hidden_dim*2)
        return output


# -------------------------------
# 🎧 음성 인코더: HuggingFaceAudioEncoder
# -------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, model_name="kresnik/wav2vec2-large-xlsr-korean", freeze=True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, attention_mask=None):
        # x: [B, T], attention_mask: [B, T]
        output = self.model(input_values=x, attention_mask=attention_mask, return_dict=True)
        return output.last_hidden_state  # [B, T, output_dim]
