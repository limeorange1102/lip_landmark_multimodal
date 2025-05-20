import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model

# -------------------------------
# 📌 영상 인코더: VisualEncoder
# -------------------------------
class VisualEncoder(nn.Module):
    def __init__(self, pretrained_path=None, hidden_dim=128, lstm_layers=2, bidirectional=True):
        super().__init__()

        self.resnet = models.resnet34(weights=None)
        self.resnet.fc = nn.Identity()  # feature만 추출

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            try:
                self.resnet.load_state_dict(state_dict, strict=False)
                print(f"✅ VisualEncoder weights loaded from {pretrained_path}")
            except Exception as e:
                print(f"❌ Failed to load VisualEncoder weights: {e}")

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x)         # (B*T, 512)
        feats = feats.view(B, T, -1)   # (B, T, 512)
        output, _ = self.rnn(feats)
        return output  # (B, T, output_dim)

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
