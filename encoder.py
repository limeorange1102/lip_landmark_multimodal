import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model

# -------------------------------
# 📌 영상 인코더: VisualEncoder
# -------------------------------
class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=54, hidden_dim=128, lstm_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x):
        # x: (B, T, 27, 2) → (B, T, 54)
        B, T, N, D = x.shape
        x = x.view(B, T, N * D)

        output, _ = self.rnn(x)
        output = self.dropout(output)  # Dropout after LSTM
        return output  # [B, T, output_dim]

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
