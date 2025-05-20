import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, audio_dim, fused_dim):
        super().__init__()

        self.query_v = nn.Linear(visual_dim, fused_dim)
        self.key_a = nn.Linear(audio_dim, fused_dim)
        self.value_a = nn.Linear(audio_dim, fused_dim)

        self.query_a = nn.Linear(audio_dim, fused_dim)
        self.key_v = nn.Linear(visual_dim, fused_dim)
        self.value_v = nn.Linear(visual_dim, fused_dim)

        self.lstm = nn.LSTM(
            input_size=fused_dim * 2,  # concat(fused_v, fused_a)
            hidden_size=fused_dim,
            num_layers=1,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )

        self.output_dim = fused_dim * 2

    def forward(self, visual_feat, audio_feat):
        # visual_feat: (B, Tv, Dv), audio_feat: (B, Ta, Da)

        B, Tv, Dv = visual_feat.shape
        B, Ta, Da = audio_feat.shape

        # ⏱️ 시간 정렬: audio_feat → visual_feat 길이로 보간
        if Tv != Ta:
            audio_feat = F.interpolate(
                audio_feat.permute(0, 2, 1),  # (B, Da, Ta)
                size=Tv, mode='linear', align_corners=True
            ).permute(0, 2, 1)  # (B, Tv, Da)

        # Visual attends to Audio
        Qv = self.query_v(visual_feat)             # (B, Tv, Df)
        Ka = self.key_a(audio_feat)                # (B, Tv, Df)
        Va = self.value_a(audio_feat)              # (B, Tv, Df)
        attn_score_va = torch.matmul(Qv, Ka.transpose(-2, -1)) / (Qv.size(-1) ** 0.5)
        attn_weight_va = F.softmax(attn_score_va, dim=-1)
        fused_va = torch.matmul(attn_weight_va, Va)  # (B, Tv, Df)

        # Audio attends to Visual
        Qa = self.query_a(audio_feat)              # (B, Tv, Df)
        Kv = self.key_v(visual_feat)               # (B, Tv, Df)
        Vv = self.value_v(visual_feat)             # (B, Tv, Df)
        attn_score_av = torch.matmul(Qa, Kv.transpose(-2, -1)) / (Qa.size(-1) ** 0.5)
        attn_weight_av = F.softmax(attn_score_av, dim=-1)
        fused_av = torch.matmul(attn_weight_av, Vv)  # (B, Tv, Df)

        # Concat both directions
        fused = torch.cat([fused_va, fused_av], dim=-1)  # (B, Tv, 2*Df)
        output, _ = self.lstm(fused)  # (B, Tv, 2*fused_dim)

        return output
