import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, audio_dim, fused_dim, num_heads=4):
        super().__init__()

        # Step 1: 두 modality를 동일한 차원으로 projection
        self.visual_proj = nn.Linear(visual_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)

        # Step 2: 서로 교차로 attention
        self.cross_attn_visual = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)

        # Step 3: concat된 결과를 하나의 fused vector로 projection
        self.fusion_proj = nn.Linear(fused_dim * 2, fused_dim)

        # ✅ Step 4: Temporal modeling (BiLSTM + projection)
        self.temporal_model = nn.LSTM(
            input_size=fused_dim,
            hidden_size=fused_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_proj = nn.Linear(fused_dim * 2, fused_dim)  # 다시 projection

    def forward(self, visual_feat, audio_feat):
        """
        visual_feat: [B, T_v, D_v]
        audio_feat:  [B, T_a, D_a]
        Returns: fused_feat [B, T, fused_dim]
        """
        T_v = visual_feat.size(1)
        T_a = audio_feat.size(1)

        if T_v != T_a:
            audio_feat = F.interpolate(audio_feat.permute(0, 2, 1), size=T_v, mode='linear', align_corners=True)
            audio_feat = audio_feat.permute(0, 2, 1)

        v = self.visual_proj(visual_feat)  # [B, T, D]
        a = self.audio_proj(audio_feat)    # [B, T, D]

        v2a, _ = self.cross_attn_visual(query=v, key=a, value=a)  # visual attends to audio
        a2v, _ = self.cross_attn_audio(query=a, key=v, value=v)   # audio attends to visual

        fused = torch.cat([v2a, a2v], dim=-1)     # [B, T, 2D]
        fused = self.fusion_proj(fused)           # [B, T, D]

        # ✅ Temporal modeling
        fused_seq, _ = self.temporal_model(fused)       # [B, T, 2D]
        fused_seq = self.temporal_proj(fused_seq)       # [B, T, D]

        return fused_seq
