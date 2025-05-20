import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, audio_dim, hidden_dim):
        super(CrossAttentionFusion, self).__init__()
        self.visual_fc = nn.Linear(visual_dim, hidden_dim)
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def match_time_resolution(self, audio_feat, visual_feat):
        """
        Resample audio feature to match visual feature length
        audio_feat: (B, T_audio, D_audio)
        visual_feat: (B, T_visual, D_visual)
        """
        B, T_v, _ = visual_feat.shape
        B, T_a, D_a = audio_feat.shape

        if T_a != T_v:
            # Permute to (B, D, T) for interpolation
            audio_feat = audio_feat.permute(0, 2, 1)
            audio_feat = F.interpolate(audio_feat, size=T_v, mode='linear', align_corners=True)
            audio_feat = audio_feat.permute(0, 2, 1)
        return audio_feat

    def forward(self, visual_feat, audio_feat):
        # Align time dimension
        audio_feat = self.match_time_resolution(audio_feat, visual_feat)

        visual_emb = self.visual_fc(visual_feat)  # (B, T, H)
        audio_emb = self.audio_fc(audio_feat)     # (B, T, H)

        fused = torch.cat([visual_emb, audio_emb], dim=-1)  # (B, T, 2H)
        output = self.output_fc(fused)  # (B, T, H)
        return output