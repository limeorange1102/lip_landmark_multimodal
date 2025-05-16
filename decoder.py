import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCDecoder(nn.Module):
    def __init__(self, input_dim, vocab_size, blank_id=0):
        super().__init__()
        self.linear = nn.Linear(input_dim, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(self, x, target=None, input_lengths=None, target_lengths=None):
        """
        x: [B, T, D]
        target: [B, L] or None
        input_lengths: [B]
        target_lengths: [B]

        If target is provided, returns CTC loss.
        If target is None, returns log-probabilities.
        """
        logits = self.linear(x)                 # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

        if target is not None:
            return self.ctc_loss(
                log_probs.transpose(0, 1),      # [T, B, V]
                target,
                input_lengths,
                target_lengths
            )
        else:
            return log_probs  # [B, T, V]
