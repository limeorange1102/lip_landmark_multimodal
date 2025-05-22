import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id=0, use_landmark=False):
    """
    batch: list of dicts from Dataset
    Each dict contains:
        - lip1: [T, H, W, C] or [T, 27, 2]
        - label1: [L]
        - audio: [T]
    """

    # ✅ 입술 입력 (영상 crop vs landmark)
    if use_landmark:
        lip1_seqs = [torch.tensor(item["lip1"], dtype=torch.float32) for item in batch]  # [T, 27, 2]
    else:
        # crop 영상인 경우: [T, H, W, C] → [T, C, H, W]
        lip1_seqs = [torch.tensor(item["lip1"], dtype=torch.float32).permute(0, 3, 1, 2) for item in batch]

    lip1_lengths = [seq.shape[0] for seq in lip1_seqs]
    lip1_padded = pad_sequence(lip1_seqs, batch_first=True)  # [B, T, ...] with auto shape alignment

    # ✅ 텍스트 레이블
    text1_seqs = [torch.tensor(item["label1"], dtype=torch.long) for item in batch]
    text1_lengths = [len(seq) for seq in text1_seqs]
    text1_padded = pad_sequence(text1_seqs, batch_first=True, padding_value=pad_id)

    # ✅ 오디오
    audio_seqs = [torch.tensor(item["audio"], dtype=torch.float32) for item in batch]
    audio_lengths = [seq.shape[0] for seq in audio_seqs]
    audio_padded = pad_sequence(audio_seqs, batch_first=True)

    attention_mask = torch.zeros_like(audio_padded, dtype=torch.bool)
    for i, length in enumerate(audio_lengths):
        attention_mask[i, :length] = 1

    return {
        "lip1": lip1_padded,                       # [B, T, 27, 2] or [B, T, C, H, W]
        "lip1_lengths": torch.tensor(lip1_lengths, dtype=torch.long),

        "text1": text1_padded,
        "text1_lengths": torch.tensor(text1_lengths),

        "audio": audio_padded,
        "audio_attention_mask": attention_mask,

        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long)
    }
