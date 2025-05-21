import random
import torch
import numpy as np
import librosa
import cv2
import torch
import numpy as np
import librosa

class SingleSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_list, tokenizer, use_landmark=True):
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer
        self.use_landmark = use_landmark  # True: [T, 27, 2], False: crop image

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        s = self.sentence_list[idx]

        # 🟠 Load waveform
        audio, sr = librosa.load(s["audio_path"], sr=None)
        start_sample = int(s["start_time"] * sr)
        end_sample = int(s["end_time"] * sr)
        audio = audio[start_sample:end_sample]
        audio = np.asarray(audio).flatten()

        # 🟠 Load visual input
        lip = np.load(s["lip_path"])
        if not self.use_landmark:
            lip = np.stack([frame for frame in lip])  # [T, H, W, C]

        # 🟠 Load label
        with open(s["text_path"], "r", encoding="utf-8") as f:
            label = self.tokenizer.encode(f.read().strip())

        return {
            "audio": audio.astype(np.float32),
            "lip1": lip.astype(np.float32),
            "label1": np.array(label, dtype=np.int64),
            "lip1_len": lip.shape[0],
        }


class RandomSentencePairDataset(SingleSpeakerDataset):
    def __init__(self, sentence_list, tokenizer, num_pairs_per_epoch=10000):
        super().__init__(sentence_list, tokenizer)
        self.num_pairs_per_epoch = num_pairs_per_epoch

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class FixedSentencePairDataset(SingleSpeakerDataset):
    def __init__(self, pair_list, tokenizer):
        super().__init__(pair_list, tokenizer)
        self.pair_list = pair_list  # for clarity; used below

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        s1, s2 = self.pair_list[idx]

        a1, sr1 = librosa.load(s1["audio_path"], sr=None)
        a2, sr2 = librosa.load(s2["audio_path"], sr=None)

        # 🎯 sr mismatch 확인
        assert sr1 == sr2, f"[오류] Sampling rates do not match! sr1={sr1}, sr2={sr2}"
        a1 = np.asarray(a1).flatten()
        a2 = np.asarray(a2).flatten()

        max_len = max(len(a1), len(a2))
        if len(a1) < max_len:
            a1 = np.pad(a1, (0, max_len - len(a1)), mode="constant")
        if len(a2) < max_len:
            a2 = np.pad(a2, (0, max_len - len(a2)), mode="constant")

        mix = a1 + a2
        mix = mix.astype(np.float32)
        mix = mix / (np.max(np.abs(mix)) + 1e-6)

        lip1 = np.load(s1["lip_path"])
        #(128x128)->(64x64)로 resize
        lip1 = np.stack([cv2.resize(frame, (64, 64)) for frame in lip1])

        with open(s1["text_path"], "r", encoding="utf-8") as f:
            label1 = self.tokenizer.encode(f.read().strip())

        return {
            "audio": mix.astype(np.float32),
            "audio1_raw": a1.astype(np.float32),
            "audio2_raw": a2.astype(np.float32),
            "lip1": lip1.astype(np.float32),
            "label1": np.array(label1, dtype=np.int64),
            "lip1_len": lip1.shape[0],
        }
