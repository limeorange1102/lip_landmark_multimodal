import torch
import numpy as np
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

        # 🟠 Load visual input
        lip = np.load(s["lip_path"])  # [T, H, W, 3] expected
        if not self.use_landmark:
            # 1. RGB → Grayscale
            lip = np.mean(lip, axis=-1, keepdims=True)  # [T, H, W, 1]
            
            # 2. z-score normalization (per sequence)
            lip = lip.astype(np.float32)
            lip = (lip - lip.mean()) / (lip.std() + 1e-6)  # [T, H, W, 1]

        # 🟠 Load label
        with open(s["text_path"], "r", encoding="utf-8") as f:
            label = self.tokenizer.encode(f.read().strip())

        return {
            "lip1": lip.astype(np.float32),
            "label1": np.array(label, dtype=np.int64),
            "lip1_len": lip.shape[0],
            "index": idx
        }

