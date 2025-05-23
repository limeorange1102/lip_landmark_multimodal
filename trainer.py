import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
from tqdm import tqdm
import numpy as np
import random

class MultimodalTrainer:
    def __init__(self, visual_encoder, audio_encoder, fusion_module,
                 decoder1, decoder_audio, decoder_visual,
                 tokenizer, learning_rate=1e-4, device="cuda"):
        
        self.visual_encoder = visual_encoder.to(device)
        self.audio_encoder = audio_encoder.to(device)
        self.fusion_module = fusion_module.to(device)

        self.decoder1 = decoder1.to(device)
        self.decoder_audio = decoder_audio.to(device)
        self.decoder_visual = decoder_visual.to(device)

        self.tokenizer = tokenizer
        self.device = device

        self.ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

        self.parameters = (
            list(self.visual_encoder.parameters()) +
            list(self.decoder_visual.parameters())
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)

    def ctc_decode(self, pred_ids):
        result = []
        prev = None
        for idx in pred_ids:
            if idx == self.tokenizer.blank_id:
                continue
            if idx != prev:
                result.append(idx)
            prev = idx
        return result

    def train_epoch(self, dataloader, epoch):
        print("✅ train_epoch() 짱입")

        self.visual_encoder.train()
        self.decoder_visual.train()

        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", ncols=100)):
            self.optimizer.zero_grad()

            lip1 = batch["lip1"].to(self.device)
            text1 = batch["text1"].to(self.device)
            len1 = batch["text1_lengths"].to(self.device)
            lip1_lengths = batch["lip1_lengths"].to(self.device)
            try:
                visual_feat1 = self.visual_encoder(lip1)

                input_lengths_visual1 = torch.full((visual_feat1.size(0),), visual_feat1.size(1), dtype=torch.long).to(self.device)

                log_probs_visual1 = self.decoder_visual(visual_feat1)

                loss_visual1 = self.ctc_loss(log_probs_visual1.transpose(0, 1), text1, input_lengths_visual1, len1)

                loss = loss_visual1

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ OOM at Batch {batch_idx} — fallback to batch size 1")
                print(f"🧭 Fallback 유발 sample indices: {batch['index']}")
                torch.cuda.empty_cache()
                B = lip1.size(0)
                safe_loss_total = 0
                for i in range(B):
                    try:
                        self.optimizer.zero_grad()
                        l1 = lip1[i:i+1]
                        t1 = text1[i:i+1]
                        l1_len = len1[i:i+1]
                        lip1_len = lip1_lengths[i:i+1]

                        vf1 = self.visual_encoder(l1)

                        in_len = torch.full((1,), vf1.size(1), dtype=torch.long).to(self.device)
                        in_len_v = torch.full((1,), vf1.size(1), dtype=torch.long).to(self.device)

                        logp_v = self.decoder_visual(vf1)

                        loss_visual_1 = self.ctc_loss(logp_v.transpose(0, 1), t1, in_len_v, l1_len)

                        loss_f = loss_visual_1

                        loss_f.backward()
                        self.optimizer.step()
                        safe_loss_total += loss_f.item()
                    except torch.cuda.OutOfMemoryError:
                        print(f"❌ Sample {i} still caused OOM — skipped")
                        torch.cuda.empty_cache()
                total_loss += safe_loss_total / max(B, 1)

            if batch_idx % 100 == 0:
                pred_ids = torch.argmax(log_probs_visual1[0], dim=-1).cpu().tolist()
                unique_ids = sorted(set(pred_ids))
                print(f"[진단] Batch {batch_idx} - 예측 토큰 ID (앞 20개): {pred_ids[:20]}", flush=True)
                print(f"[진단] 고유 토큰 ID들: {unique_ids}", flush=True)
                print(f"\n🔎 [Batch {batch_idx}] 예측 결과 확인", flush = True)
                with torch.no_grad():
                    pred1_ids = torch.argmax(log_probs_visual1, dim=-1)
                    for i in range(min(2, pred1_ids.size(0))):
                        pred_ids1 = self.ctc_decode(pred1_ids[i].cpu().tolist())
                        decoded1 = self.tokenizer.decode(pred_ids1)
                        true1 = self.tokenizer.decode(text1[i][:len1[i]].cpu().tolist())
                        print(f"[화자1 예측] {decoded1}", flush=True)
                        print(f"[화자1 정답] {true1}", flush=True)

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.visual_encoder.eval()
        self.decoder_visual.eval()

        all_refs1, all_hyps1 = [], []
        global_index = 0
        sampled_indices = set(random.sample(range(len(dataloader.dataset)), min(10, len(dataloader.dataset))))

        with torch.no_grad():
            for batch in dataloader:
                lip1 = batch["lip1"].to(self.device)
                text1 = batch["text1"].to(self.device)
                len1 = batch["text1_lengths"].to(self.device)

                visual_feat1 = self.visual_encoder(lip1)
                log_probs_visual = self.decoder_visual(visual_feat1)

                pred_visual = torch.argmax(log_probs_visual, dim=-1).cpu().numpy()

                input_lengths_visual1 = torch.full((visual_feat1.size(0),), visual_feat1.size(1), dtype=torch.long).to(self.device)

                for i in range(len(pred_visual)):
                    p_visual_ids = self.ctc_decode(pred_visual[i][:input_lengths_visual1[i]])
                    ref = self.tokenizer.decode(text1[i][:len1[i]].cpu().numpy())
                    hyp = self.tokenizer.decode(p_visual_ids)

                    all_hyps1.append(hyp)
                    all_refs1.append(ref)

                    if global_index in sampled_indices:
                        print(f"[👄 입모양 전용] {hyp}")
                        print(f"[✅ 정답 문장] {ref}\n")

                    global_index += 1

        wer1 = wer(all_refs1, all_hyps1)
        sentence_acc1 = np.mean([ref.strip() == hyp.strip() for ref, hyp in zip(all_refs1, all_hyps1)])

        print(f"✅ Eval Results: WER1={wer1:.3f}, SentenceAcc={sentence_acc1:.3f}", flush=True)

        return wer1, sentence_acc1
