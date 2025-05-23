import torch
from torch.utils.data import DataLoader
import os, random, numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from single_speaker_dataset import SingleSpeakerDataset
from collate_fn import collate_fn
from encoder import VisualEncoder
from decoder import CTCDecoder
from trainer import MultimodalTrainer
from tokenizer import Tokenizer
from preprocessing import build_data_list

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_checkpoint(epoch, trainer, path):
    torch.save({
        'epoch': epoch,
        'visual_encoder': trainer.visual_encoder.state_dict(),
        'decoder_visual': trainer.decoder_visual.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, path)

def load_checkpoint(trainer, path):
    checkpoint = torch.load(path, map_location=trainer.device)
    trainer.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
    trainer.decoder_visual.load_state_dict(checkpoint['decoder_visual'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] + 1

def main():
    set_seed()

    json_folder = "input_texts"
    npy_dir = "processed_dataset/npy"
    text_dir = "processed_dataset/text"
    wav_dir = "input_videos"

    tokenizer = Tokenizer(vocab_path="input_videos/tokenizer800.vocab")
    sentence_list = build_data_list(json_folder, npy_dir, text_dir, wav_dir)

    # ✅ 전체 문장에서 train/val/test 분할
    train_sent, temp_sent = train_test_split(sentence_list, test_size=0.2, random_state=42)
    val_sent, test_sent = train_test_split(temp_sent, test_size=0.5, random_state=42)

    train_dataset = SingleSpeakerDataset(train_sent, tokenizer, use_landmark=False)
    val_dataset = SingleSpeakerDataset(val_sent, tokenizer, use_landmark=False)


    # ✅ problem sample 제거
    problem_indices = [13173,  3711]
    keep_indices = [i for i in range(len(train_dataset)) if i not in problem_indices]
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, keep_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4,
                            collate_fn=lambda x: collate_fn(x, use_landmark=False))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4,
                            collate_fn=lambda x: collate_fn(x, use_landmark=False))

    visual_encoder = VisualEncoder(output_dim=512)
    visual_encoder.load_state_dict(torch.load("weights/Video_only_model.pt"), strict=True)

    decoder_visual = CTCDecoder(
        input_dim=visual_encoder.output_dim,
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = MultimodalTrainer(
        visual_encoder, decoder_visual,
        tokenizer,
        learning_rate=1e-4,
        device=device
    )

    drive_ckpt_dir = "/content/drive/MyDrive/lip_audio_multimodal/checkpoints_single"
    os.makedirs(drive_ckpt_dir, exist_ok=True)

    last_ckpt_path = os.path.join(drive_ckpt_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(drive_ckpt_dir, "best_checkpoint.pt")
    wer_log_path = os.path.join(drive_ckpt_dir, "wer_log.csv")
    sentence_acc_log_path = os.path.join(drive_ckpt_dir, "sentence_acc_log.csv")
    loss_log_path = os.path.join(drive_ckpt_dir, "loss_log.csv")
    start_epoch = 1
    best_wer = 1.0
    wer_history = []
    loss_history = []

    patience = 5
    no_improve_counter = 0
    max_epochs = 40

    if os.path.exists(last_ckpt_path):
        logging.info("🔁 기전 체크포인트 불러오는 중...")
        print("🔁 기전 체크포인트 불러오는 중...", flush=True)
        start_epoch = load_checkpoint(trainer, last_ckpt_path)
        logging.info(f"➞️  Epoch {start_epoch}부터 재개")
        print(f"➞️  Epoch {start_epoch}부터 재개", flush=True)
    print(f"🧪 start_epoch={start_epoch}")

    with open(wer_log_path, "w") as f:
        f.write("epoch,wer\n")
    with open(loss_log_path, "w") as f:
        f.write("epoch,loss\n")
    with open(sentence_acc_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,sentence_acc\n")

    print("▶️ for epoch 진입", flush=True)
    for epoch in range(start_epoch, max_epochs + 1):

        # ✅ ResNet freeze/unfreeze 조절
        if epoch < 5:
            trainer.visual_encoder.freeze_resnet()
            print(f"🧊 Epoch {epoch}: ResNet frozen")
        else:
            trainer.visual_encoder.unfreeze_resnet()

            print(f"🔥 Epoch {epoch}: ResNet unfrozen")
        logging.info(f"\n📚 Epoch {epoch}/{max_epochs}")
        print(f"\n📚 Epoch {epoch}/{max_epochs}", flush=True)
        loss = trainer.train_epoch(train_loader, epoch)
        loss_history.append(loss)

        wer_score, sentence_acc = trainer.evaluate(val_loader)
        wer_history.append(wer_score)

        with open(wer_log_path, "a") as f:
            f.write(f"{epoch},{wer_score:.4f}\n")
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch},{loss:.4f}\n")
        with open(sentence_acc_log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{sentence_acc:.4f}\n")

        save_checkpoint(epoch, trainer, last_ckpt_path)
        logging.info("📎 마지막 체크포인트 저장 완료")
        print("📎 마지막 체크포인트 저장 완료", flush=True)

        if wer_score < best_wer:
            best_wer = wer_score
            no_improve_counter = 0
            save_checkpoint(epoch, trainer, best_ckpt_path)
            logging.info("🏅 Best 모델 갱신 및 저장 완료")
            print("🏅 Best 모델 갱신 및 저장 완료", flush=True)
        else:
            no_improve_counter += 1
            print(f"🔻 성능 감소 무: {no_improve_counter}/{patience}, best wer/werscore = {best_wer}/{wer_score}", flush=True)
    
if __name__ == "__main__":
    main()
