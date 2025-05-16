import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import mediapipe as mp

def extract_lip_landmarks(video_path, json_path, save_dir, fps=30):
    import mediapipe as mp
    os.makedirs(save_dir, exist_ok=True)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n🎬 대상 영상: {video_filename}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)[0]
    except Exception as e:
        print(f"❌ JSON 로딩 실패: {e}")
        return

    sentence_info = metadata.get("Sentence_info", [])
    if not sentence_info:
        print("⚠️ 문장 정보가 비어있습니다.")
        return

    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # MediaPipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Lip landmark indices
    lip_indices = list(set([
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
        310, 311, 312, 13, 82, 81, 42, 183, 78
    ]))

    skipped_count = 0

    for sentence in tqdm(sentence_info, desc=f"Processing {video_filename}"):
        sent_id = sentence["ID"]
        start_frame = int(sentence["start_time"] * fps)
        end_frame = int(sentence["end_time"] * fps)

        landmarks_seq = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        invalid = False

        for frame_idx in range(start_frame, end_frame):
            success, frame = cap.read()
            if not success or frame is None:
                print(f"❗ 프레임 읽기 실패: frame {frame_idx}")
                invalid = True
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                print(f"❗ 얼굴 감지 실패: frame {frame_idx}")
                invalid = True
                break

            landmarks = results.multi_face_landmarks[0].landmark
            points = [[landmarks[i].x, landmarks[i].y] for i in lip_indices]  # (27, 2)
            landmarks_seq.append(points)

        if invalid or not landmarks_seq:
            print(f"⚠️ 문장 {sent_id} 스킵됨")
            skipped_count += 1
            continue

        arr = np.array(landmarks_seq, dtype=np.float32)  # shape = [T, 27, 2]
        save_path = os.path.join(save_dir, f"{video_filename}_sentence_{sent_id}.npy")
        np.save(save_path, arr)
        print(f"✅ 문장 {sent_id} 저장 완료 → {save_path}")

    cap.release()
    face_mesh.close()
    print(f"\n🟢 전체 완료: {video_filename} / 스킵된 문장 수: {skipped_count}")


def save_sentence_labels(json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)[0]

    video_filename = os.path.splitext(os.path.basename(json_path))[0]
    sentence_info = metadata["Sentence_info"]

    for sentence in sentence_info:
        sent_id = sentence["ID"]
        text = sentence["sentence_text"].strip()

        save_path = os.path.join(save_dir, f"{video_filename}_sentence_{sent_id}.txt")
        with open(save_path, 'w', encoding='utf-8') as f_out:
            f_out.write(text + "\n")

    print(f"✅ {len(sentence_info)}개의 문장 텍스트 라벨을 저장했습니다: {save_dir}")

def build_data_list(json_folder, npy_dir, text_dir, wav_dir):
    data_list = []

    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)[0]

        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(wav_dir, base_name + ".wav")

        for sent in metadata["Sentence_info"]:
            sent_id = sent["ID"]
            lip_path = os.path.join(npy_dir, f"{base_name}_sentence_{sent_id}.npy")
            text_path = os.path.join(text_dir, f"{base_name}_sentence_{sent_id}.txt")

            if not os.path.exists(lip_path) or not os.path.exists(text_path):
                print(f"⚠️ 파일 누락 → 제외: {lip_path}, {text_path}")
                continue

            data_list.append({
                "lip_path": lip_path,
                "text_path": text_path,
                "audio_path": wav_path,
                "start_time": float(sent["start_time"]),
                "end_time": float(sent["end_time"]),
            })

    return data_list

def extract_lip_landmarks_all(json_folder, video_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_folder, "*.json"))
    for json_path in json_files:
        filename = os.path.splitext(os.path.basename(json_path))[0]
        video_path = os.path.join(video_folder, filename + ".mp4")
        if os.path.exists(video_path):
            extract_lip_landmarks(video_path, json_path, save_dir)
        else:
            print(f"❌ 영상 파일 없음: {filename}.mp4")
            
def save_all_sentence_labels(json_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_folder, "*.json"))
    for json_path in json_files:
        save_sentence_labels(json_path, save_dir)


# 입력 경로
json_folder = "/content/drive/MyDrive/lip_audio_multimodal/input_texts"
video_folder = "/content/drive/MyDrive/lip_audio_multimodal/input_videos"

# 출력 경로
landmark_npy_dir = "C:\Users\vips1\OneDrive\Desktop\과제\종설\processed_dataset\landmark_npy"

# landmark 좌표 추출 및 저장 실행
extract_lip_landmarks_all(json_folder, video_folder, landmark_npy_dir)
