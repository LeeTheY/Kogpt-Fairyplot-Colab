import os
import json

# 경로 설정
drive_dir = "/content/drive/MyDrive/Colab Notebooks/storytellerAI"
data_json_path = os.path.join(drive_dir, "processed_fairytales_merged.jsonl")  # 입력 JSONL
train_txt_path = os.path.join(drive_dir, "train.txt")                          # 출력 TXT

# JSONL 파일 읽고 학습용 텍스트로 변환
train_data = []
with open(data_json_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        if prompt and completion:
            characters = prompt.split(":", 1)[-1].strip()
            story = completion.strip()
            train_data.append(f"등장요소: {characters}\n{story}")

# TXT 파일 저장
with open(train_txt_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_data))

print("train.txt 생성됐습니다.")
