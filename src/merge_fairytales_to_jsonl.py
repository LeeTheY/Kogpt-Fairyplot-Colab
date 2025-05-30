import os
import json
import jsonlines

# 데이터 경로 설정
folder_path = '/content/drive/MyDrive/Colab Notebooks/storytellerAI/동화줄거리학습데이터'
output_path = '/content/drive/MyDrive/Colab Notebooks/storytellerAI/processed_fairytales_merged.jsonl'

all_data = []

# 모든 JSON 파일에서 character와 plotSummary 추출
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                paragraphs = data.get("paragraphInfo", [])
                for para in paragraphs:
                    character = para.get("character", "")
                    plot_summary = para.get("plotSummaryInfo", {}).get("plotSummaryText", "")
                    if character and plot_summary:
                        prompt = f"character: {character}"
                        all_data.append({
                            "prompt": prompt,
                            "completion": plot_summary
                        })
        except Exception as e:
            print(f"{filename} 처리 중 오류 발생: {e}")

# JSONL 형식으로 저장
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(all_data)

print(f"총 {len(all_data)}개의 데이터가 {output_path}에 저장되었습니다.")
