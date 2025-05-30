import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextDataset
)

# 드라이브 내 작업 디렉토리 및 모델 저장 경로
drive_dir = "/content/drive/MyDrive/Colab Notebooks/storytellerAI"
model_dir = os.path.join(drive_dir, "kogpt-finetuned-epoch3")

# KoGPT2 모델 및 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 학습 데이터 로딩 함수
def get_dataset(file_path, tokenizer, block_size=512):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

# 학습 데이터 로딩
train_txt_path = os.path.join(drive_dir, "train.txt")
train_dataset = get_dataset(train_txt_path, tokenizer)

# MLM 없이 GPT용 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 학습 설정 정의
training_args = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=0,
    save_total_limit=None,
    logging_steps=100,
    prediction_loss_only=True,
    report_to=[]  # W&B 사용 안 함
)

# 기존 모델 불러오기 또는 새로 학습
if os.path.exists(model_dir):
    print("이미 학습된 모델을 불러옵니다.")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
else:
    print("모델을 새로 학습합니다.")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("학습된 모델과 토크나이저를 저장했습니다.")
