from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import re
import torch

# 모델과 토크나이저 로드
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>'
)

def generate_story(characters):
    prompt = (
        "아래의 등장요소를 바탕으로 한줄 동화 줄거리를 작성하세요.\n"
        f"등장요소: {characters}\n"
        "줄거리:\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors='pt')  # 입력 인코딩

    output = model.generate(
        input_ids,
        max_length=200,
        repetition_penalty=1.2,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )  # 텍스트 생성

    generated = tokenizer.decode(output[0], skip_special_tokens=True) 
    story = generated.replace(prompt, "").strip()

    first_sentence = re.split(r'(?<=[.!?])\s', story)[0]  # 첫 문장 추출

    return first_sentence


# 테스트
characters_input = "수프, 빵"
print(f"등장요소: {characters_input} (으)로 생성된 동화입니다.\n")
print(generate_story(characters_input))
