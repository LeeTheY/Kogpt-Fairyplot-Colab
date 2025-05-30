import re

# 8. 등장요소 기반 동화 줄거리 생성 함수
def generate_story(characters):
    # 입력 문장 구성
    input_text = (
        f"등장요소: {characters}\n"
        "줄거리:\n"
    )

    # 입력 토큰화 및 모델 입력
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 텍스트 생성
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 결과 디코딩 및 정제
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = generated.replace(input_text, "").strip()

    # 첫 문장만 추출
    first_sentence = re.split(r'(?<=[.!?])\s', story)[0]
    return first_sentence

# 9. 테스트 실행
characters_input = "수프, 빵"
print(f"등장요소: {characters_input} (으)로 생성된 동화입니다.\n")
print(generate_story(characters_input))
