import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 필요한 리소스 다운로드
for res in ['punkt', 'wordnet']:
    nltk.download(res, quiet=True)

nltk.download('punkt_tab')

smooth_fn = SmoothingFunction().method1

# Prompt 매핑
prompt_labels = {
    "소년, 소녀": "Boy, Girl", "할어버지, 할머니": "Grandfather, Grandmother",
    "수프,빵": "Soup, Bread", "평민,노예": "Commoner, Slave",
    "첫째,둘째,셋째": "First, Second, Third"
}

# 참조문장
references = {
    "소년, 소녀": "소년은 음악을 연주하고, 귀여운 하와이 소녀는 훌라 춤을 추고 있었다.",
    "할어버지, 할머니": "두 할머니의 할아버지 리케 노인은 옛날에 댄스 선생님이었다고 하며 이제 96세가 되어 달과 검은 박쥐가 수놓아진 무릎 덮개에 싸여 안락의자에 앉아 있었다.",
    "수프,빵": "내가 깨난 것을 알자 한 할머니가 수프와 딱딱해 보이는 빵을 내밀어서 정신없이 먹었고 접시 바닥에는 그림이 그려져 있었다.",
    "평민,노예": "너는 하류층 사회의 평민일지라도 노예는 아니라는 것을 잘 기억해 두라고 하면서, 잘 가라고 인사했다.",
    "첫째,둘째,셋째": "첫째와 둘째는 크고 튼튼한 나무에 집을 지었는데 셋째는 가장 멋진 집을 짓는다며 집을 짓지 않고 왔다 갔다 했습니다."
}

# 생성된 문장 (epoch별, prompt별)
generated_texts = {
    3: {
        "소년, 소녀": [
            "어느 날 밤 한 소년이 찾아와 줄리에 대해 이야기를 해 주었습니다.",
            "소년은 무뚝뚝하게 내달렸고, 어느 날 저녁, 뽀르뚜까는 한길에서 혼자 길을 걷고 있는 소녀를 발견하고 겁을 먹고 있었습니다.",
            "소년은 소녀가 자기를 따라와 주었다고 말하자 그녀는 고개를 끄덕였습니다.",
            "소녀는 숨을 죽이고 있는 것 같은 기분이 되어서 천천히 고개를 돌렸습니다.",
            "소년은 소름이 끼쳐서 입을 삐죽거렸고, 소녀는 눈물을 글썽이며 말했습니다."
        ],
        "할어버지, 할머니": [
            "할머니는 꼼짝하지 않고 앉아 있었고, 나는 할머니를 보려고 몸을 웅크린 채 서 있었다.",
            "이런 이야기를 듣고 할아버지는 감격스러워했습니다.",
            "할아버지가 돌아오자, 할아버지는 다시 기분이 좋아져서 할아버지에게 이번에 집에 놀러 오자고 말했어요.",
            "할아버지는 할머니를 만나러 가기 전에 한 번 더 말씀드리겠다며 저도 모르게 손을 흔들었습니다.",
            "할아버지는 할머니에게 오늘은 무슨 일이 있는지 알려 드리자고 제안했고 할아버지 말대로 하는 건 좋았지만 한 달은 더 기다려야 한다고 했습니다.",
        ],
        "수프,빵": [
            "이 집에는 빵도 많고 주인님이 뭘 좋아하는지 알면 어떨까라고 묻자, 아주머니는 웃으며 말했어요.",
            "오늘은 우리가 만든 빵을 먹으러 오겠다고 하자, 나는 그게 뭐냐며 물었습니다.",
            "다리기 같은 게임이니 내일 아침이 좋겠지만 오늘은 정말 배가 고프다며 우리도 점심을 먹자고 말했습니다.",
            "내가 아까워할 때 우리는 꼼짝도 하지 않았는데 빵을 조금만 먹고 싶어졌다.",
            "마음이 놓인다면 수프를 먹을 테니, 괜찮을 거라는 말에 나는 고개를 끄덕였다.",
        ],
        "평민,노예": [
            "어느 날 마을에 노예가 사는 집이 나타났는데 그곳은 아주 가난해 보이는 집이었습니다.",
            "대부분 평민이었던 도이칠란트 사람들은 노예가 되면 살 수 없다는 사실을 알고 있었으므로 전쟁에 반대했으며, 결국 전쟁이 끝난 뒤 고향으로 돌아가기로 했습니다.",
            "이야기가 계속되며 줄거리가 늘어갔고 그 중 가장 유명한 이야기는 평민인 노예의 이야기였다.",
            "평민인 노예가 줄행랑치다가 붙잡히면 무슨 소용이 있냐고 묻자 자신은 그냥 놔두면 안 된다고 말했습니다.",
            "1897년 8월 13일, 마침내 노예가 자유의 몸이 되어 고향으로 돌아왔다는 소식이 전해졌고, 이 소식을 들은 많은 사람이 몰려왔습니다.",
        ],
       "첫째,둘째,셋째": [
            "다음 날부터 셋째는 첫째에게 공부를 가르치기 위해 함께 산으로 갔습니다.",
            "셋째는 우리 집에 가서 놀자고 했습니다.",
            "다음날부터 셋째와 셋은 날마다 아침 일찍 일어나서 길을 떠났다.",
            "첫째가 말하자, 첫째의 대답은 옳았지만, 둘째는 그 자리에서 꼬리를 흔들며 버럭 소리를 질렀습니다.",
            "무슨 말인지 알 수가 없던 셋째는 첫째의 말을 듣고 깜짝 놀랐어요.",
        ],
    },
    6: {
        "소년, 소녀": [
            "이름도 없는 소년의 이야기를 듣고 싶지 않았던 나는 소녀를 생각하며 집으로 돌아왔다.",
            "모두 한숨을 쉬며 이야기하는 동안에도 소녀는 웃으며 언덕 꼭대기에 서 있었다.",
            "어느 마을에 살고 있는 소녀는 숲속에서 혼자 살며 가끔씩 숲을 지나가는 것이 재미있었습니다.",
            "소년은 고개를 끄덕이며 여기 있는 것을 모두 이곳에서 지내자고 말했습니다.",
            "소년은 그 자리에 멈추고 말았고, 곧 문이 열리고 누군가가 나타나 문을 두드렸습니다.",
        ],
        "할어버지, 할머니": [
            "아름다운 처녀를 만나고 난 후 마음이 무거웠던 할아버지가 이번에는 왜 왔냐며 놀랐어요.",
            "한번은 할아버지가 오시던 날이 지나가자마자 들떠 있었습니다.",
            "할머니는 할아버지가 안 계시니 집에서 놀자고 말했습니다.",
            "옛날, 어느 마을에 할아버지가 살고 있었는데 할머니가 돌아가시자 먹을 것이 없어서 산속 빈집에 숨어 지냈습니다.",
            "할아버지가 한숨을 쉬며 이 근처를 지나다 보니, 커다란 마을이 나왔습니다.",
        ],
        "수프,빵": [
            "아직 저녁은 아니니 점심을 먹자고 했더니 그는 어서 나가라고 했고 나는 빵을 다 먹고 나자 정신이 들었다.",
            "내가 제일 좋아하는 거라고 대답하자 내가 빵을 먹고 싶어서 왔다고 했다.",
            "마지막에는 어떤 음식이든 제일 먼저 먹어 보기로 했고, 우리는 빈 그릇을 받아 들고 집으로 향했습니다.",
            "수프를 조금만 먹었을 뿐이라며 다른 음식을 먹고 싶다고 하자 빵을 달라고 했다.",
            "수프를 먹고 싶다고 했더니, 빵도 먹고 싶은데 돈이 없어서 못 사준다며 그럼 나중에 얘기해도 괜찮을 거라고 했다.",
        ],
        "평민,노예": [
            "어느 날, 어느 마을에 노예가 살았는데, 그 사람은 다정한 사람이었는데, 그는 한 번도 본 적이 없는 말쑥한 옷차림이었습니다.",
            "한번은 귀족의 집에 들어가기도 하고, 여러 곳을 구경하고 다녔으며, 특히 요하네스버그는 경치가 아름다워 한 달 내내 이곳을 찾아다녔습니다.",
            "이어서 평민은 그 자리에 있었고 노예들은 넓은 땅을 차지했고, 힘센 귀족들은 힘을 들여서라도 싸움을 계속했습니다.",
            "평민인 줄거리가 궁금했던 토닌은 자신이 무슨 짓을 했는지 물었고, 노예는 자기가 누군지 알고 싶어 안달이 난다고 대답했습니다.",
            "",
        ],
       "첫째,둘째,셋째": [
            "그곳은 노예가 사는 곳인 데다 집이 가난해서 어쩔 수 없이 이곳저곳의 문화를 받아들여야 했습니다.",
            "첫째가 첫째의 옆구리를 찰싹 때려주자, 둘째는 셋째를 쓰다듬어 주며 같이 놀자고 말했습니다.",
            "첫째는 셋째의 집에서 같이 살자고 했지만, 둘째와 셋째는 거절했습니다.",
            "첫째와 둘째는 대답을 기다렸고, 셋째가 말을 이었다.",
            "첫째와 둘째는 첫째의 말을 듣고 고개를 끄덕였고, 셋째는 셋째의 말이 옳은 말이라는 것을 알고 있었습니다.",
        ],
    },
    12: {
        "소년, 소녀": [
            "소년은 자기가 맡은 구역을 다른 곳으로 옮기지 않고 혼자서 맡아서 해야 한다며 반대했습니다.",
            "그때마다 두 소녀는 서로를 향해 눈물을 흘렸습니다.",
            "소년은 소녀에게 다가가 손을 내밀어 인사를 했고, 두 사람은 한동안 말이 없었다.",
            "무섭게 으르렁대는 목소리에 겁에 질린 소년은 벌떡 일어나서 주위를 둘러보았다.",
            "소년이 웃으며 고개를 끄덕였고, 우리는 동굴을 향해 걸어갔습니다.",
        ],
        "할어버지, 할머니": [
            "마을 어귀에 도착하니 할아버지가 우리를 기다리고 있었다며 할머니가 울음을 터뜨렸습니다.",
            "할머니는 내가 좋아하는 게 아니고 자기가 좋아하는 건 뭐든지 다 하니까 좋다고 하셨어요.",
            "할머니는 이제 그만 쉬라고 했지만 아빠의 눈에는 걱정스러운 표정이 떠올랐습니다.",
            "할아버지는 무섭다며 당장 나가라고 말했고 할머니는 어서 일어나라는 말에 놀랐습니다.",
            "할아버지가 큰 소리로 외쳤고, 그 소리에 화가 난 할머니가 왜 울고 있냐며 소리쳤다.",
        ],
        "수프,빵": [
            "수프가 다 먹어 버리자 빵을 조금 먹으라고 했지만, 그는 또 먹고 싶다고 했습니다.",
            "나는 레스토랑에 가서 빵을 한 조각 먹으면 기분이 좋겠다고 말했다.",
            "첫 번째 식당에서는 음식을 가득 차려 놓고 각자 다른 음식으로 메뉴판을 만들었습니다.",
            "수프가 내미는 빵은 조금 더 짭짤했는데 나는 이게 뭐냐고 말했다.",
            "나는 수프가 먹고 싶을 때 말고는 아무것도 하지 않았는데 내일 아침 빵 한 조각 줄 테니 잠깐 기다리라고 했다.",
        ],
        "평민,노예": [
            "어느 날, 한 신부는 마을에서 가장 유명한 화가에게 노예를 어떻게 다루는지 물었습니다.",
            "이번에는 평민 출신의 노예가 뽑혀 궁궐로 올 것이라는 말에 왕은 깜짝 놀랐습니다.",
            "한 번도 보지 못한 신기하고 새로운 세계로, 마녀가 사는 오두막에 갇힌 노예들은 한밤중에 잠을 이루지 못하고, 그 밤이 끝나도 깨어나지 못했습니다.",
            "어느 날 밤 갑자기 한 백작이 마차를 타고 나타나 노예가 어떻게 살고 있는지 물었습니다.",
            "평민인 노예는 줄거리에 따라 행동하기도 하고, 말을 걸기도 하며 남의 이야기를 들을 때도 있어요.",
        ],
       "첫째,둘째,셋째": [
            "첫째는 둘째에게 다가가서 인사했습니다.",
            "첫째가 둘째에게 넌 정말 재미있는 아이지만 어쩐지 궁금하다고 말했어요.",
            "첫째와 둘째는 서로 눈치만 보며 시시콜콜한 이야기꽃을 피웠고, 셋째는 누가 먼저 말을 걸었는지 궁금했습니다.",
            "첫째와 둘째도 재미있다는 듯 웃었고, 셋째는 입이 찢어질 것 같았습니다.",
            "첫째는 셋째네 집 앞에서 둘째네 집으로 가자, 첫째와 셋째는 깜짝 놀라 뒤를 돌아봤습니다.",
        ],
    }
}

# 점수 계산 함수들
def compute_bleu(ref, cand):
    return sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(cand), smoothing_function=smooth_fn)

def compute_meteor(ref, cand):
    return single_meteor_score(nltk.word_tokenize(ref), nltk.word_tokenize(cand))

def compute_cider(ref, cand):
    vec = TfidfVectorizer().fit_transform([ref, cand])
    return cosine_similarity(vec[0], vec[1])[0][0]

# 평가
results = defaultdict(lambda: defaultdict(list))

for epoch, prompts in generated_texts.items():
    for prompt, cands in prompts.items():
        ref = references[prompt]
        for cand in filter(None, map(str.strip, cands)):
            results[epoch][prompt].append({
                "candidate": cand,
                "BLEU": compute_bleu(ref, cand),
                "METEOR": compute_meteor(ref, cand),
                "CIDEr": compute_cider(ref, cand)
            })

# 통계 요약
summary = []
for epoch, prompts in results.items():
    for prompt, scores in prompts.items():
        for metric in ['BLEU', 'METEOR', 'CIDEr']:
            metric_vals = [r[metric] for r in scores]
            locals()[f"{metric.lower()}_mean"] = np.mean(metric_vals)
            locals()[f"{metric.lower()}_std"] = np.std(metric_vals)
        summary.append({
            "epoch": epoch,
            "prompt": prompt,
            "prompt_en": prompt_labels[prompt],
            "BLEU_mean": bleu_mean, "BLEU_std": bleu_std,
            "METEOR_mean": meteor_mean, "METEOR_std": meteor_std,
            "CIDEr_mean": cider_mean, "CIDEr_std": cider_std
        })

df_summary = pd.DataFrame(summary)
df_summary.to_csv('korean_text_evaluation_results.csv', index=False, encoding='utf-8')
print("평가 요약 저장 완료: 'korean_text_evaluation_results.csv'\n")

# 시각화
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = [("BLEU_mean", "BLEU"), ("METEOR_mean", "METEOR"), ("CIDEr_mean", "CIDEr")]

for ax, (metric, name) in zip(axes, metrics):
    sns.lineplot(data=df_summary, x="epoch", y=metric, hue="prompt_en", marker="o", ax=ax)
    ax.set_title(f"{name} Score by Epoch", fontsize=14)
    ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.grid(True, alpha=0.3)
    ax.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# 전체 평균 출력
print("=== 전체 평균 성능 ===")
print(df_summary.groupby("epoch")[["BLEU_mean", "METEOR_mean", "CIDEr_mean"]].mean().round(4))
