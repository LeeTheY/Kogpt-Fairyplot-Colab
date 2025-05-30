import matplotlib.pyplot as plt
import numpy as np
import nltk
from konlpy.tag import Okt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from collections import Counter
import math

# NLTK 리소스
nltk.download('punkt')
nltk.download('wordnet')

# 형태소 분석 및 토큰화
tokenizer = Okt()
tokenize = lambda s: tokenizer.morphs(s)

# 원문 및 생성문
ref = tokenize("15명이 생활하기에는 동굴은 비좁아서, 소년들은 벼랑에 다른 동굴을 찾아 헤맸지만 허사였고, 지금 사는 동굴을 넓히기로 했다.")
gen_pre = tokenize("작가가 자신의 캐릭터를 좋아하는 것이 아닌 다른 사람과 함께 모험을 즐길 수 있게 도와주기 위해 노력하는 등장인물이다.")
gen_post = tokenize("그때 한 소년이 창밖을 내다보고는 놀라서 주위를 둘러보았는데 그 모습을 본 소녀가 자신이 여기서 살아 돌아온 게 틀림없다고 말했다.")

# BLEU & METEOR
smooth = SmoothingFunction().method1
bleu = [sentence_bleu([ref], gen_pre, smoothing_function=smooth),
        sentence_bleu([ref], gen_post, smoothing_function=smooth)]
meteor = [single_meteor_score(ref, gen_pre),
          single_meteor_score(ref, gen_post)]

# CIDEr
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tfidf_vecs(sents, n):
    df = Counter(ng for s in sents for ng in set(get_ngrams(s, n)))
    idf = {ng: math.log((len(sents)+1) / (df[ng]+1)) + 1 for ng in df}
    return [{ng: Counter(get_ngrams(s, n))[ng] * idf[ng] for ng in get_ngrams(s, n)} for s in sents]

def cosine_sim(v1, v2):
    keys = v1.keys() & v2.keys()
    dot = sum(v1[k]*v2[k] for k in keys)
    norm = lambda v: math.sqrt(sum(x**2 for x in v.values()))
    return dot / (norm(v1) * norm(v2)) if norm(v1) and norm(v2) else 0

def cider_score(gen, refs, max_n=4):
    return np.mean([
        np.mean([cosine_sim(tfidf_vecs(refs + [gen], n)[-1], v) for v in tfidf_vecs(refs, n)])
        for n in range(1, max_n+1)
    ])

cider = [cider_score(gen_pre, [ref]), cider_score(gen_post, [ref])]

# 결과 출력
metrics = ['BLEU', 'METEOR', 'CIDEr']
before = [bleu[0], meteor[0], cider[0]]
after = [bleu[1], meteor[1], cider[1]]

for m, b, a in zip(metrics, before, after):
    print(f"{m}: Before={b:.4f}, After={a:.4f}")

# 시각화
x = np.arange(len(metrics))
plt.figure(figsize=(8,5))
plt.bar(x, before, width=0.35, label='Before', color='skyblue')
plt.bar(x + 0.35, after, width=0.35, label='After', color='salmon')
plt.xticks(x + 0.175, metrics)
plt.ylabel("Score")
plt.title("Text Generation Evaluation")
plt.ylim(0, max(before + after) + 0.1)
plt.legend()
plt.tight_layout()
plt.show()
