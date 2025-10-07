# utils.py
from dotenv import load_dotenv
import os
from gradio_client import Client

# .env 파일 불러오기 (환경 변수 설정)
load_dotenv()

# ===============================
# HuggingFace OCR 클라이언트 초기화
# ===============================
client = Client(
    "santakan/Pix2Text-Demo",
    hf_token=os.getenv("HF_TOKEN")
)


# ===============================
# OCR (이미지 → 텍스트)
# ===============================
def image_to_text(image_path: str) -> str:
    """이미지를 입력받아 OCR을 통해 텍스트로 변환"""
    from gradio_client import handle_file

    result = client.predict(
        ["Korean"],       # 언어
        "mfd",            # Detection 모델
        "mfr",            # Recognition 모델
        "text_formula",   # 인식 타입
        512,              # 리사이즈 크기
        handle_file(image_path),
        api_name="/recognize",
    )
    return result[0]


# ===============================
# 모델 로드
# ===============================
def load_model_params(model_path: str):
    """모델 디렉토리에서 토크나이저, 모델, 라벨 매핑 로드"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import json

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    with open(f"{model_path}/label_mapping.json", "r", encoding="utf-8") as f:
        id2label = json.load(f)["id2label"]

    return tokenizer, model, id2label


# ===============================
# 텍스트 분류 (문제 단원 예측)
# ===============================
def text_to_category(text: str):
    """문제 텍스트를 입력받아 단원을 예측"""
    from transformers import pipeline

    tokenizer, model, id2label = load_model_params("saved_model")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    result = classifier(text, top_k=1)[0]
    label_index = int(result["label"].split("_")[1])
    real_label = id2label[str(label_index)]

    return real_label, result["score"]


# ===============================
# 이미지 분류 (OCR + 분류)
# ===============================
def predict(image_path: str):
    """이미지를 입력받아 OCR → 텍스트 분류를 수행"""
    text = image_to_text(image_path)
    label, score = text_to_category(text)
    return label, text, score


# ===============================
# 유사 문제 추천
# ===============================
def find_similar_problems(target_category, user_text, df, tfidf_dict, vectorizer_dict):
    """예측된 단원 내에서 TF-IDF 코사인 유사도 기반 유사 문제 추천"""
    from sklearn.metrics.pairwise import cosine_similarity

    df_filtered = df[df["math_category"] == target_category]
    vectorizer = vectorizer_dict[target_category]
    tfidf_matrix = tfidf_dict[target_category]

    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    return df_filtered.iloc[top_indices][["ocr_text_raw"]]