# app.py (Gradio 버전)
import gradio as gr
import pandas as pd
from utils import predict, text_to_category, find_similar_problems
from sklearn.feature_extraction.text import TfidfVectorizer


# ===============================
# 데이터 로드 및 TF-IDF 사전 계산
# ===============================
def load_data():
    info_data = pd.read_csv("data/수학시험_데이터_with_category.csv")
    df = pd.read_csv("data/tmp.csv", index_col=0)

    tfidf_dict, vectorizer_dict = {}, {}
    for category, subset in df.groupby("math_category"):
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(subset["ocr_text_raw"].fillna(""))
        tfidf_dict[category] = tfidf_matrix
        vectorizer_dict[category] = vectorizer
    return df, tfidf_dict, vectorizer_dict, info_data


# ===============================
# 문제 추천 표시용 마크다운
# ===============================
def format_problem_markdown(problems, info_data):
    markdown = ""
    for i, row in problems.iterrows():
        info = info_data.loc[i, ["answer", "exam_date", "exam_type", "subject", "question_number"]]
        problem_text = (
            row["ocr_text_raw"]
            .replace("\\[", "$$")
            .replace("\\]", "$$")
            .replace("\\(", "$")
            .replace("\\)", "$")
            .replace("(1)", "\n\n(1)")
            .replace("(2)", "\n\n(2)")
            .replace("(3)", "\n\n(3)")
            .replace("(4)", "\n\n(4)")
            .replace("(5)", "\n\n(5)")
        )
        markdown += f"""
### 📘 {info.exam_date} {info.exam_type} [{info.subject}] {info.question_number}번

{problem_text}

<details><summary>정답 보기 🔍</summary>

✅ **정답:** {info.answer}

</details>

---
            """
    return markdown


# ===============================
# 이미지 분석 처리
# ===============================
def analyze_image(image):
    if image is None:
        yield "❌ 이미지를 업로드하세요.", "", ""
        return

    yield "⏳ 이미지를 분석 중입니다... (OCR 수행 중)", "", ""
    label, ocr_text, _ = predict(image)

    result = f"📘 예측된 단원: **{label}**"
    yield result, label, ocr_text


# ===============================
# 텍스트 분석 처리
# ===============================
def analyze_text(text):
    if not text.strip():
        yield "❌ 문제 텍스트를 입력하세요.", "", ""
        return

    yield "⏳ 문제를 분석 중입니다...", "", ""
    label, _ = text_to_category(text)

    result = f"📘 예측된 단원: **{label}**"
    yield result, label, text


# ===============================
# 유사 문제 추천
# ===============================
def recommend_problems(category, text):
    if not category or not text:
        yield "❌ 먼저 문제를 분석하세요."
        return

    yield "⏳ 유사 문제를 탐색 중입니다..."
    problems = find_similar_problems(category, text, df, tfidf_dict, vectorizer_dict)
    markdown = format_problem_markdown(problems, info_data)

    yield markdown


# ===============================
# Gradio 인터페이스
# ===============================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:

    # ✅ 수식 구분 기호 설정
    custom_delimiters = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
    ]

    gr.Markdown(
        """
        ## 📊 수학 문제 분석기 (Math Problem Analyzer)
        이미지 또는 텍스트로 수학 문제를 입력하면 단원을 예측하고 유사한 문제를 추천합니다.
        """
    )

    df, tfidf_dict, vectorizer_dict, info_data = load_data()

    with gr.Tab("📷 이미지로 분석"):
        img_input = gr.Image(type="filepath", label="문제 이미지 업로드")
        analyze_btn1 = gr.Button("유형 분석하기")
        result1 = gr.Markdown()
        pred_category1 = gr.State()
        ocr_text1 = gr.State()

        analyze_btn1.click(
            analyze_image,
            inputs=img_input,
            outputs=[result1, pred_category1, ocr_text1],
        )

        recommend_btn1 = gr.Button("문제 추천받기")

        # ✅ latex_delimiters 옵션을 gr.Markdown에 적용
        recommend_output1 = gr.Markdown(latex_delimiters=custom_delimiters)

        recommend_btn1.click(
            recommend_problems,
            inputs=[pred_category1, ocr_text1],
            outputs=recommend_output1,
        )

    with gr.Tab("✍️ 텍스트로 분석"):
        text_input = gr.Textbox(
            label="문제 텍스트 입력",
            placeholder="여기에 수학 문제를 입력하세요...",
            lines=10,
        )
        analyze_btn2 = gr.Button("유형 분석하기")
        result2 = gr.Markdown()
        pred_category2 = gr.State()
        user_text2 = gr.State()

        analyze_btn2.click(
            analyze_text,
            inputs=text_input,
            outputs=[result2, pred_category2, user_text2],
        )

        recommend_btn2 = gr.Button("문제 추천받기")
        recommend_output2 = gr.Markdown(latex_delimiters=custom_delimiters)

        recommend_btn2.click(
            recommend_problems,
            inputs=[pred_category2, user_text2],
            outputs=recommend_output2,
        )

    gr.Markdown("---")
    gr.Markdown(
        """
        <div style='text-align: center; font-size: 15px; color: #555;'>**📘 develop by skh**</div>
        <div style='text-align: center; font-size: 15px; color: #555;'>
            [github] <a href="https://github.com/sonkeehoon">https://github.com/sonkeehoon</a>
        </div> 
        <div style='text-align: center; font-size: 15px; color: #555;'>
            [blog]   <a href = "https://blog.naver.com/djfkfk12345">https://blog.naver.com/djfkfk12345</a>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
