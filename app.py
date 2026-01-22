import streamlit as st
import requests
import re
import time
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract
import google.generativeai as genai

# ================= CONFIG =================
# HARD-CODED API KEY (FOR DEMO PURPOSE ONLY)
GEMINI_API_KEY = "AIzaSyDlTlRiTmQgLn1l_kGJfSGg-PtfjJQiXSc"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ================= GEMINI HELPERS =================
def call_gemini(prompt, retries=2):
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_error = str(e)
            time.sleep(2)
    return f"ERROR_CALLING_GEMINI: {last_error}"


def translate_to_english(text):
    prompt = f"""
Detect the language of the following text.
If it is not English, translate it to English.
Return ONLY the translated English text.

Text:
{text}
"""
    return call_gemini(prompt)


def classify_news(text):
    prompt = f"""
Classify the following news as REAL or FAKE.

Rules:
- Answer ONLY REAL or FAKE in first line
- Then explain shortly (2–3 sentences)

Text:
{text}
"""
    raw = call_gemini(prompt)
    lines = raw.split("\n", 1)
    verdict = lines[0].strip().upper()
    explanation = lines[1].strip() if len(lines) > 1 else ""
    return verdict, explanation


# ================= LOCAL MODELS =================
@st.cache_resource
def load_bert():
    model = AutoModelForSequenceClassification.from_pretrained(
        "omykhailiv/bert-fake-news-recognition"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "omykhailiv/bert-fake-news-recognition"
    )
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


@st.cache_resource
def load_roberta():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")


bert_pipe = load_bert()
roberta_pipe = load_roberta()


def local_signals(text):
    short = text[:512]

    try:
        b = bert_pipe(short)[0]
        bert_label = "FAKE" if "FAKE" in b["label"].upper() else "REAL"
        bert_score = round(b["score"], 3)
    except:
        bert_label, bert_score = "ERROR", 0.0

    try:
        r = roberta_pipe(short, candidate_labels=["REAL", "FAKE"])
        rob_label = r["labels"][0]
        rob_score = round(r["scores"][0], 3)
    except:
        rob_label, rob_score = "ERROR", 0.0

    return bert_label, bert_score, rob_label, rob_score


# ================= UTILITIES =================
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 30]
        return clean_text(" ".join(paragraphs)[:4000])
    except:
        return ""


def extract_text_from_image(img):
    try:
        return pytesseract.image_to_string(img)
    except:
        return ""


# ================= STREAMLIT UI =================
st.set_page_config(page_title="Multilingual Fake News Detector", layout="wide")
st.title("📰 Multilingual Fake News Detection")

input_type = st.radio("Choose input type", ["Text", "URL", "Image"])

news_text = ""

if input_type == "Text":
    news_text = st.text_area("Enter news text", height=200)

elif input_type == "URL":
    url = st.text_input("Enter news URL")
    if url:
        news_text = scrape_url(url)
        st.text_area("Extracted Article", news_text, height=250)

elif input_type == "Image":
    img_file = st.file_uploader("Upload image with news text", type=["png", "jpg", "jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        news_text = extract_text_from_image(img)
        st.text_area("Extracted Text (OCR)", news_text, height=250)

# ================= ANALYSIS =================
if st.button("Analyze"):
    if not news_text.strip():
        st.warning("Please provide valid input")
    else:
        with st.spinner("Detecting language & translating..."):
            translated = translate_to_english(news_text)

        with st.expander("📝 English Text Used for Analysis"):
            st.write(translated)

        with st.spinner("Classifying news..."):
            verdict, explanation = classify_news(translated)

        bert_l, bert_s, rob_l, rob_s = local_signals(translated)

        if verdict == "REAL":
            st.success("🟢 FINAL VERDICT: REAL")
        elif verdict == "FAKE":
            st.error("🔴 FINAL VERDICT: FAKE")
        else:
            st.warning("⚠️ FINAL VERDICT: UNSURE")

        st.markdown("### 📌 Explanation")
        st.write(explanation)

        st.markdown("### 🧪 Local Model Signals")
        st.write(f"- BERT: {bert_l} ({bert_s})")
        st.write(f"- RoBERTa: {rob_l} ({rob_s})")

        st.markdown("### ✅ How to verify yourself")
        st.markdown("1. Check official government or reputed news sites")
        st.markdown("2. Search same headline in Google News")
        st.markdown("3. Look for sensational or urgent language")
