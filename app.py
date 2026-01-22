import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ===== MULTILINGUAL + IMAGE =====
from langdetect import detect
from PIL import Image
import pytesseract

# ---------- CONFIG ----------
API_KEY = "AIzaSyDlTlRiTmQgLn1l_kGJfSGg-PtfjJQiXSc"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ---------- GEMINI HELPERS ----------
def safe_extract_text(result):
    try:
        candidate = result.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        if isinstance(content, list):
            content = content[0]
        parts = content.get("parts", [{}])
        return parts[0].get("text", "").strip()
    except:
        return ""


def call_gemini(prompt, timeout=30):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        return safe_extract_text(resp.json())
    except Exception as e:
        return f"ERROR_CALLING_GEMINI: {e}"

# ---------- MULTILINGUAL (NO googletrans) ----------
def detect_and_translate(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    if lang != "en" and lang != "unknown":
        prompt = f"""Translate the following text to English.
Do not explain. Do not summarize. Preserve meaning.

Text:
{text}"""
        translated = call_gemini(prompt)
        return translated if translated else text, lang

    return text, lang

# ---------- IMAGE OCR ----------
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return clean_text(text)
    except:
        return None

# ---------- GEMINI CLASSIFICATION ----------
def query_api_classify(text):
    prompt = f"""Classify the following news as REAL or FAKE.
Answer format:
1st line -> REAL or FAKE
2nd+ lines -> Explanation (2–4 sentences).

Text:
{text}"""
    raw = call_gemini(prompt)
    lines = raw.split("\n", 1)
    cls = lines[0].strip().upper() if lines else "UNSURE"
    expl = lines[1].strip() if len(lines) > 1 else raw

    if "REAL" in cls:
        return "REAL", expl
    elif "FAKE" in cls:
        return "FAKE", expl
    else:
        return "UNSURE", expl


def query_api_simple_explain(text, cls):
    prompt = f"""Explain why the news is {cls}.
- One sentence summary
- Exactly 3 short bullet points

Text:
{text}"""
    return call_gemini(prompt)


def query_api_detailed_explain(text, cls):
    prompt = f"""Explain in detail why the news is {cls}.
Mention evidence, sources, and 3 verification steps.

Text:
{text}"""
    return call_gemini(prompt)


def get_true_info(fake_text):
    prompt = f"""The following claim is FAKE.
Give the correct factual information in 1–2 sentences and mention a reliable source.

Fake claim:
{fake_text}"""
    return call_gemini(prompt)

# ---------- LOCAL MODELS ----------
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "omykhailiv/bert-fake-news-recognition"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "omykhailiv/bert-fake-news-recognition"
    )
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")


bert_pipeline = load_bert_model()
roberta_pipeline = load_roberta_model()


def local_model_signals(text):
    short_text = text[:512]

    try:
        bert_res = bert_pipeline(short_text)[0]
        bert_label = "REAL" if "REAL" in bert_res["label"].upper() else "FAKE"
        bert_score = round(float(bert_res["score"]), 3)
    except:
        bert_label, bert_score = "ERROR", 0.0

    try:
        rob = roberta_pipeline(short_text, candidate_labels=["REAL", "FAKE"])
        rob_label = rob["labels"][0]
        rob_score = round(float(rob["scores"][0]), 3)
    except:
        rob_label, rob_score = "ERROR", 0.0

    return {
        "bert_label": bert_label,
        "bert_score": bert_score,
        "roberta_label": rob_label,
        "roberta_score": rob_score,
    }

# ---------- UTILITIES ----------
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|read more)", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        paragraphs = [
            p.get_text() for p in soup.find_all("p") if len(p.get_text().split()) > 5
        ]
        text = " ".join(paragraphs)
        return clean_text((title + "\n" + text)[:4000])
    except:
        return None


trusted_sources = {
    "bbc.com": "BBC",
    "reuters.com": "Reuters",
    "thehindu.com": "The Hindu",
    "ndtv.com": "NDTV",
}


def get_source_name(url):
    for d, n in trusted_sources.items():
        if d in url.lower():
            return n
    return None

# ---------- FINAL DECISION ----------
def final_decision(text, url=""):
    text = clean_text(text)
    translated_text, lang = detect_and_translate(text)

    if url:
        src = get_source_name(url)
        if src:
            msg = f"Article from trusted source: {src}"
            return "REAL", msg, msg, msg, lang

    cls, expl = query_api_classify(translated_text)
    simple = query_api_simple_explain(translated_text, cls)
    detailed = query_api_detailed_explain(translated_text, cls)
    return cls, expl, simple, detailed, lang

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Multilingual Fake News Detection")

input_type = st.radio("Choose input type", ["Text", "URL", "Image"])

user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text", height=200)

elif input_type == "URL":
    page_url = st.text_input("Enter article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=300)
            user_input = scraped

elif input_type == "Image":
    img = st.file_uploader("Upload news image", type=["png", "jpg", "jpeg"])
    if img:
        image = Image.open(img)
        st.image(image, use_column_width=True)
        extracted = extract_text_from_image(image)
        if extracted:
            st.text_area("Extracted Text from Image", extracted, height=300)
            user_input = extracted

if st.button("Analyze"):
    if not user_input:
        st.warning("Please provide input.")
    else:
        signals = local_model_signals(user_input)
        result, explanation, simple, detailed, lang = final_decision(user_input, page_url)

        st.subheader(f"Detected Language: {lang.upper()}")

        if result == "REAL":
            st.success("🟢 REAL NEWS")
        elif result == "FAKE":
            st.error("🔴 FAKE NEWS")
        else:
            st.warning("⚠️ UNSURE")

        st.markdown("### Simple Explanation")
        st.info(simple)

        st.markdown("### Detailed Explanation")
        st.write(explanation)

        st.markdown("### Local Model Signals")
        st.write(f"BERT: {signals['bert_label']} ({signals['bert_score']})")
        st.write(f"RoBERTa: {signals['roberta_label']} ({signals['roberta_score']})")

        if result == "FAKE":
            st.markdown("### Correct Information")
            st.info(get_true_info(user_input))
