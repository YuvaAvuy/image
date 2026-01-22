import streamlit as st
import requests
import re
import time
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import google.generativeai as genai

# ================= CONFIG =================
# API KEY (Hardcoded for demo/sample)
GEMINI_API_KEY = "AIzaSyAId2g6llR4nDrViTb_Jd_6ZQm155-4v4g"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ================= GEMINI CALL =================
def call_gemini(prompt, retries=2):
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            time.sleep(2)
    return f"ERROR: {err}"

# ================= CORE AI LOGIC =================
def multilingual_fake_news_check(text):
    prompt = f"""
You are a professional fact-checking AI.

Steps you MUST follow internally:
1. Detect the language.
2. Translate to English if needed.
3. Reason carefully.
4. Decide clearly if the news is REAL or FAKE.

Output format (STRICT):
Language: <language name>
Verdict: REAL or FAKE
Explanation: 3–4 clear sentences explaining why.

News:
{text}
"""
    return call_gemini(prompt)

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

def extract_text_from_image(image):
    try:
        return pytesseract.image_to_string(image)
    except:
        return ""

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Multilingual Fake News Detection", layout="wide")
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
    img_file = st.file_uploader("Upload image containing news text", type=["png", "jpg", "jpeg"])
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        news_text = extract_text_from_image(image)
        st.text_area("Extracted Text (OCR)", news_text, height=250)

# ================= ANALYZE =================
if st.button("Analyze"):
    if not news_text.strip():
        st.warning("Please provide valid input")
    else:
        with st.spinner("Analyzing using AI reasoning..."):
            result = multilingual_fake_news_check(news_text)

        if result.startswith("ERROR"):
            st.error("⚠️ Gemini API Error or Rate Limit")
            st.error(result)
        else:
            st.markdown("## 🔍 Analysis Result")
            st.write(result)

            st.markdown("## ✅ How to verify yourself")
            st.markdown("1. Check official government or reputed news websites")
            st.markdown("2. Search the same headline on Google News")
            st.markdown("3. Be cautious of urgent or sensational language")
