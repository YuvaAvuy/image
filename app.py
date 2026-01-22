import streamlit as st
from groq import Groq
from langdetect import detect
from PIL import Image
import pytesseract
import requests

# ================== CONFIG ==================
GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="Multilingual Fake News Detection",
    layout="centered"
)

st.title("📰 Multilingual Fake News Detection")

# ================== FUNCTIONS ==================

def call_groq(news_text):
    prompt = f"""
You are an expert fake news detection system.

Analyze the following news and respond with:

FINAL VERDICT: REAL / FAKE / UNSURE
Explanation: short and simple (2–3 lines)
Verification Tips: how users can verify it

News:
\"\"\"{news_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a professional fact-checker."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        return r.text[:4000]
    except:
        return ""


# ================== UI ==================

input_type = st.radio(
    "Choose input type",
    ["Text", "URL", "Image"]
)

news_text = ""

if input_type == "Text":
    news_text = st.text_area(
        "Enter news text",
        height=180,
        placeholder="Enter news content here..."
    )

elif input_type == "URL":
    url = st.text_input("Enter news URL")
    if url:
        news_text = extract_text_from_url(url)

elif input_type == "Image":
    uploaded_image = st.file_uploader(
        "Upload news image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        news_text = extract_text_from_image(image)

# ================== PROCESS ==================

if st.button("🔍 Detect Fake News"):

    if not news_text.strip():
        st.warning("Please provide valid input")
    else:
        with st.spinner("Analyzing using AI..."):
            try:
                lang = detect(news_text)

                st.subheader("📝 Detected Language")
                st.write(lang.upper())

                result = call_groq(news_text)

                st.subheader("⚠️ FINAL VERDICT")
                st.success(result)

                st.subheader("✅ How to verify yourself")
                st.markdown("""
- Check official government or reputed news websites  
- Search the same headline on Google News  
- Avoid urgent or sensational messages  
- Verify the date and original source  
""")

            except Exception as e:
                st.error(f"API Error: {e}")
