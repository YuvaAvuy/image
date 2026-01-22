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
You are an expert fact-checking AI.

Analyze the following news and give:
1. Detected Language
2. FINAL VERDICT (REAL / FAKE / UNSURE)
3. Short explanation (2–3 lines)
4. How a user can verify it themselves

News:
\"\"\"{news_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You detect fake news."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        return response.text[:4000]
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
                language = detect(news_text)
                result = call_groq(news_text)

                st.subheader("📝 Detected Language")
                st.write(language.upper())

                st.subheader("📊 AI Analysis Result")
                st.success(result)

                st.subheader("✅ How to verify yourself")
                st.markdown("""
- Check official government or reputed news websites  
- Search the headline on Google News  
- Avoid messages with urgent or sensational tone  
- Verify date and source authenticity
""")

            except Exception as e:
                st.error(f"API Error: {e}")
