import streamlit as st
import google.generativeai as genai
import json
import time

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.0-flash")

# ================= GEMINI FUNCTION =================
def gemini_reasoning(news_text):
    prompt = f"""
You are an expert fake news detection system.

Analyze the following news content and decide whether it is REAL, FAKE, or UNSURE.

Rules:
- If the claim is extreme, urgent, or lacks official sources, mark FAKE.
- If information cannot be verified, mark UNSURE.
- Be conservative.

Return ONLY valid JSON in this format:
{{
  "verdict": "REAL | FAKE | UNSURE",
  "simple_explanation": "short explanation for common people",
  "detailed_explanation": "technical reasoning explanation"
}}

News content:
\"\"\"{news_text}\"\"\"
"""

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)

    except Exception as e:
        if "429" in str(e):
            return {
                "verdict": "UNSURE",
                "simple_explanation": "AI service is temporarily busy. Please try again.",
                "detailed_explanation": "Gemini API rate limit reached (HTTP 429)."
            }
        else:
            return {
                "verdict": "ERROR",
                "simple_explanation": "AI analysis failed.",
                "detailed_explanation": str(e)
            }

# ================= STREAMLIT UI =================
st.title("📰 Multilingual Fake News Detection (AI-Based)")

news_text = st.text_area("Enter news text")

if st.button("Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter news text.")
    else:
        with st.spinner("AI is reasoning..."):
            result = gemini_reasoning(news_text)

        st.subheader("Final Verdict")
        if result["verdict"] == "FAKE":
            st.error("🚨 FAKE")
        elif result["verdict"] == "REAL":
            st.success("✅ REAL")
        else:
            st.warning("⚠️ UNSURE")

        st.subheader("Simple Explanation")
        st.write(result["simple_explanation"])

        st.subheader("Detailed Explanation")
        st.write(result["detailed_explanation"])
