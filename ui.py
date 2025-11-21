# ui.py - Streamlit UI for CrewAI + Gemini Car Assistant

import json
import streamlit as st
from app import crew, direct_car_lookup_text, llm   # Import from your app.py

st.set_page_config(page_title="Car Recommendation AI", page_icon="🚗", layout="centered")

# ---------------- UI Header ----------------
st.markdown("""
# 🚗 Car Recommendation AI (CrewAI + Gemini 2.0 Flash)
Enter your requirements, and the AI will suggest the best car for you.
""")

# ---------------- User Input ----------------
user_query = st.text_area(
    "Enter your car requirement / model name:",
    placeholder="Example: I want a petrol automatic SUV under 12 lakh for family...",
    height=150
)

run_button = st.button("🔍 Get Recommendation")

# ---------------- Processing ----------------
if run_button:
    if not user_query.strip():
        st.warning("Please enter some text!")
        st.stop()

    with st.spinner("Processing your query... ⏳"):

        # Short queries (< 3 words) -> direct lookup
        if len(user_query.split()) <= 3:
            result = direct_car_lookup_text(user_query)

            st.subheader("📝 Car Details (Direct Lookup)")

            try:
                st.json(json.loads(result))
            except:
                st.code(result)
            st.stop()

        # Otherwise -> run CrewAI workflow
        try:
            result = crew.kickoff(inputs={"query": user_query})

            st.subheader("🤖 CrewAI Output")

            # Try JSON decode (Crew sometimes returns dict or string)
            try:
                st.json(json.loads(result))
            except:
                st.text(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")

            # Fallback extraction
            st.info("Running fallback...")
            try:
                prompt = "Extract JSON requirements from this user query:\n" + user_query + "\nReturn JSON only."
                req = llm.call(prompt)
                st.code(req)
            except Exception as e2:
                st.error(f"Fallback failed: {e2}")
