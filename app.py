# app.py  -- CrewAI v1.5 + Google Gemini 2.0 Flash example
# Requirements:
#   pip install "crewai[google-genai]" google-generativeai python-dotenv requests
# Put GEMINI_API_KEY=... in .env

import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

import os

# Try reading from Streamlit secrets (only works on cloud)
try:
    import streamlit as st
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    # Local development fallback
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing! Add it in Streamlit Secrets or .env")


GEMINI_MODEL = "gemini-2.0-flash"

# Create CrewAI LLM (IMPORTANT → PROMPT MUST BE STRING)
llm = LLM(
    provider="google",
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.2,
    max_tokens=1024
)


# ------------------ Agents ------------------

requirement_agent = Agent(
    role="User Requirements Analyzer",
    goal="Extract clear car requirements from user text and return JSON with budget_min, budget_max, fuel_preference, usage, transmission, seats, car_type, brand_preference, extra_requirements.",
    backstory="You extract structured requirements from user queries.",
    llm=llm,
    verbose=False
)

car_expert = Agent(
    role="Automobile Expert",
    goal="Given requirements, suggest top 5 cars available in India with brief specs and reasons.",
    backstory="You know the Indian car market.",
    llm=llm,
    verbose=False
)

comparison_agent = Agent(
    role="Comparison Specialist",
    goal="Compare the candidate cars and pick the best 3 with pros, cons and score.",
    backstory="You evaluate value-for-money, safety, mileage, reliability.",
    llm=llm,
    verbose=False
)

final_agent = Agent(
    role="Final Recommendation Expert",
    goal="From top 3, give the final best recommendation with buying tips.",
    backstory="You summarise and guide purchases.",
    llm=llm,
    verbose=False
)


# ------------------ Tasks ------------------

task1 = Task(
    description="Parse user text into a JSON requirements object.",
    agent=requirement_agent,
    expected_output="JSON with required fields."
)

task2 = Task(
    description="Given requirements, produce JSON of 5 candidate cars with fields model, price_range_in_inr, mileage, fuel_type, transmission, seating, reason_short.",
    agent=car_expert,
    expected_output="JSON array of 5 cars."
)

task3 = Task(
    description="Compare the 5 candidates and return top 3 with pros, cons and score.",
    agent=comparison_agent,
    expected_output="JSON array of up to 3 objects."
)

task4 = Task(
    description="Select final recommended car and provide reason & buying tips.",
    agent=final_agent,
    expected_output="JSON object."
)

crew = Crew(
    agents=[requirement_agent, car_expert, comparison_agent, final_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True
)


# ------------------ Direct Car Lookup (Fixed Gemini 2.0 Format) ------------------

def direct_car_lookup_text(car_name: str) -> str:
    prompt = (
        f"You are an expert on Indian cars. Provide structured JSON for '{car_name}'.\n"
        "Include: model, manufacturer, price_range_in_inr, mileage, engine_disp_or_power, "
        "fuel_type, transmission, seating, ground_clearance_mms, boot_space_liters, safety_rating_if_known, "
        "popular_trims, key_features, pros, cons.\nReturn ONLY JSON."
    )

    # Prefer CrewAI's LLM call
    try:
        if hasattr(llm, "call"):
            result = llm.call(prompt)
            return result if isinstance(result, str) else json.dumps(result)
    except Exception:
        pass

    # Fallback direct Gemini API 2.0
    try:
        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 512
            }
        }

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract text
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return json.dumps(data)

    except Exception as e:
        return f"Error calling Gemini: {e}"


# ------------------ Main loop ------------------

if __name__ == "__main__":
    print("🚗 CrewAI Car Assistant (Gemini 2.0 Flash). Type 'exit' to quit.")
    while True:
        user_query = input("\nUser: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Short queries = direct lookup
        if len(user_query.split()) <= 3:
            raw = direct_car_lookup_text(user_query)
            print("\n--- Car Details (raw) ---")
            try:
                print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False))
            except:
                print(raw)
            continue

        # Long queries = Crew flow
        try:
            result = crew.kickoff(inputs={"query": user_query})
            print("\nAgent Output:\n")
            print(result)
        except Exception as e:
            print("Crew run error:", e)
            print("\nFALLBACK: Extracting requirements using LLM directly...")

            try:
                prompt = "Extract JSON requirements from this user query:\n" + user_query + "\nReturn JSON only."
                req = llm.call(prompt)
                print("Requirements:", req)
            except Exception as e2:
                print("Fallback error:", e2)

