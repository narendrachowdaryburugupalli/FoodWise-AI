import os
import re
import requests
from google import genai
from google.genai import types
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --fw-radius: 18px; }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    .main-title {
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -0.045em;
        margin: 0;
    }

    .main-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        opacity: 0.72;
        max-width: 920px;
        margin-top: 0.55rem;
    }

    .hero {
        padding: 1.55rem 1.65rem;
        border: 1px solid rgba(128,128,128,0.16);
        border-radius: 24px;
        background:
            radial-gradient(circle at 90% 10%, rgba(46,204,113,0.13), transparent 35%),
            linear-gradient(135deg, rgba(128,128,128,0.055), rgba(128,128,128,0.015));
        box-shadow: 0 12px 35px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .badge {
        display: inline-flex;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.055);
        font-size: 0.78rem;
        font-weight: 650;
    }

    .section-note {
        padding: 0.85rem 1rem;
        border-left: 4px solid currentColor;
        border-radius: 12px;
        background: rgba(128,128,128,0.055);
        margin: 0.8rem 0 1.1rem;
    }

    div[data-testid="stMetric"] {
        min-height: 108px;
        padding: 0.9rem 1rem;
        border-radius: var(--fw-radius);
        border: 1px solid rgba(128,128,128,0.16);
        background: rgba(128,128,128,0.035);
        box-shadow: 0 5px 18px rgba(0,0,0,0.035);
    }

    div[data-testid="stMetric"] label { font-weight: 650; }
    div[data-testid="stMetricValue"] {
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        padding: 0.35rem;
        border-radius: 14px;
        background: rgba(128,128,128,0.055);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.65rem;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(0,0,0,0.09);
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] > div { border-radius: 11px; }

    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.15);
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.13);
    }

    .sidebar-brand {
        padding: 0.9rem 0.95rem;
        border-radius: 16px;
        background: rgba(128,128,128,0.055);
        border: 1px solid rgba(128,128,128,0.14);
        margin-bottom: 0.8rem;
    }

    .sidebar-brand-title { font-size: 1.1rem; font-weight: 800; }
    .sidebar-brand-sub { font-size: 0.78rem; opacity: 0.68; margin-top: 0.2rem; }

    .workflow-card {
        text-align: center;
        padding: 0.8rem 0.45rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.14);
        background: rgba(128,128,128,0.035);
        min-height: 82px;
    }

    .footer-note {
        text-align: center;
        opacity: 0.58;
        font-size: 0.8rem;
        padding: 1.4rem 0 0.5rem;
    }

    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_TITLE = "🌱 FoodWise AI — Smart Campus Food Sustainability System"

APP_SUBTITLE = (
    "AI-powered food demand forecasting, waste reduction & "
    "sustainable redistribution platform for campuses and institutions."
)

COST_PER_MEAL_INR = 45
CO2_PER_MEAL_KG = 0.4
OPERATIONAL_BUFFER = 0.05

KNOWLEDGE_BASE_DIR = "knowledge_base"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# SAMPLE REDISTRIBUTION PARTNERS
# ============================================================

SAMPLE_BENEFICIARIES = [
    {
        "name": "Campus Volunteer Group",
        "capacity": 15,
        "distance_km": 0.5,
        "type": "Volunteers",
    },
    {
        "name": "Vellore Food Bank",
        "capacity": 50,
        "distance_km": 2.5,
        "type": "NGO",
    },
    {
        "name": "Street Smile Foundation",
        "capacity": 30,
        "distance_km": 3.0,
        "type": "NGO",
    },
    {
        "name": "Hope Community Shelter",
        "capacity": 100,
        "distance_km": 5.2,
        "type": "NGO",
    },
]


# ============================================================
# DATA / MODEL LOADING
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("food_data.csv")


@st.cache_resource
def load_model():
    return joblib.load("food_prediction_model.pkl")


@st.cache_resource
def load_encoder():
    return joblib.load("menu_encoder.pkl")


# ============================================================
# RAG ENGINE
# ============================================================

@st.cache_data
def load_knowledge_documents():
    documents = []

    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        return documents

    for filename in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):
        if not filename.lower().endswith(".txt"):
            continue

        path = os.path.join(KNOWLEDGE_BASE_DIR, filename)

        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            if text:
                documents.append(
                    {
                        "source": filename,
                        "text": text,
                    }
                )
        except OSError:
            continue

    return documents


def split_text(text, chunk_size=900):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current) + len(paragraph) <= chunk_size:
            current += paragraph + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = paragraph + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


@st.cache_data
def build_rag_chunks():
    chunks = []

    for document in load_knowledge_documents():
        for chunk in split_text(document["text"]):
            chunks.append(
                {
                    "source": document["source"],
                    "text": chunk,
                }
            )

    return chunks


@st.cache_resource
def build_rag_vectorizer():
    chunks = build_rag_chunks()

    if not chunks:
        return None, None

    texts = [item["text"] for item in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def retrieve_documents(question, top_k=3):
    chunks = build_rag_chunks()
    vectorizer, matrix = build_rag_vectorizer()

    if not chunks or vectorizer is None or matrix is None:
        return []

    try:
        question_vector = vectorizer.transform([question])
        scores = cosine_similarity(question_vector, matrix)[0]
    except Exception:
        return []

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(scores[index])

        if score <= 0:
            continue

        results.append(
            {
                "source": chunks[index]["source"],
                "text": chunks[index]["text"],
                "score": score,
            }
        )

    return results


def create_rag_context(results):
    if not results:
        return ""

    return "\n\n---\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in results
    )


# ============================================================
# LOCAL LLM
# ============================================================

def get_gemini_api_key():
    """Read the Gemini key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return (key or os.getenv("GEMINI_API_KEY", "")).strip()


def generate_with_gemini(prompt, api_key):
    """Generate a response using Gemini for cloud deployment."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=700,
        ),
    )
    answer = (getattr(response, "text", "") or "").strip()
    if answer:
        return answer
    return "The cloud AI model returned an empty response."


def generate_with_ollama(prompt):
    """Generate a response using the local Ollama model."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("response", "").strip()
    if answer:
        return answer
    return "The local AI model returned an empty response."


def generate_ai_answer(
    question,
    context,
    students,
    predicted_demand,
    food_prepared,
    surplus,
    risk,
):
    prompt = f"""
You are FoodWise AI, a campus food sustainability
decision-support assistant.

Your job is to provide concise, practical and responsible
answers about food demand forecasting, food waste reduction,
sustainability and redistribution.

IMPORTANT NUMERICAL RULES:
- Treat the CURRENT FOODWISE DATA below as authoritative.
- NEVER change, reinterpret, multiply, divide, or invent numerical values.
- NEVER perform new calculations unless explicitly requested by the user.
- If you mention predicted demand, prepared food, surplus, students, cost, CO2, or risk, copy the exact value provided below.
- If a calculation is requested, show the calculation clearly using only the provided values.
- Do not invent operational facts.
- Do not claim that food is safe for redistribution.
- Food safety and redistribution decisions must be made by authorized human staff.
- Clearly distinguish ML predictions from measured or actual waste.
- Use the retrieved knowledge as supporting context, not as a source for changing the current dashboard numbers.

CURRENT FOODWISE DATA:
Students: {students}
Predicted Demand: {predicted_demand} meals
Food Prepared: {food_prepared} meals
Potential Surplus: {surplus} meals
Current Risk: {risk}

RETRIEVED KNOWLEDGE:
{context}

USER QUESTION:
{question}
"""

    gemini_key = get_gemini_api_key()

    # Cloud deployment path: Streamlit secrets -> Gemini.
    if gemini_key:
        try:
            return generate_with_gemini(prompt, gemini_key), True
        except Exception as exc:
            return (
                "The cloud AI service could not generate a response right now. "
                "The RAG retrieval above is still working. "
                f"Technical detail: {type(exc).__name__}",
                False,
            )

    # Local development path: Ollama on localhost.
    try:
        return generate_with_ollama(prompt), True
    except requests.exceptions.ConnectionError:
        return (
            "No cloud AI key is configured and Ollama is not running locally. "
            "The RAG retrieval above is still working. For the deployed app, "
            "add GEMINI_API_KEY to Streamlit Secrets.",
            False,
        )
    except requests.exceptions.Timeout:
        return (
            "The local AI model took too long to respond. Try a shorter question.",
            False,
        )
    except requests.exceptions.RequestException as exc:
        return (
            "The local AI model could not be reached. The RAG retrieval above "
            f"is still working. Technical detail: {type(exc).__name__}",
            False,
        )
    except Exception as exc:
        return (
            f"The AI generation layer encountered an error: {type(exc).__name__}. "
            "The RAG retrieval above is still working.",
            False,
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def get_day_name(day_number):
    return DAY_NAMES[int(day_number) - 1]


def calculate_risk(surplus):
    if surplus > 25:
        return "HIGH", "🔴"
    if surplus > 10:
        return "MEDIUM", "🟠"
    if surplus > 0:
        return "LOW", "🟡"
    return "NO SURPLUS", "🟢"


def recommended_preparation(predicted_demand):
    return max(
        0,
        int(round(predicted_demand * (1 + OPERATIONAL_BUFFER))),
    )


def calculate_mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return 0.0

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def create_prediction_dataframe(
    students,
    event,
    special,
    day_number,
    menu_type,
    encoder,
):
    try:
        menu_encoded = encoder.transform([menu_type])[0]
    except Exception as exc:
        raise ValueError(
            f"Menu encoding failed for '{menu_type}'. "
            f"Available encoder classes may not contain this value."
        ) from exc

    return pd.DataFrame(
        [
            [
                int(students),
                int(event),
                int(special),
                int(day_number),
                menu_encoded,
            ]
        ],
        columns=[
            "No_of_Students",
            "Event",
            "Special",
            "Day_Number",
            "Menu_Encoded",
        ],
    )


def get_redistribution_plan(surplus):
    remaining = int(max(0, surplus))
    plan = []

    for partner in sorted(
        SAMPLE_BENEFICIARIES,
        key=lambda item: item["distance_km"],
    ):
        if remaining <= 0:
            break

        amount = min(
            remaining,
            int(partner["capacity"]),
        )

        if amount > 0:
            plan.append(
                {
                    "Partner": partner["name"],
                    "Type": partner["type"],
                    "Distance": partner["distance_km"],
                    "Capacity": partner["capacity"],
                    "Meals": amount,
                }
            )
            remaining -= amount

    return plan, remaining


# ============================================================
# LOAD APPLICATION FILES
# ============================================================

try:
    df = load_data()
    model = load_model()
    le = load_encoder()
except Exception as exc:
    st.error("FoodWise AI could not load the required project files.")
    st.exception(exc)
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

feature_cols = [
    "No_of_Students",
    "Event",
    "Special",
    "Day_Number",
    "Menu_Encoded",
]

try:
    if "Menu_Encoded" not in df.columns:
        if "Menu_Type" not in df.columns:
            st.error(
                "The dataset needs either 'Menu_Encoded' "
                "or 'Menu_Type' for model prediction."
            )
            st.stop()

        df["Menu_Encoded"] = le.transform(df["Menu_Type"])

    missing_features = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing_features:
        st.error(
            "The dataset is missing required model features: "
            + ", ".join(missing_features)
        )
        st.stop()

except Exception as exc:
    st.error("Dataset preparation failed.")
    st.exception(exc)
    st.stop()


# ============================================================
# SIDEBAR — CURRENT DAY
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🌱 FoodWise AI</div>
        <div class="sidebar-brand-sub">Smart campus food planning</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("⚙️ Planning Inputs")

st.sidebar.markdown(
    "Enter campus meal information to estimate food demand."
)

st.sidebar.divider()

st.sidebar.markdown("### 📅 Today's Inputs")

students = st.sidebar.number_input(
    "👥 Number of Students",
    min_value=50,
    max_value=500,
    value=120,
    step=1,
)

event = st.sidebar.selectbox(
    "🎉 Is there an Event?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

special = st.sidebar.selectbox(
    "🌟 Special Menu Day?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

day_number = st.sidebar.selectbox(
    "📅 Day of Week",
    list(range(1, 8)),
    format_func=lambda x: DAY_NAMES[x - 1],
)

menu_options = ["Veg", "Non-Veg", "Special"]

try:
    encoder_classes = list(le.classes_)
    if all(item in encoder_classes for item in menu_options):
        menu_options = menu_options
    else:
        menu_options = encoder_classes
except Exception:
    pass

menu_type = st.sidebar.selectbox(
    "🍽️ Menu Type",
    menu_options,
)

prepared = st.sidebar.number_input(
    "🍲 Total Food Prepared (Meals)",
    min_value=50,
    max_value=500,
    value=130,
    step=1,
)


# ============================================================
# TODAY'S PREDICTION
# ============================================================

try:
    input_data = create_prediction_dataframe(
        students,
        event,
        special,
        day_number,
        menu_type,
        le,
    )

    predicted = float(model.predict(input_data)[0])
except Exception as exc:
    st.error("Today's prediction could not be generated.")
    st.exception(exc)
    st.stop()

surplus = int(round(prepared - predicted))
risk, risk_icon = calculate_risk(surplus)
today_recommended = recommended_preparation(predicted)


# ============================================================
# SIDEBAR — NEXT DAY
# ============================================================

st.sidebar.divider()
st.sidebar.markdown("### 🔮 Next Day Forecast")

st.sidebar.caption(
    "Estimate tomorrow's meal demand before preparation."
)

next_students = st.sidebar.number_input(
    "👥 Expected Students Tomorrow",
    min_value=50,
    max_value=500,
    value=int(students),
    step=1,
)

next_event = st.sidebar.selectbox(
    "🎉 Event Tomorrow?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    key="next_event",
)

next_special = st.sidebar.selectbox(
    "🌟 Special Menu Tomorrow?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    key="next_special",
)

next_day = st.sidebar.selectbox(
    "📅 Tomorrow's Day",
    list(range(1, 8)),
    format_func=lambda x: DAY_NAMES[x - 1],
    key="next_day",
)

next_menu = st.sidebar.selectbox(
    "🍽️ Tomorrow's Menu",
    menu_options,
    key="next_menu",
)

try:
    next_input = create_prediction_dataframe(
        next_students,
        next_event,
        next_special,
        next_day,
        next_menu,
        le,
    )

    next_predicted = float(
        model.predict(next_input)[0]
    )

    next_recommended = recommended_preparation(
        next_predicted
    )

except Exception as exc:
    st.sidebar.error("Next-day prediction failed.")
    next_predicted = 0.0
    next_recommended = 0
    st.sidebar.exception(exc)

st.sidebar.success(
    f"📊 Tomorrow's predicted demand: "
    f"**{int(round(next_predicted))} meals**"
)

st.sidebar.caption(
    f"Suggested preparation: **{next_recommended} meals** "
    f"(+{int(OPERATIONAL_BUFFER * 100)}% buffer)"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="main-title">{APP_TITLE}</div>
        <div class="main-subtitle">{APP_SUBTITLE}</div>
        <div class="hero-badges">
            <span class="badge">🌍 UN SDG 12</span>
            <span class="badge">🎯 Target 12.3</span>
            <span class="badge">🤖 ML Forecasting</span>
            <span class="badge">🧠 RAG + AI Assistant</span>
            <span class="badge">♻️ Sustainability Analytics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-note">'
    '🎯 <b>Decision-support dashboard:</b> FoodWise combines historical '
    'campus food data, machine-learning demand forecasting, surplus '
    'detection, sustainability analytics and retrieval-augmented AI.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# TOP SUMMARY METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "👥 Students",
    int(students),
    help="Current number of students in the planning scenario.",
)

m2.metric(
    "🤖 Predicted Demand",
    f"{int(round(predicted))} meals",
    help="Machine-learning estimate of food consumption.",
)

m3.metric(
    "🍲 Food Prepared",
    f"{int(prepared)} meals",
    help="Number of meals entered as prepared.",
)

m4.metric(
    "♻️ Potential Surplus",
    f"{max(0, surplus)} meals",
    help="Prepared meals minus predicted demand when positive.",
)

st.markdown("### 🌱 Sustainability Snapshot")

impact_cols = st.columns(4)

impact_cols[0].metric(
    "Waste Risk",
    f"{risk_icon} {risk}",
)

impact_cols[1].metric(
    "Potential Meals Diverted",
    f"{max(0, surplus)} meals",
)

impact_cols[2].metric(
    "Illustrative Food Value",
    f"₹{max(0, surplus) * COST_PER_MEAL_INR:,}",
)

impact_cols[3].metric(
    "Illustrative CO₂e",
    f"{max(0, surplus) * CO2_PER_MEAL_KG:.1f} kg",
)

st.caption(
    "Current sustainability figures are illustrative estimates based on "
    f"₹{COST_PER_MEAL_INR}/meal and {CO2_PER_MEAL_KG} kg CO₂e/meal. "
    "They are not measured savings or measured avoided emissions."
)


# ============================================================
# RISK MESSAGE
# ============================================================

if surplus > 25:
    st.error(
        f"🔴 HIGH WASTE RISK — approximately "
        f"**{surplus} meals** may be surplus."
    )
elif surplus > 10:
    st.warning(
        f"🟠 MEDIUM WASTE RISK — approximately "
        f"**{surplus} meals** may be surplus."
    )
elif surplus > 0:
    st.info(
        f"🟡 LOW SURPLUS — approximately "
        f"**{surplus} meals** may remain."
    )
else:
    st.success(
        "🟢 No positive surplus is estimated from the current inputs."
    )


# ============================================================
# QUICK WORKFLOW
# ============================================================

st.markdown("### 🔄 FoodWise Decision Workflow")

w1, w2, w3, w4, w5 = st.columns(5)

w1.markdown('<div class="workflow-card"><b>1️⃣ Input</b><br><small>Students · Menu · Events</small></div>', unsafe_allow_html=True)
w2.markdown('<div class="workflow-card"><b>2️⃣ Predict</b><br><small>ML demand forecast</small></div>', unsafe_allow_html=True)
w3.markdown('<div class="workflow-card"><b>3️⃣ Detect</b><br><small>Potential surplus</small></div>', unsafe_allow_html=True)
w4.markdown('<div class="workflow-card"><b>4️⃣ Retrieve</b><br><small>FoodWise knowledge</small></div>', unsafe_allow_html=True)
w5.markdown('<div class="workflow-card"><b>5️⃣ Assist</b><br><small>AI decision support</small></div>', unsafe_allow_html=True)

st.divider()


# ============================================================
# SHARED HISTORICAL REPORT DATA
# ============================================================

# Normalize the dataset once so all analytics/report tabs can use it.
report_df = df.copy()

if "Food_Prepared" in report_df.columns:
    report_df["Prepared"] = pd.to_numeric(
        report_df["Food_Prepared"], errors="coerce"
    ).fillna(0)
elif "Prepared" in report_df.columns:
    report_df["Prepared"] = pd.to_numeric(
        report_df["Prepared"], errors="coerce"
    ).fillna(0)
else:
    report_df["Prepared"] = 0

if "Food_Consumed" in report_df.columns:
    report_df["Consumed"] = pd.to_numeric(
        report_df["Food_Consumed"], errors="coerce"
    ).fillna(0)
elif "Consumed" in report_df.columns:
    report_df["Consumed"] = pd.to_numeric(
        report_df["Consumed"], errors="coerce"
    ).fillna(0)
else:
    report_df["Consumed"] = 0

if "Waste" in report_df.columns:
    report_df["Wasted"] = pd.to_numeric(
        report_df["Waste"], errors="coerce"
    ).fillna(0)
elif "Wasted" in report_df.columns:
    report_df["Wasted"] = pd.to_numeric(
        report_df["Wasted"], errors="coerce"
    ).fillna(
        (report_df["Prepared"] - report_df["Consumed"]).clip(lower=0)
    )
else:
    report_df["Wasted"] = (
        report_df["Prepared"] - report_df["Consumed"]
    ).clip(lower=0)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🎯 Prediction Engine",
        "📊 Waste Analytics",
        "🤖 AI Assistant",
        "🧠 AI Explainability",
        "🏗️ System Architecture",
        "🌍 Impact & Report",
        "🛡️ Trust Center",
    ]
)


# ============================================================

# ============================================================
# DECISION INTELLIGENCE UI HELPERS
# ============================================================

def render_progress_bar(value, maximum, label, caption=""):
    maximum = max(float(maximum), 1.0)
    pct = max(0.0, min(float(value) / maximum * 100.0, 100.0))
    st.markdown(
        f"""
        <div class="progress-wrap">
            <div class="progress-head">
                <span>{label}</span>
                <b>{pct:.0f}%</b>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct:.1f}%"></div>
            </div>
            <div class="progress-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_card(title, value, detail, tone="neutral"):
    st.markdown(
        f"""
        <div class="decision-card {tone}">
            <div class="decision-title">{title}</div>
            <div class="decision-value">{value}</div>
            <div class="decision-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# V6 — DECISION INTELLIGENCE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .hero-panel {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.18);
        background: linear-gradient(
            135deg,
            rgba(46,125,50,.12),
            rgba(128,128,128,.045)
        );
        margin: .4rem 0 1.1rem 0;
    }
    .hero-kicker {
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .72;
        margin-bottom: .3rem;
    }
    .hero-heading {
        font-size: 1.65rem;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: .35rem;
    }
    .hero-copy {
        font-size: .96rem;
        opacity: .78;
        line-height: 1.5;
    }
    .decision-card {
        padding: 1rem 1.05rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,.18);
        background: rgba(128,128,128,.035);
        min-height: 132px;
        margin-bottom: .65rem;
    }
    .decision-card.good {
        border-left: 5px solid #2e7d32;
    }
    .decision-card.warn {
        border-left: 5px solid #ef6c00;
    }
    .decision-card.danger {
        border-left: 5px solid #c62828;
    }
    .decision-card.neutral {
        border-left: 5px solid #607d8b;
    }
    .decision-title {
        font-size: .78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .04em;
        opacity: .72;
    }
    .decision-value {
        font-size: 1.45rem;
        font-weight: 850;
        margin: .2rem 0;
    }
    .decision-detail {
        font-size: .86rem;
        opacity: .72;
        line-height: 1.4;
    }
    .progress-wrap {
        margin: .65rem 0 1rem 0;
    }
    .progress-head {
        display:flex;
        justify-content:space-between;
        font-size:.86rem;
        margin-bottom:.35rem;
    }
    .progress-track {
        height: 10px;
        border-radius: 99px;
        overflow:hidden;
        background: rgba(128,128,128,.16);
    }
    .progress-fill {
        height:100%;
        border-radius:99px;
        background: currentColor;
    }
    .progress-caption {
        font-size:.75rem;
        opacity:.62;
        margin-top:.3rem;
    }
    .action-panel {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(46,125,50,.25);
        background: rgba(46,125,50,.07);
        margin: .6rem 0 1rem 0;
    }
    .action-title {
        font-size: 1.05rem;
        font-weight: 850;
        margin-bottom: .45rem;
    }
    .action-text {
        line-height:1.55;
        opacity:.86;
    }
    .mini-badge {
        display:inline-block;
        padding:.3rem .62rem;
        border-radius:999px;
        border:1px solid rgba(128,128,128,.2);
        margin:.15rem .25rem .15rem 0;
        font-size:.76rem;
        font-weight:700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# TAB 1 — PREDICTION ENGINE
# ============================================================

with tab1:

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Decision Intelligence</div>
            <div class="hero-heading">🎯 Plan the right amount of food</div>
            <div class="hero-copy">
                FoodWise AI converts today's campus conditions into a demand forecast,
                surplus signal, preparation recommendation and sustainability context.
            </div>
            <div style="margin-top:.55rem;">
                <span class="mini-badge">🤖 ML Forecast</span>
                <span class="mini-badge">♻️ Waste Prevention</span>
                <span class="mini-badge">🌱 SDG 12.3</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Primary decision row
    # ------------------------------------------------------------

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("### 📈 Demand vs Preparation")

        max_scale = max(int(prepared), int(round(predicted)), 1)
        render_progress_bar(
            predicted,
            max_scale,
            "Predicted demand",
            f"{int(round(predicted))} meals expected to be consumed",
        )
        render_progress_bar(
            prepared,
            max_scale,
            "Food prepared",
            f"{int(prepared)} meals entered for today's scenario",
        )

        st.markdown(
            f"""
            <div class="action-panel">
                <div class="action-title">💡 What should I do now?</div>
                <div class="action-text">
                    Based on the current ML forecast, the planning recommendation is
                    <b>{int(today_recommended)} meals</b>. This includes the configured
                    <b>{int(OPERATIONAL_BUFFER * 100)}% operational buffer</b>.
                    The current scenario has <b>{max(0, int(surplus))} potential surplus meals</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### 🧭 Decision Status")

        if surplus > 25:
            tone = "danger"
            status_detail = "Review preparation quantity and consider surplus planning."
        elif surplus > 10:
            tone = "warn"
            status_detail = "Potential surplus is present; review before serving."
        elif surplus > 0:
            tone = "neutral"
            status_detail = "Small potential surplus detected."
        else:
            tone = "good"
            status_detail = "No positive surplus estimated from this scenario."

        render_decision_card(
            "Waste risk",
            f"{risk_icon} {risk}",
            status_detail,
            tone,
        )

        render_decision_card(
            "Recommended preparation",
            f"{int(today_recommended)} meals",
            "Forecast-based planning quantity with operational buffer.",
            "good",
        )

    st.divider()

    # ------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------

    st.markdown("### 📊 Current Scenario")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Students",
        int(students),
        help="Current number of students in the planning scenario.",
    )
    c2.metric(
        "🤖 Predicted Demand",
        f"{int(round(predicted))} meals",
        help="Machine-learning estimate of food consumption.",
    )
    c3.metric(
        "🍲 Food Prepared",
        f"{int(prepared)} meals",
        help="Number of meals entered as prepared.",
    )
    c4.metric(
        "♻️ Potential Surplus",
        f"{max(0, int(surplus))} meals",
        help="Prepared meals minus predicted demand when positive.",
    )

    st.markdown("### 🧩 Planning Context")

    context_df = pd.DataFrame(
        {
            "Planning factor": [
                "Day",
                "Menu",
                "Event",
                "Special menu",
            ],
            "Current input": [
                DAY_NAMES[day_number - 1],
                menu_type,
                "Yes" if event == 1 else "No",
                "Yes" if special == 1 else "No",
            ],
        }
    )

    st.dataframe(
        context_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The forecast is a machine-learning estimate. Actual consumption and waste "
        "should be measured operationally and reviewed by authorized staff."
    )

    # ------------------------------------------------------------
    # Sustainability micro-summary
    # ------------------------------------------------------------

    st.markdown("### 🌱 Immediate Sustainability Signal")

    s1, s2, s3 = st.columns(3)

    potential = max(0, int(surplus))
    s1.metric("Potential meals to divert", f"{potential}")
    s2.metric(
        "Illustrative food value",
        f"₹{potential * COST_PER_MEAL_INR:,}",
    )
    s3.metric(
        "Illustrative CO₂e",
        f"{potential * CO2_PER_MEAL_KG:.1f} kg",
    )

    st.caption(
        f"Illustrative assumptions only: ₹{COST_PER_MEAL_INR}/meal and "
        f"{CO2_PER_MEAL_KG} kg CO₂e/meal. These are not measured savings or "
        "verified avoided emissions."
    )

    # ------------------------------------------------------------
    # Next-day planning
    # ------------------------------------------------------------

    st.divider()
    st.markdown("### 🔮 Next-Day Planning")

    n1, n2, n3 = st.columns(3)

    n1.metric(
        "Next-day predicted demand",
        f"{int(round(next_predicted))} meals",
    )
    n2.metric(
        "Next-day recommended preparation",
        f"{int(next_recommended)} meals",
    )
    n3.metric(
        "Planning buffer",
        f"{int(OPERATIONAL_BUFFER * 100)}%",
    )

    st.caption(
        "Use the Next Day Forecast controls in the sidebar to explore a different "
        "planning scenario."
    )

# TAB 2 — WASTE ANALYTICS
# ============================================================

with tab2:

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Sustainability Intelligence</div>
            <div class="hero-heading">📊 Understand where food waste happens</div>
            <div class="hero-copy">
                Explore historical food-use patterns, waste rate, current planning signals
                and illustrative sustainability impact in one view.
            </div>
            <div style="margin-top:.55rem;">
                <span class="mini-badge">📈 Historical Analytics</span>
                <span class="mini-badge">♻️ Waste Prevention</span>
                <span class="mini-badge">💰 Food Value</span>
                <span class="mini-badge">🌍 CO₂e Context</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Historical dataset metrics
    # ------------------------------------------------------------

    total_prepared = int(report_df["Prepared"].sum())
    total_consumed = int(report_df["Consumed"].sum())
    total_wasted = int(report_df["Wasted"].sum())

    historical_waste_rate = (
        (total_wasted / total_prepared) * 100
        if total_prepared > 0
        else 0
    )

    st.markdown("### 📌 Historical Performance")

    h1, h2, h3, h4 = st.columns(4)

    h1.metric("🍲 Meals prepared", f"{total_prepared:,}")
    h2.metric("🍽️ Meals consumed", f"{total_consumed:,}")
    h3.metric("🗑️ Meals wasted", f"{total_wasted:,}")
    h4.metric("📉 Waste rate", f"{historical_waste_rate:.2f}%")

    # ------------------------------------------------------------
    # Visual analytics
    # ------------------------------------------------------------

    st.markdown("### 📈 Food Flow")

    chart_df = report_df.copy()
    chart_df["Day"] = range(1, len(chart_df) + 1)

    flow_df = chart_df.melt(
        id_vars=["Day"],
        value_vars=["Prepared", "Consumed", "Wasted"],
        var_name="Metric",
        value_name="Meals",
    )

    fig_flow = px.line(
        flow_df,
        x="Day",
        y="Meals",
        color="Metric",
        markers=True,
        title="Prepared, Consumed and Wasted Meals",
    )

    fig_flow.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
        hovermode="x unified",
    )

    st.plotly_chart(fig_flow, use_container_width=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### 🗑️ Waste Composition")

        waste_chart = pd.DataFrame(
            {
                "Category": ["Consumed", "Wasted"],
                "Meals": [total_consumed, total_wasted],
            }
        )

        fig_waste = px.pie(
            waste_chart,
            names="Category",
            values="Meals",
            hole=0.58,
            title="Historical Meal Outcome",
        )

        fig_waste.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            showlegend=True,
        )

        st.plotly_chart(fig_waste, use_container_width=True)

    with c2:
        st.markdown("### 📊 Waste Rate by Day")

        rate_df = chart_df.copy()
        rate_df["Waste Rate (%)"] = (
            rate_df["Wasted"] / rate_df["Prepared"].replace(0, np.nan) * 100
        ).fillna(0)

        fig_rate = px.bar(
            rate_df,
            x="Day",
            y="Waste Rate (%)",
            title="Daily Historical Waste Rate",
            labels={"Day": "Recorded day"},
        )

        fig_rate.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
        )

        st.plotly_chart(fig_rate, use_container_width=True)

    # ------------------------------------------------------------
    # Historical impact context
    # ------------------------------------------------------------

    historical_food_value = total_wasted * COST_PER_MEAL_INR
    historical_co2e = total_wasted * CO2_PER_MEAL_KG

    st.markdown("### 🌍 Sustainability Context")

    i1, i2, i3 = st.columns(3)

    i1.metric(
        "♻️ Meals potentially avoidable",
        f"{total_wasted:,}",
        help="Historical wasted-meal count from the project dataset.",
    )

    i2.metric(
        "💰 Illustrative food value",
        f"₹{historical_food_value:,}",
        help="Illustrative calculation using the configured per-meal assumption.",
    )

    i3.metric(
        "🌱 Illustrative CO₂e",
        f"{historical_co2e:,.1f} kg",
        help="Illustrative calculation using the configured per-meal assumption.",
    )

    st.caption(
        f"Illustrative assumptions: ₹{COST_PER_MEAL_INR}/meal and "
        f"{CO2_PER_MEAL_KG} kg CO₂e/meal. These figures are not measured "
        "financial savings or verified avoided emissions."
    )

    # ------------------------------------------------------------
    # Current scenario comparison
    # ------------------------------------------------------------

    st.divider()
    st.markdown("### 🔎 Current Scenario vs Historical Baseline")

    current_surplus = max(0, int(surplus))
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Historical waste rate",
                "Current potential surplus",
                "Current predicted demand",
                "Current food prepared",
            ],
            "Value": [
                f"{historical_waste_rate:.2f}%",
                f"{current_surplus} meals",
                f"{int(round(predicted))} meals",
                f"{int(prepared)} meals",
            ],
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    if current_surplus > total_wasted:
        insight_title = "⚠️ Current planning signal"
        insight_text = (
            "The current scenario shows a potential surplus larger than the "
            "total historical wasted-meal count in the displayed dataset. "
            "Review the preparation quantity and operational context."
        )
    elif current_surplus > 0:
        insight_title = "♻️ Current planning signal"
        insight_text = (
            f"The current scenario indicates approximately {current_surplus} "
            "potential surplus meals. This can be used as an early planning "
            "signal for waste-prevention or redistribution workflows."
        )
    else:
        insight_title = "🟢 Current planning signal"
        insight_text = (
            "The current scenario does not show a positive potential surplus "
            "based on the configured prediction and preparation inputs."
        )

    st.markdown(
        f"""
        <div class="action-panel">
            <div class="action-title">{insight_title}</div>
            <div class="action-text">{insight_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Dataset table
    # ------------------------------------------------------------

    with st.expander("📋 View historical dataset records"):
        display_df = report_df.copy()
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Historical analytics are calculated from the project's recorded dataset. "
        "Potential surplus is a planning estimate derived from the current ML prediction "
        "and entered food-prepared value."
    )

# TAB 3 — AI ASSISTANT + RAG + CLOUD/LOCAL LLM
# ============================================================

with tab3:

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">FoodWise Copilot</div>
            <div class="hero-heading">🤖 Ask FoodWise AI</div>
            <div class="hero-copy">
                Ask questions about food demand, potential surplus, waste reduction,
                redistribution planning, sustainability and SDG 12.3.
                FoodWise retrieves relevant project knowledge before generating an answer.
            </div>
            <div style="margin-top:.55rem;">
                <span class="mini-badge">🧠 RAG</span>
                <span class="mini-badge">💬 Local + Cloud LLM</span>
                <span class="mini-badge">📚 Knowledge Base</span>
                <span class="mini-badge">🛡️ Human Oversight</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 💡 Try a question")

    suggested_questions = [
        "How can FoodWise reduce food waste?",
        "What is SDG 12.3 and how does FoodWise support it?",
        "How much food should we prepare?",
        "What should we do with potential surplus?",
    ]

    qcols = st.columns(4)
    for i, question in enumerate(suggested_questions):
        if qcols[i].button(
            question,
            key=f"copilot_suggestion_{i}",
            use_container_width=True,
        ):
            st.session_state["foodwise_question"] = question

    question = st.text_area(
        "Ask FoodWise AI",
        value=st.session_state.get("foodwise_question", ""),
        placeholder="Example: How can FoodWise reduce potential food surplus?",
        height=90,
        key="foodwise_question_box",
        label_visibility="collapsed",
    )

    ask_col, clear_col = st.columns([1, 5])

    with ask_col:
        ask_clicked = st.button(
            "🚀 Ask FoodWise",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        if st.button(
            "Clear",
            use_container_width=False,
        ):
            st.session_state.pop("foodwise_question", None)
            st.session_state.pop("foodwise_answer", None)
            st.session_state.pop("foodwise_sources", None)
            st.session_state.pop("foodwise_llm_ok", None)
            st.rerun()

    st.divider()

    # Current scenario context shown separately from generated AI text.
    st.markdown("### 📊 Current FoodWise Data")

    data_cols = st.columns(5)
    data_cols[0].metric("Students", int(students))
    data_cols[1].metric("Predicted demand", f"{int(round(predicted))} meals")
    data_cols[2].metric("Food prepared", f"{int(prepared)} meals")
    data_cols[3].metric("Potential surplus", f"{max(0, int(surplus))} meals")
    data_cols[4].metric("Risk", f"{risk_icon} {risk}")

    st.caption(
        "These values are calculated by the FoodWise application. "
        "The AI assistant should explain them, not replace or reinterpret them."
    )

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("🧠 Retrieving FoodWise knowledge and generating an answer..."):
                retrieved = retrieve_documents(question)
                retrieved_context = create_rag_context(retrieved)

                answer, llm_ok = generate_ai_answer(
                    question,
                    retrieved_context,
                    students,
                    predicted,
                    prepared,
                    surplus,
                    risk,
                )

            st.session_state["foodwise_answer"] = answer
            st.session_state["foodwise_sources"] = retrieved
            st.session_state["foodwise_llm_ok"] = llm_ok

    answer = st.session_state.get("foodwise_answer")
    sources = st.session_state.get("foodwise_sources", [])
    llm_ok = st.session_state.get("foodwise_llm_ok", False)

    if answer:
        st.markdown("### 💬 FoodWise Response")

        if llm_ok:
            st.success("🤖 Local AI response generated")
        else:
            st.info("📚 Knowledge-based response available")

        st.markdown(
            f"""
            <div class="action-panel">
                <div class="action-title">🤖 FoodWise AI</div>
                <div class="action-text">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📚 Knowledge Retrieved")

        if sources:
            source_cols = st.columns(min(3, len(sources)))

            for i, source in enumerate(sources):
                # Support common tuple/dict formats used by the retrieval engine.
                if isinstance(source, dict):
                    name = source.get("source", source.get("name", f"Source {i+1}"))
                    score = source.get("score", source.get("similarity", 0))
                    snippet = source.get("text", source.get("content", ""))
                else:
                    name = str(source[0]) if len(source) > 0 else f"Source {i+1}"
                    score = source[1] if len(source) > 1 else 0
                    snippet = source[2] if len(source) > 2 else ""

                try:
                    score_pct = float(score) * 100
                except Exception:
                    score_pct = 0.0

                source_cols[i % len(source_cols)].markdown(
                    f"""
                    <div class="decision-card neutral">
                        <div class="decision-title">SOURCE {i+1}</div>
                        <div class="decision-value" style="font-size:1rem;">
                            {name}
                        </div>
                        <div class="decision-detail">
                            Relevance: <b>{score_pct:.1f}%</b><br>
                            {str(snippet)[:220]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No matching knowledge-base passages were returned.")

        with st.expander("🔎 View retrieved context"):
            if sources:
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**Source {i}**")
                    st.write(source)
                    if i < len(sources):
                        st.divider()
            else:
                st.write("No retrieved passages.")

        st.divider()

        st.markdown("### 🛡️ AI Safety & Transparency")

        safety1, safety2 = st.columns(2)

        with safety1:
            st.info(
                "**AI limitation**\n\n"
                "FoodWise AI provides decision support. It does not make final "
                "food-safety, redistribution or operational decisions."
            )

        with safety2:
            st.info(
                "**Numerical transparency**\n\n"
                "The current scenario values above come from the application. "
                "Generated text should be treated as an explanation of those values, "
                "not as a new source of numerical truth."
            )

# TAB 4 — AI EXPLAINABILITY
# ============================================================

with tab4:

    st.subheader("🧠 AI Explainability")

    st.markdown(
        "Understand model performance, feature importance "
        "and responsible-AI considerations."
    )

    st.divider()

    st.markdown("### 📈 Model Performance")

    try:
        actual = df["Food_Consumed"].values

        predictions = model.predict(
            df[feature_cols]
        )

        mae = mean_absolute_error(
            actual,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions,
            )
        )

        r2 = r2_score(
            actual,
            predictions,
        )

        mape = calculate_mape(
            actual,
            predictions,
        )

        e1, e2, e3, e4 = st.columns(4)

        e1.metric("R² Score", f"{r2:.3f}")
        e2.metric("MAE", f"{mae:.1f} meals")
        e3.metric("RMSE", f"{rmse:.1f} meals")
        e4.metric("MAPE", f"{mape:.1f}%")

        st.warning(
            "⚠️ These metrics are calculated on the dataset loaded "
            "by this application. Unless the model was evaluated "
            "on a separate held-out test set, they should not be "
            "presented as independent test performance."
        )

    except Exception as exc:
        st.error("Unable to calculate model metrics.")
        st.exception(exc)

    st.divider()

    st.markdown("### 🔍 What Influences the Prediction?")

    try:
        importances = model.feature_importances_

        feature_names = [
            "No. of Students",
            "Event Day",
            "Special Menu",
            "Day of Week",
            "Menu Type",
        ]

        if len(importances) == len(feature_names):

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importances,
                }
            ).sort_values(
                "Importance",
                ascending=True,
            )

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importance",
                text=(
                    importance_df["Importance"] * 100
                ).round(1).astype(str) + "%",
            )

            fig_importance.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )

            st.caption(
                "Feature importance indicates how much each input "
                "contributes to the Random Forest prediction process. "
                "It should not be interpreted as causal influence."
            )

        else:
            st.warning(
                "The number of model features does not match "
                "the expected FoodWise feature list."
            )

    except Exception:
        st.warning(
            "Feature importance is not available for this model."
        )

    st.divider()

    st.markdown("### ⚖️ Model Comparison")

    try:
        comparison_df = joblib.load(
            "model_comparison.pkl"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Model selection should be based on appropriate "
            "evaluation metrics and a separate validation/test methodology."
        )

    except Exception:
        st.info(
            "model_comparison.pkl was not available."
        )

    st.divider()

    st.markdown("### 🛡️ Responsible AI Considerations")

    r1, r2, r3, r4 = st.columns(4)

    r1.markdown(
        """
        **🔎 Transparency**

        Explain prediction inputs, assumptions and limitations.
        """
    )

    r2.markdown(
        """
        **⚖️ Fairness**

        Avoid unnecessary sensitive personal information.
        """
    )

    r3.markdown(
        """
        **🔐 Privacy**

        Prefer aggregated operational data over individual student data.
        """
    )

    r4.markdown(
        """
        **👤 Human Oversight**

        Authorized staff make final operational decisions.
        """
    )

    st.divider()

    st.markdown("### 🌍 Sustainability Impact Assumptions")

    st.write(
        f"Illustrative food-value assumption: "
        f"**₹{COST_PER_MEAL_INR} per meal**"
    )

    st.write(
        f"Illustrative emissions factor: "
        f"**{CO2_PER_MEAL_KG} kg CO₂e per meal**"
    )

    st.caption(
        "These are demonstration assumptions, not direct measurements "
        "of actual financial savings or avoided emissions."
    )


# ============================================================
# TAB 6 — IMPACT & REPORT
# ============================================================

with tab6:

    st.subheader("🌍 Sustainability Impact & Report")

    st.markdown(
        "Convert FoodWise data into a clear sustainability summary "
        "for campus planning, project evaluation and demonstration."
    )

    st.divider()

    # ------------------------------------------------------------
    # Historical metrics
    # ------------------------------------------------------------

    total_prepared_report = (
        int(report_df["Food_Prepared"].sum())
        if "Food_Prepared" in report_df.columns
        else 0
    )

    total_consumed_report = (
        int(report_df["Food_Consumed"].sum())
        if "Food_Consumed" in report_df.columns
        else 0
    )

    total_waste_report = (
        int(report_df["Waste"].sum())
        if "Waste" in report_df.columns
        else max(
            0,
            total_prepared_report - total_consumed_report,
        )
    )

    waste_rate_report = (
        (total_waste_report / total_prepared_report) * 100
        if total_prepared_report > 0
        else 0.0
    )

    historical_value_report = (
        total_waste_report * COST_PER_MEAL_INR
    )

    historical_co2_report = (
        total_waste_report * CO2_PER_MEAL_KG
    )

    # ------------------------------------------------------------
    # Report header
    # ------------------------------------------------------------

    st.markdown("### 📋 FoodWise Sustainability Summary")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "📦 Meals Prepared",
        f"{total_prepared_report:,}",
    )

    r2.metric(
        "🍽️ Meals Consumed",
        f"{total_consumed_report:,}",
    )

    r3.metric(
        "♻️ Meals Wasted",
        f"{total_waste_report:,}",
    )

    r4.metric(
        "📉 Waste Rate",
        f"{waste_rate_report:.2f}%",
    )

    st.divider()

    # ------------------------------------------------------------
    # Impact indicators
    # ------------------------------------------------------------

    st.markdown("### 🌱 Potential Sustainability Impact")

    i1, i2, i3 = st.columns(3)

    i1.metric(
        "♻️ Meals Potentially Diverted",
        f"{max(0, surplus)} meals",
    )

    i2.metric(
        "💰 Historical Illustrative Food Value",
        f"₹{historical_value_report:,}",
    )

    i3.metric(
        "🌿 Historical Illustrative CO₂e",
        f"{historical_co2_report:,.1f} kg",
    )

    st.caption(
        "Impact values are illustrative assumptions for demonstration. "
        f"Food value uses ₹{COST_PER_MEAL_INR}/meal and CO₂e uses "
        f"{CO2_PER_MEAL_KG} kg CO₂e/meal. They are not measured "
        "financial savings or measured avoided emissions."
    )

    st.divider()

    # ------------------------------------------------------------
    # Current scenario
    # ------------------------------------------------------------

    st.markdown("### 🎯 Current Planning Scenario")

    current_report = pd.DataFrame(
        {
            "Metric": [
                "Students",
                "Predicted Demand",
                "Food Prepared",
                "Potential Surplus",
                "Waste Risk",
                "Recommended Preparation",
            ],
            "Value": [
                int(students),
                int(round(predicted)),
                int(prepared),
                max(0, int(surplus)),
                f"{risk_icon} {risk}",
                int(today_recommended),
            ],
        }
    )

    st.dataframe(
        current_report,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ------------------------------------------------------------
    # SDG 12.3 alignment
    # ------------------------------------------------------------

    st.markdown("### 🌍 SDG 12 Alignment")

    st.success(
        """
        **UN SDG 12 — Responsible Consumption and Production**

        **Target 12.3:** Reduce food waste and food losses by 2030.

        FoodWise AI contributes to this objective by using demand
        forecasting, surplus detection, waste analytics and
        sustainability reporting to support more informed food
        preparation and redistribution decisions.
        """
    )

    st.divider()

    # ------------------------------------------------------------
    # Downloadable CSV report
    # ------------------------------------------------------------

    st.markdown("### 📥 Download Sustainability Report")

    export_rows = [
        ["Report", "FoodWise AI Sustainability Report"],
        ["Recorded Days", len(report_df)],
        ["Meals Prepared", total_prepared_report],
        ["Meals Consumed", total_consumed_report],
        ["Meals Wasted", total_waste_report],
        ["Historical Waste Rate (%)", round(waste_rate_report, 2)],
        ["Current Students", int(students)],
        ["Current Predicted Demand", int(round(predicted))],
        ["Current Food Prepared", int(prepared)],
        ["Current Potential Surplus", max(0, int(surplus))],
        ["Current Risk", risk],
        ["Recommended Preparation", int(today_recommended)],
        ["Illustrative Food Value per Meal (INR)", COST_PER_MEAL_INR],
        ["Illustrative CO2e per Meal (kg)", CO2_PER_MEAL_KG],
        ["Historical Illustrative Food Value (INR)", historical_value_report],
        ["Historical Illustrative CO2e (kg)", round(historical_co2_report, 2)],
        ["SDG", "UN SDG 12"],
        ["SDG Target", "Target 12.3"],
    ]

    export_df = pd.DataFrame(
        export_rows,
        columns=["Metric", "Value"],
    )

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download FoodWise Sustainability Report (CSV)",
        data=csv_data,
        file_name="FoodWise_Sustainability_Report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "The downloaded report contains historical dataset metrics "
        "and the current planning scenario. It is intended for "
        "project demonstration and analysis."
    )

    st.divider()

    # ------------------------------------------------------------
    # Reporting methodology
    # ------------------------------------------------------------

    st.markdown("### 🧮 Reporting Methodology")

    methodology_df = pd.DataFrame(
        {
            "Indicator": [
                "Waste Rate",
                "Historical Illustrative Food Value",
                "Historical Illustrative CO₂e",
                "Potential Surplus",
            ],
            "Method": [
                "Meals Wasted ÷ Meals Prepared × 100",
                f"Meals Wasted × ₹{COST_PER_MEAL_INR}",
                f"Meals Wasted × {CO2_PER_MEAL_KG} kg CO₂e",
                "Food Prepared − ML Predicted Demand, when positive",
            ],
        }
    )

    st.dataframe(
        methodology_df,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "FoodWise is a decision-support prototype. Actual food safety, "
        "redistribution, financial savings and emissions outcomes require "
        "validated operational data and authorized human review."
    )


# ============================================================
# TAB 5 — SYSTEM ARCHITECTURE
# ============================================================

with tab5:

    st.subheader("🏗️ FoodWise AI System Architecture")

    st.markdown(
        "End-to-end architecture showing how campus data, "
        "machine learning, RAG and generative AI work together."
    )

    st.divider()

    try:
        st.graphviz_chart(
            """
            digraph FoodWise {
                rankdir=LR;

                node [
                    shape=box,
                    style="rounded,filled",
                    fontname="Arial",
                    fontsize=10
                ];

                A [
                    label="1. Campus Data\\nStudents + Menu + Events",
                    fillcolor="#D6EAF8"
                ];

                B [
                    label="2. Feature Engineering\\nDay + Menu + Event",
                    fillcolor="#D5F5E3"
                ];

                C [
                    label="3. ML Prediction Engine\\nRandom Forest",
                    fillcolor="#FEF9E7"
                ];

                D [
                    label="4. Demand Forecast\\nPredicted Consumption",
                    fillcolor="#FCF3CF"
                ];

                E [
                    label="5. Surplus Detection\\nPrepared - Predicted",
                    fillcolor="#FDEBD0"
                ];

                F [
                    label="6. Risk Classification\\nLow / Medium / High",
                    fillcolor="#FDEDEC"
                ];

                G [
                    label="7. Knowledge Base\\nFoodWise Documents",
                    fillcolor="#E8F8F5"
                ];

                H [
                    label="8. RAG Retrieval\\nTF-IDF + Similarity",
                    fillcolor="#E8DAEF"
                ];

                I [
                    label="9. Local LLM\\nQwen 2.5 1.5B",
                    fillcolor="#FADBD8"
                ];

                J [
                    label="10. AI Assistant\\nDecision Support",
                    fillcolor="#D4EFDF"
                ];

                K [
                    label="11. Streamlit Dashboard",
                    fillcolor="#EBDEF0"
                ];

                A -> B;
                B -> C;
                C -> D;
                D -> E;
                E -> F;

                D -> K;
                E -> K;
                F -> K;

                G -> H;
                H -> I;
                D -> I;
                E -> I;
                F -> I;

                I -> J;
                J -> K;
            }
            """,
        )
    except Exception as exc:
        st.warning(
            "Architecture diagram could not be rendered. "
            "The system architecture is still shown below."
        )

    st.divider()

    st.markdown("### 🧩 Core Components")

    architecture_df = pd.DataFrame(
        {
            "Component": [
                "Data Layer",
                "Machine Learning",
                "Prediction",
                "Waste Analytics",
                "Knowledge Base",
                "RAG Retrieval",
                "Local Generative AI",
                "AI Assistant",
                "Redistribution",
                "Responsible AI",
                "Interface",
            ],
            "Purpose": [
                "Historical campus food records",
                "Random Forest demand model",
                "Estimate food consumption",
                "Identify waste and surplus patterns",
                "FoodWise sustainability documents",
                "Retrieve relevant knowledge using TF-IDF",
                "Generate contextual answers with Qwen",
                "Answer operational and sustainability questions",
                "Suggest sample allocation options",
                "Privacy, transparency and human oversight",
                "Streamlit dashboard",
            ],
        }
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("### 📊 Full Dataset Summary")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Recorded Days",
        len(df),
    )

    d2.metric(
        "Meals Prepared",
        f"{int(df['Food_Prepared'].sum()):,}"
        if "Food_Prepared" in df.columns
        else "N/A",
    )

    d3.metric(
        "Meals Consumed",
        f"{int(df['Food_Consumed'].sum()):,}"
        if "Food_Consumed" in df.columns
        else "N/A",
    )

    d4.metric(
        "Meals Wasted",
        f"{int(df['Waste'].sum()):,}"
        if "Waste" in df.columns
        else "N/A",
    )

    st.divider()

    st.markdown("### 🌱 SDG Alignment")

    st.success(
        """
        **UN SDG 12 — Responsible Consumption and Production**

        **Target 12.3:** Reduce food waste and food losses by 2030.

        FoodWise AI supports this objective through demand forecasting,
        waste analytics, surplus detection and redistribution planning.
        """
    )

    st.divider()

    st.markdown("### 🔄 End-to-End Workflow")

    workflow = [
        "1️⃣ Enter campus conditions",
        "2️⃣ AI predicts food demand",
        "3️⃣ Compare predicted demand with food prepared",
        "4️⃣ Estimate potential surplus",
        "5️⃣ Classify waste risk",
        "6️⃣ Provide preparation recommendation",
        "7️⃣ Retrieve relevant sustainability knowledge",
        "8️⃣ Generate an AI-assisted contextual response",
        "9️⃣ Suggest sample redistribution options",
        "🔟 Display sustainability indicators",
        "1️⃣1️⃣ Human staff make final operational decisions",
    ]

    for step in workflow:
        st.write(step)



# ============================================================
# TAB 7 — RESPONSIBLE AI & TRUST CENTER
# ============================================================

with tab7:

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Responsible AI</div>
            <div class="hero-heading">🛡️ FoodWise Trust Center</div>
            <div class="hero-copy">
                FoodWise AI is designed as a decision-support prototype.
                This section makes its data boundaries, AI limitations,
                human oversight and sustainability assumptions explicit.
            </div>
            <div style="margin-top:.55rem;">
                <span class="mini-badge">🔐 Privacy</span>
                <span class="mini-badge">⚖️ Fairness</span>
                <span class="mini-badge">🧠 Explainability</span>
                <span class="mini-badge">👤 Human Oversight</span>
                <span class="mini-badge">🔢 Transparency</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Trust status overview
    st.markdown("### ✅ Responsible AI Checklist")

    trust_items = [
        ("🔐", "Data Privacy", "Use only the minimum data needed for food-demand planning.", "good"),
        ("⚖️", "Fairness & Bias", "Review performance across relevant campus conditions and groups.", "warn"),
        ("🧠", "Explainability", "Show inputs, predictions, assumptions and retrieved knowledge.", "good"),
        ("👤", "Human Oversight", "Authorized staff remain responsible for operational decisions.", "good"),
        ("🍱", "Food Safety", "The system does not determine food-safety eligibility.", "good"),
        ("🔢", "Numerical Integrity", "Application-calculated values are kept separate from generated explanations.", "good"),
    ]

    trust_cols = st.columns(3)

    for i, (icon, title, detail, tone) in enumerate(trust_items):
        with trust_cols[i % 3]:
            render_decision_card(
                f"{icon} {title}",
                "Defined",
                detail,
                tone,
            )

    st.divider()

    # ------------------------------------------------------------
    # Core principles
    # ------------------------------------------------------------

    st.markdown("### 🧭 Core Responsible AI Principles")

    p1, p2 = st.columns(2, gap="large")

    with p1:
        with st.expander("🔐 1. Data Privacy", expanded=True):
            st.markdown(
                """
                **Purpose:** Limit data collection to information relevant to
                demand planning and sustainability analysis.

                - Avoid collecting personally identifiable student information.
                - Prefer aggregate counts such as meal demand or attendance.
                - Do not use the AI assistant to infer sensitive personal attributes.
                - Keep project data focused on the operational use case.
                """
            )

        with st.expander("⚖️ 2. Fairness & Bias"):
            st.markdown(
                """
                **Purpose:** Recognize that historical data can contain patterns
                that do not represent every future situation equally.

                - Review model performance across different event and menu conditions.
                - Check whether unusual days are underrepresented.
                - Treat predictions as estimates rather than guaranteed outcomes.
                - Improve the training dataset as more representative operational
                  data becomes available.
                """
            )

        with st.expander("🧠 3. Explainability"):
            st.markdown(
                """
                **Purpose:** Help users understand why FoodWise produces a planning
                recommendation.

                The application exposes:
                - Current input conditions
                - Predicted demand
                - Food prepared
                - Potential surplus
                - Risk level
                - Recommended preparation
                - Retrieved RAG knowledge
                - Model evaluation information
                """
            )

    with p2:
        with st.expander("👤 4. Human Oversight", expanded=True):
            st.markdown(
                """
                **Purpose:** Keep people responsible for decisions that require
                operational judgment.

                FoodWise AI does **not** independently decide:
                - Whether food is safe to serve or redistribute
                - Which organization should receive food
                - Whether a redistribution action should occur
                - Whether a financial or environmental claim is verified

                Authorized campus or institutional staff make final decisions.
                """
            )

        with st.expander("🍱 5. Food-Safety Boundary"):
            st.markdown(
                """
                **Important:** FoodWise is a planning and decision-support prototype,
                not a food-safety certification system.

                Any real redistribution workflow must follow applicable institutional
                procedures, local requirements, storage controls, handling rules and
                human verification. The AI output must not be treated as proof that
                food is safe for consumption.
                """
            )

        with st.expander("🌍 6. Sustainability Transparency"):
            st.markdown(
                f"""
                Some sustainability values shown by FoodWise are **illustrative
                estimates**, not measured outcomes.

                Current project assumptions include:
                - Illustrative food value: **₹{COST_PER_MEAL_INR} per meal**
                - Illustrative CO₂e factor: **{CO2_PER_MEAL_KG} kg CO₂e per meal**

                These assumptions are useful for demonstrating the calculation,
                but they should be replaced with validated local measurements before
                making real-world impact claims.
                """
            )

    # ------------------------------------------------------------
    # AI / RAG transparency
    # ------------------------------------------------------------

    st.markdown("### 🤖 AI & RAG Transparency")

    ai1, ai2, ai3 = st.columns(3)

    with ai1:
        render_decision_card(
            "Prediction layer",
            "ML model",
            "Estimates meal demand from the project's structured input features.",
            "good",
        )

    with ai2:
        render_decision_card(
            "Knowledge layer",
            "TF-IDF RAG",
            "Retrieves relevant passages from the FoodWise project knowledge base.",
            "good",
        )

    with ai3:
        render_decision_card(
            "Generation layer",
            "Gemini / Ollama",
            f"Uses Gemini in the deployed app when GEMINI_API_KEY is configured; otherwise uses local Ollama: {OLLAMA_MODEL}.",
            "neutral",
        )

    st.info(
        "The RAG layer retrieves project knowledge; the language model generates "
        "natural-language explanations. Retrieved knowledge and application-calculated "
        "numbers should remain distinguishable from generated text."
    )

    # ------------------------------------------------------------
    # Known limitations
    # ------------------------------------------------------------

    st.markdown("### ⚠️ Known Limitations")

    limitations_df = pd.DataFrame(
        {
            "Area": [
                "Training data",
                "Model evaluation",
                "Operational data",
                "Redistribution partners",
                "Sustainability impact",
                "Food safety",
                "LLM generation",
            ],
            "Current limitation": [
                "Project dataset may not represent every campus condition.",
                "Evaluation metrics in the prototype should not be treated as independent real-world validation.",
                "Current inputs are planning/demo values rather than a live institutional data pipeline.",
                "Displayed partners are sample/demo records and are not confirmed availability.",
                "Food-value and CO₂e figures use illustrative assumptions.",
                "The system does not determine whether food is safe to serve or redistribute.",
                "Generated explanations can still contain language-model errors and require review.",
            ],
        }
    )

    st.dataframe(
        limitations_df,
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------
    # Human review workflow
    # ------------------------------------------------------------

    st.markdown("### 👤 Recommended Human Review Workflow")

    review_cols = st.columns(5)

    review_steps = [
        ("1", "Check inputs", "Confirm students, menu, event and prepared meals."),
        ("2", "Review forecast", "Inspect predicted demand and recommendation."),
        ("3", "Inspect surplus", "Review the potential surplus signal."),
        ("4", "Verify reality", "Use actual operational and food-safety information."),
        ("5", "Decide", "Authorized staff choose the appropriate action."),
    ]

    for col, (number, title, detail) in zip(review_cols, review_steps):
        with col:
            st.markdown(
                f"""
                <div class="decision-card neutral">
                    <div class="decision-title">STEP {number}</div>
                    <div class="decision-value" style="font-size:1rem;">{title}</div>
                    <div class="decision-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.success(
        "🌱 **FoodWise principle:** AI should support better decisions, "
        "while people remain accountable for decisions that affect food safety, "
        "redistribution and real-world operations."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">
        🌱 <b>FoodWise AI</b> — Smart Campus Food Sustainability System<br>
        AI-Powered Food Demand Forecasting · Waste Reduction ·
        Sustainable Redistribution<br>
        Aligned with <b>UN SDG 12 — Responsible Consumption and Production</b>
        · Focus: <b>Target 12.3</b><br>
        Random Forest ML · TF-IDF RAG · Qwen 2.5 1.5B ·
        Python · Pandas · Streamlit · Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
