import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import BaseModel, Field

# --- LLM backend (OpenAI-compatible) ---
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional
API_KEY = os.getenv("OPENAI_API_KEY")


def get_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install requirements.txt")
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    if BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return OpenAI(api_key=API_KEY)


# --- Personas ---
PERSONA_SYSTEM_PROMPTS: Dict[str, str] = {
    "Designer": (
        "You are the Designer for biomaterials: balance performance, sustainability, and aesthetics. "
        "Ask for intended application and constraints (mechanical, thermal, tactile, biodegradability). "
        "Propose candidate materials or recipes, tests, and realistic trade-offs. "
        "Prefer concise, structured answers."
    ),
    "Farmer": (
        "You are the Farmer (bioprocess operator) for biomaterials: focus on yield, quality, time-to-harvest, and bottlenecks. "
        "Request baseline SOP parameters (substrate, spawn rate, temperature, humidity, CO2, light cycles). "
        "Suggest safe, incremental process tweaks and monitoring strategies."
    ),
    "CFO": (
        "You are the CFO for biomaterials: focus on unit economics, pricing, COGS, gross margin, and runway. "
        "Request assumptions (batch size, yield%, price, labor, substrate, energy, depreciation). "
        "Provide sensitivity analysis and risks succinctly."
    ),
}


# --- Chart spec returned by the LLM ---
class ChartSpec(BaseModel):
    title: str = Field(default="KPI Projection")
    x: List[float]
    y: List[float]
    x_label: str = Field(default="X")
    y_label: str = Field(default="Y")
    kind: str = Field(default="line")  # line | bar | scatter


# --- Simple KPI simulators ---
def simulate_designer_kpis(n_points: int = 12) -> pd.DataFrame:
    months = np.arange(1, n_points + 1)
    tensile_mpa = 20 + 5 * np.log1p(months) + np.random.normal(0, 0.6, size=n_points)
    biodeg_days = 120 - 2.5 * months + np.random.normal(0, 2.0, size=n_points)
    score = 0.6 * (tensile_mpa / max(tensile_mpa)) + 0.4 * (1 - biodeg_days / max(biodeg_days))
    return pd.DataFrame({"month": months, "tensile_mpa": tensile_mpa, "biodeg_days": biodeg_days, "readiness": score})


def simulate_farmer_kpis(n_points: int = 12) -> pd.DataFrame:
    weeks = np.arange(1, n_points + 1)
    yield_kg = 50 + 8 * np.log1p(weeks) + np.random.normal(0, 0.8, size=n_points)
    defect_rate = np.clip(0.12 - 0.006 * weeks + np.random.normal(0, 0.003, size=n_points), 0, 0.2)
    cycle_days = 18 - 0.3 * weeks + np.random.normal(0, 0.4, size=n_points)
    return pd.DataFrame({"week": weeks, "yield_kg": yield_kg, "defect_rate": defect_rate, "cycle_days": cycle_days})


def simulate_cfo_kpis(n_points: int = 12) -> pd.DataFrame:
    months = np.arange(1, n_points + 1)
    price = 25 + np.sin(months / 2)  # hypothetical market drift
    cogs = 14 - 0.3 * np.log1p(months)
    gm = (price - cogs) / price
    return pd.DataFrame({"month": months, "price": price, "cogs": cogs, "gross_margin": gm})


def try_parse_chart_json(text: str) -> Optional[ChartSpec]:
    try:
        data = json.loads(text)
        return ChartSpec(**data)
    except Exception:
        return None


def render_chart(spec: ChartSpec):
    df = pd.DataFrame({spec.x_label: spec.x, spec.y_label: spec.y})
    if spec.kind == "bar":
        fig = px.bar(df, x=spec.x_label, y=spec.y_label, title=spec.title)
    elif spec.kind == "scatter":
        fig = px.scatter(df, x=spec.x_label, y=spec.y_label, title=spec.title)
    else:
        fig = px.line(df, x=spec.x_label, y=spec.y_label, title=spec.title)
    st.plotly_chart(fig, use_container_width=True)


def call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# --- Streamlit UI ---
st.set_page_config(page_title="AI Designer • Farmer • CFO", layout="wide")
st.title("AI for Biomaterials: Designer • Farmer • CFO")
st.caption("How AI helps you design, grow, and bring biomaterials forward")

with st.sidebar:
    persona = st.selectbox("Persona", list(PERSONA_SYSTEM_PROMPTS.keys()))
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.markdown("Environment: set `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`).")


def init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = {p: [] for p in PERSONA_SYSTEM_PROMPTS.keys()}


init_history()

col_left, col_right = st.columns([2, 1])
with col_right:
    st.subheader("KPI Simulators")
    if persona == "Designer":
        df = simulate_designer_kpis()
        fig = px.line(df, x="month", y=["tensile_mpa", "biodeg_days", "readiness"], title="Designer KPIs")
        st.plotly_chart(fig, use_container_width=True)
    elif persona == "Farmer":
        df = simulate_farmer_kpis()
        fig = px.line(df, x="week", y=["yield_kg", "defect_rate", "cycle_days"], title="Farmer KPIs")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = simulate_cfo_kpis()
        fig = px.line(df, x="month", y=["price", "cogs", "gross_margin"], title="CFO KPIs")
        st.plotly_chart(fig, use_container_width=True)

with col_left:
    st.subheader(f"Chat — {persona}")

    history: List[Dict[str, str]] = st.session_state["history"][persona]
    for msg in history:
        st.chat_message(msg["role"]).markdown(msg["content"])  # type: ignore

    user_input = st.chat_input("Ask a question or share assumptions…")
    if user_input:
        history.append({"role": "user", "content": user_input})

        system_prompt = PERSONA_SYSTEM_PROMPTS[persona]
        messages = [{"role": "system", "content": system_prompt}] + history

        with st.spinner("Thinking…"):
            reply = call_llm(messages, model=model)

        # Try to parse a chart spec if the model returned JSON
        chart_spec = None
        if reply.strip().startswith("{"):
            chart_spec = try_parse_chart_json(reply)

        history.append({"role": "assistant", "content": reply if not chart_spec else "(Rendered chart below)"})
        st.chat_message("assistant").markdown(history[-1]["content"])  # type: ignore

        if chart_spec:
            render_chart(chart_spec)

    st.button("Clear chat", on_click=lambda: st.session_state["history"].update({persona: []}))


