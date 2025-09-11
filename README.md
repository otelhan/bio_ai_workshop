AI as Designer, Farmer, and CFO for Biomaterials — Workshop App

Overview
This repo contains a minimal Streamlit app that lets participants chat with three AI personas — Designer, Farmer, and CFO — and generate simple KPI projections and charts using an LLM backend.

Using with Dify (recommended for hosted workshop)
1) Upload or reference these JSON files via GitHub raw URLs:
- prompts/: designer.json, farmer.json, cfo.json
- style/: answer_style.json
- data/: kpis_designer.json, kpis_farmer.json, unit_economics.json
- schemas/: chart_spec.json, image_spec.json

2) In Dify Cloud, choose the "Knowledge Retrieval + Chatbot" template.
- Add Knowledge: your data/*.json (enable citations, top_k 3–5)
- Add variables: persona (Designer/Farmer/CFO), coach_mode (true/false)
- Tools: fetch_json_from_github (HTTP GET), render_chart (QuickChart), generate_image (OpenAI/Stability)
- Answer style: sections = Answer / Evidence / Assumptions / Next Questions

Quickstart
1) Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Set API key (OpenAI-compatible)
```bash
export OPENAI_API_KEY=sk-...
# Optional: use OpenAI-compatible API
# export OPENAI_BASE_URL=https://api.openai.com/v1
# export OPENAI_MODEL=gpt-4o-mini
```

3) Run the app
```bash
streamlit run app.py
```

Features
- Three personas with tailored system prompts
- Chat interface with message history per persona
- Lightweight KPI simulators (materials performance, yield/throughput, unit economics)
- LLM can request charts by returning a small JSON spec that the app renders

Persona Intents
- Designer: balances performance, sustainability, aesthetics; suggests tests and materials choices
- Farmer: scale-up, yield, time-to-harvest, bottleneck analysis; proposes SOP tweaks
- CFO: unit economics, pricing, sensitivity analyses, runway scenarios

Notes
- Keep data hypothetical and non-sensitive. This is an educational tool.
- You can swap to any OpenAI-compatible server by setting OPENAI_BASE_URL and OPENAI_MODEL.


