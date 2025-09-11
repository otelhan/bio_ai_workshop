AI as Designer, Farmer, and CFO for Biomaterials — Workshop Content (Dify)

Overview
This repo contains JSON prompts, style guides, schemas, and sample data to power a Dify-hosted Knowledge Retrieval + Chatbot app. Participants will interact with three AI personas (Designer, Farmer, CFO), see evidence-based answers, and generate charts/images.

Using with Dify (hosted workshop)
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

Features
- Three personas with tailored system prompts
- Evidence-first answers with citations to your JSON data
- Chart/image generation via simple JSON contracts

Persona Intents
- Designer: balances performance, sustainability, aesthetics; suggests tests and materials choices
- Farmer: scale-up, yield, time-to-harvest, bottleneck analysis; proposes SOP tweaks
- CFO: unit economics, pricing, sensitivity analyses, runway scenarios

Notes
- Keep data hypothetical and non-sensitive. This is an educational tool.


