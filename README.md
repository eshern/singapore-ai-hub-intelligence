# Singapore AI Hub Investment Intelligence
## Multi-Agent Intelligence Framework — Developer & User Guide

> **Scope:** This document covers the full system introduced in `Singapore AI Hub Intelligence.ipynb` and its companion FastAPI deployment (`api.py`).  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Dataset](#3-dataset)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Environment Configuration](#6-environment-configuration)
7. [Quick Start — Jupyter Notebook](#7-quick-start--jupyter-notebook)
8. [FastAPI Deployment](#8-fastapi-deployment)
9. [API Reference](#9-api-reference)
10. [LangSmith Observability](#10-langsmith-observability)
11. [User Guide — Investor Chatbot](#11-user-guide--investor-chatbot)
12. [Extending the System](#12-extending-the-system)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Overview

This project builds a **grounded multi-agent RAG system** that powers an investment intelligence
chatbot. Its central question:

> *As a tech AI company planning a regional AI hub in Asia, why Singapore — and how does it
> compare to neighbouring ASEAN countries?*

The system ingests twenty-one authoritative PDF documents (Singapore government, EDB, CSA Singapore,
and regional research reports), embeds them into domain-specific vector collections, and routes
investor queries to the most appropriate specialist agent(s) before synthesising a single
executive-level answer.

**Key capabilities:**

| Capability | Detail |
|---|---|
| Grounded RAG | All answers cite the source PDFs — no hallucination from training data |
| Multi-agent routing | LLM intent classifier dispatches to 1–5 agents (4 RAG domains + live web search) per query |
| Multi-document synthesis | Specialist outputs are merged into one coherent briefing |
| LangSmith tracing | Full observability: every retrieval, LLM call, and routing decision is traced |
| Dual interface | Gradio chatbot (notebook) + REST API (FastAPI) |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│         INVESTOR QUERY INTERFACE  (Gradio Chatbot  /  FastAPI REST)              │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     │
                                     ▼
          ┌──────────────────────────────────────────────────────────────────┐
          │                   ORCHESTRATOR / ROUTER                          │
          │          (LLM Intent Classifier + Keyword Fallback)              │
          │                selects 1 to 5 agents per query                   │
          └──┬────────────┬─────────────┬─────────────┬────────────────┬─────┘
             │            │             │             │                │
             ▼            ▼             ▼             ▼                ▼
      ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  ┌──────────────┐
      │ SG Policy │ │ Invest.   │ │ Regional  │ │ Cybersec. │  │ Web Search   │
      │   Agent   │ │ Ecosystem │ │   Comp.   │ │ Compliance│  │    Agent     │
      │   (RAG)   │ │   Agent   │ │   Agent   │ │   Agent   │  │              │
      └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘  └───────┬──────┘
            │             │             │             │                │
            ▼             ▼             ▼             ▼                ▼
       ChromaDB       ChromaDB       ChromaDB       ChromaDB       DuckDuckGo
      sg_policy_     invest_eco_   regional_cmp_  cybersec_cmp_  (live internet)
      collection     collection    collection     collection
            │              │              │              │               │
            └──────────────┴──────────────┴──────────────┴───────────────┘
                                          │
                                          ▼  (all agent outputs)
                            ┌─────────────────────────────┐
                            │       SYNTHESIZER NODE       │  <- LangGraph Final Node
                            │   (Executive-Level Answer)   │
                            └─────────────┬───────────────┘
                                          │
                                          ▼
                                 Final investor answer
                           ┌──────────────┴──────────────┐
                           ▼                             ▼
                   Gradio Chatbot             FastAPI JSON Response
```

### LangGraph Node Responsibilities

| Node | Inputs | Outputs |
|---|---|---|
| `router_node` | Raw query | `selected_agents`, `routing_reason` |
| `execute_agents_node` | Query + agent list | `agent_outputs` (per-agent answers + doc counts) |
| `synthesize_node` | Query + all agent outputs | `response` (final answer) |

### Agent–Collection Mapping

| Agent | Source | Knowledge Base |
|---|---|---|
| `sg_policy_agent` | ChromaDB RAG | `sg_policy_collection` — 3 SG policy PDFs |
| `investment_ecosystem_agent` | ChromaDB RAG | `investment_ecosystem_collection` — 3 EDB/workforce PDFs |
| `regional_comparison_agent` | ChromaDB RAG | `regional_comparison_collection` — 9 ASEAN PDFs |
| `cybersecurity_compliance_agent` | ChromaDB RAG | `cybersecurity_compliance_collection` — 6 CSA PDFs |
| `web_search_agent` | Live internet (DuckDuckGo) | Real-time web results — no ChromaDB |
| `general_agent` | LLM only | No retrieval — base model answer |

---

## 3. Dataset

All source documents are in `data_agents/`:

| File | Publisher | Knowledge Domain |
|---|---|---|
| `Singapore AI Opportunity.pdf` | Singapore Govt / IMDA | NAIS 2.0 strategy, AI governance, Smart Nation |
| `CSET-Examining-Singapores-AI-Progress.pdf` | CSET (Georgetown) | Independent assessment of Singapore's AI capabilities and global standing |
| `Model-AI-Governance-Framework-for-Generative-AI-19-June-2024.pdf` | IMDA / PDPC Singapore | Model AI Governance Framework for Generative AI — responsible GenAI deployment guidelines |
| `EDB_Singapore-Tech-Ecosystem.pdf` | EDB Singapore | Tech clusters, industries, R&D ecosystem |
| `EDB_Guide-to-Hiring-Your-Dream-Tech-Team-in-Singapore.pdf` | EDB Singapore | Talent acquisition, workforce, hiring market |
| `Gen-AI_Artificial Intelligence and the Future of Work _sdnea2024001.pdf` | SDNEA / Research | Generative AI impact on workforce, jobs, and skills across SEA |
| `State_of_Ai_SEA_Digital.pdf` | Industry research | AI adoption and state across Southeast Asia |
| `The AI Readiness Barometer-ASEAN AI Landscape.pdf` | Research institute | ASEAN AI readiness rankings and scores |
| `unlocking-southeast-asias-ai-potential.pdf` | Consulting firm | SEA AI growth potential and investment outlook |
| `ASEAN-Guide-on-AI-Governance-and-Ethics_beautified_201223_v2.pdf` | ASEAN | Regional AI governance principles, ethics guidelines across ASEAN |
| `e_conomy_sea_2025_report.pdf` | Google / Temasek / Bain | Digital economy size, growth, and AI trends across Southeast Asia |
| `ERIA-One-ASEAN-Start-up-White-Paper-2024.pdf` | ERIA | ASEAN startup ecosystems, cross-border investment, and policy enablers |
| `Accelerating_SME_AI_Adoption_Through_Open_Source_in_Malaysia_s_Digital_Future_NAIO.pdf` | Ecosystm / NAIO | Malaysia SME AI adoption via open source, digital transformation |
| `THE-NATIONAL-GUIDELINES-ON-AI-GOVERNANCE-ETHICS.pdf` | National govt | National-level AI governance and ethics guidelines across the region |
| `Accelerating-AI-Discussions-in-ASEAN-.pdf` | ASEAN / Policy body | Key AI policy discussions, priorities, and collaboration being accelerated across ASEAN |
| `Cybersecurity_Playbook_for_Large_Language_Model_LLM_Applications.pdf` | CSA Singapore | LLM security, OWASP, compliance for AI deployments |
| `Companion Guide on Securing AI Systems.pdf` | CSA Singapore | Practical guidance for securing AI infrastructure and pipelines |
| `Guidelines on Securing AI Systems.pdf` | CSA Singapore | Security guidelines and controls for AI system deployment |
| `large-language-model-starter-kit.pdf` | CSA Singapore | Safe LLM implementation practices, risks, and mitigations |
| `Securing LLM Backed Systems_ Essential Authorization Practices 20240823.pdf` | CSA Singapore | Authorization and access control practices for LLM-backed systems |
| `Singapore Cyber Landscape 2024_2025.pdf` | CSA Singapore | Singapore cybersecurity threat landscape, incidents, and national posture |

---

## 4. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| pip | ≥ 23 |
| [Groq API key](https://console.groq.com) | Free tier sufficient |
| [LangSmith API key](https://smith.langchain.com) | Free tier sufficient |
| RAM | ≥ 8 GB (embedding model loads into memory) |
| Disk | ≥ 2 GB (ChromaDB + embedding model cache) |

---

## 5. Installation

### pip (recommended for FastAPI deployment)

```bash
# Clone or navigate to the project directory
cd singapore-ai-hub-intelligence

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import langchain, langgraph, chromadb, fastapi; print('OK')"
```

---

## 6. Environment Configuration

Create a `.env` file in the project root (same folder as `api.py`):

```env
# ── Required ──────────────────────────────────────────────────────────────────
GROQ_API_KEY=gsk_...your_groq_key...

# ── LangSmith observability (add these three lines to enable tracing) ─────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...your_langsmith_key...
LANGCHAIN_PROJECT=sg-ai-hub-intelligence

# ── Optional: other models / services ─────────────────────────────────────────
HF_TOKEN=hf_...your_huggingface_token...
GEMINI_API_KEY=...
OPENAI_API_KEY=...

# ── Optional: FastAPI admin endpoint protection ────────────────────────────────
ADMIN_API_KEY=choose_a_strong_secret_here
```

> **Security note:** Never commit `.env` to version control. It is already listed in `.gitignore`.

### What LangSmith captures automatically

Once `LANGCHAIN_TRACING_V2=true` is set, **every** LangChain and LangGraph call is traced with
zero code changes:

- Router LLM prompt + agent selection decision
- Per-agent ChromaDB similarity search queries and retrieved chunks
- Synthesizer prompt and final LLM output
- End-to-end latency and token usage per Groq call

View traces at [smith.langchain.com](https://smith.langchain.com) → project `sg-ai-hub-intelligence`.

---

## 7. Quick Start — Jupyter Notebook

```bash
# Activate your environment
conda activate genai_local   # or source .venv/bin/activate

# Launch Jupyter
jupyter lab
```

Open `Singapore AI Hub Intelligence.ipynb` and **run cells top-to-bottom**:

| Cell | Action | Expected output |
|---|---|---|
| 3 | Install deps (if needed) | `Dependencies ready.` |
| 4 | Imports + config | `Environment configured.` |
| 6 | Define PDF→collection map | Prints mapping table |
| 7 | `build_knowledge_base()` | Ingests 21 PDFs on first run, shows chunk counts; skips on subsequent runs |
| 9 | Define agents | `Agent tools and routing functions defined.` |
| 11–15 | Build LangGraph | `Singapore AI Hub Intelligence graph compiled.` |
| 17 | Define investor queries | 8 sample queries ready |
| 18 | Run flagship query | Full investor briefing printed |
| 21 | Launch Gradio | Browser tab opens at `http://127.0.0.1:7860` |

> **First run:** Cell 7 downloads the `all-mpnet-base-v2` embedding model (~420 MB) and
> processes all 21 PDFs. This takes 15–25 minutes. Subsequent runs skip ingestion automatically.

---

## 8. FastAPI Deployment

### Development server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag restarts the server on code changes (do not use in production).

### Production server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

> **Note:** `--workers > 1` requires the ChromaDB path to be on a shared or network filesystem,
> or use a single worker with async handling. For a single-server deployment, `--workers 1` is safe.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sg-ai-hub .
docker run --env-file .env -p 8000:8000 -v $(pwd)/chroma_db_agents:/app/chroma_db_agents sg-ai-hub
```

### Startup sequence

On startup `api.py` automatically:
1. Loads all environment variables
2. Initialises `ChatGroq`, `HuggingFaceEmbeddings`, and `chromadb.PersistentClient`
3. Calls `build_knowledge_base(force_rebuild=False)` — skips ingestion if collections exist
4. Compiles the LangGraph workflow
5. Returns `Ready.` — API is now accepting requests

---

## 9. API Reference

Interactive docs available at `http://localhost:8000/docs` (Swagger UI) after server start.

### `GET /`
Health check.

```bash
curl http://localhost:8000/
```
```json
{"status": "ok", "service": "Singapore AI Hub Intelligence API", "version": "1.0.0"}
```

---

### `POST /query`
Submit an investor question. Returns a structured JSON response.

**Request body:**
```json
{
  "question": "Why should I set up my AI company in Singapore over Malaysia?",
  "include_debug": false
}
```

**Response:**
```json
{
  "question": "Why should I set up my AI company in Singapore over Malaysia?",
  "answer": "Based on the source documents...",
  "agents_used": ["investment_ecosystem_agent", "regional_comparison_agent"],
  "routing_reason": "Comparison + investment query triggers two agents",
  "agent_outputs": {
    "investment_ecosystem_agent": "...",
    "regional_comparison_agent": "..."
  },
  "debug_log": null,
  "latency_ms": 4821.3
}
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is NAIS 2.0?", "include_debug": true}'
```

---

### `POST /query/stream`
Same as `/query` but streams the answer as Server-Sent Events (SSE).

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare Singapore and Vietnam for AI investment."}'
```

SSE event types: `agents` → `token` (repeated) → `done`

---

### `GET /agents`
List all available agents and their descriptions.

```bash
curl http://localhost:8000/agents
```

---

### `GET /collections`
List ChromaDB collections and document counts.

```bash
curl http://localhost:8000/collections
```

---

### `POST /rebuild-kb`
Force re-ingestion of all PDFs (drops and rebuilds all collections).  
Protected by `X-Admin-Key` header if `ADMIN_API_KEY` is set in `.env`.

```bash
curl -X POST http://localhost:8000/rebuild-kb \
  -H "X-Admin-Key: your_admin_key"
```

---

## 10. LangSmith Observability

LangSmith provides full tracing of every agent run with **no code changes required**
— only the three `.env` variables described in Section 6.

### Accessing the LangSmith UI

1. Go to [smith.langchain.com](https://smith.langchain.com) and sign in
2. In the left sidebar click **Projects** → select **`sg-ai-hub-intelligence`**
3. Each row in the runs table is one `app.invoke()` call (one user query end-to-end)
4. Click any run to expand the **trace tree** showing all three nodes

### What you see in the trace tree

```
Run: LangGraph  (total latency shown here)
  │
  ├── router_node                            ← click to inspect routing decision
  │     LLM call: intent classification
  │       Input:  router_prompt + user query
  │       Output: {"agents": ["regional_comparison_agent",
  │                            "investment_ecosystem_agent"], ...}
  │       Tokens: 312 in / 28 out  |  Latency: 0.8 s
  │
  ├── execute_agents_node                    ← click to inspect retrieval + agent answers
  │     ├── ChromaDB similarity_search (regional_comparison_agent, k=5)
  │     │     Query:     "Singapore vs Malaysia AI investment"
  │     │     Retrieved: 5 chunks from regional_comparison_collection
  │     ├── LLM call: regional_comparison_agent answer
  │     │     Tokens: 2 140 in / 380 out  |  Latency: 2.1 s
  │     ├── ChromaDB similarity_search (investment_ecosystem_agent, k=5)
  │     └── LLM call: investment_ecosystem_agent answer
  │           Tokens: 1 890 in / 290 out  |  Latency: 1.8 s
  │
  │   (for web_search_agent queries)
  │     └── DuckDuckGoSearchRun              ← shows raw search results string
  │           LLM call: web_search_agent synthesis
  │           Tokens: 3 100 in / 420 out  |  Latency: 3.4 s
  │
  └── synthesize_node                        ← click to see the final merge prompt
        Input:  all agent outputs + user query
        Output: synthesised executive briefing
        Tokens: 1 842 in / 412 out  |  Latency: 3.2 s
```

### Typical debugging workflow

| Problem | LangSmith trace to inspect | What to look for |
|---|---|---|
| Wrong agent selected | `router_node` → LLM call output | Is the JSON valid? Did `web_search_agent` get included for a "latest" query? |
| RAG answer not grounded in PDF | `execute_agents_node` → ChromaDB call | Are retrieved chunks relevant? Try rephrasing the question |
| Web search returned no results | `execute_agents_node` → DuckDuckGoSearchRun | Check the raw results string — may be empty or rate-limited |
| Synthesizer ignores agent output | `synthesize_node` → LLM input | Check `specialist_reports` block — is the agent output present? |
| High latency | Top-level run waterfall | Expand each node; the LLM call with most tokens is usually the bottleneck |
| JSON parse failure (keyword fallback used) | `router_node` → LLM output | The raw LLM string — look for markdown fences or extra text around the JSON |

---

## 11. User Guide — Investor Chatbot

### Starting the Gradio interface (notebook)

Run all cells in `Singapore AI Hub Intelligence.ipynb`, then navigate to the URL
printed in Cell 21 (typically `http://127.0.0.1:7860`).

### Suggested questions to try

| Category | Sample Question |
|---|---|
| Policy | *What is latest Singapore's NAIS (National Artificial Intelligence Strategy) and its key pillars?* |
| Investment | *What government incentives and EDB programs does Singapore offer to AI companies?* |
| Comparison | *How does Singapore compare to Malaysia and Thailand on AI readiness?* |
| Cybersecurity | *What LLM compliance and security frameworks apply to AI deployments in Singapore?* |
| Comprehensive | *As a tech AI company, why should I choose Singapore over Indonesia or Vietnam as my regional hub?* |
| Talent | *What is the tech talent landscape in Singapore vs other ASEAN countries?* |
| Market opportunity | *What are the major AI growth opportunities across Southeast Asia?* |
| Executive summary | *Give me a full executive briefing on Singapore's AI competitive advantages vs all ASEAN nations.* |
| AI Security | *What does the Companion Guide on Securing AI Systems recommend for protecting AI infrastructure?* |
| AI Security | *What are the key guidelines for securing AI systems against adversarial threats?* |
| LLM Implementation | *What does the LLM Starter Kit recommend for safely implementing large language models?* |
| LLM Authorization | *What essential authorization practices should be applied to LLM-backed systems?* |
| ASEAN Governance | *How does the ASEAN Guide on AI Governance and Ethics shape responsible AI deployment across the region?* |
| Singapore AI Progress | *How does the CSET assessment evaluate Singapore's AI capabilities and global ranking?* |
| Digital Economy | *What does the e-Conomy SEA 2025 report reveal about digital economy growth and AI opportunities across Southeast Asia?* |
| Startup Ecosystem | *What does the ERIA One ASEAN Startup White Paper say about startup ecosystems and cross-border investment opportunities in ASEAN?* |
| Malaysia / Open Source | *How is Malaysia accelerating SME AI adoption through open source, and what does this mean for regional competitiveness?* |
| Future of Work | *What are the key impacts of Generative AI on the future of work in Southeast Asia, and how should companies prepare their workforce?* |
| Cyber Landscape | *What does the Singapore Cyber Landscape 2024/2025 report reveal about key threats and Singapore's national cybersecurity posture?* |
| National AI Ethics | *What national AI governance and ethics guidelines are shaping AI deployment standards across the ASEAN region?* |
| GenAI Governance | *What does Singapore's Model AI Governance Framework for Generative AI recommend for responsible GenAI deployment?* |
| ASEAN AI Policy | *What are the key AI policy discussions and priorities being accelerated across ASEAN member states?* |
| Live News | *What are the latest AI investment announcements and funding rounds in Singapore in 2025/2026?* |
| Live News | *What recent AI policy updates or new regulations has Singapore announced this year?* |

### Understanding the response

Each answer is grounded **only** in the twenty-one source PDFs. The chatbot will explicitly state
*"This specific information is not available in my knowledge base"* rather than hallucinate.

The expandable **Orchestration Details** section in each chat response shows:
- Which agents were activated and why
- How many document chunks were retrieved per agent
- The full debug log of the LangGraph run

---

## 12. Extending the System

### Add a new knowledge domain

1. Place new PDF(s) in `data_agents/`
2. Add entries to `PDF_COLLECTION_MAP` in both `Singapore AI Hub Intelligence.ipynb` (Cell 6) and `api.py`
3. Add a new agent entry to `AGENT_TO_COLLECTION` and `AGENT_PERSONA`
4. Update `VALID_AGENTS` and `AGENT_DESCRIPTIONS`
5. Add keywords to `keyword_fallback_route()`
6. Call `build_knowledge_base(force_rebuild=True)` or `POST /rebuild-kb`

### Switch the LLM

Change `LLM_MODEL` in Cell 4 / `api.py`. Groq-supported models include:

| Model | Speed | Context | Notes |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Fast | 128k | Default — best balance |
| `llama-3.1-8b-instant` | Very fast | 128k | Lower quality, good for testing |
| `mixtral-8x7b-32768` | Fast | 32k | Alternative |

### Enable conditional deep-agent loops

For multi-hop ASEAN comparison queries, add a re-retrieval step in `synthesize_node`:
if the synthesized answer contains `"not available in my knowledge base"`, re-invoke
`execute_agents_node` with a rephrased query before producing the final answer.

### Swap the vector database

Replace `langchain-chroma` with `langchain-pinecone` or `langchain-qdrant`.
The agent executor (`run_grounded_agent`) only calls `.similarity_search(query, k=5)` — 
swap the `Chroma(...)` constructor call and the rest of the code is unchanged.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY missing` | Check `.env` file exists and key is spelled correctly |
| `chromadb.errors.InvalidCollectionException` | Run `build_knowledge_base(force_rebuild=True)` to rebuild |
| PDF not found warning during ingestion | Verify filename in `data_agents/` matches exactly (spaces, case) |
| Embedding model download fails | Set `HF_TOKEN` in `.env`; or run `huggingface-cli login` |
| `503 System not ready` from API | Server is still loading the embedding model on startup; wait 30–60 s |
| Gradio `OSError: Port already in use` | Kill the existing Gradio process or change port: `demo.launch(server_port=7861)` |
| LangSmith traces not appearing | Confirm `LANGCHAIN_TRACING_V2=true` (lowercase, no quotes) in `.env`; if using `= 'True'` format, change to `LANGCHAIN_TRACING_V2=true` |
| Slow first query (30–60 s) | Normal — embedding model is warming up; subsequent queries are 3–8 s |

