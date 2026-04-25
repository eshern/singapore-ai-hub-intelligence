# Singapore AI Hub Intelligence — Copilot Handoff Context

> **How to use this file:** When starting a new chat session in this workspace, attach this file.
> It gives GitHub Copilot full context of all decisions, architecture, and in-progress work from
> the previous sessions so you can continue seamlessly.

---

## 1. Project Purpose

Build a **grounded multi-agent RAG + live web search system** to power an investment intelligence
chatbot. Central question:

> *As a tech AI company planning a regional AI hub in Asia, why Singapore — and how does it
> compare to neighbouring ASEAN countries?*

The system ingests 21 authoritative PDFs (Singapore government, EDB, CSA Singapore, ASEAN/ERIA
research), embeds them into domain-specific ChromaDB collections, and routes investor queries to
specialist agents before synthesising a single executive-level answer.

---

## 2. Files in This Folder

| File / Folder | Purpose |
|---|---|
| `Singapore AI Hub Intelligence.ipynb` | **Main notebook** — full multi-agent system (22 cells) |
| `api.py` | FastAPI REST deployment of the same system |
| `README.md` | Full developer guide with PDF sourcing instructions |
| `requirements.txt` | pip dependencies (Python 3.10+) |
| `environment.yml` | Conda environment spec |
| `.env` | API keys — **never commit** (gitignored) |
| `.gitignore` | Excludes `.env`, `data_agents/`, `chroma_db_agents/`, `*.zip` |
| `data_agents/` | 21 source PDFs — **gitignored** (109 MB); must be downloaded manually (see README § 3) |
| `chroma_db_agents/` | ChromaDB vector store — **gitignored** (55 MB); rebuilt by running Cell 7 |

---

## 3. System Architecture

```
Query → Gatekeeper → [BLOCK → polite refusal]
                   → [ALLOW → Router → Executor → Synthesizer → Answer]
```

### LangGraph Nodes

| Node | Function |
|---|---|
| `gatekeeper_node` | LLM topic classifier: ALLOW or BLOCK before any RAG/search work |
| `blocked_response_node` | Polite refusal listing 5 supported domains; zero LLM/ChromaDB calls |
| `router_node` | LLM intent classifier; selects 1–5 agents; keyword fallback on JSON parse failure |
| `execute_agents_node` | Runs each selected agent sequentially; collects grounded answers |
| `synthesize_node` | Merges multi-agent outputs into one executive-level briefing |

### Graph Wiring

```python
workflow.set_entry_point('gatekeeper')
workflow.add_conditional_edges('gatekeeper',
    lambda s: 'router' if s.get('allowed', True) else 'blocked')
workflow.add_edge('blocked',     END)   # short-circuit — no RAG consumed
workflow.add_edge('router',      'executor')
workflow.add_edge('executor',    'synthesizer')
workflow.add_edge('synthesizer', END)
```

### Agents

| Agent | Source | Collection |
|---|---|---|
| `sg_policy_agent` | ChromaDB RAG | `sg_policy_collection` — 3 SG policy PDFs |
| `investment_ecosystem_agent` | ChromaDB RAG | `investment_ecosystem_collection` — 3 EDB PDFs |
| `regional_comparison_agent` | ChromaDB RAG | `regional_comparison_collection` — 9 ASEAN PDFs |
| `cybersecurity_compliance_agent` | ChromaDB RAG | `cybersecurity_compliance_collection` — 6 CSA PDFs |
| `web_search_agent` | DuckDuckGo live search | No ChromaDB — real-time internet |
| `general_agent` | LLM only | No retrieval — base model fallback |

### `InvestorState` TypedDict

```python
class InvestorState(TypedDict):
    query:           str
    allowed:         bool          # set by gatekeeper_node
    selected_agents: List[str]
    routing_reason:  str
    agent_outputs:   Dict[str, str]
    response:        str
    debug_log:       str
```

---

## 4. Notebook Cell Map (22 cells)

| Cell # | Type | Content |
|---|---|---|
| 1 | Markdown | Architecture diagram + data sources table |
| 2 | Markdown | Section 1 — Setup header |
| 3 | Python | pip install comment (includes `ddgs`) |
| 4 | Python | Imports + config + `search_tool = DuckDuckGoSearchRun()` |
| 5 | Markdown | Section 2 — Knowledge base header |
| 6 | Python | `PDF_COLLECTION_MAP` — all 21 PDFs → collection mapping |
| 7 | Python | `build_knowledge_base()` — ingest PDFs into ChromaDB |
| 8 | Markdown | Section 3 — Agent definitions header |
| 9 | Python | `AGENT_TO_COLLECTION`, `AGENT_PERSONA`, `keyword_fallback_route()`, `run_web_search_agent()`, `run_grounded_agent()`, `run_general_agent()` |
| 10 | Markdown | Section 4 — LangGraph orchestration header |
| 11 | Python | `InvestorState` TypedDict (includes `allowed: bool`) |
| 12 | Python | **`gatekeeper_node` + `blocked_response_node`** (new — adapted from LangGraph5/6/7) |
| 13 | Python | `router_node` |
| 14 | Python | `execute_agents_node` |
| 15 | Python | `synthesize_node` |
| 16 | Python | Graph compile — entry: `gatekeeper`, conditional edges ALLOW/BLOCK |
| 17 | Markdown | Section 5 — Investor query testing header |
| 18 | Python | `INVESTOR_QUERIES` list (8 test queries) |
| 19 | Python | `query_hub()` + runs flagship query (`INVESTOR_QUERIES[4]`) |
| 20 | Python | Full suite runner (commented out) + custom query example |
| 21 | Markdown | Section 6 — Gradio chatbot header |
| 22 | Python | Full Gradio UI — Inter + JetBrains Mono fonts, `position: fixed` sticky input bar |

---

## 5. Key Technical Decisions

### Models
```python
EMBEDDING_MODEL = 'all-mpnet-base-v2'          # HuggingFace sentence-transformers
LLM_MODEL = "llama-3.3-70b-versatile"          # Groq — primary (131K ctx, 100K TPD quota)
# Fallbacks if quota exhausted:
# LLM_MODEL = "openai/gpt-oss-120b"
# LLM_MODEL = "openai/gpt-oss-20b"
# LLM_MODEL = "llama-3.1-8b-instant"
```

### ChromaDB paths (relative to notebook)
```python
DATA_AGENTS_DIR = os.path.join(_BASE, 'data_agents')
CHROMA_DB_PATH  = os.path.join(_BASE, 'chroma_db_agents')
```

### Gatekeeper bias: `BLOCK` not in result → ALLOW (broad bias toward helpfulness)
```python
allowed = 'BLOCK' not in result   # ALLOW if response is not exactly BLOCK
```

### Web search routing triggers
Keywords that auto-include `web_search_agent`: `latest`, `recent`, `current`, `news`,
`2025`, `2026`, `update`, `live`, `breaking`, `announce`, `new report`, `this week/month/year`,
`just released`

### Gradio UI notes
- Theme: `gr.themes.Soft` with `Inter` + `JetBrains Mono` Google Fonts
- Input bar: `position: fixed; bottom: 0` via `elem_id="input_area"` — sticky across all scroll positions
- Chat column: `padding-bottom: 130px` via `elem_id="chat_col"` — prevents fixed bar from covering messages
- Chatbot uses Gradio 6.x messages format: `[{'role': 'user', 'content': ...}, ...]`
- 18 suggested questions in the sidebar (4 RAG domains + 2 live web search)

---

## 6. Environment Setup (fresh clone)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate sg-ai-hub

# Or with pip:
pip install -r requirements.txt

# 2. Create .env with your keys
echo "GROQ_API_KEY=gsk_..." > .env
# Optional LangSmith tracing:
echo "LANGCHAIN_API_KEY=lsv2_..." >> .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_PROJECT=singapore-ai-hub-intelligence" >> .env

# 3. Download the 21 PDFs (see README § 3 for sources)
mkdir -p data_agents
# ... place all 21 PDFs in data_agents/

# 4. Open notebook and run Cell 7 to build ChromaDB
# build_knowledge_base(force_rebuild=False)
# → Skips if collections already exist; use force_rebuild=True to re-ingest

# 5. Run Cell 22 to launch the Gradio chatbot
```

---

## 7. Git State (as of handoff)

- Remote: `https://github.com/eshern/singapore-ai-hub-intelligence.git`
- Branch: `main`
- Repo was freshly `git init`-ed in this folder and remote added
- `data_agents/` and `chroma_db_agents/` are gitignored — never push
- Next step: `git add` the 5 files, commit, and `git push -u origin main`

```bash
git add "Singapore AI Hub Intelligence.ipynb" api.py requirements.txt README.md .gitignore
git commit -m "feat: Singapore AI Hub Intelligence — multi-agent RAG + gatekeeper guardrail"
git push -u origin main
```

---

## 8. Completed Work (all sessions)

- [x] 21 PDFs ingested into 4 ChromaDB collections
- [x] 4 RAG agents (sg_policy, investment_ecosystem, regional_comparison, cybersecurity)
- [x] `web_search_agent` via `DuckDuckGoSearchRun` (live internet — 5th agent)
- [x] `general_agent` fallback (6th agent)
- [x] LLM router with keyword fallback
- [x] `synthesize_node` — multi-agent output merger
- [x] `gatekeeper_node` — topic classifier (LangGraph5/6/7 pattern), ALLOW/BLOCK
- [x] `blocked_response_node` — polite refusal, zero RAG calls consumed
- [x] `InvestorState.allowed: bool` field
- [x] Graph entry point: `gatekeeper` → conditional → `[blocked | router]`
- [x] Gradio UI: Inter + JetBrains Mono fonts
- [x] Gradio UI: `position: fixed` sticky input bar
- [x] FastAPI `api.py` — REST deployment with `/query` POST endpoint
- [x] `.gitignore` — excludes `data_agents/`, `chroma_db_agents/`, `chroma_db/`, `*.zip`, `.env`
- [x] `README.md` — full developer guide with PDF download sources table

## 9. Potential Next Steps

- [ ] `git push` to `eshern/singapore-ai-hub-intelligence`
- [ ] Add a `download_pdfs.py` helper script for collaborators
- [ ] Add `environment.yml` for conda users
- [ ] Add LangSmith tracing section to README
- [ ] Explore streaming responses in Gradio (yield-based)
- [ ] Add a `docker-compose.yml` for containerised FastAPI + Gradio deployment
