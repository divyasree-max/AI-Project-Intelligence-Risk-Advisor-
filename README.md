# AI Project Intelligence & Risk Advisor

An evidence-grounded project workspace that turns project documents into a searchable knowledge base and surfaces documented scope, risks, blockers, decisions, and action items. The Streamlit dashboard includes project health insights, a cited project assistant, and downloadable report drafts.

## Features

- **Document ingestion:** Upload PDF, DOCX, CSV, Markdown, and plain-text (`.txt`) project files.
- **Local semantic search:** Extracted text is split into overlapping chunks, embedded with the local BGE-small model, and indexed in a persistent Chroma database.
- **Evidence-based project analysis:** Identify goals, deliverables, milestones, timelines, responsibilities, schedule signals, dependencies, blockers, decisions, and actions, with source-file and chunk citations.
- **Delivery outlook:** Summarize documented delay signals, dependencies, and other challenges. The outlook is rule-based and does not predict delivery dates.
- **Project health dashboard:** Review scope clarity, timeline risk, blockers, and evidence coverage. Missing evidence is reported as partial or insufficient rather than treated as a positive signal.
- **Project assistant:** Search indexed files with natural-language questions and receive relevant passages with citations. Answers are extractive and limited to retrieved evidence.
- **Exportable drafts:** Generate user stories, a risk register, and action items as Markdown or CSV, retaining their source references.
- **Source management:** Add files incrementally, replace an indexed source by uploading a changed file with the same name, or remove a source from the local index.

## Requirements

- Python 3.10 or newer
- The packages listed in [`requirements.txt`](requirements.txt)

The first indexing run downloads the BGE embedding model. After it has been downloaded, embeddings are generated locally on CPU. The app does not require a hosted LLM or API key.

## Run locally

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). In **Workspace & Ingestion**, upload one or more files and submit them for indexing. Then explore the **Overview**, **Intelligence Agents**, **Project Health & Radar**, **Generated Documents**, and **Project Assistant** sections.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The tests cover document extraction and indexing, retrieval behavior, evidence-based analysis, and generated project artifacts.

## How it works

1. Uploaded files are converted to text by format-specific extractors.
2. LangChain's recursive text splitter creates overlapping chunks.
3. FastEmbed generates local embeddings, and Chroma stores the indexed chunks and source metadata.
4. Retrieval, rule-based analysis, and project health calculations operate on the indexed evidence.
5. Search results, findings, and generated drafts retain source-file and chunk references.

The analysis is intended to support project review, not replace it. Risk categories, suggested responses, generated stories, and health scores are heuristic or draft outputs; review them against the source documents before making project decisions.

## Data and privacy

The vector index is persisted locally under `data/chroma/`. Uploaded content is processed for extraction and indexing; extracted text and embeddings are stored in that local database so the index can be reused between app sessions. The app uses a local embedding model and does not send project content to a hosted language model.

The Chroma database and common Python-generated files are excluded from Git by [`.gitignore`](.gitignore). Do not commit project documents, secrets, or local data unless you have explicitly checked that they are appropriate to publish.

## Project structure

```text
.
├── advisor/
│   ├── agents.py       # Evidence-grounded project findings
│   ├── ingestion.py    # File extraction and text chunking
│   ├── milestone3.py   # Health scoring, drafts, and cited answers
│   ├── retrieval.py    # Local Chroma knowledge base and search
│   └── ui.py           # Streamlit dashboard and workflows
├── tests/              # Unit and integration tests
├── app.py              # Streamlit application entry point
├── requirements.txt
└── README.md
```
