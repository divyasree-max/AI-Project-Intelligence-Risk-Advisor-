from __future__ import annotations

import csv
import hashlib
import html
import io
import re
from typing import Iterable

import plotly.graph_objects as go
import streamlit as st

from advisor.agents import (
    AgentFinding,
    DeliveryForecast,
    ScopeFindings,
    detect_risks_and_delivery_challenges,
    extract_scope_and_deliverables,
    forecast_delivery_outlook,
    identify_blockers_and_action_items,
)
from advisor.ingestion import SUPPORTED_EXTENSIONS, extract_text
from advisor.milestone3 import (
    ProjectHealth,
    answer_from_evidence,
    build_markdown_document,
    extract_due_date,
    extract_owner,
    generate_action_items,
    generate_risk_register,
    generate_user_stories,
    score_project_health,
)
from advisor.retrieval import LocalKnowledgeBase, SearchResult

PAGES = (
    "Overview",
    "Workspace & Ingestion",
    "Intelligence Agents",
    "Project Health & Radar",
    "Generated Documents",
    "Project Assistant",
)
DIMENSION_WEIGHTS = {
    "Scope": 30,
    "Schedule Risk": 30,
    "Blockers": 25,
    "Quality": 15,
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #20332e;
          --muted: #64736d;
          --paper: #F4F5F3;
          --panel: #FFFFFF;
          --line: #E2E5E1;
          --forest: #134E4A;
          --teal: #0F766E;
          --mint: #A7F3D0;
        }
        #MainMenu, footer { visibility: hidden; }
        [data-testid="stHeader"] { height: 2rem; background: transparent; }
        .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
        [data-testid="stAppViewContainer"] .main .block-container {
          padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1500px;
        }
        html, body, [class*="css"] { color: var(--ink); }
        h1, h2, h3, h4 { color: var(--forest) !important; }
        [data-testid="stSidebar"] {
          background: #0F1E1A; border-right: 1px solid #253b34;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stText"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label { color: #F3F8F5; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #B5C9C0; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
          border-radius: 8px; padding: 5px 8px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
          background: #19332B;
        }
        .brand-title { color: #F3F8F5; font-size: 20px; font-weight: 800; line-height: 1.15; }
        .status-pill { display: inline-block; margin-top: 9px; padding: 4px 8px;
          border: 1px solid #31594A; border-radius: 999px; background: #16352A;
          color: #A7F3D0; font: 10px 'DM Mono', monospace; letter-spacing: .7px; }
        .eyebrow { color: #52716A; font: 10px 'DM Mono', monospace;
          letter-spacing: 1.15px; text-transform: uppercase; }
        .page-title { color: var(--forest); font-size: 30px; line-height: 1.15;
          font-weight: 800; margin: 3px 0 5px; }
        .page-subtitle { color: var(--muted); font-size: 14px; margin-bottom: 15px; }
        .metric-card, .panel-card {
          background: #FFFFFF; border: 1px solid #E2E5E1; border-radius: 12px;
        }
        .metric-card { padding: 14px 16px; min-height: 112px; }
        .metric-label { color: #64736D; font-size: 10px; font-weight: 700;
          letter-spacing: .8px; text-transform: uppercase; }
        .metric-value { color: #134E4A; font-size: 27px; font-weight: 800; margin: 5px 0; }
        .metric-subtitle { color: #64736D; font-size: 11px; line-height: 1.4; }
        .badge { display: inline-block; border-radius: 999px; padding: 4px 9px;
          font-size: 10px; font-weight: 800; letter-spacing: .6px; }
        .badge-green { background: #DCFCE7; color: #166534; }
        .badge-amber { background: #FEF3C7; color: #92400E; }
        .badge-red { background: #FEE2E2; color: #991B1B; }
        .badge-teal { background: #CCFBF1; color: #115E59; }
        .panel-card { padding: 15px 17px; }
        .factor-card { background: #FFFFFF; border: 1px solid #E2E5E1;
          border-radius: 10px; padding: 10px 12px; margin: 0 0 8px; }
        .factor-row { display: flex; justify-content: space-between; gap: 8px;
          align-items: center; font-size: 12px; }
        .factor-name { color: #20332E; font-weight: 700; }
        .factor-score { color: #134E4A; font-weight: 800; }
        .factor-meta { color: #687871; font-size: 10px; margin-top: 3px; }
        .factor-track { height: 6px; border-radius: 99px; background: #E8EEEA; margin-top: 7px; }
        .factor-fill { height: 6px; border-radius: 99px; background: #0F766E; }
        .source-pill { display: inline-flex; gap: 8px; align-items: center;
          max-width: 100%; padding: 7px 11px; border: 1px solid #D5E4DD;
          border-radius: 999px; background: #EAF4EF; color: #134E4A; font-size: 12px; }
        .source-pill-name { overflow-wrap: anywhere; font-weight: 700; }
        .source-pill-meta { color: #64736D; white-space: nowrap; font-size: 10px; }
        .pipeline-step { border-left: 2px solid #31594A; padding: 3px 0 3px 9px;
          margin: 3px 0; color: #D8E5DE; font-size: 11px; }
        .stButton > button { border-radius: 8px; }
        .stButton > button[kind="primary"] {
          background: #134E4A; border-color: #134E4A; color: white;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: #E2E5E1; background: #FFFFFF; border-radius: 12px;
        }
        [data-testid="stExpander"] { border-color: #E2E5E1; border-radius: 9px; }
        [data-testid="stChatMessage"] { background: #FFFFFF; border: 1px solid #E2E5E1; }
        .agent-section-title { color: #134E4A; font-size: 20px; font-weight: 800; margin-bottom: 3px; }
        .agent-section-subtitle { color: #64736D; font-size: 12px; margin-bottom: 10px; }
        .health-diagnostic { background: #FFFFFF; border: 1px solid #E2E5E1;
          border-radius: 12px; padding: 13px 15px; margin: 0 0 9px; }
        @media (max-width: 750px) { .page-title { font-size: 25px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session() -> None:
    if (
        st.session_state.get("rag_session_version") != 4
        or not isinstance(st.session_state.get("knowledge_base"), LocalKnowledgeBase)
    ):
        knowledge_base = LocalKnowledgeBase()
        index_restored = knowledge_base.load_index()
        st.session_state.knowledge_base = knowledge_base
        st.session_state.indexed_sources = knowledge_base.source_summaries
        st.session_state.ingestion_errors = []
        st.session_state.pipeline_state = "READY" if index_restored else "WAITING"
        st.session_state.upload_hashes = {}
        st.session_state.messages = []
        st.session_state.rag_session_version = 4
    defaults = {
        "indexed_sources": st.session_state.knowledge_base.source_summaries,
        "ingestion_errors": [],
        "pipeline_state": "READY" if st.session_state.knowledge_base.chunk_count else "WAITING",
        "upload_hashes": {},
        "messages": [],
        "active_page": PAGES[0],
        "analysis_cache": None,
        "analysis_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _invalidate_analysis() -> None:
    st.session_state.analysis_cache = None
    st.session_state.analysis_signature = None


def _index_uploaded_documents(uploads: Iterable) -> None:
    knowledge_base = st.session_state.knowledge_base
    ingestion_errors = []
    changed = False
    processed_hashes = {}

    for upload in uploads:
        content = upload.getvalue()
        content_hash = hashlib.sha256(content).hexdigest()
        if st.session_state.upload_hashes.get(upload.name) == content_hash:
            continue
        try:
            extracted = extract_text(upload.name, content)
            chunk_total = knowledge_base.add_document(upload.name, extracted)
            if chunk_total:
                processed_hashes[upload.name] = content_hash
                changed = True
            else:
                ingestion_errors.append(f"{upload.name}: no searchable text was found.")
        except Exception as error:
            ingestion_errors.append(f"{upload.name}: {error}")

    if changed:
        try:
            knowledge_base.build_index()
            st.session_state.upload_hashes.update(processed_hashes)
            st.session_state.messages = []
            _invalidate_analysis()
        except Exception as error:
            ingestion_errors.append(f"Vector indexing failed: {error}")

    st.session_state.indexed_sources = knowledge_base.source_summaries
    st.session_state.ingestion_errors = ingestion_errors
    if ingestion_errors:
        st.session_state.pipeline_state = "ATTENTION"
    elif knowledge_base.document_count:
        st.session_state.pipeline_state = "READY"
    else:
        st.session_state.pipeline_state = "WAITING"


def _remove_source(source: str) -> None:
    knowledge_base = st.session_state.knowledge_base
    if knowledge_base.remove_document(source):
        st.session_state.indexed_sources = knowledge_base.source_summaries
        st.session_state.upload_hashes.pop(source, None)
        st.session_state.messages = []
        st.session_state.ingestion_errors = []
        st.session_state.pipeline_state = "READY" if knowledge_base.document_count else "WAITING"
        _invalidate_analysis()


def _sidebar() -> None:
    knowledge_base = st.session_state.knowledge_base
    with st.sidebar:
        st.markdown(
            "<div class='brand-title'>Project Intelligence Advisor</div>"
            "<div class='status-pill'>LOCAL RAG &bull; READY</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='eyebrow' style='margin-top:22px'>Navigate</div>", unsafe_allow_html=True)
        st.radio("Primary navigation", PAGES, key="active_page", label_visibility="collapsed")

        st.markdown("<div class='eyebrow' style='margin:18px 0 7px'>Workspace status</div>", unsafe_allow_html=True)
        st.caption(
            f"{knowledge_base.document_count} indexed source(s) · "
            f"{knowledge_base.chunk_count} chunks · {st.session_state.pipeline_state}"
        )
        steps = (
            "01 Document Intake",
            "02 Text Extraction",
            "03 Recursive Chunking",
            "04 BGE Embeddings",
            "05 Chroma Vector Search",
        )
        ready = bool(knowledge_base.chunk_count)
        for step in steps:
            icon = "✓" if ready else "○"
            st.markdown(
                f"<div class='pipeline-step'>{icon} &nbsp; {step}</div>",
                unsafe_allow_html=True,
            )


def _render_source_management() -> None:
    st.subheader("Project files")
    st.caption("Upload local project documents, or remove sources from the active Chroma index.")
    with st.form("ingestion_form", clear_on_submit=True):
        uploads = st.file_uploader(
            "Add project files",
            type=[extension.removeprefix(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
            help="PDF, DOCX, CSV, TXT and Markdown. Indexing begins only when you submit.",
            key="project_uploads",
        )
        submitted = st.form_submit_button("Add files to workspace", type="primary")
    if submitted and uploads:
        with st.spinner("Extracting text, embedding, and updating Chroma..."):
            _index_uploaded_documents(uploads)
    elif submitted:
        st.info("Choose one or more files before submitting.")

    if st.session_state.ingestion_errors:
        for error in st.session_state.ingestion_errors:
            st.error(error)

    sources = st.session_state.indexed_sources
    st.markdown("**Active indexed sources**")
    if not sources:
        st.caption("No files are indexed yet.")
        return
    for source in sources:
        source_name = str(source["name"])
        left, right = st.columns([12, 1], vertical_alignment="center")
        left.markdown(
            "<div class='source-pill'><span class='source-pill-name'>"
            f"{html.escape(source_name)}</span><span class='source-pill-meta'>"
            f"{int(source['chunks'])} chunks</span></div>",
            unsafe_allow_html=True,
        )
        if right.button(
            "✖",
            key=f"remove_{hashlib.sha1(source_name.encode()).hexdigest()}",
            help=f"Remove {source_name} from the local index",
        ):
            _remove_source(source_name)
            st.rerun()


def _analysis() -> dict:
    knowledge_base = st.session_state.knowledge_base
    chunks = knowledge_base.all_chunks()
    fingerprint = hashlib.sha256(
        "\n".join(f"{chunk.source}:{chunk.chunk_number}:{chunk.text}" for chunk in chunks).encode()
    ).hexdigest()
    if (
        st.session_state.analysis_cache is not None
        and st.session_state.analysis_signature == fingerprint
    ):
        return st.session_state.analysis_cache

    scope = extract_scope_and_deliverables(chunks)
    risks = detect_risks_and_delivery_challenges(chunks)
    actions = identify_blockers_and_action_items(chunks)
    analysis = {
        "chunks": chunks,
        "scope": scope,
        "risks": risks,
        "actions": actions,
        "forecast": forecast_delivery_outlook(chunks),
        "stories": generate_user_stories(scope),
        "risk_register": generate_risk_register(risks),
        "generated_actions": generate_action_items(actions),
        "health": score_project_health(scope, risks, actions),
    }
    st.session_state.analysis_cache = analysis
    st.session_state.analysis_signature = fingerprint
    return analysis


def _short_summary(text: str, limit: int = 260) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentence for sentence in sentences[:2] if sentence)
    if len(summary) > limit:
        return summary[: limit - 1].rsplit(" ", 1)[0] + "…"
    return summary


def _render_evidence(finding: AgentFinding) -> None:
    with st.expander("View Chunk Citation & Raw Evidence"):
        st.caption(f"{finding.source} · chunk {finding.chunk_number}")
        st.write(finding.text)


def _scope_card(finding: AgentFinding, item_number: int) -> None:
    labels = {
        "goals": "Project goal",
        "deliverables": f"Deliverable D-{item_number:02d}",
        "milestones": f"Milestone M-{item_number:02d}",
        "timelines": "Timeline",
        "responsibilities": "Responsibility",
    }
    with st.container(border=True):
        st.markdown(f"**{labels[finding.category]}**")
        target = extract_due_date(finding.text)
        if finding.category in {"deliverables", "milestones", "timelines"}:
            label = f"TARGET {target.upper()}" if target != "Not specified" else "TARGET DATE NOT SPECIFIED"
            st.markdown(f":blue-background[{label}]")
        if finding.category == "responsibilities":
            st.markdown(f":gray-background[{extract_owner(finding.text)}]")
        st.write(_short_summary(finding.text))
        _render_evidence(finding)


def _risk_badge(finding: AgentFinding) -> str:
    text = finding.text.casefold()
    if any(term in text for term in ("critical", "blocked", "overdue", "miss the deadline", "will fail")):
        return ":red-background[CRITICAL RISK]"
    if finding.category == "Dependency":
        return ":orange-background[DEPENDENCY]"
    if "throughput" in text or "capacity" in text:
        return ":violet-background[THROUGHPUT]"
    if finding.category == "Schedule":
        return ":orange-background[SCHEDULE RISK]"
    return ":violet-background[DELIVERY RISK]"


def _risk_card(finding: AgentFinding) -> None:
    with st.container(border=True):
        st.markdown(f"**{finding.category} challenge**")
        st.markdown(_risk_badge(finding))
        st.write(_short_summary(finding.text))
        _render_evidence(finding)


def _action_card(finding: AgentFinding) -> None:
    priority = "P0" if finding.category == "Blocker" else "P1" if finding.category == "Decision" else "P2"
    badge = "BLOCKER" if finding.category == "Blocker" else "DECISION" if finding.category == "Decision" else "ACTION"
    with st.container(border=True):
        st.markdown(f"**{finding.category}**")
        st.markdown(
            f":blue-background[{priority}] :gray-background[{extract_owner(finding.text)}] "
            f":green-background[{badge}]"
        )
        st.write(_short_summary(finding.text))
        _render_evidence(finding)


def _page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='eyebrow'>{eyebrow}</div><div class='page-title'>{title}</div>"
        f"<div class='page-subtitle'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def _navigate_to(page: str) -> None:
    st.session_state.active_page = page


def _render_overview(data: dict | None) -> None:
    kb = st.session_state.knowledge_base
    _page_header("Project overview", "One workspace for project evidence.", "A quick, evidence-grounded view of your indexed project artifacts.")
    _render_source_management()
    if data is None:
        st.info("Add project files above to build your local workspace.")
        return
    health: ProjectHealth = data["health"]
    forecast: DeliveryForecast = data["forecast"]
    columns = st.columns(4)
    cards = (
        ("INDEXED SOURCES", str(kb.document_count), "Local project files"),
        ("SEARCHABLE CHUNKS", str(kb.chunk_count), "Indexed with BGE embeddings"),
        ("OVERALL HEALTH", f"{health.score}/100" if health.score is not None else "N/E", health.status),
        ("DELIVERY OUTLOOK", forecast.outlook, "Based on explicit document signals"),
    )
    for column, (label, value, detail) in zip(columns, cards):
        column.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value' style='font-size:21px'>{html.escape(value)}</div>"
            f"<div class='metric-subtitle'>{html.escape(detail)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1])
    with left:
        with st.container(border=True):
            st.subheader("Project signals")
            st.markdown(f"**Delivery outlook:** {forecast.outlook}")
            if forecast.evidence:
                for finding in forecast.evidence[:3]:
                    st.markdown(f"- {_short_summary(finding.text, 180)}")
                    st.caption(f"{finding.source} · chunk {finding.chunk_number}")
            else:
                st.caption("No explicit delay or blocker signal found. This is not a prediction that delivery is on track.")
    with right:
        with st.container(border=True):
            st.subheader("Start here")
            st.write("Explore the extracted evidence, health indicators, draft documents, or ask a question grounded in your files.")
            st.button(
                "Open Project Assistant",
                type="primary",
                key="overview_assistant",
                on_click=_navigate_to,
                args=("Project Assistant",),
            )
            st.button(
                "Review Intelligence Agents",
                key="overview_agents",
                on_click=_navigate_to,
                args=("Intelligence Agents",),
            )
    st.caption("All project data and retrieval stay local in the persistent Chroma index.")


def _render_workspace() -> None:
    kb = st.session_state.knowledge_base
    _page_header(
        "Workspace & ingestion",
        "Your local project knowledge base.",
        "Add files, track each ingestion stage, and manage sources in the active index.",
    )
    _render_source_management()
    columns = st.columns(3)
    values = (("Indexed sources", kb.document_count), ("Total chunks", kb.chunk_count), ("Pipeline status", st.session_state.pipeline_state))
    for column, (label, value) in zip(columns, values):
        column.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div><div class='metric-subtitle'>"
            f"{'Persisted in Chroma' if label != 'Pipeline status' else 'Upload and indexing state'}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Ingestion pipeline")
    pipeline_columns = st.columns(5)
    steps = (
        ("01", "Document Intake", "PDF · DOCX · CSV · TXT · MD"),
        ("02", "Text Extraction", "Local parsers"),
        ("03", "Recursive Chunking", "LangChain splitters"),
        ("04", "BGE Embeddings", "BAAI/bge-small-en-v1.5"),
        ("05", "Chroma Vector Search", "Persistent local retrieval"),
    )
    for column, (number, title, detail) in zip(pipeline_columns, steps):
        with column:
            with st.container(border=True):
                st.markdown(f"**{number} · {title}**")
                st.caption(detail)
                st.markdown(":green-background[READY]" if kb.chunk_count else ":gray-background[WAITING]")

    st.caption("Removing a source deletes only that source's vectors and in-memory chunks.")


def _render_agents(data: dict | None) -> None:
    _page_header(
        "Intelligence agents",
        "Evidence extracted from project documents.",
        "Rule-based Milestone 2 findings remain linked to the exact source file and chunk.",
    )
    if data is None:
        st.info("Index project documents to run the extraction agents.")
        return
    scope: ScopeFindings = data["scope"]
    risks: tuple[AgentFinding, ...] = data["risks"]
    actions: tuple[AgentFinding, ...] = data["actions"]
    forecast: DeliveryForecast = data["forecast"]
    agent_tabs = st.tabs(["All Agents (Sequential)", "1. Scope", "2. Risks", "3. Blockers"])
    with agent_tabs[0]:
        _render_scope_agent(scope)
        _render_risk_agent(risks, forecast)
        _render_blocker_agent(actions, risks)
    with agent_tabs[1]:
        _render_scope_agent(scope)
    with agent_tabs[2]:
        _render_risk_agent(risks, forecast)
    with agent_tabs[3]:
        _render_blocker_agent(actions, risks)


def _render_scope_agent(scope: ScopeFindings) -> None:
    with st.container(border=True):
        st.markdown("<div class='agent-section-title'>1. Scope & Deliverables Agent</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='agent-section-subtitle'>Goals, promised outputs, planning dates, and ownership found in project files.</div>",
            unsafe_allow_html=True,
        )
        categories = (
            ("Goals", scope.goals),
            ("Deliverables", scope.deliverables),
            ("Milestones & Timelines", (*scope.milestones, *scope.timelines)),
            ("Responsibilities", scope.responsibilities),
        )
        for heading, findings in categories:
            st.markdown(f"#### {heading} <span class='badge badge-teal'>{len(findings)}</span>", unsafe_allow_html=True)
            if not findings:
                st.caption(f"No explicit {heading.casefold()} evidence was identified.")
            for number, finding in enumerate(findings, start=1):
                _scope_card(finding, number)


def _render_risk_agent(risks: tuple[AgentFinding, ...], forecast: DeliveryForecast) -> None:
    technical_terms = (
        "technical", "integration", "api", "performance", "architecture",
        "compatibility", "infrastructure", "security", "migration",
    )
    schedules = tuple(finding for finding in risks if finding.category == "Schedule")
    dependencies = tuple(finding for finding in risks if finding.category == "Dependency")
    technical = tuple(
        finding for finding in risks
        if finding.category == "Delivery" and any(term in finding.text.casefold() for term in technical_terms)
    )
    categorized = set(schedules + dependencies + technical)
    delivery = tuple(finding for finding in risks if finding not in categorized)
    with st.container(border=True):
        st.markdown("<div class='agent-section-title'>2. Risks & Delivery Challenges Agent</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='agent-section-subtitle'>Documented delivery signals are grouped by their evidence category; no predictive claims are added.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Delivery Outlook")
        st.markdown(f"**{forecast.outlook}** · {len(forecast.evidence)} supporting signal(s)")
        if forecast.evidence:
            for finding in forecast.evidence:
                _risk_card(finding)
        else:
            st.caption("No explicit delivery outlook signal was extracted.")
        for heading, findings in (
            ("Schedule Risks", schedules),
            ("Dependency Gaps", dependencies),
            ("Technical Challenges", technical),
            ("Other Delivery Challenges", delivery),
        ):
            st.markdown(f"#### {heading} <span class='badge badge-amber'>{len(findings)}</span>", unsafe_allow_html=True)
            if not findings:
                st.caption(f"No explicit {heading.casefold()} evidence was identified.")
            for finding in findings:
                _risk_card(finding)


def _render_blocker_agent(
    actions: tuple[AgentFinding, ...],
    risks: tuple[AgentFinding, ...],
) -> None:
    blockers = tuple(finding for finding in actions if finding.category == "Blocker")
    decisions = tuple(finding for finding in actions if finding.category == "Decision")
    open_actions = tuple(finding for finding in actions if finding.category == "Action")
    chains = tuple(
        finding
        for finding in (*blockers, *risks)
        if finding.category in {"Blocker", "Dependency"}
        and re.search(r"\b(dependency|depends on|waiting on|blocked by)\b", finding.text, re.I)
    )
    with st.container(border=True):
        st.markdown("<div class='agent-section-title'>3. Blockers & Action Items Agent</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='agent-section-subtitle'>Active blockers, documented dependency chains, pending decisions, and prioritized follow-ups.</div>",
            unsafe_allow_html=True,
        )
        for heading, findings in (
            ("Active Blockers", blockers),
            ("Blocker Chains", chains),
            ("Pending Decisions", decisions),
        ):
            st.markdown(f"#### {heading} <span class='badge badge-red'>{len(findings)}</span>", unsafe_allow_html=True)
            if not findings:
                st.caption(f"No explicit {heading.casefold()} evidence was identified.")
            for finding in findings:
                _action_card(finding) if finding.category == "Blocker" else _risk_card(finding)
        st.markdown("#### Prioritized Action Items")
        for priority, findings in (
            ("P0", blockers),
            ("P1", decisions),
            ("P2", open_actions),
        ):
            st.markdown(f"**{priority} · {len(findings)} item(s)**")
            if not findings:
                st.caption(f"No {priority} items were extracted.")
            for finding in findings:
                _action_card(finding)


def _risk_matrix(findings: Iterable[AgentFinding]) -> tuple[int | None, str]:
    scores = []
    for finding in findings:
        text = finding.text.casefold()
        if any(term in text for term in ("critical", "blocked", "overdue", "will fail", "miss the deadline")):
            likelihood, impact = 5, 5
        else:
            likelihood = 4 if any(term in text for term in ("delay", "late", "behind schedule", "at risk", "slip")) else 3
            impact = {"Schedule": 4, "Dependency": 5, "Delivery": 3}.get(finding.category, 3)
        scores.append(likelihood * impact)
    if not scores:
        return None, "No explicit risk signals; this is not proof of no risk."
    return max(scores), "Highest documented signal · heuristic Likelihood × Impact (1–5 each)."


def _health_axes(data: dict) -> dict[str, dict[str, int | str | None]]:
    health: ProjectHealth = data["health"]
    dimensions = {dimension.name: dimension for dimension in health.dimensions}
    scope: ScopeFindings = data["scope"]
    risks: tuple[AgentFinding, ...] = data["risks"]
    actions: tuple[AgentFinding, ...] = data["actions"]
    findings = (
        *scope.goals,
        *scope.deliverables,
        *scope.milestones,
        *scope.timelines,
        *scope.responsibilities,
        *risks,
        *actions,
    )
    chunk_references = {
        (chunk.source, chunk.chunk_number)
        for chunk in data["chunks"]
    }
    grounded_findings = sum(
        (finding.source, finding.chunk_number) in chunk_references
        for finding in findings
    )
    quality_score = round(100 * grounded_findings / len(findings)) if findings else None
    scope_count = sum(
        len(group)
        for group in (
            scope.goals,
            scope.deliverables,
            scope.milestones,
            scope.timelines,
            scope.responsibilities,
        )
    )
    schedule_count = len(scope.timelines) + sum(
        finding.category in {"Schedule", "Dependency"}
        for finding in risks
    )
    blocker_count = sum(finding.category == "Blocker" for finding in actions)
    scope_dimension = dimensions.get("Scope clarity")
    schedule_dimension = dimensions.get("Timeline risk")
    blocker_dimension = dimensions.get("Blocker count")
    return {
        "Scope": {
            "score": scope_dimension.score if scope_dimension else None,
            "count": scope_count,
            "recommendation": (
                "Fill the documented gaps in goals, deliverables, milestones, timelines, or ownership."
                if scope_dimension is None or scope_dimension.score < 100
                else "Keep the scope baseline and its ownership traceable as project plans change."
            ),
        },
        "Schedule Risk": {
            "score": schedule_dimension.score if schedule_dimension else None,
            "count": schedule_count,
            "recommendation": (
                "Document milestones and dependency owners with required-by dates."
                if schedule_dimension is None or schedule_dimension.score < 75
                else "Keep milestone dates and dependency status current in the source documents."
            ),
        },
        "Blockers": {
            "score": blocker_dimension.score if blocker_dimension else None,
            "count": blocker_count,
            "recommendation": (
                f"Resolve the {blocker_count} active blocker(s), each with an owner and next step."
                if blocker_count
                else "Record blocker owners and resolution dates when issues are identified."
            ),
        },
        "Quality": {
            "score": quality_score,
            "count": grounded_findings,
            "recommendation": (
                "Add project evidence that can be extracted and tied back to indexed source chunks."
                if quality_score is None
                else "Review the cited source chunks and confirm extracted statements remain accurate."
            ),
        },
    }


def _metric_card(label: str, value: str, subtitle: str, badge: str, badge_class: str) -> str:
    return (
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<span class='badge {badge_class}'>{badge}</span>"
        f"<div class='metric-subtitle' style='margin-top:7px'>{subtitle}</div></div>"
    )


def _render_health(data: dict | None) -> None:
    _page_header(
        "Project health & radar",
        "A four-dimension project diagnostic.",
        "The overall score remains the existing weighted health heuristic; diagnostic scores use only extracted project evidence and citation coverage.",
    )
    if data is None:
        st.info("Index project documents to calculate project health.")
        return

    health: ProjectHealth = data["health"]
    dimensions = _health_axes(data)
    schedule_findings = tuple(
        finding for finding in data["risks"]
        if finding.category in {"Schedule", "Dependency"}
    )
    risk_index, _ = _risk_matrix(schedule_findings)
    active_blockers = sum(finding.category == "Blocker" for finding in data["actions"])
    health_label = (
        "INSUFFICIENT EVIDENCE" if health.score is None
        else "HEALTHY" if health.score >= 80
        else "NEEDS ATTENTION" if health.score >= 60
        else "CRITICAL"
    )
    health_class = (
        "badge-green" if health.score is not None and health.score >= 80
        else "badge-amber" if health.score is not None and health.score >= 60
        else "badge-red"
    )
    summary = st.columns(5)
    summary_cards = (
        ("OVERALL PROJECT HEALTH", f"{health.score}/100" if health.score is not None else "N/E", health.status, "SCORE", "badge-teal"),
        ("SCHEDULE RISK INDEX", f"{risk_index}/25" if risk_index is not None else "N/E", "Higher index means a stronger documented signal", "EVIDENCE", "badge-amber"),
        ("ACTIVE BLOCKERS", str(active_blockers), "Explicit blocker findings", "P0", "badge-red" if active_blockers else "badge-green"),
        ("QUALITY READINESS", f"{dimensions['Quality']['score']}/100" if dimensions["Quality"]["score"] is not None else "N/E", "Grounding coverage of extracted findings", "CITATION CHECK", "badge-teal"),
        ("STATUS", health_label, "Not a delivery prediction", health_label, health_class),
    )
    for column, (label, value, subtitle, badge, badge_class) in zip(summary, summary_cards):
        column.markdown(
            _metric_card(label, html.escape(value), html.escape(subtitle), badge, badge_class),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    chart_column, diagnostic_column = st.columns([1.15, 1])
    axis_labels = tuple(DIMENSION_WEIGHTS)

    def score_color(score: int | None) -> str:
        if score is None:
            return "#94A3A8"
        if score >= 75:
            return "#0F766E"
        if score >= 45:
            return "#D99A22"
        return "#C2413B"

    with chart_column:
        with st.container(border=True):
            st.markdown("### Project Health Diamond")
            st.caption("Four evidence-based dimensions · Green ≥75 · Amber 45–74 · Red <45")
            values = [dimensions[label]["score"] for label in axis_labels]
            figure = go.Figure()
            figure.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=list(axis_labels) + [axis_labels[0]],
                    mode="lines+markers",
                    name="Project evidence",
                    line={"color": "#134E4A", "width": 3},
                    marker={"color": [score_color(value) for value in values] + [score_color(values[0])], "size": 10, "line": {"color": "#FFFFFF", "width": 2}},
                    fill="toself",
                    fillcolor="rgba(15, 118, 110, 0.20)",
                    customdata=[
                        f"{value}/100" if value is not None else "N/E · no supporting evidence"
                        for value in values
                    ] + [f"{values[0]}/100" if values[0] is not None else "N/E · no supporting evidence"],
                    hovertemplate="%{theta}: %{customdata}<extra></extra>",
                    connectgaps=False,
                )
            )
            figure.update_layout(
                height=400,
                margin={"l": 35, "r": 35, "t": 25, "b": 25},
                showlegend=False,
                paper_bgcolor="#FFFFFF",
                polar={
                    "bgcolor": "#FFFFFF",
                    "radialaxis": {"visible": True, "range": [0, 100], "tickvals": [25, 50, 75, 100], "gridcolor": "#E2E5E1", "tickfont": {"color": "#64736D", "size": 9}},
                    "angularaxis": {"gridcolor": "#D6E4DE", "tickfont": {"color": "#20332E", "size": 10}},
                },
                annotations=[
                    {
                        "text": f"<b>{health.score}/100</b>" if health.score is not None else "<b>N/E</b>",
                        "x": 0.5,
                        "y": 0.5,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 16, "color": "#134E4A"},
                    }
                ],
            )
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
            st.caption("N/E indicates insufficient evidence, not a score of zero. Quality measures source-chunk grounding coverage.")
    with diagnostic_column:
        st.markdown("### Dimension Diagnostics")
        for label in axis_labels:
            dimension = dimensions[label]
            score = dimension["score"]
            score_label = f"{score}/100" if score is not None else "N/E"
            score_style = score_color(score)
            count_label = (
                f"{dimension['count']} extracted finding(s)"
                if label != "Quality"
                else f"{dimension['count']} grounded finding(s)"
            )
            with st.expander(f"{label} · {score_label} · {DIMENSION_WEIGHTS[label]}% weight"):
                st.markdown(
                    "<div class='health-diagnostic'>"
                    f"<div class='factor-row'><span class='factor-name'>{html.escape(label)}</span>"
                    f"<span class='factor-score' style='color:{score_style}'>{score_label}</span></div>"
                    f"<div class='factor-meta'>{count_label}</div>"
                    f"<div class='metric-subtitle' style='margin-top:7px'>{html.escape(str(dimension['recommendation']))}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
    st.caption("Dimension weights are diagnostic display weights; they do not alter the existing overall health calculation.")


def _csv_content(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def _render_documents(data: dict | None) -> None:
    _page_header(
        "Generated documents",
        "Drafts for your team to review.",
        "Markdown and CSV exports retain the original document and chunk citation for each generated item.",
    )
    if data is None:
        st.info("Index project documents to generate draft stories, risks, and actions.")
        return

    stories = data["stories"]
    risks = data["risk_register"]
    actions = data["generated_actions"]
    tabs = st.tabs(["User Stories", "Risk Register", "Action Items"])
    story_sections = [
        (story.story_id, [f"**{story.title}**", story.story, f"Source: {story.source}, chunk {story.chunk_number}."])
        for story in stories
    ]
    risk_sections = [
        (risk.risk_id, [f"**{risk.severity} · {risk.category}:** {risk.description}", f"Suggested response: {risk.suggested_response}", f"Source: {risk.source}, chunk {risk.chunk_number}."])
        for risk in risks
    ]
    action_sections = [
        (item.item_id, [f"**{item.status} · {item.category}:** {item.description}", f"Owner: {item.owner} · Due: {item.due_date}", f"Source: {item.source}, chunk {item.chunk_number}."])
        for item in actions
    ]

    with tabs[0]:
        if stories:
            for story in stories:
                with st.container(border=True):
                    st.markdown(f"**{story.story_id} · {story.title}**")
                    st.write(story.story)
                    st.caption(f"Source: {story.source} · chunk {story.chunk_number}")
            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Download Markdown",
                build_markdown_document("Draft User Stories", story_sections),
                file_name="user-stories.md",
                mime="text/markdown",
                key="download_user_stories_md",
            )
            download_columns[1].download_button(
                "Download CSV",
                _csv_content(
                    ["ID", "Title", "Story", "Source", "Chunk"],
                    ((item.story_id, item.title, item.story, item.source, item.chunk_number) for item in stories),
                ),
                file_name="user-stories.csv",
                mime="text/csv",
                key="download_user_stories_csv",
            )
        else:
            st.info("No explicit deliverables or goals were found to draft user stories from.")

    with tabs[1]:
        if risks:
            for risk in risks:
                with st.container(border=True):
                    st.markdown(f"**{risk.risk_id} · {risk.severity} {risk.category}**")
                    st.write(risk.description)
                    st.write(f"Suggested response: {risk.suggested_response}")
                    st.caption(f"Source: {risk.source} · chunk {risk.chunk_number}")
            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Download Markdown",
                build_markdown_document("Draft Risk Register", risk_sections),
                file_name="risk-register.md",
                mime="text/markdown",
                key="download_risk_register_md",
            )
            download_columns[1].download_button(
                "Download CSV",
                _csv_content(
                    ["ID", "Category", "Severity", "Description", "Suggested response", "Source", "Chunk"],
                    ((item.risk_id, item.category, item.severity, item.description, item.suggested_response, item.source, item.chunk_number) for item in risks),
                ),
                file_name="risk-register.csv",
                mime="text/csv",
                key="download_risk_register_csv",
            )
        else:
            st.info("No explicit risk statements were found in the uploaded text.")

    with tabs[2]:
        if actions:
            for item in actions:
                with st.container(border=True):
                    st.markdown(f"**{item.item_id} · {item.status} · {item.category}**")
                    st.write(item.description)
                    st.caption(f"Owner: {item.owner} · Due: {item.due_date} · Source: {item.source}, chunk {item.chunk_number}")
            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Download Markdown",
                build_markdown_document("Draft Action Items", action_sections),
                file_name="action-items.md",
                mime="text/markdown",
                key="download_action_items_md",
            )
            download_columns[1].download_button(
                "Download CSV",
                _csv_content(
                    ["ID", "Category", "Description", "Owner", "Due date", "Status", "Source", "Chunk"],
                    ((item.item_id, item.category, item.description, item.owner, item.due_date, item.status, item.source, item.chunk_number) for item in actions),
                ),
                file_name="action-items.csv",
                mime="text/csv",
                key="download_action_items_csv",
            )
        else:
            st.info("No explicit blockers, decisions, or action items were found.")


def _ask_question(prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    knowledge_base: LocalKnowledgeBase = st.session_state.knowledge_base
    st.session_state.messages.append({"role": "user", "content": prompt})
    results: list[SearchResult] = knowledge_base.search(prompt, top_k=5)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_from_evidence(results),
            "evidence": [
                {
                    "source": result.source,
                    "chunk_number": result.chunk_number,
                    "score": result.score,
                    "excerpt": result.excerpt,
                }
                for result in results
            ],
        }
    )


def _render_assistant() -> None:
    knowledge_base: LocalKnowledgeBase = st.session_state.knowledge_base
    _page_header(
        "Project assistant",
        "Ask questions grounded in your files.",
        "Answers are extractive: each response uses locally retrieved chunks and shows its citations.",
    )
    suggestions = ("Are we on track?", "What are our P0 action items?", "What is the main blocker chain?")
    prompt = None
    suggestion_columns = st.columns(3)
    for column, suggestion in zip(suggestion_columns, suggestions):
        if column.button(suggestion, key=f"starter_{suggestion}"):
            prompt = suggestion

    with st.container(height=340, border=True):
        if not knowledge_base.chunk_count:
            st.info("Add and index project documents to enable questions.")
        elif not st.session_state.messages:
            st.markdown(
                "<div class='panel-card'><strong>Welcome to your project assistant</strong>"
                "<p>Ask a question to search indexed project evidence. Try:</p>"
                "<ul><li>Which milestones have dates?</li><li>What dependencies are unresolved?</li>"
                "<li>Who owns the next action?</li></ul></div>",
                unsafe_allow_html=True,
            )
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                for evidence in message.get("evidence", []):
                    with st.expander("View Chunk Citation & Raw Evidence"):
                        st.caption(
                            f"{evidence['source']} · chunk {evidence['chunk_number']} · "
                            f"match {evidence['score']:.0%}"
                        )
                        st.write(evidence["excerpt"])

    typed_prompt = st.chat_input(
        "Ask about project scope, risks, or blockers...",
        disabled=not knowledge_base.chunk_count,
        key="project_chat_input",
    )
    prompt = typed_prompt or prompt
    if prompt:
        _ask_question(prompt)
        st.rerun()


def run_app() -> None:
    st.set_page_config(
        page_title="Project Intelligence Advisor",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _initialize_session()
    _sidebar()

    active_page = st.session_state.active_page
    data = _analysis() if st.session_state.knowledge_base.chunk_count else None
    if active_page == "Overview":
        _render_overview(data)
    elif active_page == "Workspace & Ingestion":
        _render_workspace()
    elif active_page == "Intelligence Agents":
        _render_agents(data)
    elif active_page == "Project Health & Radar":
        _render_health(data)
    elif active_page == "Generated Documents":
        _render_documents(data)
    else:
        _render_assistant()
