"""Evidence-based project analysis agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from advisor.retrieval import IndexedChunk


@dataclass(frozen=True)
class AgentFinding:
    category: str
    text: str
    source: str
    chunk_number: int


@dataclass(frozen=True)
class ScopeFindings:
    goals: tuple[AgentFinding, ...]
    deliverables: tuple[AgentFinding, ...]
    milestones: tuple[AgentFinding, ...]
    timelines: tuple[AgentFinding, ...]
    responsibilities: tuple[AgentFinding, ...]


@dataclass(frozen=True)
class DeliveryForecast:
    outlook: str
    evidence: tuple[AgentFinding, ...]


_SCOPE_PATTERNS = {
    "goals": re.compile(r"\b(objective|goal|aim|purpose|intends? to|seeks? to)\b", re.I),
    "deliverables": re.compile(r"\b(deliverables?|output|feature|module|report|user stor(?:y|ies)|risk register)\b", re.I),
    "milestones": re.compile(r"\b(milestone|phase|sprint|iteration)\b", re.I),
    "timelines": re.compile(r"\b(timeline|deadline|due date|by (?:week|month|day)|before \d|within \d|week \d)\b", re.I),
    "responsibilities": re.compile(r"\b(owner|assigned to|responsible for|led by)\b", re.I),
}

_RISK_PATTERN = re.compile(
    r"\b(risk(?!\s+register)|delay(?:ed)?|late|behind schedule|dependency|dependencies|"
    r"blocked|blocker|threat|challenge|slip(?:ped)?|overdue|at risk|"
    r"may fail|might miss|bottleneck|issue|uncertain)\b",
    re.I,
)
_BLOCKER_PATTERN = re.compile(
    r"\b(blocker|blocked|unresolved|waiting on|cannot proceed|can't proceed|"
    r"obstacle|impediment|dependency gap)\b",
    re.I,
)
_DECISION_PATTERN = re.compile(
    r"\b(decision|decide|approval needed|needs? approval|tbd|to be determined)\b",
    re.I,
)
_ACTION_PATTERN = re.compile(
    r"\b(action item|todo|follow[- ]?up|must|should|needs? to|need to|"
    r"assigned to|responsible for|owner|due by)\b",
    re.I,
)
_DELAY_SIGNAL_PATTERN = re.compile(
    r"\b(delay(?:ed|s)?|late|behind schedule|overdue|slipp?(?:ed|ing)?|"
    r"miss(?:ed|ing)? (?:the )?deadline)\b",
    re.I,
)
_TABLE_HEADERS = {
    "action",
    "assignee",
    "date",
    "deadline",
    "deliverable",
    "dependency",
    "description",
    "details",
    "due date",
    "issue",
    "milestone",
    "notes",
    "owner",
    "priority",
    "responsible",
    "risk",
    "risks",
    "status",
    "task",
    "timeline",
}


def _statements(chunks: Iterable[IndexedChunk]) -> Iterable[AgentFinding]:
    for chunk in chunks:
        for statement in re.split(r"(?<=[.!?])\s+|\r?\n+", chunk.text):
            text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", statement).strip()
            cells = [cell.strip().casefold() for cell in text.split("|")]
            if text and not (len(cells) > 1 and all(cell in _TABLE_HEADERS for cell in cells)):
                yield AgentFinding("", text, chunk.source, chunk.chunk_number)


def _unique(findings: Iterable[AgentFinding]) -> tuple[AgentFinding, ...]:
    unique: dict[tuple[str, str], AgentFinding] = {}
    for finding in findings:
        key = (finding.source.casefold(), re.sub(r"\s+", " ", finding.text).casefold())
        unique.setdefault(key, finding)
    return tuple(unique.values())


def extract_scope_and_deliverables(chunks: Iterable[IndexedChunk]) -> ScopeFindings:
    """Extract scope statements and project-planning details with source references."""
    findings: dict[str, list[AgentFinding]] = {name: [] for name in _SCOPE_PATTERNS}
    for statement in _statements(chunks):
        for category, pattern in _SCOPE_PATTERNS.items():
            if pattern.search(statement.text):
                findings[category].append(
                    AgentFinding(category, statement.text, statement.source, statement.chunk_number)
                )
    return ScopeFindings(**{name: _unique(items) for name, items in findings.items()})


def detect_risks_and_delivery_challenges(
    chunks: Iterable[IndexedChunk],
) -> tuple[AgentFinding, ...]:
    """Find explicitly documented risks, dependencies, and delivery challenges."""
    findings = []
    for statement in _statements(chunks):
        if not _RISK_PATTERN.search(statement.text):
            continue
        if re.search(r"\b(dependency|dependencies|dependency gap)\b", statement.text, re.I):
            category = "Dependency"
        elif re.search(
            r"\b(delay|late|behind schedule|deadline|slip|overdue|schedule)\b",
            statement.text,
            re.I,
        ):
            category = "Schedule"
        else:
            category = "Delivery"
        findings.append(
            AgentFinding(category, statement.text, statement.source, statement.chunk_number)
        )
    return _unique(findings)


def forecast_delivery_outlook(chunks: Iterable[IndexedChunk]) -> DeliveryForecast:
    """Summarize documented delivery signals without estimating unsupported dates."""
    indexed_chunks = tuple(chunks)
    risks = detect_risks_and_delivery_challenges(indexed_chunks)
    blockers = tuple(
        finding
        for finding in identify_blockers_and_action_items(indexed_chunks)
        if finding.category == "Blocker"
    )

    delay_evidence = tuple(
        finding for finding in risks if _DELAY_SIGNAL_PATTERN.search(finding.text)
    )
    if delay_evidence:
        return DeliveryForecast("Delay signal", delay_evidence)

    dependency_evidence = tuple(
        finding for finding in risks if finding.category == "Dependency"
    )
    if blockers or dependency_evidence:
        return DeliveryForecast(
            "At risk",
            _unique((*blockers, *dependency_evidence)),
        )

    if risks:
        return DeliveryForecast("Challenges identified", risks)
    return DeliveryForecast("Insufficient evidence", ())


def identify_blockers_and_action_items(
    chunks: Iterable[IndexedChunk],
) -> tuple[AgentFinding, ...]:
    """Identify explicitly stated blockers, pending decisions, and assigned actions."""
    findings = []
    for statement in _statements(chunks):
        if _BLOCKER_PATTERN.search(statement.text):
            category = "Blocker"
        elif _DECISION_PATTERN.search(statement.text):
            category = "Decision"
        elif _ACTION_PATTERN.search(statement.text):
            category = "Action"
        else:
            continue
        findings.append(
            AgentFinding(category, statement.text, statement.source, statement.chunk_number)
        )
    return _unique(findings)
