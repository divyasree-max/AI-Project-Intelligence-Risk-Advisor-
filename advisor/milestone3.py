"""Evidence-grounded documentation and project health helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from advisor.agents import AgentFinding, ScopeFindings
from advisor.retrieval import SearchResult

_EXPLICIT_STORY_SOURCE = re.compile(
    r"\b(deliverables?|user stor(?:y|ies)|risk register)\b|"
    r"\b(?:outputs?|features?|modules?)\s+(?:include|are|will|must|should|to)\b",
    re.I,
)


@dataclass(frozen=True)
class GeneratedUserStory:
    story_id: str
    title: str
    story: str
    source: str
    chunk_number: int


@dataclass(frozen=True)
class RiskRegisterEntry:
    risk_id: str
    category: str
    severity: str
    description: str
    suggested_response: str
    source: str
    chunk_number: int


@dataclass(frozen=True)
class GeneratedActionItem:
    item_id: str
    category: str
    description: str
    owner: str
    due_date: str
    status: str
    source: str
    chunk_number: int


@dataclass(frozen=True)
class HealthDimension:
    name: str
    score: int | None
    detail: str


@dataclass(frozen=True)
class ProjectHealth:
    score: int | None
    status: str
    dimensions: tuple[HealthDimension, ...]


def extract_owner(text: str) -> str:
    patterns = (
        r"\bowner\s*(?:is|:)\s*([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?)",
        r"\bassigned\s+to\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?)",
        r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?)\s+"
        r"(?:is\s+)?(?:responsible\s+for|owns|leads)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "Not specified"


def extract_due_date(text: str) -> str:
    patterns = (
        r"\b(?:by|before|due(?:\s+on)?|deadline(?:\s+is)?)\s+"
        r"((?:week|month|day)\s+\d+|"
        r"\d{4}-\d{2}-\d{2}|"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?)\b",
        r"\b(week\s+\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return "Not specified"


def generate_user_stories(scope: ScopeFindings) -> tuple[GeneratedUserStory, ...]:
    """Create reviewable story drafts that retain their source statements."""
    explicit_deliverables = tuple(
        finding
        for finding in scope.deliverables
        if len(finding.text) >= 25 and _EXPLICIT_STORY_SOURCE.search(finding.text)
    )
    source_findings = explicit_deliverables or scope.goals
    stories = []
    for number, finding in enumerate(source_findings, start=1):
        title = re.sub(r"\s+", " ", finding.text).strip()
        if len(title) > 80:
            title = title[:77].rsplit(" ", 1)[0] + "..."
        stories.append(
            GeneratedUserStory(
                story_id=f"US-{number:02d}",
                title=title,
                story=(
                    "As a project stakeholder, I want the documented scope captured "
                    f'and tracked: "{finding.text}" so that delivery can be reviewed '
                    "against the project documents."
                ),
                source=finding.source,
                chunk_number=finding.chunk_number,
            )
        )
    return tuple(stories)


def _risk_severity(finding: AgentFinding) -> str:
    text = finding.text.casefold()
    if any(term in text for term in ("critical", "blocked", "overdue", "miss the deadline", "will fail")):
        return "High"
    if finding.category == "Schedule" or finding.category == "Dependency":
        return "Medium"
    return "Review"


def _risk_response(category: str) -> str:
    if category == "Schedule":
        return "Confirm remaining work, owner, and due date; agree a recovery action."
    if category == "Dependency":
        return "Confirm the dependency owner, required-by date, and fallback; track it to resolution."
    return "Assign an owner and review the issue at the next project check-in."


def generate_risk_register(
    findings: Iterable[AgentFinding],
) -> tuple[RiskRegisterEntry, ...]:
    return tuple(
        RiskRegisterEntry(
            risk_id=f"R-{number:02d}",
            category=finding.category,
            severity=_risk_severity(finding),
            description=finding.text,
            suggested_response=_risk_response(finding.category),
            source=finding.source,
            chunk_number=finding.chunk_number,
        )
        for number, finding in enumerate(findings, start=1)
    )


def generate_action_items(
    findings: Iterable[AgentFinding],
) -> tuple[GeneratedActionItem, ...]:
    items = []
    for number, finding in enumerate(findings, start=1):
        status = {
            "Blocker": "Blocked",
            "Decision": "Needs decision",
            "Action": "Open",
        }[finding.category]
        items.append(
            GeneratedActionItem(
                item_id=f"A-{number:02d}",
                category=finding.category,
                description=finding.text,
                owner=extract_owner(finding.text),
                due_date=extract_due_date(finding.text),
                status=status,
                source=finding.source,
                chunk_number=finding.chunk_number,
            )
        )
    return tuple(items)


def score_project_health(
    scope: ScopeFindings,
    risks: Iterable[AgentFinding],
    action_items: Iterable[AgentFinding],
) -> ProjectHealth:
    """Score only dimensions supported by extracted evidence; higher is better."""
    risk_findings = tuple(risks)
    actions = tuple(action_items)
    dimensions: list[HealthDimension] = []

    scope_weights = {
        "goals": 30,
        "deliverables": 30,
        "milestones": 15,
        "timelines": 15,
        "responsibilities": 10,
    }
    scope_values = {
        name: getattr(scope, name)
        for name in scope_weights
    }
    if any(scope_values.values()):
        scope_score = sum(
            weight for name, weight in scope_weights.items() if scope_values[name]
        )
        missing = [name for name, values in scope_values.items() if not values]
        detail = (
            "Documented planning areas are present."
            if not missing
            else "Missing documented areas: " + ", ".join(missing) + "."
        )
        dimensions.append(HealthDimension("Scope clarity", scope_score, detail))

    schedule_risks = sum(finding.category == "Schedule" for finding in risk_findings)
    dependency_risks = sum(finding.category == "Dependency" for finding in risk_findings)
    other_risks = sum(finding.category not in {"Schedule", "Dependency"} for finding in risk_findings)
    if scope.timelines or risk_findings:
        timeline_score = max(
            0,
            100 - 30 * schedule_risks - 20 * dependency_risks - 10 * other_risks,
        )
        dimensions.append(
            HealthDimension(
                "Timeline risk",
                timeline_score,
                f"{schedule_risks} schedule, {dependency_risks} dependency, "
                f"and {other_risks} other documented risk signal(s); "
                "no recorded risk is not proof of no risk.",
            )
        )

    if actions:
        blocker_count = sum(finding.category == "Blocker" for finding in actions)
        blocker_score = max(0, 100 - 30 * blocker_count)
        dimensions.append(
            HealthDimension(
                "Blocker count",
                blocker_score,
                f"{blocker_count} explicit blocker(s) identified in the uploaded text.",
            )
        )

    if not dimensions:
        return ProjectHealth(None, "Insufficient evidence", ())

    weights = {"Scope clarity": 0.40, "Timeline risk": 0.35, "Blocker count": 0.25}
    available_weight = sum(weights[dimension.name] for dimension in dimensions)
    weighted_score = sum(
        dimension.score * weights[dimension.name]
        for dimension in dimensions
        if dimension.score is not None
    ) / available_weight
    score = round(weighted_score)
    status = "Healthy" if score >= 80 else "Watch" if score >= 60 else "Needs attention"
    if len(dimensions) < len(weights):
        status += " · partial evidence"
    return ProjectHealth(score, status, tuple(dimensions))


def answer_from_evidence(results: Iterable[SearchResult]) -> str:
    """Return an extractive answer whose claims are limited to retrieved passages."""
    evidence = tuple(results)
    if not evidence:
        return (
            "I couldn't find supporting evidence in the uploaded project documents. "
            "Try a different question or add a relevant document."
        )

    lines = [
        "Here are the most relevant passages I found in your uploaded documents:"
    ]
    for result in evidence[:3]:
        excerpt = re.sub(r"\s+", " ", result.excerpt).strip()
        lines.append(
            f'- “{excerpt}” — {result.source}, chunk {result.chunk_number}.'
        )
    lines.append(
        "This answer is extractive; it does not add facts beyond these cited passages."
    )
    return "\n\n".join(lines)


def build_markdown_document(title: str, sections: Iterable[tuple[str, Iterable[str]]]) -> str:
    lines = [f"# {title}", "", "_Draft generated from uploaded project documents. Review before use._"]
    for heading, entries in sections:
        lines.extend(("", f"## {heading}", ""))
        lines.extend(entries)
    return "\n".join(lines).rstrip() + "\n"
