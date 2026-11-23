import unittest
from io import BytesIO

from advisor.agents import (
    detect_risks_and_delivery_challenges,
    extract_scope_and_deliverables,
    forecast_delivery_outlook,
    identify_blockers_and_action_items,
)
from advisor.ingestion import extract_text
from advisor.retrieval import IndexedChunk
from docx import Document


def _pdf_with_text(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\n"
        b"stream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body + b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


class ProjectAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            IndexedChunk(
                source="project.txt",
                chunk_number=1,
                text=(
                    "The project goal is to deliver a reliable reporting system. "
                    "Deliverables include a risk register and a weekly report. "
                    "Milestone 2 begins in week 3. "
                    "The deadline is by week 4. "
                    "Sam is responsible for API testing. "
                    "The API dependency is late, creating delivery risk. "
                    "The integration is blocked waiting on vendor credentials. "
                    "The product owner must approve the design decision. "
                    "Sam must update the sprint plan."
                ),
            )
        ]

    def test_extracts_scope_categories_with_source_references(self) -> None:
        findings = extract_scope_and_deliverables(self.chunks)

        self.assertIn("project goal", findings.goals[0].text)
        self.assertIn("risk register", findings.deliverables[0].text)
        self.assertTrue(findings.milestones)
        self.assertTrue(findings.timelines)
        self.assertIn("Sam", findings.responsibilities[0].text)
        self.assertEqual(findings.goals[0].source, "project.txt")
        self.assertEqual(findings.goals[0].chunk_number, 1)

    def test_detects_risks_and_classifies_dependencies(self) -> None:
        findings = detect_risks_and_delivery_challenges(self.chunks)

        self.assertEqual(
            {finding.category for finding in findings},
            {"Dependency", "Delivery"},
        )
        self.assertTrue(any("late" in finding.text for finding in findings))

    def test_forecasts_delivery_outlook_from_documented_signals(self) -> None:
        forecast = forecast_delivery_outlook(self.chunks)

        self.assertEqual(forecast.outlook, "Delay signal")
        self.assertTrue(forecast.evidence)
        self.assertTrue(all(finding.source == "project.txt" for finding in forecast.evidence))

    def test_does_not_claim_on_track_without_delivery_evidence(self) -> None:
        chunks = [IndexedChunk("notes.txt", "The team met on Monday.", 1)]

        forecast = forecast_delivery_outlook(chunks)

        self.assertEqual(forecast.outlook, "Insufficient evidence")
        self.assertEqual(forecast.evidence, ())

    def test_forecast_distinguishes_dependency_risk_from_other_challenges(self) -> None:
        scenarios = (
            ("The API dependency remains unresolved.", "At risk"),
            ("The team faces a significant delivery challenge.", "Challenges identified"),
        )

        for text, expected_outlook in scenarios:
            with self.subTest(expected_outlook=expected_outlook):
                forecast = forecast_delivery_outlook(
                    [IndexedChunk("notes.txt", text, 1)]
                )
                self.assertEqual(forecast.outlook, expected_outlook)
                self.assertTrue(forecast.evidence)

    def test_identifies_blockers_decisions_and_actions(self) -> None:
        findings = identify_blockers_and_action_items(self.chunks)

        self.assertEqual(
            {finding.category for finding in findings},
            {"Blocker", "Decision", "Action"},
        )
        self.assertTrue(any("blocked" in finding.text for finding in findings))
        self.assertTrue(any("must approve" in finding.text for finding in findings))
        self.assertTrue(any("update the sprint plan" in finding.text for finding in findings))

    def test_deduplicates_overlapping_chunks_from_same_source(self) -> None:
        chunks = [
            IndexedChunk("notes.txt", "The API is late, creating a delivery risk.", 1),
            IndexedChunk("notes.txt", "The API is late, creating a delivery risk.", 2),
        ]

        findings = detect_risks_and_delivery_challenges(chunks)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].chunk_number, 1)

    def test_analyzes_samples_from_txt_csv_docx_and_pdf(self) -> None:
        word_document = Document()
        word_document.add_paragraph("Sam is responsible for the delivery milestone.")
        docx_content = BytesIO()
        word_document.save(docx_content)

        samples = {
            "sample.txt": b"The project goal is to deliver the migration.",
            "sample.csv": b"risk,details\nDependency,The API dependency is blocked.\n",
            "sample.docx": docx_content.getvalue(),
            "sample.pdf": _pdf_with_text(
                "The release is delayed and the deadline may slip by week 4."
            ),
        }
        chunks = [
            IndexedChunk(
                source=filename,
                text=extract_text(filename, content),
                chunk_number=1,
            )
            for filename, content in samples.items()
        ]

        scope = extract_scope_and_deliverables(chunks)
        risks = detect_risks_and_delivery_challenges(chunks)
        blockers = identify_blockers_and_action_items(chunks)

        self.assertEqual(scope.goals[0].source, "sample.txt")
        self.assertEqual(scope.milestones[0].source, "sample.docx")
        self.assertEqual(risks[0].source, "sample.csv")
        self.assertTrue(
            any(
                finding.source == "sample.pdf" and finding.category == "Schedule"
                for finding in risks
            )
        )
        self.assertEqual(blockers[0].source, "sample.csv")


if __name__ == "__main__":
    unittest.main()
