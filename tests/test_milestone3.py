import unittest

from advisor.agents import (
    ScopeFindings,
    detect_risks_and_delivery_challenges,
    extract_scope_and_deliverables,
    identify_blockers_and_action_items,
)
from advisor.milestone3 import (
    answer_from_evidence,
    build_markdown_document,
    extract_due_date,
    extract_owner,
    generate_action_items,
    generate_risk_register,
    generate_user_stories,
    score_project_health,
)
from advisor.retrieval import IndexedChunk, SearchResult


class MilestoneThreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            IndexedChunk(
                "project.txt",
                "The project goal is to deliver a portal. "
                "Deliverables include an analytics dashboard. "
                "Milestone 2 is due by week 4. "
                "Sam is responsible for testing. "
                "The API dependency is late and blocked.",
                1,
            )
        ]
        self.scope = extract_scope_and_deliverables(self.chunks)
        self.risks = detect_risks_and_delivery_challenges(self.chunks)
        self.actions = identify_blockers_and_action_items(self.chunks)

    def test_generates_cited_stories_from_deliverables(self) -> None:
        stories = generate_user_stories(self.scope)

        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0].story_id, "US-01")
        self.assertIn("analytics dashboard", stories[0].story)
        self.assertEqual((stories[0].source, stories[0].chunk_number), ("project.txt", 1))

    def test_skips_generic_report_and_output_mentions_when_drafting_stories(self) -> None:
        noisy_scope = extract_scope_and_deliverables(
            [IndexedChunk("notes.txt", "Report shows 25 users and output.", 1)]
        )

        self.assertEqual(generate_user_stories(noisy_scope), ())

    def test_generates_register_and_action_items_with_source_data(self) -> None:
        risks = generate_risk_register(self.risks)
        actions = generate_action_items(self.actions)

        self.assertTrue(risks)
        self.assertTrue(all(risk.source == "project.txt" for risk in risks))
        self.assertTrue(all(risk.suggested_response for risk in risks))
        self.assertTrue(any(item.status == "Blocked" for item in actions))

    def test_extracts_owner_and_due_date_without_inventing_missing_values(self) -> None:
        self.assertEqual(extract_owner("Sam is responsible for testing."), "Sam")
        self.assertEqual(extract_owner("Update the sprint plan."), "Not specified")
        self.assertEqual(extract_due_date("The milestone is due by week 4."), "week 4")
        self.assertEqual(extract_due_date("The team met on Monday."), "Not specified")

    def test_scores_health_with_dimension_breakdown_and_partial_evidence(self) -> None:
        health = score_project_health(self.scope, self.risks, self.actions)

        self.assertIsNotNone(health.score)
        self.assertEqual(
            {dimension.name for dimension in health.dimensions},
            {"Scope clarity", "Timeline risk", "Blocker count"},
        )
        self.assertTrue(any("blocker" in item.detail for item in health.dimensions))

        sparse_scope = ScopeFindings((), (), (), (), ())
        partial = score_project_health(sparse_scope, self.risks, ())
        self.assertIn("partial evidence", partial.status)
        self.assertEqual(len(partial.dimensions), 1)

    def test_health_is_unavailable_without_any_scoring_evidence(self) -> None:
        health = score_project_health(ScopeFindings((), (), (), (), ()), (), ())

        self.assertIsNone(health.score)
        self.assertEqual(health.status, "Insufficient evidence")

    def test_chat_answers_with_cited_retrieved_text_or_admits_no_evidence(self) -> None:
        results = [
            SearchResult("project.txt", "The API dependency is late.", 0.9, 1)
        ]

        answer = answer_from_evidence(results)

        self.assertIn("The API dependency is late.", answer)
        self.assertIn("project.txt, chunk 1", answer)
        self.assertIn("does not add facts", answer)
        self.assertIn("couldn't find supporting evidence", answer_from_evidence(()))

    def test_markdown_export_contains_document_title_and_citations(self) -> None:
        document = build_markdown_document(
            "Draft Actions",
            [("A-01", ["Confirm the API owner.", "Source: project.txt, chunk 1."])],
        )

        self.assertIn("# Draft Actions", document)
        self.assertIn("Source: project.txt, chunk 1.", document)


if __name__ == "__main__":
    unittest.main()
