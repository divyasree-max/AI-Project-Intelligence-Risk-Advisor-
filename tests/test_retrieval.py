import unittest
from pathlib import Path

import chromadb
from langchain_core.embeddings import Embeddings

from advisor.retrieval import COLLECTION_NAME, LocalKnowledgeBase


class TestEmbeddings(Embeddings):
    vocabulary = [
        "api", "delivery", "schedule", "risk", "late", "project", "team",
        "meets", "monday", "review", "design", "notes", "kickoff", "today",
    ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        tokens = text.lower().split()
        return [float(tokens.count(word)) for word in cls.vocabulary]


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.persist_directory = Path("tests/.chroma") / self._testMethodName
        client = chromadb.PersistentClient(path=str(self.persist_directory))
        if COLLECTION_NAME in {collection.name for collection in client.list_collections()}:
            client.delete_collection(COLLECTION_NAME)

    def test_incremental_upload_preserves_other_sources_and_replaces_same_source(self) -> None:
        knowledge_base = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        knowledge_base.add_document(
            "schedule.txt",
            "The API delivery is late and threatens the project schedule. " * 80,
        )
        knowledge_base.build_index()

        knowledge_base.add_document("team.txt", "The team meets every Monday.")
        knowledge_base.build_index()

        self.assertEqual(knowledge_base.document_count, 2)
        self.assertTrue(knowledge_base.search("API delivery schedule"))

        knowledge_base.add_document("schedule.txt", "The project kickoff is today.")
        knowledge_base.build_index()

        self.assertEqual(knowledge_base.document_count, 2)
        self.assertEqual(knowledge_base.chunk_count, 2)
        self.assertEqual(
            [chunk.text for chunk in knowledge_base.all_chunks() if chunk.source == "schedule.txt"],
            ["The project kickoff is today."],
        )
        self.assertTrue(knowledge_base.search("team Monday"))

    def test_ranks_relevant_evidence_and_returns_source(self) -> None:
        knowledge_base = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        knowledge_base.add_document("schedule.txt", "The API delivery is late and threatens the project schedule.")
        knowledge_base.add_document("team.txt", "The team meets every Monday to review design notes.")
        knowledge_base.build_index()

        results = knowledge_base.search("API delivery schedule risk")

        self.assertTrue(results)
        self.assertEqual(results[0].source, "schedule.txt")
        self.assertIn("schedule", results[0].excerpt)

    def test_empty_or_unmatched_queries_return_no_results(self) -> None:
        knowledge_base = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        knowledge_base.add_document("notes.txt", "A project kickoff happened today.")
        knowledge_base.build_index()

        self.assertEqual(knowledge_base.search(""), [])

    def test_persistent_collection_is_written(self) -> None:
        knowledge_base = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        knowledge_base.add_document("notes.txt", "A project kickoff happened today.")
        knowledge_base.build_index()

        self.assertTrue(any(self.persist_directory.iterdir()))
        restored = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        self.assertTrue(restored.load_index())
        self.assertEqual(restored.document_count, 1)
        self.assertTrue(restored.search("project kickoff"))
        self.assertEqual(restored.all_chunks()[0].source, "notes.txt")

    def test_removing_source_updates_memory_and_persisted_search(self) -> None:
        knowledge_base = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        knowledge_base.add_document("schedule.txt", "The API delivery is late.")
        knowledge_base.add_document("team.txt", "The team meets every Monday.")
        knowledge_base.build_index()

        self.assertTrue(knowledge_base.remove_document("schedule.txt"))

        self.assertEqual(knowledge_base.document_count, 1)
        self.assertEqual([chunk.source for chunk in knowledge_base.all_chunks()], ["team.txt"])
        self.assertFalse(knowledge_base.search("API delivery late"))
        self.assertFalse(knowledge_base.remove_document("schedule.txt"))

        restored = LocalKnowledgeBase(self.persist_directory, embeddings=TestEmbeddings())
        self.assertTrue(restored.load_index())
        self.assertEqual(restored.document_count, 1)
        self.assertEqual([chunk.source for chunk in restored.all_chunks()], ["team.txt"])


if __name__ == "__main__":
    unittest.main()