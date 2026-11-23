from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from advisor.ingestion import chunk_text


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "project_artifacts"


@dataclass(frozen=True)
class SearchResult:
    source: str
    excerpt: str
    score: float
    chunk_number: int


@dataclass(frozen=True)
class IndexedChunk:
    source: str
    text: str
    chunk_number: int


class LocalKnowledgeBase:
    """LangChain and Chroma index backed by local semantic embeddings."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        embeddings: Embeddings | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self._embeddings = embeddings
        self._documents: list[Document] = []
        self._source_names: set[str] = set()
        self._pending_sources: set[str] = set()
        self._vector_store: Chroma | None = None

    @property
    def document_count(self) -> int:
        return len(self._source_names)

    @property
    def chunk_count(self) -> int:
        return len(self._documents)

    @property
    def source_summaries(self) -> list[dict[str, str | int]]:
        chunks_by_source = Counter(document.metadata["source"] for document in self._documents)
        return [
            {"name": source, "chunks": count}
            for source, count in sorted(chunks_by_source.items())
        ]

    def all_chunks(self) -> list[IndexedChunk]:
        """Return indexed text and source metadata for local project analysis."""
        return [
            IndexedChunk(
                source=document.metadata["source"],
                text=document.page_content,
                chunk_number=int(document.metadata["chunk_number"]),
            )
            for document in self._documents
        ]

    def add_document(self, source: str, text: str) -> int:
        chunks = chunk_text(text)
        if not chunks:
            return 0

        self._documents = [
            document
            for document in self._documents
            if document.metadata["source"] != source
        ]
        self._documents.extend(
            Document(
                page_content=chunk,
                metadata={"source": source, "chunk_number": number},
            )
            for number, chunk in enumerate(chunks, start=1)
        )
        self._source_names.add(source)
        self._pending_sources.add(source)
        return len(chunks)

    def remove_document(self, source: str) -> bool:
        """Remove a source from the in-memory documents and persisted collection."""
        if source not in self._source_names:
            return False

        if self._vector_store is not None:
            self._vector_store.delete(where={"source": source})
        elif self.persist_directory.exists():
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            collection_names = {collection.name for collection in client.list_collections()}
            if COLLECTION_NAME in collection_names:
                client.get_collection(COLLECTION_NAME).delete(where={"source": source})

        self._documents = [
            document
            for document in self._documents
            if document.metadata["source"] != source
        ]
        self._source_names.remove(source)
        self._pending_sources.discard(source)
        return True

    def build_index(self) -> int:
        """Persist only newly added or replaced sources in the Chroma collection."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_directory))
        collection_names = {collection.name for collection in client.list_collections()}
        if not self._pending_sources:
            return 0

        if self._embeddings is None:
            self._embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

        if COLLECTION_NAME in collection_names:
            self._vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self._embeddings,
            )
        else:
            self._vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self._embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )

        for source in self._pending_sources:
            self._vector_store.delete(where={"source": source})

        pending_documents = [
            document
            for document in self._documents
            if document.metadata["source"] in self._pending_sources
        ]
        if pending_documents:
            self._vector_store.add_documents(
                documents=pending_documents,
                ids=[
                    f"{document.metadata['source']}:{document.metadata['chunk_number']}"
                    for document in pending_documents
                ],
            )
        self._pending_sources.clear()
        return self.chunk_count

    def load_index(self) -> bool:
        """Reopen a previously persisted collection for a new app session."""
        if not self.persist_directory.exists():
            return False

        client = chromadb.PersistentClient(path=str(self.persist_directory))
        collection_names = {collection.name for collection in client.list_collections()}
        if COLLECTION_NAME not in collection_names:
            return False

        collection = client.get_collection(COLLECTION_NAME)
        stored = collection.get(include=["documents", "metadatas"])
        self._documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(stored["documents"], stored["metadatas"])
            if text is not None and metadata is not None
        ]
        self._source_names = {document.metadata["source"] for document in self._documents}
        self._pending_sources.clear()
        if not self._documents:
            return False

        if self._embeddings is None:
            self._embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        self._vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        return True

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or self._vector_store is None or top_k <= 0:
            return []

        results: list[SearchResult] = []
        matches = self._vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        for document, score in matches:
            if score > 0:
                results.append(
                    SearchResult(
                        source=document.metadata["source"],
                        excerpt=document.page_content,
                        score=float(score),
                        chunk_number=int(document.metadata["chunk_number"]),
                    )
                )
        return results