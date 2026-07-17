"""
Vector store for SEC filing chunks.

Wraps ChromaDB to provide embedding, persistent storage, and semantic
search over chunked filing documents.
"""

import chromadb
from chromadb.utils import embedding_functions
from src.chunker import Chunk


class VectorStore:
    """ChromaDB-backed vector store with cosine similarity search."""

    def __init__(self, persist_dir: str = "./data/chroma_db", collection_name: str = "sec_filings"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the store. Skips duplicates. Returns count added."""
        if not chunks:
            return 0

        existing_ids = set()
        if self.collection.count() > 0:
            try:
                result = self.collection.get()
                existing_ids = set(result["ids"])
            except Exception:
                pass

        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            if chunk.chunk_id in existing_ids:
                continue
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)

        if not ids:
            return 0

        batch_size = 500
        added = 0
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
            added += len(ids[i:i + batch_size])

        return added

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Semantic search over stored chunks. Returns ranked results."""
        if self.collection.count() == 0:
            return []

        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self.collection.count()),
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self.collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
                "id": results["ids"][0][i],
            })
        return output

    def list_companies(self) -> list[str]:
        """Return all unique company names in the store."""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        return sorted({m["company"] for m in result["metadatas"] if "company" in m})

    def list_filings(self, company: str | None = None) -> list[dict]:
        """Return unique filings, optionally filtered by company."""
        if self.collection.count() == 0:
            return []

        kwargs = {"include": ["metadatas"]}
        if company:
            kwargs["where"] = {"company": company}

        result = self.collection.get(**kwargs)
        seen = set()
        filings = []
        for meta in result["metadatas"]:
            key = (meta.get("company", ""), meta.get("form_type", ""), meta.get("filed_date", ""))
            if key not in seen:
                seen.add(key)
                filings.append({
                    "company": meta.get("company", ""),
                    "ticker": meta.get("ticker", ""),
                    "form_type": meta.get("form_type", ""),
                    "filed_date": meta.get("filed_date", ""),
                })
        return sorted(filings, key=lambda x: x["filed_date"], reverse=True)

    def delete_company(self, company: str) -> int:
        """Remove all chunks for a given company. Returns count deleted."""
        if self.collection.count() == 0:
            return 0
        result = self.collection.get(where={"company": company})
        if result["ids"]:
            self.collection.delete(ids=result["ids"])
            return len(result["ids"])
        return 0

    def reset(self):
        """Delete all stored data."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
