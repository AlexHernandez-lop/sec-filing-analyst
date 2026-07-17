"""
Unit tests for the SEC Filing Analyst core modules.

Run with: pytest tests/ -v
"""

import os
import shutil
import pytest
from src.chunker import chunk_text, chunk_filing, detect_section, Chunk
from src.edgar_client import EdgarClient
from src.vector_store import VectorStore


class TestChunker:

    def test_short_text_returns_single_chunk(self):
        chunks = chunk_text("Apple reported revenue of $100 billion.")
        assert len(chunks) == 1
        assert "Apple" in chunks[0].text

    def test_long_text_produces_multiple_chunks(self):
        text = "This is a sentence about financials. " * 200
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        text = "Repeated content here. " * 200
        chunks = chunk_text(text, chunk_size=200)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_is_preserved(self):
        chunks = chunk_text("Some text", metadata={"company": "Apple"})
        assert chunks[0].metadata["company"] == "Apple"

    def test_detect_section_risk_factors(self):
        assert detect_section("ITEM 1A. Risk Factors The company faces...") == "Risk Factors"

    def test_detect_section_mda(self):
        assert detect_section("ITEM 7. Management's Discussion and Analysis...") == "MD&A"

    def test_detect_section_unknown_text(self):
        assert detect_section("Some paragraph without any item header.") == "Unknown"

    def test_chunk_filing_attaches_metadata(self):
        chunks = chunk_filing(
            text="Apple sells iPhones and services.",
            company="Apple Inc.",
            ticker="AAPL",
            form_type="10-K",
            filed_date="2025-10-31",
        )
        meta = chunks[0].metadata
        assert meta["company"] == "Apple Inc."
        assert meta["ticker"] == "AAPL"
        assert meta["form_type"] == "10-K"


class TestEdgarClient:

    def test_search_finds_apple(self):
        client = EdgarClient()
        results = client.search_company("AAPL")
        assert len(results) > 0
        assert any(r["ticker"] == "AAPL" for r in results)

    def test_search_returns_empty_for_nonsense(self):
        client = EdgarClient()
        results = client.search_company("XYZNOTREAL123")
        assert len(results) == 0

    def test_get_filings_returns_10k(self):
        client = EdgarClient()
        filings = client.get_filings(cik="0000320193", form_types=["10-K"], limit=1)
        assert len(filings) == 1
        assert filings[0].form_type == "10-K"
        assert filings[0].company != ""


class TestVectorStore:

    TEST_DIR = "./data/test_chroma"

    def setup_method(self):
        self.store = VectorStore(persist_dir=self.TEST_DIR, collection_name="test")

    def teardown_method(self):
        if os.path.exists(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)

    def test_add_and_search(self):
        chunks = [
            Chunk(text="Apple reported $394 billion in revenue", metadata={"company": "Apple"}),
            Chunk(text="Tesla delivered 1.8 million vehicles", metadata={"company": "Tesla"}),
            Chunk(text="Microsoft cloud revenue grew 22%", metadata={"company": "Microsoft"}),
        ]
        self.store.add_chunks(chunks)
        assert self.store.count == 3

        results = self.store.search("How much revenue did Apple make?", n_results=1)
        assert len(results) == 1
        assert "Apple" in results[0]["text"]

    def test_list_companies(self):
        chunks = [
            Chunk(text="Apple data", metadata={"company": "Apple"}),
            Chunk(text="Tesla data", metadata={"company": "Tesla"}),
        ]
        self.store.add_chunks(chunks)
        companies = self.store.list_companies()
        assert "Apple" in companies
        assert "Tesla" in companies

    def test_delete_company(self):
        chunks = [
            Chunk(text="Apple data", metadata={"company": "Apple"}),
            Chunk(text="Tesla data", metadata={"company": "Tesla"}),
        ]
        self.store.add_chunks(chunks)
        self.store.delete_company("Apple")
        assert self.store.count == 1
        assert "Apple" not in self.store.list_companies()

    def test_empty_store_returns_no_results(self):
        results = self.store.search("anything")
        assert results == []
