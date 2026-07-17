"""
RAG pipeline for SEC filing analysis.

Orchestrates retrieval of relevant filing chunks and answer generation
via the Anthropic API.
"""

import anthropic
from src.vector_store import VectorStore

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250116",
}

SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions about 
public companies strictly based on excerpts from their official SEC filings 
(10-K, 10-Q, 8-K).

Rules:
- Only use information present in the provided excerpts. If the answer is not 
  covered by the excerpts, state that clearly. Never fabricate financial data.
- Cite the source of each claim: company name, filing type, filing date, and 
  section (e.g. Risk Factors, MD&A, Financial Statements).
- Reproduce exact figures (revenue, expenses, ratios) as they appear in the filing.
- If a question is ambiguous, ask for clarification before answering.
- Lead with the direct answer, then provide supporting detail.
- Note when data may be outdated and recommend checking more recent filings."""


class RAGPipeline:
    """Retrieve-then-generate pipeline over indexed SEC filings."""

    def __init__(
        self,
        vector_store: VectorStore,
        model: str = "haiku",
        api_key: str | None = None,
    ):
        self.vector_store = vector_store
        self.model_id = MODELS.get(model, model)
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def retrieve(self, query: str, n_results: int = 8, company: str | None = None) -> list[dict]:
        """Fetch the most relevant chunks for a query."""
        filter_meta = {"company": company} if company else None
        return self.vector_store.search(query, n_results=n_results, filter_metadata=filter_meta)

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        if not chunks:
            return "No relevant filing excerpts found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            header = (
                f"[Excerpt {i}] "
                f"{meta.get('company', 'Unknown')} | "
                f"{meta.get('form_type', '?')} | "
                f"Filed: {meta.get('filed_date', '?')} | "
                f"Section: {meta.get('section', '?')}"
            )
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def query(
        self,
        question: str,
        n_results: int = 8,
        company: str | None = None,
        stream: bool = False,
    ):
        """
        Run the full RAG pipeline: retrieve context, then generate an answer.

        Returns a dict with keys: answer, sources, model, chunks_used.
        If stream=True, returns a generator yielding text tokens.
        """
        chunks = self.retrieve(question, n_results=n_results, company=company)
        context = self._format_context(chunks)

        user_message = (
            f"Based on the following SEC filing excerpts, answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Filing Excerpts:\n{context}"
        )

        messages = [{"role": "user", "content": user_message}]

        if stream:
            return self._stream_response(messages)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        answer = response.content[0].text

        sources = []
        seen = set()
        for chunk in chunks:
            meta = chunk["metadata"]
            key = (meta.get("company"), meta.get("form_type"), meta.get("filed_date"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "company": meta.get("company", ""),
                    "form_type": meta.get("form_type", ""),
                    "filed_date": meta.get("filed_date", ""),
                    "section": meta.get("section", ""),
                })

        return {
            "answer": answer,
            "sources": sources,
            "model": self.model_id,
            "chunks_used": len(chunks),
        }

    def _stream_response(self, messages):
        """Yield response tokens as they arrive."""
        with self.client.messages.stream(
            model=self.model_id,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
