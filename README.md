# SEC Filing Analyst

A retrieval-augmented generation (RAG) pipeline that answers questions about public companies using their official SEC filings. It downloads 10-K, 10-Q, and 8-K reports from EDGAR, indexes them in a local vector store, and uses an LLM to generate cited answers grounded exclusively in filing data.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Architecture

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  SEC EDGAR   │────>│   Chunker    │────>│  ChromaDB    │────>│  Claude API  │
 │  (fetch)     │     │  (split)     │     │  (embed)     │     │  (generate)  │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
  Download filing      1000-char chunks     all-MiniLM-L6-v2     Retrieve top-k
  HTML, strip tags     with overlap and     local embeddings     chunks, generate
                       section detection    cosine similarity    cited answer
```

1. **Ingest** -- Fetches filings from the SEC EDGAR API (free, no key required).
2. **Chunk** -- Splits documents into overlapping segments with automatic section detection (Item 1, 1A, 7, 7A, 8, etc.).
3. **Embed** -- Encodes chunks with all-MiniLM-L6-v2, stored in ChromaDB. Runs locally at zero cost.
4. **Retrieve** -- For each question, retrieves the top 8 most relevant chunks by cosine similarity.
5. **Generate** -- Passes retrieved chunks as context to Claude, which produces an answer citing the specific filing, date, and section.

## Tech Stack

| Component    | Technology              | Rationale                                      |
|------------- |------------------------ |------------------------------------------------|
| Data source  | SEC EDGAR API           | Official, free, no authentication              |
| Chunking     | Custom, section-aware   | Preserves 10-K/10-Q document structure         |
| Embeddings   | all-MiniLM-L6-v2        | Local inference, no API cost, no data exfiltration |
| Vector store | ChromaDB (persistent)   | Lightweight, serverless, file-based            |
| LLM          | Claude Haiku 4.5        | Strong document comprehension at ~$0.002/query |
| Interface    | Streamlit / CLI         | Rapid prototyping with built-in chat widget    |

## Setup

```bash
git clone https://github.com/AlexHernandez-lop/sec-filing-analyst.git
cd sec-filing-analyst

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Usage

### Index filings

```bash
python cli.py ingest AAPL --forms 10-K --limit 1
python cli.py ingest MSFT --forms 10-K,10-Q --limit 3
python cli.py ingest TSLA
```

### Ask questions

```bash
python cli.py ask "What were Apple's total net sales in fiscal 2025?"
python cli.py ask "What are the main risk factors?" --company "Apple Inc."
```

### Interactive chat

```bash
python cli.py chat
```

### Web interface

```bash
streamlit run app.py
```

### List indexed data

```bash
python cli.py list
```

## Sample Output

```
Question: What were Apple's total net sales in the last fiscal year?
Model: haiku

Apple's total net sales for fiscal year 2025 (ended September 27, 2025)
were $416,161 million ($416.161 billion).

Breakdown:
  Products: $307,003 million (iPhone $209,586M, Mac $33,708M, iPad $28,023M)
  Services: $109,158 million

This represents a 6% increase over fiscal year 2024 ($391,035 million).

------------------------------------------------------------
Sources:
  - Apple Inc. | 10-K | 2025-10-31 | Financial Statements
```

## Configuration

| Variable           | Default                              | Description                              |
|--------------------|--------------------------------------|------------------------------------------|
| `ANTHROPIC_API_KEY`| (required)                           | Anthropic API key                        |
| `LLM_MODEL`        | `haiku`                              | `haiku` (~$0.002/query) or `sonnet` (~$0.01/query) |
| `SEC_USER_AGENT`   | `SECFilingAnalyst contact@example.com` | Contact info for EDGAR rate-limit compliance |

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
sec-filing-analyst/
├── app.py                  Streamlit web interface
├── cli.py                  Command-line interface
├── src/
│   ├── edgar_client.py     EDGAR API client (search, fetch, download)
│   ├── chunker.py          Section-aware document chunking
│   ├── vector_store.py     ChromaDB embedding and search
│   └── rag.py              RAG pipeline (retrieve + generate)
├── tests/
│   └── test_core.py        Unit tests
├── requirements.txt
├── Makefile
└── .env.example
```

## Limitations

- Extracts text only; tabular data and charts may lose formatting during HTML stripping.
- Embedding model is general-purpose. Domain-specific models (e.g., FinBERT) would improve retrieval precision on financial terminology.
- No XBRL parsing. Structured financial data from inline XBRL is not leveraged.
- Single-user local deployment. Production use would require hosted vector storage and authentication.

## License

MIT
