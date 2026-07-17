"""
Document chunking for SEC filings.

Splits filing text into overlapping chunks with automatic SEC section
detection (Item 1, 1A, 7, etc.), optimized for retrieval in a RAG pipeline.
"""

import re
import hashlib
from dataclasses import dataclass, field

_chunk_counter = 0


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            global _chunk_counter
            _chunk_counter += 1
            raw = f"{_chunk_counter}:{self.text}"
            self.chunk_id = hashlib.md5(raw.encode()).hexdigest()[:16]


SEC_SECTION_MAP = {
    "1.": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2.": "Properties",
    "3.": "Legal Proceedings",
    "4.": "Mine Safety",
    "5.": "Market Info",
    "6.": "Selected Financial Data",
    "7.": "MD&A",
    "7A": "Quantitative Disclosures",
    "8.": "Financial Statements",
    "9.": "Accountant Disagreements",
    "9A": "Controls & Procedures",
    "10": "Directors & Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Related Transactions",
    "14": "Accountant Fees",
    "15": "Exhibits",
}


def detect_section(text: str) -> str:
    """Identify which SEC filing section a text block belongs to."""
    match = re.search(r"(?i)\bITEM\s+(1A|1B|7A|9A|\d{1,2})[.\s]", text[:500])
    if match:
        key = match.group(1).upper()
        return SEC_SECTION_MAP.get(key + ".", SEC_SECTION_MAP.get(key, "Unknown"))
    return "Unknown"


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks at paragraph boundaries.

    For SEC filings, automatically detects section headers (ITEM 1, 1A, 7, etc.)
    and attaches section labels to chunk metadata.
    """
    if metadata is None:
        metadata = {}

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < chunk_size:
        return [Chunk(text=text, metadata={**metadata, "section": detect_section(text)})]

    chunks = []
    paragraphs = re.split(r"\n\s*\n|\.\s{2,}", text)
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    current_chunk = ""
    current_section = "Unknown"

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        section = detect_section(para)
        if section != "Unknown":
            current_section = section

        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                metadata={**metadata, "section": current_section},
            ))
            words = current_chunk.split()
            overlap_words = words[-chunk_overlap // 5:] if len(words) > chunk_overlap // 5 else []
            current_chunk = " ".join(overlap_words) + " " + para
        else:
            current_chunk += " " + para

    if current_chunk.strip():
        chunks.append(Chunk(
            text=current_chunk.strip(),
            metadata={**metadata, "section": current_section},
        ))

    return chunks


def chunk_filing(
    text: str,
    company: str,
    ticker: str,
    form_type: str,
    filed_date: str,
    **extra,
) -> list[Chunk]:
    """Chunk a full filing with structured metadata."""
    metadata = {
        "company": company,
        "ticker": ticker,
        "form_type": form_type,
        "filed_date": filed_date,
        **extra,
    }
    return chunk_text(text, metadata=metadata)
