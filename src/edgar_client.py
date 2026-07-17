"""
SEC EDGAR API client.

Handles company lookup, filing retrieval, and document downloading
via the free EDGAR Full-Text Search and Company Submissions APIs.
No authentication required — only a User-Agent header with contact info.
"""

import re
import time
import requests
from dataclasses import dataclass


@dataclass
class Filing:
    company: str
    ticker: str
    cik: str
    form_type: str
    filed_date: str
    accession_number: str
    document_url: str
    description: str = ""


class EdgarClient:
    """Client for the SEC EDGAR public API."""

    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, user_agent: str = "SECFilingAnalyst contact@example.com"):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    def search_company(self, query: str) -> list[dict]:
        """Search for a company by name or ticker symbol."""
        resp = self.session.get(self.TICKERS_URL)
        resp.raise_for_status()
        data = resp.json()

        query_lower = query.lower().strip()
        results = []
        for entry in data.values():
            ticker = entry.get("ticker", "").lower()
            title = entry.get("title", "").lower()
            if query_lower in ticker or query_lower in title:
                results.append({
                    "cik": str(entry["cik_str"]).zfill(10),
                    "ticker": entry["ticker"],
                    "company": entry["title"],
                })
        return results[:10]

    def get_filings(
        self,
        cik: str,
        form_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Filing]:
        """Retrieve recent filings for a company by CIK number."""
        if form_types is None:
            form_types = ["10-K", "10-Q", "8-K"]

        cik_padded = cik.zfill(10)
        url = f"{self.SUBMISSIONS_URL}/CIK{cik_padded}.json"

        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()

        company_name = data.get("name", "Unknown")
        tickers = data.get("tickers", [])
        ticker = tickers[0] if tickers else ""

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings = []
        for i in range(len(forms)):
            if forms[i] not in form_types:
                continue
            acc_no_raw = accessions[i].replace("-", "")
            doc_url = f"{self.ARCHIVES_URL}/{cik_padded}/{acc_no_raw}/{primary_docs[i]}"
            filings.append(Filing(
                company=company_name,
                ticker=ticker,
                cik=cik_padded,
                form_type=forms[i],
                filed_date=dates[i],
                accession_number=accessions[i],
                document_url=doc_url,
                description=descriptions[i] if i < len(descriptions) else "",
            ))
            if len(filings) >= limit:
                break

        return filings

    def download_filing_text(self, filing: Filing) -> str:
        """Download a filing document and extract its text content."""
        time.sleep(0.11)  # EDGAR rate limit: 10 requests/second
        resp = self.session.get(filing.document_url)
        resp.raise_for_status()

        text = resp.text
        if filing.document_url.endswith((".htm", ".html")):
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r"&amp;", "&", text)
            text = re.sub(r"&lt;", "<", text)
            text = re.sub(r"&gt;", ">", text)
            text = re.sub(r"&#\d+;", " ", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
