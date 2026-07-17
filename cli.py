#!/usr/bin/env python3
"""
SEC Filing Analyst -- Command-line interface.

Usage:
    python cli.py ingest AAPL --forms 10-K --limit 2
    python cli.py ask "What are Apple's main risk factors?"
    python cli.py list
    python cli.py chat
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.edgar_client import EdgarClient
from src.chunker import chunk_filing
from src.vector_store import VectorStore
from src.rag import RAGPipeline


def cmd_ingest(args):
    """Download and index SEC filings for a company."""
    edgar = EdgarClient()
    store = VectorStore()

    print(f"Searching for '{args.company}'...")
    results = edgar.search_company(args.company)

    if not results:
        print(f"No company found matching '{args.company}'.")
        return

    company = results[0]
    print(f"Found: {company['ticker']} -- {company['company']} (CIK: {company['cik']})")

    forms = args.forms.split(",") if args.forms else ["10-K"]
    print(f"Fetching up to {args.limit} filing(s) ({', '.join(forms)})...\n")
    filings = edgar.get_filings(cik=company["cik"], form_types=forms, limit=args.limit)

    if not filings:
        print("No filings found for the specified criteria.")
        return

    total = 0
    for filing in filings:
        print(f"  [{filing.form_type}] {filing.filed_date}")
        try:
            text = edgar.download_filing_text(filing)
            print(f"    Downloaded {len(text):,} characters")

            chunks = chunk_filing(
                text=text,
                company=filing.company,
                ticker=filing.ticker,
                form_type=filing.form_type,
                filed_date=filing.filed_date,
            )
            added = store.add_chunks(chunks)
            total += added
            print(f"    Indexed {added} chunks")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nDone. {total} chunks indexed ({store.count} total in store).")


def cmd_ask(args):
    """Ask a question using RAG over indexed filings."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    store = VectorStore()
    if store.count == 0:
        print("No filings indexed. Run 'python cli.py ingest <company>' first.")
        sys.exit(1)

    model = os.getenv("LLM_MODEL", "haiku")
    rag = RAGPipeline(vector_store=store, model=model)

    company = args.company if args.company != "all" else None
    question = " ".join(args.question)

    print(f"\nQuestion: {question}")
    if company:
        print(f"Filtered to: {company}")
    print(f"Model: {model}\n")

    result = rag.query(question=question, company=company)

    print(result["answer"])
    print("\n" + "-" * 60)
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src['company']} | {src['form_type']} | {src['filed_date']} | {src['section']}")


def cmd_list(args):
    """List all indexed companies and filings."""
    store = VectorStore()
    companies = store.list_companies()

    if not companies:
        print("No filings indexed yet.")
        return

    print(f"{len(companies)} company(ies) indexed ({store.count} total chunks)\n")
    for comp in companies:
        filings = store.list_filings(company=comp)
        print(f"  {comp}")
        for f in filings:
            print(f"    {f['form_type']}  {f['filed_date']}")
        print()


def cmd_chat(args):
    """Interactive chat session."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    store = VectorStore()
    if store.count == 0:
        print("No filings indexed. Run 'python cli.py ingest <company>' first.")
        sys.exit(1)

    model = os.getenv("LLM_MODEL", "haiku")
    rag = RAGPipeline(vector_store=store, model=model)

    print("SEC Filing Analyst -- Interactive Mode")
    print(f"Model: {model} | Indexed chunks: {store.count}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break

        result = rag.query(question=question)
        print(f"\nAnalyst: {result['answer']}\n")


def main():
    parser = argparse.ArgumentParser(description="SEC Filing Analyst")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Download and index filings")
    p_ingest.add_argument("company", help="Company name or ticker symbol")
    p_ingest.add_argument("--forms", default="10-K", help="Comma-separated form types (default: 10-K)")
    p_ingest.add_argument("--limit", type=int, default=3, help="Max filings to retrieve (default: 3)")

    p_ask = sub.add_parser("ask", help="Ask a question about indexed filings")
    p_ask.add_argument("question", nargs="+", help="Your question")
    p_ask.add_argument("--company", default="all", help="Filter by company name")

    sub.add_parser("list", help="List indexed companies and filings")
    sub.add_parser("chat", help="Start an interactive chat session")

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "list": cmd_list,
        "chat": cmd_chat,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
