"""
SEC Filing Analyst -- Streamlit web interface.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from src.edgar_client import EdgarClient
from src.chunker import chunk_filing
from src.vector_store import VectorStore
from src.rag import RAGPipeline

load_dotenv()

st.set_page_config(page_title="SEC Filing Analyst", page_icon="none", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 900px; }
    .source-tag {
        display: inline-block;
        background: #1e293b;
        color: #94a3b8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_edgar_client():
    email = os.getenv("SEC_USER_AGENT", "SECFilingAnalyst contact@example.com")
    return EdgarClient(user_agent=email)

@st.cache_resource
def get_rag_pipeline(_vector_store):
    model = os.getenv("LLM_MODEL", "haiku")
    return RAGPipeline(vector_store=_vector_store, model=model)


store = get_vector_store()
edgar = get_edgar_client()
rag = get_rag_pipeline(store)

# -- Sidebar: Filing ingestion --

with st.sidebar:
    st.header("Load SEC Filings")

    search_query = st.text_input("Search company (name or ticker)", placeholder="AAPL, Tesla, MSFT...")

    if search_query:
        with st.spinner("Searching EDGAR..."):
            results = edgar.search_company(search_query)

        if not results:
            st.warning("No companies found.")
        else:
            options = {f"{r['ticker']} -- {r['company']}": r for r in results[:5]}
            selected = st.selectbox("Select company", list(options.keys()))
            company = options[selected]

            form_types = st.multiselect("Filing types", ["10-K", "10-Q", "8-K"], default=["10-K"])
            max_filings = st.slider("Max filings to load", 1, 10, 3)

            if st.button("Download and Index", type="primary"):
                with st.status("Processing filings...", expanded=True) as status:
                    filings = edgar.get_filings(
                        cik=company["cik"],
                        form_types=form_types,
                        limit=max_filings,
                    )

                    if not filings:
                        st.error("No filings found for this company.")
                    else:
                        total_chunks = 0
                        for filing in filings:
                            st.write(f"Downloading {filing.form_type} ({filing.filed_date})...")
                            try:
                                text = edgar.download_filing_text(filing)
                                st.write(f"  {len(text):,} characters")

                                chunks = chunk_filing(
                                    text=text,
                                    company=filing.company,
                                    ticker=filing.ticker,
                                    form_type=filing.form_type,
                                    filed_date=filing.filed_date,
                                )
                                added = store.add_chunks(chunks)
                                total_chunks += added
                                st.write(f"  {added} chunks indexed")
                            except Exception as e:
                                st.error(f"  Error: {e}")

                        status.update(label=f"Done. {total_chunks} chunks added.", state="complete")

    st.divider()

    st.subheader("Indexed Filings")
    companies = store.list_companies()
    if companies:
        for comp in companies:
            with st.expander(comp):
                filings = store.list_filings(company=comp)
                for f in filings:
                    st.text(f"{f['form_type']}  {f['filed_date']}")
                if st.button(f"Remove {comp}", key=f"del_{comp}"):
                    store.delete_company(comp)
                    st.rerun()

        st.caption(f"Total chunks: {store.count}")
    else:
        st.info("No filings indexed yet. Search for a company above.")

# -- Main chat area --

st.title("SEC Filing Analyst")
st.caption("Ask questions about public companies using their official SEC filings (10-K, 10-Q, 8-K)")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning("Set your ANTHROPIC_API_KEY in the .env file to enable the analyst.")
    st.code("echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env", language="bash")
    st.stop()

if store.count == 0:
    st.info("Start by loading filings from the sidebar.")
    st.stop()

companies = store.list_companies()
company_filter = st.selectbox("Filter by company (optional)", ["All companies"] + companies)
active_company = None if company_filter == "All companies" else company_filter

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-tag">{src["company"]} | {src["form_type"]} | {src["filed_date"]} | {src["section"]}</span>',
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("Ask about financials, risks, strategy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            try:
                result = rag.query(question=prompt, company=active_company, n_results=8)
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("Sources"):
                        for src in result["sources"]:
                            st.markdown(
                                f'<span class="source-tag">{src["company"]} | {src["form_type"]} | {src["filed_date"]} | {src["section"]}</span>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.error(f"Error: {e}")
