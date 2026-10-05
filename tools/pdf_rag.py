import os
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

PDF_FILENAME = "tata_motors_pv_trimmed.pdf"

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    PDF_FILENAME,
)

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        pages = PyPDFLoader(PDF_PATH).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        _retriever = BM25Retriever.from_documents(chunks)
        _retriever.k = 3
    return _retriever


def _original_page(trimmed_page):
    """Map a page of the trimmed PDF (1-based) to the page of the original report PDF."""
    if trimmed_page <= 18:        # trimmed 1-18  -> original 20-37
        return trimmed_page + 19
    if trimmed_page <= 30:        # trimmed 19-30 -> original 154-165
        return trimmed_page + 135
    return trimmed_page + 232     # trimmed 31-51 -> original 263-283


@tool
def search_company_pdf(query: str) -> str:
    """Search the company's private annual report PDF and return the most relevant
    passages with page numbers. Input must be a plain-text search query, for example
    'electric vehicle sales' or 'revenue growth'. Use this for the company's own
    internal figures and strategy."""
    docs = get_retriever().invoke(query)
    if not docs:
        return "No relevant passages found in the PDF."
    return "\n\n".join(
        f"[Page {_original_page(d.metadata.get('page', 0) + 1)}] {d.page_content}"
        for d in docs
    )