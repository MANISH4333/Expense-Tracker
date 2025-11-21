from typing import List, Optional

# Make FAISS optional at runtime. If unavailable, raise a friendly error
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _FAISS_AVAILABLE = True
except Exception:
    FAISS = None
    HuggingFaceEmbeddings = None
    _FAISS_AVAILABLE = False


def create_faiss_index(texts: List[str]):
    """Create a FAISS vectorstore from a list of texts.

    If FAISS is not available in the environment, raise a RuntimeError
    with a friendly message instructing how to enable it or use an
    alternative deployment.
    """
    if not _FAISS_AVAILABLE:
        raise RuntimeError(
            "FAISS is not available in this environment.\n"
            "To enable FAISS, deploy with a conda environment that installs 'faiss-cpu'\n"
            "or use a Docker image that includes prebuilt FAISS binaries.\n"
            "As a quick workaround, you can skip document indexing in this environment."
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts, embeddings)


def retrive_relevant_docs(vectorstore, query: str, k: int = 4):
    if not _FAISS_AVAILABLE or vectorstore is None:
        raise RuntimeError(
            "Vector search is not available because FAISS is missing or vectorstore is not initialized."
        )
    return vectorstore.similarity_search(query, k=k)

