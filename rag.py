import logging
import os
import uuid
from itertools import zip_longest

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from github_loader import load_repository_documents

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = logging.getLogger(__name__)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

INITIAL_RETRIEVAL_K = int(os.getenv("INITIAL_RETRIEVAL_K", "10"))

# reranking with a cross-encoder
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))   # chunks sent to the LLM

bm25_retriever = None
vector_retriever = None

vectorstore = None
loaded_repository = None
_embeddings = None
_reranker = None
_reranker_name = None
_llm = None

_LANGUAGE_BY_EXTENSION = {
    ".py": "PYTHON", ".ipynb": "PYTHON", ".js": "JS", ".jsx": "JS",
    ".ts": "TS", ".tsx": "TS", ".java": "JAVA", ".html": "HTML", ".htm": "HTML",
    ".md": "MARKDOWN", ".go": "GO", ".rs": "RUST", ".cpp": "CPP", ".hpp": "CPP",
    ".c": "C", ".h": "C", ".cs": "CSHARP", ".php": "PHP", ".rb": "RUBY",
    ".kt": "KOTLIN", ".swift": "SWIFT", ".scala": "SCALA",
}

PROMPT_TEMPLATE = """You are a helpful GitHub Repository Question Answering Assistant.
Answer the user's question using only the repository context below.
If the answer is not available in the context, say: "I could not find the answer in the repository."

Repository Context:
{context}

User Question:
{question}

Answer:"""

def get_embeddings():
    """Return the (cached) Hugging Face embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0)
    return _llm


def _get_reranker():
    """Return the (cached) cross-encoder. Reloaded if RERANKER_MODEL changes."""
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != RERANKER_MODEL:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
        _reranker_name = RERANKER_MODEL
    return _reranker


def split_documents(documents):
    """
    Split file documents into chunks. Code/markdown files use a language-aware
    splitter; every chunk keeps the file metadata and starts with its file path
    so the retriever and the LLM always know where it came from.
    """
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splitters = {}
    splits = []

    for document in documents:
        file_type = document.metadata.get("file_type", "")
        language_name = _LANGUAGE_BY_EXTENSION.get(file_type)

        if language_name and hasattr(Language, language_name):
            if language_name not in splitters:
                splitters[language_name] = RecursiveCharacterTextSplitter.from_language(
                    language=getattr(Language, language_name),
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
            splitter = splitters[language_name]
        else:
            splitter = default_splitter

        source = document.metadata.get("source", "unknown")
        for chunk in splitter.split_documents([document]):
            chunk.page_content = f"File: {source}\n{chunk.page_content}"
            splits.append(chunk)

    return splits


def rerank_documents(question, documents, top_n=None):
    """
    Re-score the candidate chunks with a cross-encoder and keep the best ones.
    The relevance score is stored in ``metadata["rerank_score"]``.
    If the reranker cannot be used, the original order is kept.
    """
    top_n = top_n or RERANK_TOP_N
    if not documents:
        return []

    try:
        reranker = _get_reranker()
        scores = reranker.predict([(question, doc.page_content) for doc in documents])
    except Exception as exc:  # model download / inference problems
        logger.warning("Reranking failed (%s). Using the initial retrieval order.", exc)
        return documents[:top_n]

    ranked = sorted(zip(documents, scores), key=lambda pair: float(pair[1]), reverse=True)

    reranked = []
    for doc, score in ranked[:top_n]:
        metadata = dict(doc.metadata)
        metadata["rerank_score"] = float(score)
        reranked.append(Document(page_content=doc.page_content, metadata=metadata))
    return reranked


def retrieve_documents(question):
    """
    Initial retrieval (BM25 + ChromaDB) -> duplicate removal -> reranking.
    Returns the final chunks that are sent to the LLM.
    """
    bm25_docs = bm25_retriever.invoke(question)
    vector_docs = vector_retriever.invoke(question)



    candidates = []
    seen = set()
    for pair in zip_longest(bm25_docs, vector_docs):
        for doc in pair:
            if doc is not None and doc.page_content not in seen:
                seen.add(doc.page_content)
                candidates.append(doc)

    if RERANK_ENABLED:
        return rerank_documents(question, candidates, RERANK_TOP_N)
    return candidates[:RERANK_TOP_N]
def load_repository(github_url):

    global bm25_retriever
    global vector_retriever
    global vectorstore
    global loaded_repository
    documents = load_repository_documents(github_url)

    splits = split_documents(documents)

    new_bm25 = BM25Retriever.from_documents(splits)
    new_bm25.k = INITIAL_RETRIEVAL_K

    embeddings = get_embeddings()
    new_vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=f"github_repo_{uuid.uuid4().hex[:12]}",
    )
    new_vector_retriever = new_vectorstore.as_retriever(
        search_kwargs={"k": INITIAL_RETRIEVAL_K}
    )

    
    old_vectorstore = vectorstore
    bm25_retriever = new_bm25
    vector_retriever = new_vector_retriever
    vectorstore = new_vectorstore
    loaded_repository = github_url

    if old_vectorstore is not None:
        try:
            old_vectorstore.delete_collection()
        except Exception as exc:
            logger.warning("Could not delete the previous collection: %s", exc)

    file_count = len({doc.metadata.get("source") for doc in documents})
    return f"Repository loaded successfully! ({file_count} files, {len(splits)} chunks)"


def answer_question(question):
    """
    Full RAG answer. Returns a dict with the answer, the reranked context
    chunks that were given to the LLM, and their source files.
    """
    if bm25_retriever is None or vector_retriever is None:
        return {
            "answer": "Please load a GitHub repository first.",
            "contexts": [],
            "sources": [],
        }

    final_docs = retrieve_documents(question)
    context = "\n\n".join(doc.page_content for doc in final_docs)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )
    response = _get_llm().invoke(final_prompt)

    sources = []
    for doc in final_docs:
        entry = {"source": doc.metadata.get("source", "unknown")}
        if "rerank_score" in doc.metadata:
            entry["rerank_score"] = round(doc.metadata["rerank_score"], 4)
        sources.append(entry)

    return {
        "answer": response.content,
        "contexts": [doc.page_content for doc in final_docs],
        "sources": sources,
    }

def ask_question(question):
    return answer_question(question)["answer"]