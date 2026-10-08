import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from qdrant_client import QdrantClient

import pytest

from hash5_rag.config import Settings
from hash5_rag.guards import Hash5Guards
from hash5_rag.vectorstore import Hash5VectorStore
from tests.fakes import BoundedEmbeddings, ZeroEmbeddings


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.qdrant_collection = "hash5_kb_test"
    s.top_k = 2
    s.max_iterations = 2
    return s


@pytest.fixture
def guards() -> Hash5Guards:
    return Hash5Guards()


@pytest.fixture
def store(settings: Settings):
    s = Hash5VectorStore(settings, QdrantClient(":memory:"))
    yield s
    s.close()


@pytest.fixture
def seeded_store(settings: Settings):
    from langchain_core.documents import Document

    store = Hash5VectorStore(settings, QdrantClient(":memory:"))
    docs = [
        Document(
            page_content="Building setbacks define minimum open space between plot boundary and building to ensure light, ventilation and safety.",
            metadata={"source": "tncdrbr.pdf", "page": 1, "title": "TNCDRBR 2019", "chunk_id": 0},
        ),
        Document(
            page_content="Floor area ratio (FAR) is the ratio of built-up area to plot area as per development regulations.",
            metadata={"source": "nbc_ch3.pdf", "page": 2, "title": "NBC", "chunk_id": 1},
        ),
    ]
    store.create_collection(384)
    store.add_documents(docs, [[1.0] * 384, [1.0] * 384])
    yield store
    store.close()


@pytest.fixture
def embeddings() -> ZeroEmbeddings:
    return ZeroEmbeddings()


@pytest.fixture
def field_embeddings() -> BoundedEmbeddings:
    return BoundedEmbeddings(["nato"])