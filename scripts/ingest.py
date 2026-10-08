"""Ingest documents from data/raw into the Qdrant defence knowledge base.

Usage:
  uv run python scripts/ingest.py [--reset]
"""
import click

from hash5_rag.config import get_settings
from hash5_rag.documents import load_sources, split_documents
from hash5_rag.llm import get_cached_embeddings
from hash5_rag.vectorstore import Hash5VectorStore


@click.command()
@click.option("--reset", is_flag=True, help="Drop and recreate the collection")
def main(reset: bool):
    settings = get_settings()
    store = Hash5VectorStore(settings)
    try:
        embeddings = get_cached_embeddings()
        docs = load_sources(settings.data_dir)
        chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
        click.echo(f"Loaded {len(docs)} page-documents -> {len(chunks)} chunks")

        if reset:
            store.delete_collection()
            click.echo("dropped existing collection")

        vectors = embeddings.embed_documents([c.page_content for c in chunks])
        store.create_collection(len(vectors[0]))
        added = store.add_documents(chunks, vectors)
        click.echo(f"Indexed {added} chunks into qdrant://{settings.qdrant_collection} "
                   f"(store now holds {store.count})")
    finally:
        store.close()


if __name__ == "__main__":
    main()