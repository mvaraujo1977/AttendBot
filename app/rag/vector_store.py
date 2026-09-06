"""Criação do vector store (ChromaDB persistido em disco)."""

from app.config import Configuracoes
from app.rag.embeddings import criar_embeddings


def criar_vector_store(config: Configuracoes):
    """Abre (ou cria) a coleção do Chroma configurada.

    A coleção usa distância de cosseno para que a conversão em similaridade
    feita no ``Retriever`` seja simples e previsível.
    """
    from langchain_chroma import Chroma

    config.diretorio_chroma.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.colecao_chroma,
        embedding_function=criar_embeddings(config.modelo_embedding),
        persist_directory=str(config.diretorio_chroma),
        collection_metadata={"hnsw:space": "cosine"},
    )
