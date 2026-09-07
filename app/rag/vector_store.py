"""Criação do vector store (ChromaDB)."""

from app.config import Configuracoes
from app.rag.embedding.base import AdaptadorLangChain, ProvedorEmbedding


def criar_vector_store(config: Configuracoes, embedding: ProvedorEmbedding):
    """Abre (ou cria) a coleção do Chroma configurada.

    A coleção usa distância de cosseno para que a conversão em similaridade
    feita no ``Retriever`` seja simples e previsível. O provedor de embedding
    chega pronto de fora (``app/dependencias.py``): quem escolhe entre modelo
    local e API é o composition root, não esta função.
    """
    from langchain_chroma import Chroma

    config.diretorio_chroma.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.colecao_chroma,
        embedding_function=AdaptadorLangChain(embedding),
        persist_directory=str(config.diretorio_chroma),
        collection_metadata={"hnsw:space": "cosine"},
    )
