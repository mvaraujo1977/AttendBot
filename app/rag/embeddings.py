"""Função de embeddings usada tanto na ingestão quanto na busca.

O modelo padrão roda localmente (sentence-transformers), então não há custo de
API nem dependência de rede depois do primeiro download. Para trocar por
embeddings de API, basta devolver outro objeto compatível com a interface
``Embeddings`` do LangChain — nada mais no projeto precisa mudar.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings


class EmbeddingsE5(Embeddings):
    """Aplica os prefixos que a família de modelos E5 espera.

    Os modelos ``intfloat/*-e5-*`` são treinados com ``query:`` nas consultas e
    ``passage:`` nos documentos indexados. Sem esses prefixos a qualidade da
    busca cai bastante, e é um detalhe fácil de esquecer porque o modelo
    continua "funcionando".
    """

    def __init__(self, base: Embeddings) -> None:
        self._base = base

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return self._base.embed_documents([f"passage: {texto}" for texto in textos])

    def embed_query(self, texto: str) -> list[float]:
        return self._base.embed_query(f"query: {texto}")


@lru_cache
def criar_embeddings(modelo: str) -> Embeddings:
    """Carrega o modelo de embeddings local (cacheado por nome de modelo)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    base = HuggingFaceEmbeddings(
        model_name=modelo,
        encode_kwargs={"normalize_embeddings": True},
    )
    if "e5" in modelo.lower():
        return EmbeddingsE5(base)
    return base
