"""Busca semântica das perguntas de FAQ mais próximas da mensagem do cliente."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultadoBusca:
    """Uma entrada de FAQ recuperada, com o quanto ela casa com a pergunta."""

    pergunta: str
    resposta: str
    categoria: str
    tags: tuple[str, ...]
    similaridade: float


class Retriever:
    """Recupera as entradas de FAQ mais similares no vector store.

    Depende apenas do método ``similarity_search_with_score``, o que mantém a
    classe testável com um vector store falso e independente do Chroma.
    """

    def __init__(self, vector_store, top_k: int = 3) -> None:
        self._vector_store = vector_store
        self._top_k = top_k

    def buscar(self, pergunta: str) -> list[ResultadoBusca]:
        """Devolve os resultados ordenados da maior para a menor similaridade."""
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return []

        brutos = self._vector_store.similarity_search_with_score(
            pergunta, k=self._top_k
        )
        resultados = [
            self._converter(documento, distancia) for documento, distancia in brutos
        ]
        return sorted(resultados, key=lambda item: item.similaridade, reverse=True)

    @staticmethod
    def _converter(documento, distancia: float) -> ResultadoBusca:
        metadados = getattr(documento, "metadata", None) or {}
        tags = tuple(
            tag.strip()
            for tag in str(metadados.get("tags", "")).split(",")
            if tag.strip()
        )
        return ResultadoBusca(
            pergunta=metadados.get("pergunta") or documento.page_content,
            resposta=metadados.get("resposta", ""),
            categoria=metadados.get("categoria", "geral"),
            tags=tags,
            similaridade=converter_para_similaridade(distancia),
        )


def converter_para_similaridade(distancia: float) -> float:
    """Converte a distância de cosseno do Chroma em similaridade de 0 a 1.

    O Chroma devolve distância (0 = idêntico, 2 = oposto). Trabalhar com
    similaridade deixa o threshold de transbordo muito mais legível.
    """
    return max(0.0, min(1.0, 1.0 - float(distancia)))
