"""Dublês usados nos testes.

Nenhum teste sobe ChromaDB, baixa modelo de embedding ou chama API: as
fronteiras do sistema são todas interfaces pequenas, então dá para trocá-las
por dublês e testar a lógica de verdade.
"""

from dataclasses import dataclass, field

import pytest

from app.llm.base import ErroGeracao, ProvedorLLM
from app.rag.retriever import ResultadoBusca


@dataclass
class DocumentoFalso:
    """Imita o ``Document`` do LangChain (só o que o Retriever usa)."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class VectorStoreFalso:
    """Devolve pares (documento, distância) pré-definidos."""

    def __init__(self, retorno: list[tuple[DocumentoFalso, float]] | None = None):
        self.retorno = retorno or []
        self.chamadas: list[tuple[str, int]] = []

    def similarity_search_with_score(self, consulta: str, k: int = 3):
        self.chamadas.append((consulta, k))
        return self.retorno[:k]


class LLMFalso(ProvedorLLM):
    """Registra os prompts recebidos e devolve um texto fixo."""

    def __init__(self, resposta: str = "Resposta gerada pelo LLM.") -> None:
        self.resposta = resposta
        self.prompts: list[tuple[str, str]] = []

    def gerar(self, prompt_sistema: str, prompt_usuario: str) -> str:
        self.prompts.append((prompt_sistema, prompt_usuario))
        return self.resposta


class LLMQuebrado(ProvedorLLM):
    """Simula indisponibilidade do provedor de LLM."""

    def gerar(self, prompt_sistema: str, prompt_usuario: str) -> str:
        raise ErroGeracao("provedor indisponível")


def criar_resultado(
    similaridade: float,
    pergunta: str = "Qual o prazo de entrega?",
    resposta: str = "De 3 a 7 dias úteis.",
    categoria: str = "entrega",
    tags: tuple[str, ...] = ("prazo", "entrega"),
) -> ResultadoBusca:
    """Atalho para montar um ``ResultadoBusca`` nos testes."""
    return ResultadoBusca(
        pergunta=pergunta,
        resposta=resposta,
        categoria=categoria,
        tags=tags,
        similaridade=similaridade,
    )


@pytest.fixture
def llm_falso() -> LLMFalso:
    return LLMFalso()
