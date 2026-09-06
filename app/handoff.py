"""Decisão de transbordo para atendimento humano.

Regra central do projeto: se a busca vetorial não trouxer nada suficientemente
parecido com a pergunta, o bot não tenta responder — ele encaminha para um
humano. É o que evita resposta genérica e alucinação.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from app.rag.retriever import ResultadoBusca


class MotivoTransbordo(str, Enum):
    """Por que a conversa foi encaminhada para um atendente."""

    SEM_RESULTADOS = "sem_resultados"
    BAIXA_SIMILARIDADE = "baixa_similaridade"
    # O contexto passou no limiar, mas o LLM avaliou que ele não responde
    # à pergunta (segunda barreira). Ver app/rag/generator.py.
    CONTEXTO_INSUFICIENTE = "contexto_insuficiente"
    ERRO_GERACAO = "erro_geracao"


@dataclass(frozen=True)
class DecisaoTransbordo:
    """Resultado da avaliação, com a similaridade que motivou a decisão."""

    transbordar: bool
    similaridade: float
    motivo: MotivoTransbordo | None = None


def avaliar_transbordo(
    resultados: Sequence[ResultadoBusca], limiar: float
) -> DecisaoTransbordo:
    """Decide se o bot responde sozinho ou passa para um atendente humano."""
    if not resultados:
        return DecisaoTransbordo(
            transbordar=True,
            similaridade=0.0,
            motivo=MotivoTransbordo.SEM_RESULTADOS,
        )

    melhor = max(resultado.similaridade for resultado in resultados)
    if melhor < limiar:
        return DecisaoTransbordo(
            transbordar=True,
            similaridade=melhor,
            motivo=MotivoTransbordo.BAIXA_SIMILARIDADE,
        )
    return DecisaoTransbordo(transbordar=False, similaridade=melhor)


def filtrar_contexto(
    resultados: Sequence[ResultadoBusca], limiar: float
) -> list[ResultadoBusca]:
    """Mantém no prompt apenas os trechos acima do limiar.

    Evita que resultados fracos do ``top_k`` entrem no contexto e puxem a
    resposta do LLM para o lado errado.
    """
    return [resultado for resultado in resultados if resultado.similaridade >= limiar]
