"""Orquestração do atendimento: busca -> decisão de transbordo -> geração.

Esta é a única classe que conhece o fluxo completo. Ela não sabe que existe
WhatsApp, HTTP, Chroma ou OpenAI — só fala com o retriever, o generator e a
regra de handoff.
"""

import logging
from dataclasses import dataclass

from app.handoff import MotivoTransbordo, avaliar_transbordo, filtrar_contexto
from app.llm.base import ErroGeracao
from app.rag.generator import Generator
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

MENSAGEM_SEM_TEXTO = (
    "Recebi sua mensagem, mas não consegui ler nenhum texto nela. "
    "Pode me escrever sua dúvida?"
)


@dataclass(frozen=True)
class RespostaAtendimento:
    """O que o bot vai responder, mais os dados de diagnóstico da decisão."""

    texto: str
    transbordo: bool
    similaridade: float
    motivo: str | None = None


class ServicoAtendimento:
    """Responde a uma pergunta de cliente aplicando o fluxo de RAG."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        limiar_similaridade: float,
        mensagem_transbordo: str,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._limiar = limiar_similaridade
        self._mensagem_transbordo = mensagem_transbordo

    def responder(self, pergunta: str) -> RespostaAtendimento:
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return RespostaAtendimento(
                texto=MENSAGEM_SEM_TEXTO, transbordo=False, similaridade=0.0
            )

        resultados = self._retriever.buscar(pergunta)
        decisao = avaliar_transbordo(resultados, self._limiar)

        if decisao.transbordar:
            logger.info(
                "Transbordo humano (motivo=%s, similaridade=%.3f, limiar=%.2f)",
                decisao.motivo.value if decisao.motivo else "?",
                decisao.similaridade,
                self._limiar,
            )
            return RespostaAtendimento(
                texto=self._mensagem_transbordo,
                transbordo=True,
                similaridade=decisao.similaridade,
                motivo=decisao.motivo.value if decisao.motivo else None,
            )

        contextos = filtrar_contexto(resultados, self._limiar)
        try:
            texto = self._generator.gerar(pergunta, contextos)
        except ErroGeracao:
            # Falhou o LLM: melhor passar para um humano do que ficar mudo.
            logger.exception("Erro ao gerar resposta; transbordando para humano.")
            return RespostaAtendimento(
                texto=self._mensagem_transbordo,
                transbordo=True,
                similaridade=decisao.similaridade,
                motivo=MotivoTransbordo.ERRO_GERACAO.value,
            )

        logger.info(
            "Resposta gerada (similaridade=%.3f, contextos=%d)",
            decisao.similaridade,
            len(contextos),
        )
        return RespostaAtendimento(
            texto=texto, transbordo=False, similaridade=decisao.similaridade
        )
