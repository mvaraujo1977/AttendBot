"""Orquestração do atendimento: busca -> decisão de transbordo -> geração.

Esta é a única classe que conhece o fluxo completo. Ela não sabe que existe
WhatsApp, HTTP, Chroma ou OpenAI — só fala com o retriever, o generator e a
regra de handoff.
"""

import logging
from dataclasses import dataclass

from app.handoff import MotivoTransbordo, avaliar_transbordo, filtrar_contexto
from app.llm.base import ErroGeracao
from app.rag.generator import SINAL_TRANSBORDO, Generator
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

MENSAGEM_SEM_TEXTO = (
    "Recebi sua mensagem, mas não consegui ler nenhum texto nela. "
    "Pode me escrever sua dúvida?"
)

MENSAGEM_BOAS_VINDAS = (
    "Olá! Sou o assistente virtual de atendimento. Respondo dúvidas sobre "
    "pedidos, entrega, frete, pagamento, trocas, reembolso, garantia e nota "
    "fiscal.\n\n"
    "É só escrever sua pergunta com suas próprias palavras. Quando eu não "
    "souber responder com segurança, encaminho você para um atendente humano."
)

# Comandos respondidos com texto fixo, sem passar pelo RAG. São convenção do
# Telegram, mas ficam aqui — e não no provedor — porque a resposta é a mesma
# em qualquer canal e nenhuma delas depende de transporte.
COMANDOS_BOAS_VINDAS = frozenset({"/start", "/help", "/ajuda"})

# Corte da pergunta antes de ela virar embedding e prompt. O custo em tokens é
# linear no tamanho, e o corte mora aqui — e não no schema da rota — para valer
# igual nos dois caminhos de entrada: o webhook de cada canal e o
# `/api/mensagem`. Uma dúvida de atendimento real não chega perto disto.
LIMITE_CARACTERES_PERGUNTA = 1000


@dataclass(frozen=True)
class RespostaAtendimento:
    """O que o bot vai responder, mais os dados de diagnóstico da decisão."""

    texto: str
    transbordo: bool
    similaridade: float
    motivo: str | None = None


def _comando(texto: str) -> str | None:
    """Extrai o comando de uma mensagem, ou ``None`` se ela não for um.

    O Telegram manda ``/start``, mas também ``/start@nome_do_bot`` em grupos
    e ``/start <payload>`` em links de convite: só a primeira palavra, sem o
    sufixo do bot, identifica o comando.
    """
    if not texto.startswith("/"):
        return None
    return texto.split(maxsplit=1)[0].split("@", 1)[0].lower()


def _e_sinal_de_transbordo(texto: str) -> bool:
    """O LLM pediu transbordo por julgar o contexto insuficiente?"""
    return texto.strip().strip(".!\"'").upper() == SINAL_TRANSBORDO


class ServicoAtendimento:
    """Responde a uma pergunta de cliente aplicando o fluxo de RAG.

    O transbordo tem duas barreiras: o limiar de similaridade descarta o que
    nem chega perto da base, e o LLM — que já está lendo o contexto — decide os
    casos de fronteira, sinalizando quando o contexto não responde de fato à
    pergunta. Uma barreira só não basta: a similaridade absoluta discrimina
    mal, e sem o limiar toda mensagem viraria chamada de API.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        limiar_similaridade: float,
        mensagem_transbordo: str,
        mensagem_boas_vindas: str | None = None,
        limite_caracteres: int = LIMITE_CARACTERES_PERGUNTA,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._limiar = limiar_similaridade
        self._mensagem_transbordo = mensagem_transbordo
        self._mensagem_boas_vindas = mensagem_boas_vindas or MENSAGEM_BOAS_VINDAS
        self._limite_caracteres = limite_caracteres

    @property
    def limiar(self) -> float:
        """Limiar em vigor. Exposto porque ele depende do modelo de embedding
        ativo (ver ``resolver_limiar``), então nem sempre está no ``.env``."""
        return self._limiar

    def responder(self, pergunta: str) -> RespostaAtendimento:
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return RespostaAtendimento(
                texto=MENSAGEM_SEM_TEXTO, transbordo=False, similaridade=0.0
            )

        if len(pergunta) > self._limite_caracteres:
            # Cortar em vez de recusar: o cliente prolixo continua atendido, e
            # o começo da mensagem é onde a pergunta costuma estar. O que o
            # corte impede é o custo crescer sem teto com o tamanho da entrada.
            logger.info(
                "Pergunta truncada de %d para %d caracteres.",
                len(pergunta),
                self._limite_caracteres,
            )
            pergunta = pergunta[: self._limite_caracteres]

        # Antes da busca: '/start' não é pergunta de cliente, e mandá-lo para o
        # RAG só produz transbordo (medido: similaridade 0.771) — um humano
        # seria chamado para responder a um clique de abertura de conversa.
        comando = _comando(pergunta)
        if comando in COMANDOS_BOAS_VINDAS:
            # Só o comando normalizado: `/start <payload>` carrega o payload do
            # link de convite, que não tem por que ficar registrado.
            logger.info("Comando respondido sem RAG: %s", comando)
            return RespostaAtendimento(
                texto=self._mensagem_boas_vindas,
                transbordo=False,
                similaridade=0.0,
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

        if _e_sinal_de_transbordo(texto):
            logger.info(
                "Transbordo humano (motivo=%s, similaridade=%.3f): o LLM avaliou "
                "que o contexto não responde à pergunta.",
                MotivoTransbordo.CONTEXTO_INSUFICIENTE.value,
                decisao.similaridade,
            )
            return RespostaAtendimento(
                texto=self._mensagem_transbordo,
                transbordo=True,
                similaridade=decisao.similaridade,
                motivo=MotivoTransbordo.CONTEXTO_INSUFICIENTE.value,
            )

        logger.info(
            "Resposta gerada (similaridade=%.3f, contextos=%d)",
            decisao.similaridade,
            len(contextos),
        )
        return RespostaAtendimento(
            texto=texto, transbordo=False, similaridade=decisao.similaridade
        )
