"""Implementação do provedor de LLM usando a API do Gemini.

Entrou ao lado da OpenAI pelo mesmo motivo do embedding: o tier gratuito
permite rodar em produção sem custo recorrente. Como o ``Generator`` só conhece
``ProvedorLLM``, o prompt, a calibração e o transbordo continuam iguais.
"""

import logging

from app.gemini_api import chamar
from app.llm.base import ErroGeracao, ProvedorLLM

logger = logging.getLogger(__name__)

# Medido em setembro/2026 contra a API: mediana de 1,15s no prompt real deste
# projeto, e emite o sinal TRANSBORDO da 2a barreira corretamente.
#
# Não fixe uma versão sem verificar: o `gemini-2.5-flash` continua aparecendo em
# `GET /v1beta/models`, mas o `generateContent` devolve 404 "no longer available
# to new users" para chaves criadas depois da retirada. A listagem não é sinal
# de disponibilidade — só a chamada real é.
MODELO_PADRAO = "gemini-3.5-flash-lite"
TIMEOUT_SEGUNDOS = 30.0


class ProvedorGemini(ProvedorLLM):
    """Chama o endpoint ``generateContent`` da API do Gemini."""

    def __init__(
        self,
        api_key: str | None,
        modelo: str = MODELO_PADRAO,
        temperatura: float = 0.3,
        orcamento_raciocinio: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não configurada. Defina a chave no .env ou use "
                "PROVEDOR_LLM=demo para rodar sem API."
            )
        self._api_key = api_key
        self._modelo = modelo if modelo.startswith("models/") else f"models/{modelo}"
        self._temperatura = temperatura
        self._orcamento_raciocinio = orcamento_raciocinio

    def gerar(self, prompt_sistema: str, prompt_usuario: str) -> str:
        try:
            resposta = chamar(
                f"{self._modelo}:generateContent",
                {
                    # O Gemini separa instrução de sistema do turno do usuário,
                    # como a OpenAI separa as roles system/user.
                    "systemInstruction": {"parts": [{"text": prompt_sistema}]},
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt_usuario}]}
                    ],
                    "generationConfig": self._configuracao(),
                },
                self._api_key,
                timeout=TIMEOUT_SEGUNDOS,
            )
        except RuntimeError as erro:
            raise ErroGeracao(f"Falha ao chamar o Gemini: {erro}") from erro

        return self._extrair_texto(resposta)

    def _configuracao(self) -> dict:
        """Monta o ``generationConfig``, omitindo o raciocínio por padrão.

        Controlar o "pensamento" do modelo é uma alavanca real de latência —
        o `gemini-flash-latest` levou 19s no mesmo prompt que o flash-lite
        resolve em ~1s. Mas o campo não é universal: os modelos 3.5+ recusam
        `thinkingBudget` com 400, enquanto o 3.1-flash-lite e o 2.5 o aceitam.
        Por isso ele é opcional (``GEMINI_ORCAMENTO_RACIOCINIO``) em vez de
        fixo: o default omite o campo, que é o que funciona no modelo padrão.
        """
        config: dict = {"temperature": self._temperatura}
        if self._orcamento_raciocinio is not None:
            config["thinkingConfig"] = {
                "thinkingBudget": self._orcamento_raciocinio
            }
        return config

    @staticmethod
    def _extrair_texto(resposta: dict) -> str:
        """Puxa o texto do primeiro candidato, explicando as respostas vazias.

        Vazio aqui não é exceção rara: o filtro de segurança pode bloquear o
        prompt (``promptFeedback.blockReason``) ou cortar a geração
        (``finishReason``). Nos dois casos o certo é ``ErroGeracao``, que o
        ``ServicoAtendimento`` já trata transbordando para um humano.
        """
        bloqueio = (resposta.get("promptFeedback") or {}).get("blockReason")
        if bloqueio:
            raise ErroGeracao(f"O Gemini bloqueou o prompt: {bloqueio}.")

        candidatos = resposta.get("candidates") or []
        if not candidatos:
            raise ErroGeracao("O Gemini não devolveu nenhum candidato.")

        partes = (candidatos[0].get("content") or {}).get("parts") or []
        texto = "".join(parte.get("text", "") for parte in partes).strip()
        if not texto:
            motivo = candidatos[0].get("finishReason") or "desconhecido"
            raise ErroGeracao(
                f"O Gemini devolveu uma resposta vazia (finishReason={motivo})."
            )
        return texto
