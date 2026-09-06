"""Implementação do provedor de LLM usando a API da OpenAI."""

import logging

from app.llm.base import ErroGeracao, ProvedorLLM

logger = logging.getLogger(__name__)


class ProvedorOpenAI(ProvedorLLM):
    """Chama o endpoint de chat completions da OpenAI."""

    def __init__(
        self,
        api_key: str | None,
        modelo: str = "gpt-4o-mini",
        temperatura: float = 0.3,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não configurada. Defina a chave no .env ou use "
                "PROVEDOR_LLM=demo para rodar sem API."
            )
        self._api_key = api_key
        self._modelo = modelo
        self._temperatura = temperatura
        self._cliente = None

    def gerar(self, prompt_sistema: str, prompt_usuario: str) -> str:
        try:
            resposta = self._openai.chat.completions.create(
                model=self._modelo,
                temperature=self._temperatura,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
            )
        except Exception as erro:  # noqa: BLE001 - a origem é sempre externa
            raise ErroGeracao(f"Falha ao chamar a OpenAI: {erro}") from erro

        texto = (resposta.choices[0].message.content or "").strip()
        if not texto:
            raise ErroGeracao("A OpenAI devolveu uma resposta vazia.")
        return texto

    @property
    def _openai(self):
        """Cliente criado sob demanda, para o import não pesar na subida da app."""
        if self._cliente is None:
            from openai import OpenAI

            self._cliente = OpenAI(api_key=self._api_key)
        return self._cliente
