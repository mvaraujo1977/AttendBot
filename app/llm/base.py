"""Contrato do provedor de LLM.

Mesma ideia da camada de mensageria: o RAG só conhece esta interface, então
trocar OpenAI por Anthropic, Gemini ou um modelo local é escrever uma nova
implementação, sem mexer em prompt, retrieval ou handoff.
"""

from abc import ABC, abstractmethod


class ProvedorLLM(ABC):
    """Gera texto a partir de uma instrução de sistema e uma do usuário."""

    @abstractmethod
    def gerar(self, prompt_sistema: str, prompt_usuario: str) -> str:
        """Devolve a resposta do modelo já como texto puro."""


class ErroGeracao(RuntimeError):
    """Falha ao chamar o LLM (rede, cota, credencial inválida...)."""
