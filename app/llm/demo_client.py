"""Provedor de LLM de demonstração, que não chama API nenhuma.

Serve para rodar o projeto inteiro (webhook, busca vetorial, transbordo) sem
chave de API: ele devolve a resposta oficial do FAQ que veio no contexto, sem
reescrever nada. Ative com ``PROVEDOR_LLM=demo``.
"""

from app.llm.base import ProvedorLLM
from app.rag.generator import MARCADOR_RESPOSTA


class ProvedorDemo(ProvedorLLM):
    """Extrai a primeira resposta oficial do prompt e a devolve verbatim."""

    def gerar(self, prompt_sistema: str, prompt_usuario: str) -> str:
        for linha in prompt_usuario.splitlines():
            linha = linha.strip()
            if linha.startswith(MARCADOR_RESPOSTA):
                return linha[len(MARCADOR_RESPOSTA) :].strip()
        return "Não consegui montar uma resposta com o contexto disponível."
