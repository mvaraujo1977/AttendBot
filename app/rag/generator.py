"""Montagem do prompt com o contexto recuperado e chamada ao LLM."""

from collections.abc import Sequence

from app.llm.base import ProvedorLLM
from app.rag.retriever import ResultadoBusca

# Prefixo usado nos blocos de contexto. Fica como constante porque o provedor
# de demonstração (sem API) precisa reconhecê-lo para devolver a resposta base.
MARCADOR_RESPOSTA = "Resposta oficial:"

PROMPT_SISTEMA = """Você é o assistente virtual de atendimento da {empresa}, \
falando com clientes pelo WhatsApp.

Regras:
- Responda usando APENAS as informações do contexto fornecido.
- Se o contexto não responder à pergunta, diga que vai encaminhar para um \
atendente humano. Nunca invente prazos, valores, políticas ou links.
- Escreva em português do Brasil, com tom cordial e direto, em no máximo três \
frases curtas.
- Não mencione "contexto", "base de dados", "FAQ" ou como você obteve a \
informação. Fale como um atendente falaria.
- Não cumprimente de novo se o cliente já foi direto ao ponto."""

TEMPLATE_USUARIO = """Contexto (perguntas frequentes mais parecidas com a \
mensagem do cliente):
{contexto}

Mensagem do cliente: {pergunta}

Escreva a resposta que deve ser enviada ao cliente."""


class Generator:
    """Transforma pergunta + contexto recuperado em uma resposta natural."""

    def __init__(self, llm: ProvedorLLM, nome_empresa: str) -> None:
        self._llm = llm
        self._nome_empresa = nome_empresa

    def gerar(self, pergunta: str, contextos: Sequence[ResultadoBusca]) -> str:
        prompt_sistema = PROMPT_SISTEMA.format(empresa=self._nome_empresa)
        return self._llm.gerar(prompt_sistema, self.montar_prompt(pergunta, contextos))

    @staticmethod
    def montar_prompt(pergunta: str, contextos: Sequence[ResultadoBusca]) -> str:
        """Monta o prompt do usuário com os trechos de FAQ recuperados."""
        blocos = [
            f"[{indice}] Pergunta frequente: {contexto.pergunta}\n"
            f"{MARCADOR_RESPOSTA} {contexto.resposta}"
            for indice, contexto in enumerate(contextos, start=1)
        ]
        return TEMPLATE_USUARIO.format(
            contexto="\n\n".join(blocos) or "(nenhuma informação disponível)",
            pergunta=pergunta,
        )
