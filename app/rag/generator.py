"""Montagem do prompt com o contexto recuperado e chamada ao LLM."""

from collections.abc import Sequence

from app.llm.base import ProvedorLLM
from app.rag.retriever import ResultadoBusca

# Prefixo usado nos blocos de contexto. Fica como constante porque o provedor
# de demonstração (sem API) precisa reconhecê-lo para devolver a resposta base.
MARCADOR_RESPOSTA = "Resposta oficial:"

# Segunda barreira de transbordo. O limiar de similaridade sozinho não separa
# bem o que a FAQ cobre do que ela não cobre — modelos de embedding como o e5
# comprimem as similaridades numa faixa estreita. Então o LLM, que já está
# lendo o contexto, também julga se ele de fato responde à pergunta; quando não
# responde, emite este sinal e o ServicoAtendimento transborda.
SINAL_TRANSBORDO = "TRANSBORDO"

PROMPT_SISTEMA = """Você é o assistente virtual de atendimento da {empresa}, \
falando com clientes pelo WhatsApp.

Antes de responder, verifique: o contexto abaixo responde de fato à pergunta \
do cliente? Se não responder — ou se responder só parcialmente, ou se o \
assunto for outro — responda EXATAMENTE com a palavra {sinal}, sozinha, sem \
nenhum outro texto. É melhor encaminhar para um humano do que arriscar.

Se o contexto responder, siga estas regras:
- Use APENAS as informações do contexto. Nunca invente prazos, valores, \
políticas ou links.
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
        """Devolve a resposta ao cliente, ou ``SINAL_TRANSBORDO``.

        Quem interpreta o sinal é o ``ServicoAtendimento`` — aqui o texto do
        modelo é repassado como veio.
        """
        prompt_sistema = PROMPT_SISTEMA.format(
            empresa=self._nome_empresa, sinal=SINAL_TRANSBORDO
        )
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
