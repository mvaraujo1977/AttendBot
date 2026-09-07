"""Corpus e rotina de medição do limiar de transbordo.

O limiar da 1a barreira é calibrado por modelo de embedding: cada modelo
distribui as similaridades numa faixa própria, então um número medido para um
não significa nada no outro. Trocar ``PROVEDOR_EMBEDDING`` sem repetir a
medição é o modo de falha mais provável desta parte do projeto — o transbordo
continua "funcionando" e decidindo errado.

O corpus mora aqui, e não no script nem no teste, porque os dois precisam
dele: ``scripts/calibrar_limiar.py`` mede e sugere o limiar,
``tests/test_integracao_transbordo.py`` vigia se as faixas continuam onde a
calibração assumiu. Duplicar as perguntas deixaria a medição e o teste que a
protege livres para divergirem.

São 16 perguntas: 10 cobertas pela FAQ e 6 fora dela. A medição original do
README usou 10 de fora; ficaram versionadas as 6 que valem como regressão —
as 2 que a similaridade separa sozinha e as 4 de fronteira, que só a 2a
barreira barra. Para reproduzir a tabela histórica do README é preciso o
conjunto de 10 descrito lá.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Perguntas cobertas pela FAQ, escritas como um cliente escreveria — com
# acentuação, e nenhuma é cópia da pergunta canônica. O segundo item é a
# pergunta canônica que o retrieval deve recuperar.
#
# A acentuação não é detalhe: medindo com as mesmas perguntas sem acento, o
# MiniLM parecia acertar tudo; com acento, ele cai para 6/10. Foi assim que uma
# medição descuidada quase fixou o modelo errado no projeto.
PERGUNTAS_COBERTAS: list[tuple[str, str]] = [
    ("quando meu pedido vai chegar?", "Qual o prazo de entrega do pedido?"),
    ("como eu acompanho a entrega?", "Como faço para rastrear meu pedido?"),
    ("posso pagar com pix?", "Quais formas de pagamento vocês aceitam?"),
    ("quero trocar um produto que não serviu", "Como solicitar a troca de um produto?"),
    ("dá pra cancelar a compra?", "Como faço para cancelar meu pedido?"),
    (
        "quanto tempo demora pra devolver meu dinheiro?",
        "Em quanto tempo recebo o reembolso?",
    ),
    ("o frete é grátis?", "Qual o valor do frete?"),
    ("meu produto veio com defeito, e agora?", "Os produtos têm garantia?"),
    ("preciso da nota fiscal do que comprei", "Vocês emitem nota fiscal?"),
    ("vocês atendem no domingo?", "Qual o horário de atendimento?"),
]

# Perguntas sem nenhuma relação com a FAQ: devem morrer já na primeira barreira.
LONGE_DA_BASE: list[str] = [
    "qual a capital da França?",
    "quem ganhou a copa do mundo de 2022?",
]

# Perguntas fora do domínio que a similaridade sozinha NÃO separa — elas passam
# do limiar e só a segunda barreira, no LLM, consegue barrar.
FRONTEIRA: list[str] = [
    "vocês vendem passagem aérea?",
    "vocês têm vaga de emprego?",
    "me ajuda a escrever um currículo",
    "quanto é 2 + 2?",
]

# Folga entre o limiar sugerido e a pergunta legítima mais fraca. Errar para o
# lado do transbordo é barato; responder errado com confiança não é — mas
# barrar cliente legítimo também custa, e é o que essa margem evita.
MARGEM = 0.02


@dataclass(frozen=True)
class Amostra:
    """Uma pergunta medida contra o índice."""

    pergunta: str
    similaridade: float
    recuperada: str
    esperada: str | None = None

    @property
    def acertou(self) -> bool:
        """Só faz sentido para as cobertas: o retrieval trouxe a entrada certa?"""
        return self.esperada is not None and self.recuperada == self.esperada


@dataclass(frozen=True)
class Medicao:
    """Resultado da medição, no formato da tabela de calibração do README."""

    cobertas: list[Amostra]
    fora: list[Amostra]

    @property
    def acertos(self) -> int:
        return sum(1 for amostra in self.cobertas if amostra.acertou)

    @property
    def faixa_cobertas(self) -> tuple[float, float]:
        return _faixa(self.cobertas)

    @property
    def faixa_fora(self) -> tuple[float, float]:
        return _faixa(self.fora)

    @property
    def sobreposicao(self) -> float:
        """Quanto a faixa das de fora invade a das corretas.

        Positivo significa que nenhum limiar separa os dois grupos — foi o que
        a medição do e5 mostrou, e o motivo de existir a 2a barreira.
        """
        return max(0.0, self.faixa_fora[1] - self.faixa_cobertas[0])

    @property
    def limiar_sugerido(self) -> float:
        """Abaixo da pergunta legítima mais fraca, com uma margem de folga.

        É um ponto de partida, não um veredito: quem confirma é rodar os testes
        de integração com o número novo.
        """
        piso = self.faixa_cobertas[0] - MARGEM
        return max(0.0, math.floor(piso * 100) / 100)


def medir(retriever) -> Medicao:
    """Roda o corpus contra um ``Retriever`` já apontando para o índice real."""
    cobertas = [
        _amostrar(retriever, pergunta, esperada)
        for pergunta, esperada in PERGUNTAS_COBERTAS
    ]
    fora = [
        _amostrar(retriever, pergunta) for pergunta in LONGE_DA_BASE + FRONTEIRA
    ]
    return Medicao(cobertas=cobertas, fora=fora)


def _amostrar(retriever, pergunta: str, esperada: str | None = None) -> Amostra:
    resultados = retriever.buscar(pergunta)
    if not resultados:
        return Amostra(pergunta, 0.0, "(nada recuperado)", esperada)
    melhor = resultados[0]
    return Amostra(pergunta, melhor.similaridade, melhor.pergunta, esperada)


def _faixa(amostras: Sequence[Amostra]) -> tuple[float, float]:
    valores = [amostra.similaridade for amostra in amostras]
    return (min(valores), max(valores)) if valores else (0.0, 0.0)
