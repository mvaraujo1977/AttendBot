"""Testes da rotina de medição do limiar.

O corpus é dado; o que se verifica aqui é a aritmética que traduz medição em
decisão — porque é dela que sai o número que decide quando o bot cala a boca.
"""

from app.rag.calibracao import (
    FRONTEIRA,
    LONGE_DA_BASE,
    MARGEM,
    PERGUNTAS_COBERTAS,
    Amostra,
    Medicao,
    medir,
)


class RetrieverFalso:
    """Devolve uma similaridade fixa por pergunta."""

    def __init__(self, por_pergunta: dict[str, tuple[float, str]]) -> None:
        self._por_pergunta = por_pergunta

    def buscar(self, pergunta: str):
        similaridade, recuperada = self._por_pergunta.get(pergunta, (0.1, "outra"))
        return [
            type(
                "R",
                (),
                {"pergunta": recuperada, "similaridade": similaridade},
            )()
        ]


def _medicao(cobertas: list[tuple[float, bool]], fora: list[float]) -> Medicao:
    return Medicao(
        cobertas=[
            Amostra("p", s, "certa" if ok else "errada", "certa")
            for s, ok in cobertas
        ],
        fora=[Amostra("q", s, "qualquer") for s in fora],
    )


def test_o_corpus_cobre_os_tres_grupos_da_calibracao() -> None:
    """Cada grupo responde por uma pergunta diferente da calibração.

    Cobertas: o retrieval acerta e a similaridade fica acima do limiar.
    Longe da base: a 1a barreira dá conta sozinha.
    Fronteira: passa do limiar e só a 2a barreira barra — se este grupo
    esvaziar, a medição para de exercitar o motivo de a 2a barreira existir.
    """
    assert len(PERGUNTAS_COBERTAS) == 10
    assert len(LONGE_DA_BASE) >= 2
    assert len(FRONTEIRA) >= 4


def test_conta_acertos_de_retrieval() -> None:
    medicao = _medicao([(0.9, True), (0.85, False), (0.88, True)], [0.5])

    assert medicao.acertos == 2


def test_faixas_sao_o_minimo_e_o_maximo() -> None:
    medicao = _medicao([(0.90, True), (0.84, True)], [0.70, 0.52])

    assert medicao.faixa_cobertas == (0.84, 0.90)
    assert medicao.faixa_fora == (0.52, 0.70)


def test_sobreposicao_quando_as_faixas_se_cruzam() -> None:
    """O caso do e5: nenhum limiar separa os dois grupos sozinho."""
    medicao = _medicao([(0.839, True)], [0.845])

    assert round(medicao.sobreposicao, 3) == 0.006


def test_sem_sobreposicao_quando_as_faixas_estao_separadas() -> None:
    medicao = _medicao([(0.90, True)], [0.40])

    assert medicao.sobreposicao == 0.0


def test_limiar_sugerido_fica_abaixo_da_coberta_mais_fraca() -> None:
    """Errar barrando cliente legítimo é o custo que a margem evita."""
    medicao = _medicao([(0.839, True), (0.891, True)], [0.70])

    assert medicao.limiar_sugerido == 0.81
    assert medicao.limiar_sugerido <= 0.839 - MARGEM


def test_limiar_sugerido_nunca_e_negativo() -> None:
    assert _medicao([(0.01, True)], [0.0]).limiar_sugerido == 0.0


def test_medir_percorre_o_corpus_inteiro() -> None:
    retriever = RetrieverFalso(
        {pergunta: (0.9, canonica) for pergunta, canonica in PERGUNTAS_COBERTAS}
    )

    medicao = medir(retriever)

    assert len(medicao.cobertas) == len(PERGUNTAS_COBERTAS)
    assert len(medicao.fora) == len(LONGE_DA_BASE) + len(FRONTEIRA)
    assert medicao.acertos == 10
