"""Testes da regra de transbordo para atendimento humano."""

from app.handoff import MotivoTransbordo, avaliar_transbordo, filtrar_contexto
from tests.conftest import criar_resultado

LIMIAR = 0.60


def test_transborda_quando_a_busca_nao_retorna_nada() -> None:
    decisao = avaliar_transbordo([], LIMIAR)

    assert decisao.transbordar is True
    assert decisao.motivo is MotivoTransbordo.SEM_RESULTADOS
    assert decisao.similaridade == 0.0


def test_transborda_quando_a_melhor_similaridade_fica_abaixo_do_limiar() -> None:
    decisao = avaliar_transbordo(
        [criar_resultado(0.42), criar_resultado(0.31)], LIMIAR
    )

    assert decisao.transbordar is True
    assert decisao.motivo is MotivoTransbordo.BAIXA_SIMILARIDADE
    assert decisao.similaridade == 0.42


def test_nao_transborda_quando_ha_resultado_acima_do_limiar() -> None:
    decisao = avaliar_transbordo(
        [criar_resultado(0.35), criar_resultado(0.88)], LIMIAR
    )

    assert decisao.transbordar is False
    assert decisao.motivo is None
    assert decisao.similaridade == 0.88


def test_similaridade_exatamente_no_limiar_nao_transborda() -> None:
    decisao = avaliar_transbordo([criar_resultado(LIMIAR)], LIMIAR)

    assert decisao.transbordar is False


def test_limiar_mais_alto_torna_o_bot_mais_conservador() -> None:
    resultados = [criar_resultado(0.70)]

    assert avaliar_transbordo(resultados, 0.60).transbordar is False
    assert avaliar_transbordo(resultados, 0.85).transbordar is True


def test_filtrar_contexto_descarta_resultados_fracos() -> None:
    resultados = [criar_resultado(0.91), criar_resultado(0.62), criar_resultado(0.20)]

    contextos = filtrar_contexto(resultados, LIMIAR)

    assert [r.similaridade for r in contextos] == [0.91, 0.62]
