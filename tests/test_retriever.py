"""Testes da busca vetorial (conversão de distância e ordenação)."""

import pytest

from app.rag.retriever import Retriever, converter_para_similaridade
from tests.conftest import DocumentoFalso, VectorStoreFalso


def _documento(pergunta: str, resposta: str = "R", tags: str = "") -> DocumentoFalso:
    return DocumentoFalso(
        page_content=pergunta,
        metadata={
            "pergunta": pergunta,
            "resposta": resposta,
            "categoria": "entrega",
            "tags": tags,
        },
    )


@pytest.mark.parametrize(
    ("distancia", "esperado"),
    [
        (0.0, 1.0),  # idêntico
        (0.25, 0.75),
        (1.0, 0.0),  # ortogonal
        (1.8, 0.0),  # oposto: nunca negativo
        (-0.1, 1.0),  # ruído numérico: nunca acima de 1
    ],
)
def test_converte_distancia_em_similaridade(distancia: float, esperado: float) -> None:
    assert converter_para_similaridade(distancia) == pytest.approx(esperado)


def test_buscar_devolve_resultados_ordenados_por_similaridade() -> None:
    store = VectorStoreFalso(
        [
            (_documento("Qual o frete?"), 0.40),  # similaridade 0.60
            (_documento("Qual o prazo de entrega?"), 0.10),  # similaridade 0.90
            (_documento("Como trocar?"), 0.70),  # similaridade 0.30
        ]
    )

    resultados = Retriever(store, top_k=3).buscar("quando chega meu pedido?")

    assert [round(r.similaridade, 2) for r in resultados] == [0.90, 0.60, 0.30]
    assert resultados[0].pergunta == "Qual o prazo de entrega?"


def test_buscar_repassa_o_top_k_configurado() -> None:
    store = VectorStoreFalso([(_documento(f"P{i}"), 0.1) for i in range(5)])

    resultados = Retriever(store, top_k=2).buscar("qualquer coisa")

    assert store.chamadas == [("qualquer coisa", 2)]
    assert len(resultados) == 2


def test_buscar_com_pergunta_vazia_nao_consulta_o_store() -> None:
    store = VectorStoreFalso([(_documento("P"), 0.1)])

    assert Retriever(store).buscar("   ") == []
    assert store.chamadas == []


def test_buscar_converte_metadados_em_resultado() -> None:
    store = VectorStoreFalso(
        [(_documento("Qual o frete?", "Grátis acima de R$ 199.", "frete, valor"), 0.2)]
    )

    resultado = Retriever(store).buscar("frete é caro?")[0]

    assert resultado.resposta == "Grátis acima de R$ 199."
    assert resultado.categoria == "entrega"
    assert resultado.tags == ("frete", "valor")


def test_buscar_usa_page_content_quando_falta_metadado() -> None:
    documento = DocumentoFalso(page_content="Pergunta sem metadados", metadata={})
    store = VectorStoreFalso([(documento, 0.3)])

    resultado = Retriever(store).buscar("algo")[0]

    assert resultado.pergunta == "Pergunta sem metadados"
    assert resultado.tags == ()


def test_buscar_colapsa_variacoes_da_mesma_entrada() -> None:
    """Variações são documentos distintos apontando para a mesma entrada."""
    canonica = _documento("Vocês emitem nota fiscal?")
    store = VectorStoreFalso(
        [
            (canonica, 0.30),  # similaridade 0.70
            (canonica, 0.10),  # a variação casou melhor: 0.90
            (_documento("Qual o frete?"), 0.40),  # 0.60
        ]
    )

    resultados = Retriever(store, top_k=3).buscar("tem NF?")

    assert [r.pergunta for r in resultados] == [
        "Vocês emitem nota fiscal?",
        "Qual o frete?",
    ]
    # Sobra a ocorrência de maior similaridade, não a primeira que apareceu.
    assert resultados[0].similaridade == pytest.approx(0.90)
