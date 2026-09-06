"""Testes da carga do dataset de FAQ."""

import json

import pytest

from app.config import obter_configuracoes
from app.rag.ingest import ItemFAQ, carregar_faq, indexar


class VectorStoreEspiao:
    """Captura os argumentos de ``add_texts`` para verificar o upsert."""

    def __init__(self) -> None:
        self.chamadas: list[dict] = []

    def add_texts(self, texts, metadatas, ids):
        self.chamadas.append({"texts": texts, "metadatas": metadatas, "ids": ids})
        return ids


def _escrever(tmp_path, dados) -> "object":
    caminho = tmp_path / "faq.json"
    caminho.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return caminho


def test_dataset_de_exemplo_tem_ao_menos_oito_perguntas() -> None:
    itens = carregar_faq(obter_configuracoes().caminho_faq)

    assert len(itens) >= 8
    assert all(item.pergunta and item.resposta for item in itens)
    # Perguntas duplicadas colidiriam no upsert e sumiriam da base.
    assert len({item.id for item in itens}) == len(itens)


def test_carregar_faq_le_todos_os_campos(tmp_path) -> None:
    caminho = _escrever(
        tmp_path,
        [
            {
                "pergunta": "Qual o frete?",
                "resposta": "Grátis acima de R$ 199.",
                "categoria": "entrega",
                "tags": ["frete", " valor "],
            }
        ],
    )

    item = carregar_faq(caminho)[0]

    assert item.pergunta == "Qual o frete?"
    assert item.categoria == "entrega"
    assert item.tags == ("frete", "valor")


def test_carregar_faq_aplica_padroes_quando_faltam_campos_opcionais(tmp_path) -> None:
    caminho = _escrever(tmp_path, [{"pergunta": "P?", "resposta": "R."}])

    item = carregar_faq(caminho)[0]

    assert item.categoria == "geral"
    assert item.tags == ()


@pytest.mark.parametrize(
    "dados",
    [
        [],
        [{"pergunta": "Sem resposta"}],
        [{"resposta": "Sem pergunta"}],
        [{"pergunta": "  ", "resposta": "R."}],
        ["nao é um objeto"],
    ],
)
def test_carregar_faq_rejeita_dataset_invalido(tmp_path, dados) -> None:
    with pytest.raises(ValueError):
        carregar_faq(_escrever(tmp_path, dados))


def test_id_do_item_e_estavel_e_ignora_caixa_e_espacos() -> None:
    base = ItemFAQ(pergunta="Qual o frete?", resposta="R.")
    variacao = ItemFAQ(pergunta="  QUAL O FRETE?  ", resposta="Outra resposta.")

    assert base.id == variacao.id


def test_indexar_usa_ids_estaveis_para_fazer_upsert() -> None:
    itens = [
        ItemFAQ(pergunta="Qual o frete?", resposta="Grátis acima de R$ 199."),
        ItemFAQ(pergunta="Qual o prazo?", resposta="De 3 a 7 dias."),
    ]
    store = VectorStoreEspiao()

    total = indexar(itens, store)

    assert total == 2
    chamada = store.chamadas[0]
    assert chamada["texts"] == ["Qual o frete?", "Qual o prazo?"]
    assert chamada["ids"] == [item.id for item in itens]
    # Reindexar o mesmo item reaproveita o id, então atualiza em vez de duplicar.
    indexar(itens, store)
    assert store.chamadas[1]["ids"] == chamada["ids"]


def test_indexar_lista_vazia_nao_toca_no_store() -> None:
    store = VectorStoreEspiao()

    assert indexar([], store) == 0
    assert store.chamadas == []


def test_metadados_serializam_tags_como_texto() -> None:
    item = ItemFAQ(pergunta="P?", resposta="R.", tags=("a", "b"))

    assert item.metadados()["tags"] == "a, b"


# --- Variações e abreviações -------------------------------------------------


def test_carregar_faq_le_as_variacoes(tmp_path) -> None:
    caminho = _escrever(
        tmp_path,
        [
            {
                "pergunta": "Vocês emitem nota fiscal?",
                "resposta": "Sim, por e-mail.",
                "variacoes": ["tem NF?", "  emitem nfe?  ", "  "],
            }
        ],
    )

    item = carregar_faq(caminho)[0]

    # Vazios são descartados e os espaços, aparados — como já acontece nas tags.
    assert item.variacoes == ("tem NF?", "emitem nfe?")


def test_dataset_de_exemplo_tem_variacoes_em_todas_as_entradas() -> None:
    itens = carregar_faq(obter_configuracoes().caminho_faq)

    assert all(item.variacoes for item in itens)


def test_indexar_cria_um_documento_por_variacao_com_a_mesma_resposta() -> None:
    item = ItemFAQ(
        pergunta="Vocês emitem nota fiscal?",
        resposta="Sim, por e-mail.",
        variacoes=("tem NF?", "emitem nfe?"),
    )
    store = VectorStoreEspiao()

    total = indexar([item], store)

    assert total == 3
    chamada = store.chamadas[0]
    assert chamada["texts"] == ["Vocês emitem nota fiscal?", "tem NF?", "emitem nfe?"]
    # O que muda é o texto que vira embedding; a resposta é sempre a canônica.
    assert {meta["resposta"] for meta in chamada["metadatas"]} == {"Sim, por e-mail."}
    assert {meta["pergunta"] for meta in chamada["metadatas"]} == {
        "Vocês emitem nota fiscal?"
    }


def test_ids_das_variacoes_sao_estaveis_e_distintos() -> None:
    item = ItemFAQ(pergunta="P?", resposta="R.", variacoes=("v1", "v2"))
    store = VectorStoreEspiao()

    indexar([item], store)
    indexar([item], store)

    ids = store.chamadas[0]["ids"]
    assert len(set(ids)) == 3  # nenhuma colisão entre pergunta e variações
    assert store.chamadas[1]["ids"] == ids  # reindexar faz upsert, não duplica


def test_mesma_variacao_em_entradas_diferentes_nao_colide() -> None:
    a = ItemFAQ(pergunta="Qual o frete?", resposta="R1.", variacoes=("quanto custa?",))
    b = ItemFAQ(pergunta="Qual o prazo?", resposta="R2.", variacoes=("quanto custa?",))

    assert a.id_da_variacao("quanto custa?") != b.id_da_variacao("quanto custa?")
