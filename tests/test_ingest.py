"""Testes da carga do dataset de FAQ."""

import json

import pytest

from app.config import obter_configuracoes
from app.rag.ingest import (
    ItemFAQ,
    carregar_faq,
    garantir_indice,
    ids_do_dataset,
    indexar,
)


class VectorStoreEspiao:
    """Captura os argumentos de ``add_texts`` para verificar o upsert."""

    def __init__(self) -> None:
        self.chamadas: list[dict] = []

    def add_texts(self, texts, metadatas, ids):
        self.chamadas.append({"texts": texts, "metadatas": metadatas, "ids": ids})
        return ids


def _escrever(tmp_path, dados) -> "object":
    tmp_path.mkdir(parents=True, exist_ok=True)
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


# --- Reindexação na subida ---------------------------------------------------
#
# O plano Free do Render não tem disco: o chroma_db some a cada deploy, restart
# ou hibernação, e o índice é reconstruído na subida. Estes testes cobrem a
# decisão de reconstruir ou não — errar para o lado de não reconstruir deixaria
# o bot no ar respondendo `sem_resultados` a todo cliente.


class VectorStoreComIds(VectorStoreEspiao):
    """Espião que também responde ``get``, como o Chroma."""

    def __init__(self, ids: list[str] | None = None) -> None:
        super().__init__()
        self.existentes = list(ids or [])

    def get(self, include=None):
        return {"ids": list(self.existentes)}


def _config_com_faq(tmp_path, dados):
    return obter_configuracoes().model_copy(
        update={"caminho_faq": _escrever(tmp_path, dados)}
    )


def test_garantir_indice_indexa_quando_a_base_esta_vazia(tmp_path) -> None:
    config = _config_com_faq(
        tmp_path, [{"pergunta": "P?", "resposta": "R.", "variacoes": ["p?"]}]
    )
    store = VectorStoreComIds()

    assert garantir_indice(config, store) == 2
    assert len(store.chamadas) == 1


def test_garantir_indice_nao_gasta_embedding_quando_ja_esta_completo(
    tmp_path,
) -> None:
    """O caso local: reiniciar o app não deve custar chamada de API nenhuma."""
    dados = [{"pergunta": "P?", "resposta": "R.", "variacoes": ["p?"]}]
    config = _config_com_faq(tmp_path, dados)
    store = VectorStoreComIds(sorted(ids_do_dataset(carregar_faq(config.caminho_faq))))

    assert garantir_indice(config, store) == 0
    assert store.chamadas == []


def test_garantir_indice_reindexa_quando_o_faq_ganhou_pergunta(tmp_path) -> None:
    """Compara ids, não contagem: pergunta nova precisa entrar no índice."""
    antigo = [{"pergunta": "P1?", "resposta": "R."}]
    config_antiga = _config_com_faq(tmp_path / "a", antigo)
    ids_antigos = sorted(ids_do_dataset(carregar_faq(config_antiga.caminho_faq)))

    novo = antigo + [{"pergunta": "P2?", "resposta": "R2."}]
    config = _config_com_faq(tmp_path / "b", novo)
    store = VectorStoreComIds(ids_antigos)

    assert garantir_indice(config, store) == 2


def test_ids_do_dataset_cobre_perguntas_e_variacoes() -> None:
    item = ItemFAQ(pergunta="P?", resposta="R.", variacoes=("v1", "v2"))

    assert ids_do_dataset([item]) == {
        item.id,
        item.id_da_variacao("v1"),
        item.id_da_variacao("v2"),
    }
