"""Testes da fronteira de embedding: contrato, adaptador e os dois provedores.

Nenhum teste aqui baixa modelo ou chama API — o provedor local recebe um
modelo falso e o do Gemini tem o transporte HTTP substituído. O que se verifica
é o que quebra em silêncio: os prefixos/taskType que separam documento de
consulta, e a normalização de que o cálculo de similaridade depende.
"""

import math

import pytest

from app.rag.embedding.base import (
    AdaptadorLangChain,
    ErroEmbedding,
    ProvedorEmbedding,
    normalizar,
)
from app.rag.embedding.gemini_client import (
    TAREFA_CONSULTA,
    TAREFA_DOCUMENTO,
    ProvedorEmbeddingGemini,
)
from app.rag.embedding.local_client import ProvedorEmbeddingLocal


class EmbeddingFalso(ProvedorEmbedding):
    """Provedor mínimo, para exercitar o adaptador."""

    def __init__(self) -> None:
        self.documentos: list[list[str]] = []
        self.consultas: list[str] = []

    def vetorizar_documentos(self, textos):
        self.documentos.append(list(textos))
        return [[1.0, 0.0] for _ in textos]

    def vetorizar_consulta(self, texto):
        self.consultas.append(texto)
        return [0.0, 1.0]

    @property
    def limiar_calibrado(self) -> float:
        return 0.5


class ModeloFalso:
    """Imita o ``SentenceTransformer`` (só o ``encode``)."""

    def __init__(self) -> None:
        self.recebidos: list[list[str]] = []

    def encode(self, textos, normalize_embeddings=True):
        self.recebidos.append(list(textos))
        return [[1.0, 0.0, 0.0] for _ in textos]


# --- normalizar --------------------------------------------------------------


def test_normalizar_devolve_vetor_unitario() -> None:
    vetor = normalizar([3.0, 4.0])

    assert math.isclose(sum(valor * valor for valor in vetor), 1.0)
    assert vetor == pytest.approx([0.6, 0.8])


def test_normalizar_nao_divide_por_zero() -> None:
    assert normalizar([0.0, 0.0]) == [0.0, 0.0]


# --- adaptador para o Chroma -------------------------------------------------


def test_adaptador_encaminha_documentos_e_consulta() -> None:
    provedor = EmbeddingFalso()
    adaptador = AdaptadorLangChain(provedor)

    assert adaptador.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert adaptador.embed_query("c") == [0.0, 1.0]
    assert provedor.documentos == [["a", "b"]]
    assert provedor.consultas == ["c"]


# --- provedor local ----------------------------------------------------------


def test_local_aplica_os_prefixos_do_e5(monkeypatch) -> None:
    """Sem `passage:`/`query:` o e5 continua funcionando, só que buscando pior."""
    modelo = ModeloFalso()
    monkeypatch.setattr(
        "app.rag.embedding.local_client._carregar_modelo", lambda _: modelo
    )
    provedor = ProvedorEmbeddingLocal(modelo="intfloat/multilingual-e5-large")

    provedor.vetorizar_documentos(["prazo de entrega"])
    provedor.vetorizar_consulta("quando chega?")

    assert modelo.recebidos == [
        ["passage: prazo de entrega"],
        ["query: quando chega?"],
    ]


def test_local_nao_prefixa_modelo_que_nao_e_e5(monkeypatch) -> None:
    modelo = ModeloFalso()
    monkeypatch.setattr(
        "app.rag.embedding.local_client._carregar_modelo", lambda _: modelo
    )
    provedor = ProvedorEmbeddingLocal(modelo="sentence-transformers/all-MiniLM-L6-v2")

    provedor.vetorizar_documentos(["prazo de entrega"])

    assert modelo.recebidos == [["prazo de entrega"]]


# --- provedor Gemini ---------------------------------------------------------


def _fixar_resposta(monkeypatch, resposta, registro: list | None = None):
    def chamar_falso(caminho, corpo, api_key, timeout=30.0):
        if registro is not None:
            registro.append({"caminho": caminho, "corpo": corpo, "chave": api_key})
        return resposta

    monkeypatch.setattr(
        "app.rag.embedding.gemini_client.chamar", chamar_falso
    )


def test_gemini_exige_chave() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        ProvedorEmbeddingGemini(api_key=None)


def test_gemini_marca_documento_e_consulta_com_task_types_diferentes(
    monkeypatch,
) -> None:
    """É o análogo dos prefixos do e5 — e some tão silenciosamente quanto."""
    registro: list = []
    _fixar_resposta(
        monkeypatch,
        {"embeddings": [{"values": [1.0, 0.0]}], "embedding": {"values": [0.0, 1.0]}},
        registro,
    )
    provedor = ProvedorEmbeddingGemini(api_key="k", dimensoes=2)

    provedor.vetorizar_documentos(["prazo de entrega"])
    provedor.vetorizar_consulta("quando chega?")

    documento, consulta = registro
    assert documento["caminho"].endswith(":batchEmbedContents")
    assert documento["corpo"]["requests"][0]["taskType"] == TAREFA_DOCUMENTO
    assert consulta["caminho"].endswith(":embedContent")
    assert consulta["corpo"]["taskType"] == TAREFA_CONSULTA


def test_gemini_pede_o_modelo_e_as_dimensoes_configurados(monkeypatch) -> None:
    registro: list = []
    _fixar_resposta(monkeypatch, {"embedding": {"values": [1.0, 0.0]}}, registro)
    provedor = ProvedorEmbeddingGemini(api_key="k", modelo="gemini-embedding-001",
                                       dimensoes=768)

    provedor.vetorizar_consulta("oi")

    assert registro[0]["corpo"]["model"] == "models/gemini-embedding-001"
    assert registro[0]["corpo"]["outputDimensionality"] == 768
    assert registro[0]["chave"] == "k"


def test_gemini_normaliza_a_saida(monkeypatch) -> None:
    """As dimensões truncadas do Matryoshka não vêm normalizadas.

    Sem isto o `1 - distância` do Retriever para de render similaridade de 0 a
    1, e o limiar de transbordo passa a comparar números sem escala.
    """
    _fixar_resposta(monkeypatch, {"embedding": {"values": [3.0, 4.0]}})
    provedor = ProvedorEmbeddingGemini(api_key="k", dimensoes=2)

    assert provedor.vetorizar_consulta("oi") == pytest.approx([0.6, 0.8])


def test_gemini_recusa_lote_com_tamanho_diferente_do_pedido(monkeypatch) -> None:
    """Alinhamento errado associaria a resposta de uma pergunta a outra."""
    _fixar_resposta(monkeypatch, {"embeddings": [{"values": [1.0, 0.0]}]})
    provedor = ProvedorEmbeddingGemini(api_key="k", dimensoes=2)

    with pytest.raises(ErroEmbedding, match="1 embeddings para 2 textos"):
        provedor.vetorizar_documentos(["a", "b"])


def test_gemini_recusa_embedding_vazio(monkeypatch) -> None:
    _fixar_resposta(monkeypatch, {"embedding": {"values": []}})
    provedor = ProvedorEmbeddingGemini(api_key="k", dimensoes=2)

    with pytest.raises(ErroEmbedding, match="vazio"):
        provedor.vetorizar_consulta("oi")


def test_gemini_traduz_falha_de_transporte(monkeypatch) -> None:
    def explodir(*_args, **_kwargs):
        raise RuntimeError("HTTP 429: cota estourada")

    monkeypatch.setattr("app.rag.embedding.gemini_client.chamar", explodir)
    provedor = ProvedorEmbeddingGemini(api_key="k")

    with pytest.raises(ErroEmbedding, match="cota estourada"):
        provedor.vetorizar_consulta("oi")


def test_gemini_quebra_lotes_grandes(monkeypatch) -> None:
    """A API aceita 100 requisições por chamada; o FAQ pode passar disso."""
    lotes: list[int] = []

    def chamar_falso(caminho, corpo, api_key, timeout=30.0):
        quantidade = len(corpo["requests"])
        lotes.append(quantidade)
        return {"embeddings": [{"values": [1.0, 0.0]}] * quantidade}

    monkeypatch.setattr("app.rag.embedding.gemini_client.chamar", chamar_falso)
    provedor = ProvedorEmbeddingGemini(api_key="k", dimensoes=2)

    vetores = provedor.vetorizar_documentos([f"p{i}" for i in range(250)])

    assert lotes == [100, 100, 50]
    assert len(vetores) == 250
