"""Testes do composition root: as três fronteiras trocáveis por variável.

O valor aqui não é cobrir os `if`: é travar o contrato que o README promete —
que trocar de provedor é configuração, e que um nome desconhecido falha na
subida com uma mensagem que diz o que fazer, em vez de mais tarde e no escuro.
"""

import pytest

from app.config import Configuracoes
from app.dependencias import criar_embedding, criar_llm, resolver_limiar
from app.llm.demo_client import ProvedorDemo
from app.llm.gemini_client import ProvedorGemini
from app.rag.embedding.gemini_client import ProvedorEmbeddingGemini


# --- LLM ---------------------------------------------------------------------


def test_provedor_llm_gemini() -> None:
    config = Configuracoes(provedor_llm="gemini", gemini_api_key="k")

    assert isinstance(criar_llm(config), ProvedorGemini)


def test_provedor_llm_demo() -> None:
    assert isinstance(criar_llm(Configuracoes(provedor_llm="demo")), ProvedorDemo)


def test_provedor_llm_desconhecido_diz_quais_existem() -> None:
    with pytest.raises(ValueError, match="'openai', 'gemini' ou 'demo'"):
        criar_llm(Configuracoes(provedor_llm="anthropic"))


def test_provedor_llm_gemini_sem_chave_falha_cedo() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        criar_llm(Configuracoes(provedor_llm="gemini", gemini_api_key=None))


# --- Embedding ---------------------------------------------------------------


def test_provedor_embedding_gemini_usa_a_configuracao() -> None:
    config = Configuracoes(
        provedor_embedding="gemini",
        gemini_api_key="k",
        gemini_modelo_embedding="gemini-embedding-001",
        gemini_dimensoes_embedding=1536,
    )

    provedor = criar_embedding(config)

    assert isinstance(provedor, ProvedorEmbeddingGemini)
    assert provedor.descricao == "gemini:gemini-embedding-001@1536d"


def test_provedor_embedding_local_nao_carrega_o_modelo_na_criacao() -> None:
    """Criar é barato; o download de ~2,2 GB só acontece no primeiro uso."""
    # Modelo explícito para o teste não herdar o .env do desenvolvedor.
    config = Configuracoes(
        provedor_embedding="local", modelo_embedding="intfloat/multilingual-e5-large"
    )

    assert criar_embedding(config).descricao == "local:intfloat/multilingual-e5-large"


def test_provedor_embedding_desconhecido_diz_quais_existem() -> None:
    with pytest.raises(ValueError, match="'local' ou 'gemini'"):
        criar_embedding(Configuracoes(provedor_embedding="cohere"))


# --- Limiar ------------------------------------------------------------------


def test_limiar_vazio_usa_o_calibrado_do_provedor_ativo() -> None:
    """É o que impede trocar de embedding e herdar o limiar do modelo antigo."""
    config = Configuracoes(
        provedor_embedding="gemini", gemini_api_key="k", limiar_similaridade=None
    )
    embedding = criar_embedding(config)

    assert resolver_limiar(config, embedding) == embedding.limiar_calibrado


def test_limiar_do_env_tem_a_palavra_final() -> None:
    config = Configuracoes(
        provedor_embedding="gemini", gemini_api_key="k", limiar_similaridade=0.42
    )

    assert resolver_limiar(config, criar_embedding(config)) == 0.42


# --- Parsing da configuração -------------------------------------------------


def test_limiar_em_branco_no_env_vale_como_nao_definido() -> None:
    """`LIMIAR_SIMILARIDADE=` é como o .env.example pede o padrão do provedor.

    Sem o tratamento de string vazia isso derrubava a subida com erro de
    parsing, em vez de cair no limiar calibrado.
    """
    assert Configuracoes(limiar_similaridade="").limiar_similaridade is None
    assert Configuracoes(limiar_similaridade="   ").limiar_similaridade is None


def test_limiar_preenchido_continua_sendo_lido() -> None:
    assert Configuracoes(limiar_similaridade="0.75").limiar_similaridade == 0.75
