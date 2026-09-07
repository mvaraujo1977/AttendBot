"""Testes do fluxo completo de atendimento (busca -> handoff -> geração)."""

import logging

import pytest

from app.atendimento import (
    MENSAGEM_BOAS_VINDAS,
    MENSAGEM_SEM_TEXTO,
    ServicoAtendimento,
)
from app.handoff import MotivoTransbordo
from app.rag.generator import SINAL_TRANSBORDO, Generator
from app.rag.retriever import Retriever
from tests.conftest import DocumentoFalso, LLMFalso, LLMQuebrado, VectorStoreFalso

MENSAGEM_TRANSBORDO = "Vou te encaminhar para um atendente."
LIMIAR = 0.60


def _documento(pergunta: str, resposta: str) -> DocumentoFalso:
    return DocumentoFalso(
        page_content=pergunta,
        metadata={
            "pergunta": pergunta,
            "resposta": resposta,
            "categoria": "entrega",
            "tags": "prazo",
        },
    )


def _criar_servico(retorno, llm, **kwargs) -> ServicoAtendimento:
    return ServicoAtendimento(
        retriever=Retriever(VectorStoreFalso(retorno), top_k=3),
        generator=Generator(llm, nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
        **kwargs,
    )


def test_responde_com_o_llm_quando_encontra_contexto_relevante() -> None:
    llm = LLMFalso("Seu pedido chega em até 7 dias úteis.")
    servico = _criar_servico(
        [(_documento("Qual o prazo de entrega?", "De 3 a 7 dias úteis."), 0.10)], llm
    )

    resposta = servico.responder("quando chega meu pedido?")

    assert resposta.transbordo is False
    assert resposta.texto == "Seu pedido chega em até 7 dias úteis."
    assert resposta.similaridade == 0.90
    # O contexto recuperado precisa chegar ao prompt do LLM.
    _, prompt_usuario = llm.prompts[0]
    assert "De 3 a 7 dias úteis." in prompt_usuario
    assert "quando chega meu pedido?" in prompt_usuario


def test_transborda_e_nao_chama_o_llm_quando_a_similaridade_e_baixa() -> None:
    llm = LLMFalso()
    servico = _criar_servico(
        [(_documento("Qual o prazo de entrega?", "De 3 a 7 dias."), 0.85)], llm
    )

    resposta = servico.responder("vocês têm vaga de emprego?")

    assert resposta.transbordo is True
    assert resposta.texto == MENSAGEM_TRANSBORDO
    assert resposta.motivo == MotivoTransbordo.BAIXA_SIMILARIDADE.value
    assert llm.prompts == []  # nada foi enviado ao LLM


def test_transborda_quando_a_base_esta_vazia() -> None:
    servico = _criar_servico([], LLMFalso())

    resposta = servico.responder("qualquer pergunta")

    assert resposta.transbordo is True
    assert resposta.motivo == MotivoTransbordo.SEM_RESULTADOS.value


def test_transborda_quando_o_llm_falha() -> None:
    servico = _criar_servico(
        [(_documento("Qual o frete?", "Grátis acima de R$ 199."), 0.05)], LLMQuebrado()
    )

    resposta = servico.responder("quanto custa o frete?")

    assert resposta.transbordo is True
    assert resposta.texto == MENSAGEM_TRANSBORDO
    assert resposta.motivo == MotivoTransbordo.ERRO_GERACAO.value


def test_contexto_fraco_nao_entra_no_prompt() -> None:
    llm = LLMFalso()
    servico = _criar_servico(
        [
            (_documento("Qual o prazo de entrega?", "De 3 a 7 dias úteis."), 0.10),
            (_documento("Como trocar um produto?", "Em até 30 dias."), 0.80),
        ],
        llm,
    )

    servico.responder("quando chega?")

    _, prompt_usuario = llm.prompts[0]
    assert "De 3 a 7 dias úteis." in prompt_usuario
    assert "Em até 30 dias." not in prompt_usuario


def test_transborda_quando_o_llm_julga_o_contexto_insuficiente() -> None:
    """Segunda barreira: a similaridade passou, mas o LLM disse que não serve."""
    servico = _criar_servico(
        [(_documento("Qual o prazo de entrega?", "De 3 a 7 dias úteis."), 0.05)],
        LLMFalso(SINAL_TRANSBORDO),
    )

    resposta = servico.responder("vocês vendem passagem aérea?")

    assert resposta.transbordo is True
    assert resposta.texto == MENSAGEM_TRANSBORDO
    assert resposta.motivo == MotivoTransbordo.CONTEXTO_INSUFICIENTE.value


@pytest.mark.parametrize(
    "resposta_do_llm",
    [SINAL_TRANSBORDO, " transbordo ", "TRANSBORDO.", '"Transbordo"', "transbordo!"],
)
def test_sinal_de_transbordo_e_reconhecido_apesar_de_variacoes(
    resposta_do_llm: str,
) -> None:
    """O modelo nem sempre devolve o sinal exatamente como pedido."""
    servico = _criar_servico(
        [(_documento("Qual o frete?", "Grátis acima de R$ 199."), 0.05)],
        LLMFalso(resposta_do_llm),
    )

    assert servico.responder("qualquer pergunta").transbordo is True


def test_resposta_normal_que_menciona_transbordo_nao_e_confundida() -> None:
    """Só o sinal isolado conta — texto que contém a palavra é resposta comum."""
    texto = "Vou transbordo? Não: seu pedido chega em até 7 dias úteis."
    servico = _criar_servico(
        [(_documento("Qual o prazo?", "De 3 a 7 dias."), 0.05)], LLMFalso(texto)
    )

    resposta = servico.responder("quando chega?")

    assert resposta.transbordo is False
    assert resposta.texto == texto


def test_mensagem_sem_texto_pede_a_duvida_por_escrito() -> None:
    llm = LLMFalso()
    servico = _criar_servico([(_documento("P", "R"), 0.1)], llm)

    resposta = servico.responder("   ")

    assert resposta.texto == MENSAGEM_SEM_TEXTO
    assert resposta.transbordo is False
    assert llm.prompts == []


# --- Comandos (respondidos sem passar pelo RAG) ------------------------------


@pytest.mark.parametrize("comando", ["/start", "/help", "/ajuda", "/START", "/Help"])
def test_comandos_respondem_boas_vindas_sem_consultar_o_rag(comando: str) -> None:
    llm = LLMFalso("nao deveria ser chamado")
    store = VectorStoreFalso([(_documento("Qual o frete?", "Grátis."), 0.10)])
    servico = ServicoAtendimento(
        retriever=Retriever(store, top_k=3),
        generator=Generator(llm, nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
    )

    resposta = servico.responder(comando)

    assert resposta.texto == MENSAGEM_BOAS_VINDAS
    assert resposta.transbordo is False
    # A prova de que o RAG foi curto-circuitado: nem busca nem LLM aconteceram.
    assert store.chamadas == []
    assert llm.prompts == []


@pytest.mark.parametrize(
    "mensagem",
    [
        "/start@attendbot_marcelo_bot",  # o Telegram sufixa o comando em grupos
        "/start abc123",  # payload de link de convite
        "  /help  ",
    ],
)
def test_comando_e_reconhecido_com_sufixo_do_bot_ou_payload(mensagem: str) -> None:
    servico = _criar_servico([], LLMFalso("x"))

    assert servico.responder(mensagem).texto == MENSAGEM_BOAS_VINDAS


def test_comando_desconhecido_segue_para_o_rag() -> None:
    servico = _criar_servico([], LLMFalso("x"))

    resposta = servico.responder("/qualquer_outra_coisa")

    # Sem resultados na busca, cai no transbordo normal — não em boas-vindas.
    assert resposta.texto == MENSAGEM_TRANSBORDO
    assert resposta.motivo == MotivoTransbordo.SEM_RESULTADOS.value


def test_mensagem_com_barra_no_meio_nao_e_comando() -> None:
    llm = LLMFalso("Aceitamos Pix.")
    servico = _criar_servico(
        [(_documento("Formas de pagamento?", "Pix e cartão."), 0.10)], llm
    )

    assert servico.responder("aceita pix e/ou boleto?").texto == "Aceitamos Pix."


def test_mensagem_de_boas_vindas_e_configuravel() -> None:
    servico = ServicoAtendimento(
        retriever=Retriever(VectorStoreFalso([]), top_k=3),
        generator=Generator(LLMFalso("x"), nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
        mensagem_boas_vindas="Oi! Sou o bot da Loja Exemplo.",
    )

    assert servico.responder("/start").texto == "Oi! Sou o bot da Loja Exemplo."


# --- Teto de tamanho da pergunta ---------------------------------------------


def test_pergunta_longa_e_cortada_antes_de_virar_embedding_e_prompt() -> None:
    """O custo em tokens é linear no tamanho, e a entrada não tinha teto.

    O corte mora no serviço, e não no schema da rota, para valer igual nos dois
    caminhos de entrada — o webhook de cada canal e o `/api/mensagem`.
    """
    llm = LLMFalso("Chega em 7 dias.")
    vector_store = VectorStoreFalso(
        [(_documento("Qual o prazo de entrega?", "De 3 a 7 dias úteis."), 0.10)]
    )
    servico = ServicoAtendimento(
        retriever=Retriever(vector_store, top_k=3),
        generator=Generator(llm, nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
        limite_caracteres=100,
    )

    servico.responder("prazo? " + "x" * 5_000)

    consulta, _ = vector_store.chamadas[0]
    assert len(consulta) == 100
    _, prompt_usuario = llm.prompts[0]
    assert len(prompt_usuario) < 500


def test_pergunta_dentro_do_limite_chega_intacta() -> None:
    llm = LLMFalso("Chega em 7 dias.")
    servico = _criar_servico(
        [(_documento("Qual o prazo de entrega?", "De 3 a 7 dias úteis."), 0.10)],
        llm,
        limite_caracteres=100,
    )

    servico.responder("quando chega meu pedido?")

    _, prompt_usuario = llm.prompts[0]
    assert "quando chega meu pedido?" in prompt_usuario


def test_comando_com_payload_nao_vai_inteiro_para_o_log(caplog) -> None:
    """`/start <payload>` carrega o payload do link de convite."""
    servico = _criar_servico([], LLMFalso())

    with caplog.at_level(logging.INFO, logger="app.atendimento"):
        resposta = servico.responder("/start ref_campanha_cliente_12345")

    assert resposta.texto == MENSAGEM_BOAS_VINDAS
    assert "ref_campanha_cliente_12345" not in caplog.text
    assert "/start" in caplog.text


# --- Saudações (respondidas sem passar pelo RAG) -----------------------------


@pytest.mark.parametrize(
    "saudacao",
    [
        "oi",
        "Oi",
        "OI",
        "oi!",
        "  oi  ",
        "olá",
        "Olá!",
        "ola",
        "OLÁ",
        "opa",
        "oie",
        "e aí",
        "E aí?",
        "eai",
        "bom dia",
        "Bom dia!",
        "BOM DIA",
        "boa tarde",
        "Boa tarde.",
        "boa noite",
        "Boa noite!",
        "oi, bom dia",  # uma mensagem, duas saudações
        "Olá! Boa tarde.",
    ],
)
def test_saudacao_responde_boas_vindas_sem_consultar_o_rag(saudacao: str) -> None:
    llm = LLMFalso("nao deveria ser chamado")
    store = VectorStoreFalso([(_documento("Qual o frete?", "Grátis."), 0.10)])
    servico = ServicoAtendimento(
        retriever=Retriever(store, top_k=3),
        generator=Generator(llm, nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
    )

    resposta = servico.responder(saudacao)

    assert resposta.texto == MENSAGEM_BOAS_VINDAS
    assert resposta.transbordo is False
    # A prova de que nada foi gasto: nem embedding da consulta, nem LLM.
    assert store.chamadas == []
    assert llm.prompts == []


@pytest.mark.parametrize(
    "mensagem",
    [
        # Saudação + pergunta: o cliente quer uma resposta, não boas-vindas.
        "bom dia, qual o prazo de entrega?",
        "Bom dia! Qual o prazo de entrega?",
        "oi, quanto custa o frete?",
        "olá, meu pedido não chegou",
        # A saudação no meio ou no fim não pode sequestrar a mensagem.
        "meu pedido não chegou, boa noite",
        "queria saber se vocês abrem de boa noite até de manhã",
        "recebi o pedido de bom dia",
        # Perguntas comuns que contêm um termo da lista como substring.
        "qual o horário de atendimento?",
        "voces entregam a noite?",
    ],
)
def test_mensagem_com_pergunta_junto_vai_para_o_rag(mensagem: str) -> None:
    """Só saudação isolada é curto-circuitada; substring não conta."""
    llm = LLMFalso("Chega em 7 dias.")
    store = VectorStoreFalso(
        [(_documento("Qual o prazo de entrega?", "De 3 a 7 dias úteis."), 0.10)]
    )
    servico = ServicoAtendimento(
        retriever=Retriever(store, top_k=3),
        generator=Generator(llm, nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
    )

    resposta = servico.responder(mensagem)

    assert resposta.texto != MENSAGEM_BOAS_VINDAS
    assert store.chamadas, "a busca vetorial precisa ter acontecido"
    assert llm.prompts, "o LLM precisa ter sido consultado"


def test_saudacao_desconhecida_segue_para_o_rag() -> None:
    """A lista é conservadora: o que ela não cobre continua no caminho normal."""
    servico = _criar_servico([], LLMFalso("x"))

    resposta = servico.responder("salve, blz?")

    assert resposta.transbordo is True
    assert resposta.motivo == MotivoTransbordo.SEM_RESULTADOS.value


def test_saudacao_usa_a_mensagem_de_boas_vindas_configurada() -> None:
    servico = ServicoAtendimento(
        retriever=Retriever(VectorStoreFalso([]), top_k=3),
        generator=Generator(LLMFalso(), nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
        mensagem_boas_vindas="Oi! Como posso ajudar?",
    )

    assert servico.responder("bom dia").texto == "Oi! Como posso ajudar?"


def test_pontuacao_sozinha_nao_e_saudacao() -> None:
    servico = _criar_servico([], LLMFalso("x"))

    resposta = servico.responder("???")

    assert resposta.texto != MENSAGEM_BOAS_VINDAS
