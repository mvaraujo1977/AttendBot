"""Testes do fluxo completo de atendimento (busca -> handoff -> geração)."""

import pytest

from app.atendimento import MENSAGEM_SEM_TEXTO, ServicoAtendimento
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


def _criar_servico(retorno, llm) -> ServicoAtendimento:
    return ServicoAtendimento(
        retriever=Retriever(VectorStoreFalso(retorno), top_k=3),
        generator=Generator(llm, nome_empresa="Loja Exemplo"),
        limiar_similaridade=LIMIAR,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
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
