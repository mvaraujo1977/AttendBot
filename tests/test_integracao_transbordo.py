"""Testes de integração com o dataset real e embeddings reais.

Os outros testes usam dublês e provam que a *lógica* está certa. Isso não basta:
o bug que apareceu neste projeto foi de **calibração** — a lógica funcionava,
mas o modelo de embeddings pontuava perguntas fora do domínio tão alto quanto
as de dentro, e o transbordo nunca disparava. Nenhum teste com dublê pegaria
isso, e foi este arquivo que pegou.

Tudo aqui roda contra o ``PROVEDOR_EMBEDDING`` configurado, e é isso que dá
sentido ao módulo: as faixas de similaridade que ele vigia são propriedade do
modelo ativo. Trocar de provedor sem recalibrar tem que fazer estes testes
falharem.

O que dá para verificar com o embedding local (determinístico, sem custo):

* o retrieval recupera a entrada certa da FAQ para perguntas parafraseadas;
* a primeira barreira (limiar de similaridade) descarta o que nem chega perto;
* a calibração registrada no README continua valendo.

A segunda barreira mora no LLM, então os testes que a cobrem exigem um provedor
real (``PROVEDOR_LLM=openai`` ou ``gemini``, com a chave correspondente) e são
pulados sem ele. Um deles também é pulado quando a 1a barreira já barra toda a
fronteira sozinha — o que acontece com o embedding do Gemini, mas não com o e5.

São lentos (carregam o modelo ou chamam a API, e indexam no ChromaDB), então
ficam fora da suíte padrão. Para rodar:

    pytest -m integracao
"""

import pytest

from app.atendimento import ServicoAtendimento
from app.config import obter_configuracoes
from app.dependencias import criar_embedding, criar_llm, resolver_limiar
from app.handoff import MotivoTransbordo
from app.llm.demo_client import ProvedorDemo
from app.rag.calibracao import (
    FRONTEIRA,
    LONGE_DA_BASE,
    PERGUNTAS_COBERTAS,
    medir,
)
from app.rag.generator import Generator
from app.rag.ingest import executar_ingestao
from app.rag.retriever import Retriever
from app.rag.vector_store import criar_vector_store

pytestmark = pytest.mark.integracao

pytest.importorskip("chromadb", reason="requer as dependências completas")

MENSAGEM_TRANSBORDO = "Vou te encaminhar para um atendente humano."

# O corpus de 16 perguntas (10 cobertas, 6 fora, todas acentuadas) mora em
# `app/rag/calibracao.py`, junto da rotina de medição: é o mesmo conjunto que
# `scripts/calibrar_limiar.py` usa para fixar o limiar. Duplicá-lo aqui
# deixaria a calibração e o teste que a vigia livres para divergirem.


@pytest.fixture(scope="module")
def indice(tmp_path_factory):
    """Indexa o dataset real em um ChromaDB temporário, uma vez por módulo.

    Roda contra o ``PROVEDOR_EMBEDDING`` configurado — é o ponto: as faixas de
    similaridade que este módulo vigia são uma propriedade do modelo ativo, e
    trocar de provedor tem que fazer estes testes falharem até a recalibração.
    """
    config = obter_configuracoes().model_copy(
        update={"diretorio_chroma": tmp_path_factory.mktemp("chroma_integracao")}
    )
    embedding = criar_embedding(config)
    executar_ingestao(config, embedding=embedding)
    retriever = Retriever(
        criar_vector_store(config, embedding), top_k=config.top_k
    )
    return config, retriever, resolver_limiar(config, embedding)


def _montar_servico(config, retriever, limiar, llm) -> ServicoAtendimento:
    return ServicoAtendimento(
        retriever=retriever,
        generator=Generator(llm, nome_empresa=config.nome_empresa),
        limiar_similaridade=limiar,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
    )


@pytest.mark.parametrize(("pergunta", "canonica_esperada"), PERGUNTAS_COBERTAS)
def test_retrieval_recupera_a_entrada_certa_da_faq(
    indice, pergunta: str, canonica_esperada: str
) -> None:
    """Qualidade do retrieval: é a base de tudo que vem depois."""
    _, retriever, _limiar = indice

    resultados = retriever.buscar(pergunta)

    assert resultados, f"'{pergunta}' não recuperou nada"
    assert resultados[0].pergunta == canonica_esperada, (
        f"'{pergunta}' recuperou '{resultados[0].pergunta}' "
        f"(similaridade={resultados[0].similaridade:.3f})"
    )


@pytest.mark.parametrize(("pergunta", "_canonica"), PERGUNTAS_COBERTAS)
def test_pergunta_coberta_passa_da_primeira_barreira(
    indice, pergunta: str, _canonica: str
) -> None:
    """Nenhuma pergunta legítima pode morrer no limiar de similaridade."""
    config, retriever, limiar = indice

    melhor = retriever.buscar(pergunta)[0].similaridade

    assert melhor >= limiar, (
        f"'{pergunta}' ficou em {melhor:.3f}, abaixo do limiar "
        f"{limiar:.2f}: o bot transbordaria sem necessidade"
    )


@pytest.mark.parametrize("pergunta", LONGE_DA_BASE)
def test_pergunta_distante_morre_na_primeira_barreira(indice, pergunta: str) -> None:
    config, retriever, limiar = indice
    servico = _montar_servico(config, retriever, limiar, ProvedorDemo())

    resposta = servico.responder(pergunta)

    assert resposta.transbordo, (
        f"'{pergunta}' devia transbordar, mas o bot respondeu "
        f"'{resposta.texto}' (similaridade={resposta.similaridade:.3f})"
    )
    assert resposta.motivo == MotivoTransbordo.BAIXA_SIMILARIDADE.value


def test_a_calibracao_do_provedor_ativo_continua_valendo(indice) -> None:
    """Regressão da calibração, no formato da tabela do README.

    Roda o mesmo corpus e a mesma aritmética de ``scripts/calibrar_limiar.py``,
    então se alguém mexer no dataset, no modelo ou no limiar sem recalibrar, é
    aqui que aparece.
    """
    _, retriever, limiar = indice

    medicao = medir(retriever)

    assert medicao.acertos == len(PERGUNTAS_COBERTAS), (
        f"o retrieval errou {len(PERGUNTAS_COBERTAS) - medicao.acertos} "
        "entrada(s): limiar nenhum conserta modelo que busca a resposta errada"
    )
    assert medicao.faixa_cobertas[0] >= limiar, (
        f"a pergunta legítima mais fraca ficou em {medicao.faixa_cobertas[0]:.3f}, "
        f"abaixo do limiar {limiar:.2f}: o bot transbordaria sem necessidade"
    )


def _llm_real(config):
    """LLM de verdade para os testes das duas barreiras, ou pula o teste.

    Serve qualquer provedor configurado — o transbordo é do `ServicoAtendimento`
    e não depende de quem gera o texto. Amarrar na OpenAI deixaria a 2a barreira
    sem cobertura justamente na configuração que vai para produção (Gemini).
    """
    if config.provedor_llm.strip().lower() == "demo":
        pytest.skip("requer um LLM real (PROVEDOR_LLM=openai ou gemini)")
    try:
        return criar_llm(config)
    except ValueError as erro:
        pytest.skip(str(erro))


@pytest.mark.parametrize("pergunta", FRONTEIRA)
def test_pergunta_de_fronteira_nunca_e_respondida(indice, pergunta: str) -> None:
    """O invariante que vale para qualquer modelo de embedding.

    QUAL barreira segura essas perguntas depende do modelo, e os dois regimes
    estão medidos no README: com o e5 local elas passam do limiar e só o LLM as
    barra; com o embedding do Gemini a 1a barreira já as separa com folga. O
    que não pode variar é o resultado — pergunta sem resposta na FAQ não é
    respondida.
    """
    config, retriever, limiar = indice
    servico = _montar_servico(config, retriever, limiar, _llm_real(config))

    resposta = servico.responder(pergunta)

    assert resposta.transbordo, (
        f"'{pergunta}' não é coberta pela FAQ, mas o bot respondeu "
        f"'{resposta.texto}'"
    )
    assert resposta.motivo in {
        MotivoTransbordo.BAIXA_SIMILARIDADE.value,
        MotivoTransbordo.CONTEXTO_INSUFICIENTE.value,
    }


def test_segunda_barreira_barra_o_que_a_similaridade_deixou_passar(indice) -> None:
    """Cobre a 2a barreira só com o que a 1a de fato deixou passar.

    Com um embedding que separa as faixas sozinho não sobra nada para ela
    julgar neste corpus, e o teste é pulado — o que não a aposenta: seis
    perguntas não são prova de cobertura, e ela não custa chamada extra.
    """
    config, retriever, limiar = indice
    passaram = [
        pergunta
        for pergunta in FRONTEIRA
        if retriever.buscar(pergunta)[0].similaridade >= limiar
    ]
    if not passaram:
        pytest.skip(
            "a 1a barreira barrou toda a fronteira neste corpus; nada chega à 2a"
        )

    servico = _montar_servico(config, retriever, limiar, _llm_real(config))

    for pergunta in passaram:
        resposta = servico.responder(pergunta)
        assert resposta.motivo == MotivoTransbordo.CONTEXTO_INSUFICIENTE.value, (
            f"'{pergunta}' passou do limiar e devia ser barrada pelo LLM, "
            f"mas o motivo foi {resposta.motivo}"
        )


@pytest.mark.parametrize(("pergunta", "_canonica"), PERGUNTAS_COBERTAS)
def test_pergunta_coberta_e_respondida_com_llm_real(
    indice, pergunta: str, _canonica: str
) -> None:
    """A segunda barreira não pode ser conservadora demais e barrar o legítimo."""
    config, retriever, limiar = indice
    servico = _montar_servico(config, retriever, limiar, _llm_real(config))

    resposta = servico.responder(pergunta)

    assert not resposta.transbordo, (
        f"'{pergunta}' é coberta pela FAQ, mas transbordou "
        f"(motivo={resposta.motivo}, similaridade={resposta.similaridade:.3f})"
    )
