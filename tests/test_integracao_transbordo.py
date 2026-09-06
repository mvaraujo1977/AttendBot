"""Testes de integração com o dataset real e embeddings reais.

Os outros testes usam dublês e provam que a *lógica* está certa. Isso não basta:
o bug que apareceu neste projeto foi de **calibração** — a lógica funcionava,
mas o modelo de embeddings pontuava perguntas fora do domínio tão alto quanto
as de dentro, e o transbordo nunca disparava. Nenhum teste com dublê pegaria
isso, e foi este arquivo que pegou.

O que dá para verificar sem chave de API (determinístico, offline):

* o retrieval recupera a entrada certa da FAQ para perguntas parafraseadas;
* a primeira barreira (limiar de similaridade) descarta o que nem chega perto.

A segunda barreira mora no LLM, então o teste que a cobre só roda quando há
``OPENAI_API_KEY`` no ambiente — caso contrário é pulado.

São lentos (carregam o modelo e indexam no ChromaDB), então ficam fora da
suíte padrão. Para rodar:

    pytest -m integracao
"""

import pytest

from app.atendimento import ServicoAtendimento
from app.config import obter_configuracoes
from app.dependencias import criar_llm
from app.handoff import MotivoTransbordo
from app.llm.demo_client import ProvedorDemo
from app.rag.generator import Generator
from app.rag.ingest import executar_ingestao
from app.rag.retriever import Retriever
from app.rag.vector_store import criar_vector_store

pytestmark = pytest.mark.integracao

pytest.importorskip("chromadb", reason="requer as dependências completas")
pytest.importorskip("sentence_transformers", reason="requer as dependências completas")

MENSAGEM_TRANSBORDO = "Vou te encaminhar para um atendente humano."

# Perguntas cobertas pela FAQ, escritas como um cliente escreveria — com
# acentuação, e nenhuma é cópia da pergunta canônica. O segundo item é a
# pergunta canônica que o retrieval deve recuperar.
#
# A acentuação não é detalhe: medindo com as mesmas perguntas sem acento, o
# MiniLM parecia acertar tudo; com acento, ele cai para 6/10. Foi assim que uma
# medição descuidada quase fixou o modelo errado no projeto.
PERGUNTAS_COBERTAS = [
    ("quando meu pedido vai chegar?", "Qual o prazo de entrega do pedido?"),
    ("como eu acompanho a entrega?", "Como faço para rastrear meu pedido?"),
    ("posso pagar com pix?", "Quais formas de pagamento vocês aceitam?"),
    ("quero trocar um produto que não serviu", "Como solicitar a troca de um produto?"),
    ("dá pra cancelar a compra?", "Como faço para cancelar meu pedido?"),
    (
        "quanto tempo demora pra devolver meu dinheiro?",
        "Em quanto tempo recebo o reembolso?",
    ),
    ("o frete é grátis?", "Qual o valor do frete?"),
    ("meu produto veio com defeito, e agora?", "Os produtos têm garantia?"),
    ("preciso da nota fiscal do que comprei", "Vocês emitem nota fiscal?"),
    ("vocês atendem no domingo?", "Qual o horário de atendimento?"),
]

# Perguntas sem nenhuma relação com a FAQ: devem morrer já na primeira barreira.
LONGE_DA_BASE = [
    "qual a capital da França?",
    "quem ganhou a copa do mundo de 2022?",
]

# Perguntas fora do domínio que a similaridade sozinha NÃO separa — elas passam
# do limiar e só a segunda barreira, no LLM, consegue barrar.
FRONTEIRA = [
    "vocês vendem passagem aérea?",
    "vocês têm vaga de emprego?",
    "me ajuda a escrever um currículo",
    "quanto é 2 + 2?",
]


@pytest.fixture(scope="module")
def indice(tmp_path_factory):
    """Indexa o dataset real em um ChromaDB temporário, uma vez por módulo."""
    config = obter_configuracoes().model_copy(
        update={"diretorio_chroma": tmp_path_factory.mktemp("chroma_integracao")}
    )
    executar_ingestao(config)
    return config, Retriever(criar_vector_store(config), top_k=config.top_k)


def _montar_servico(config, retriever, llm) -> ServicoAtendimento:
    return ServicoAtendimento(
        retriever=retriever,
        generator=Generator(llm, nome_empresa=config.nome_empresa),
        limiar_similaridade=config.limiar_similaridade,
        mensagem_transbordo=MENSAGEM_TRANSBORDO,
    )


@pytest.mark.parametrize(("pergunta", "canonica_esperada"), PERGUNTAS_COBERTAS)
def test_retrieval_recupera_a_entrada_certa_da_faq(
    indice, pergunta: str, canonica_esperada: str
) -> None:
    """Qualidade do retrieval: é a base de tudo que vem depois."""
    _, retriever = indice

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
    config, retriever = indice

    melhor = retriever.buscar(pergunta)[0].similaridade

    assert melhor >= config.limiar_similaridade, (
        f"'{pergunta}' ficou em {melhor:.3f}, abaixo do limiar "
        f"{config.limiar_similaridade:.2f}: o bot transbordaria sem necessidade"
    )


@pytest.mark.parametrize("pergunta", LONGE_DA_BASE)
def test_pergunta_distante_morre_na_primeira_barreira(indice, pergunta: str) -> None:
    config, retriever = indice
    servico = _montar_servico(config, retriever, ProvedorDemo())

    resposta = servico.responder(pergunta)

    assert resposta.transbordo, (
        f"'{pergunta}' devia transbordar, mas o bot respondeu "
        f"'{resposta.texto}' (similaridade={resposta.similaridade:.3f})"
    )
    assert resposta.motivo == MotivoTransbordo.BAIXA_SIMILARIDADE.value


def test_a_primeira_barreira_sozinha_nao_basta(indice) -> None:
    """Documenta *por que* existe a segunda barreira.

    Se algum dia a similaridade passar a separar essas perguntas sozinha, este
    teste falha — e aí a segunda barreira virou custo sem benefício.
    """
    config, retriever = indice

    passaram = [
        (p, retriever.buscar(p)[0].similaridade)
        for p in FRONTEIRA
        if retriever.buscar(p)[0].similaridade >= config.limiar_similaridade
    ]

    assert passaram, (
        "nenhuma pergunta de fronteira passou do limiar: a similaridade agora "
        "separa sozinha e a segunda barreira pode ser reavaliada"
    )


@pytest.mark.parametrize("pergunta", FRONTEIRA)
def test_segunda_barreira_barra_o_que_a_similaridade_deixou_passar(
    indice, pergunta: str
) -> None:
    """Exige LLM de verdade: é ele quem julga se o contexto responde."""
    config, retriever = indice
    if not config.openai_api_key or config.provedor_llm != "openai":
        pytest.skip("requer OPENAI_API_KEY e PROVEDOR_LLM=openai")

    servico = _montar_servico(config, retriever, criar_llm(config))
    resposta = servico.responder(pergunta)

    assert resposta.transbordo, (
        f"'{pergunta}' não é coberta pela FAQ, mas o bot respondeu "
        f"'{resposta.texto}'"
    )
    assert resposta.motivo == MotivoTransbordo.CONTEXTO_INSUFICIENTE.value


@pytest.mark.parametrize(("pergunta", "_canonica"), PERGUNTAS_COBERTAS)
def test_pergunta_coberta_e_respondida_com_llm_real(
    indice, pergunta: str, _canonica: str
) -> None:
    """A segunda barreira não pode ser conservadora demais e barrar o legítimo."""
    config, retriever = indice
    if not config.openai_api_key or config.provedor_llm != "openai":
        pytest.skip("requer OPENAI_API_KEY e PROVEDOR_LLM=openai")

    servico = _montar_servico(config, retriever, criar_llm(config))
    resposta = servico.responder(pergunta)

    assert not resposta.transbordo, (
        f"'{pergunta}' é coberta pela FAQ, mas transbordou "
        f"(motivo={resposta.motivo}, similaridade={resposta.similaridade:.3f})"
    )
