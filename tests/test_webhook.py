"""Testes do webhook HTTP, com serviço e canal substituídos por dublês.

O ``TestClient`` só executa o lifespan quando usado como context manager, então
aqui nada de ChromaDB ou OpenAI é carregado.
"""

import logging
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.atendimento import RespostaAtendimento
from app.config import Configuracoes
from app.main import (
    LimitePorRemetente,
    RegistroDeUpdates,
    app,
    obter_config_app,
    obter_limite,
    obter_mensageria,
    obter_servico,
    obter_updates,
)
from app.whatsapp.base import MensagemRecebida, ProvedorMensageria, pseudonimo
from app.whatsapp.twilio_client import ProvedorTwilio


class ServicoFalso:
    def __init__(self, resposta: RespostaAtendimento) -> None:
        self.resposta = resposta
        self.perguntas: list[str] = []

    def responder(self, pergunta: str) -> RespostaAtendimento:
        self.perguntas.append(pergunta)
        return self.resposta


class MensageriaFalsa(ProvedorMensageria):
    def __init__(self, valida: bool = True) -> None:
        self.enviados: list[tuple[str, str]] = []
        self.valida = valida

    def extrair_mensagem(self, payload) -> MensagemRecebida:
        return ProvedorTwilio(None, None, "whatsapp:+1").extrair_mensagem(payload)

    def enviar(self, destinatario: str, texto: str) -> str | None:
        self.enviados.append((destinatario, texto))
        return "SM123"

    def validar_requisicao(self, url, payload, cabecalhos) -> bool:
        return self.valida


@pytest.fixture
def resposta_padrao() -> RespostaAtendimento:
    return RespostaAtendimento(
        texto="Seu pedido chega em até 7 dias úteis.",
        transbordo=False,
        similaridade=0.91,
    )


@pytest.fixture
def montar_cliente(resposta_padrao):
    """Devolve (client, servico, mensageria) com as dependências trocadas."""
    criados = []

    def _montar(mensageria=None, config=None, resposta=None, limite=None):
        servico = ServicoFalso(resposta or resposta_padrao)
        mensageria = mensageria or MensageriaFalsa()
        config = config or Configuracoes()
        # Registro novo por teste (ver o comentário equivalente em
        # tests/test_telegram.py). O mesmo vale para o teto por remetente: um
        # limite compartilhado faria um teste consumir a cota do seguinte.
        updates = RegistroDeUpdates()
        limite = limite or LimitePorRemetente()
        app.dependency_overrides[obter_servico] = lambda: servico
        app.dependency_overrides[obter_mensageria] = lambda: mensageria
        app.dependency_overrides[obter_config_app] = lambda: config
        app.dependency_overrides[obter_updates] = lambda: updates
        app.dependency_overrides[obter_limite] = lambda: limite
        criados.append(True)
        return TestClient(app), servico, mensageria

    yield _montar
    app.dependency_overrides.clear()


def test_health_responde_ok(montar_cliente) -> None:
    client, _, _ = montar_cliente()

    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_webhook_responde_e_envia_pelo_canal(montar_cliente) -> None:
    client, servico, mensageria = montar_cliente()

    resposta = client.post(
        "/webhook/whatsapp",
        data={
            "From": "whatsapp:+5511999999999",
            "Body": "quando chega meu pedido?",
            "MessageSid": "SM1",
        },
    )

    assert resposta.status_code == 200
    assert "application/xml" in resposta.headers["content-type"]
    assert servico.perguntas == ["quando chega meu pedido?"]
    assert mensageria.enviados == [
        ("whatsapp:+5511999999999", "Seu pedido chega em até 7 dias úteis.")
    ]


def test_webhook_envia_a_mensagem_de_transbordo(montar_cliente) -> None:
    transbordo = RespostaAtendimento(
        texto="Vou te encaminhar para um atendente.",
        transbordo=True,
        similaridade=0.12,
        motivo="baixa_similaridade",
    )
    client, _, mensageria = montar_cliente(resposta=transbordo)

    client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+5511999999999", "Body": "vocês têm vaga?"},
    )

    assert mensageria.enviados[0][1] == "Vou te encaminhar para um atendente."


def test_webhook_rejeita_payload_sem_remetente(montar_cliente) -> None:
    client, _, _ = montar_cliente()

    resposta = client.post("/webhook/whatsapp", data={"Body": "oi"})

    assert resposta.status_code == 400


def test_webhook_recusa_assinatura_invalida(montar_cliente) -> None:
    config = Configuracoes(twilio_validar_assinatura=True)
    client, _, mensageria = montar_cliente(
        mensageria=MensageriaFalsa(valida=False), config=config
    )

    resposta = client.post(
        "/webhook/whatsapp", data={"From": "whatsapp:+55119", "Body": "oi"}
    )

    assert resposta.status_code == 403
    assert mensageria.enviados == []


def test_endpoint_de_teste_devolve_diagnostico(montar_cliente) -> None:
    config = Configuracoes(expor_ferramentas_de_teste=True)
    client, _, _ = montar_cliente(config=config)

    resposta = client.post("/api/mensagem", json={"pergunta": "quando chega?"})

    assert resposta.status_code == 200
    assert resposta.json() == {
        "resposta": "Seu pedido chega em até 7 dias úteis.",
        "transbordo": False,
        "similaridade": 0.91,
        "motivo": None,
    }


def test_endpoint_de_teste_nao_existe_por_padrao(montar_cliente) -> None:
    """Sem EXPOR_FERRAMENTAS_DE_TESTE ele some: é RAG sem assinatura nem segredo."""
    client, servico, _ = montar_cliente()

    resposta = client.post("/api/mensagem", json={"pergunta": "quando chega?"})

    assert resposta.status_code == 404
    assert servico.perguntas == []


# --- Isolamento entre os canais ----------------------------------------------
#
# As duas rotas de webhook são registradas sempre, mas cada uma valida com o
# guard do seu canal enquanto todas usam o `mensageria` do canal configurado.
# Sem `_exigir_canal`, a rota do canal inativo é um desvio em volta do guard do
# canal ativo. Os dois testes abaixo fixam as duas metades disso.


def test_rota_do_telegram_nao_contorna_a_assinatura_do_twilio(
    montar_cliente,
) -> None:
    """Rodando como Twilio, o payload do Twilio não passa pela rota do Telegram.

    É o desvio real: com CANAL=twilio não existe TELEGRAM_SEGREDO_WEBHOOK para
    conferir, então a rota do Telegram aceitava o POST sem validar nada e o
    entregava ao extrator do Twilio, que só precisa de 'From' e 'Body' — um
    envio real disparado por quem não sabe assinar requisição nenhuma.
    """
    config = Configuracoes(canal="twilio", twilio_validar_assinatura=True)
    client, servico, mensageria = montar_cliente(
        mensageria=MensageriaFalsa(valida=False), config=config
    )

    # O mesmo payload que a rota do Twilio recusa com 403.
    resposta = client.post(
        "/webhook/telegram",
        json={
            "From": "whatsapp:+5511999999999",
            "Body": "quanto custa o frete?",
            "MessageSid": "SM1",
        },
    )

    assert resposta.status_code == 404
    assert servico.perguntas == []
    assert mensageria.enviados == []


def test_rota_do_twilio_nao_existe_quando_o_canal_e_telegram(
    montar_cliente,
) -> None:
    config = Configuracoes(canal="telegram", telegram_segredo_webhook="s3gr3d0")
    client, servico, mensageria = montar_cliente(config=config)

    resposta = client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+5511999999999", "Body": "oi"},
    )

    assert resposta.status_code == 404
    assert servico.perguntas == []
    assert mensageria.enviados == []


# --- Autenticação de webhook obrigatória ao enviar de verdade ----------------


def test_telegram_sem_segredo_nao_sobe_fora_de_dry_run() -> None:
    """O caso do deploy: `sync: false` no render.yaml deixa o campo em branco.

    Sem esta checagem a app subia limpa e servia o webhook para qualquer um —
    a rota pula a validação quando o segredo é vazio, então nem aviso no log
    sobrava.
    """
    with pytest.raises(ValidationError, match="TELEGRAM_SEGREDO_WEBHOOK"):
        Configuracoes(
            canal="telegram", telegram_dry_run=False, telegram_segredo_webhook=None
        )


def test_twilio_sem_assinatura_nao_sobe_fora_de_dry_run() -> None:
    with pytest.raises(ValidationError, match="TWILIO_VALIDAR_ASSINATURA"):
        Configuracoes(
            canal="twilio", twilio_dry_run=False, twilio_validar_assinatura=False
        )


def test_canal_que_envia_de_verdade_sobe_com_o_webhook_autenticado() -> None:
    """O contrapeso: a exigência não atrapalha a configuração correta."""
    telegram = Configuracoes(
        canal="telegram", telegram_dry_run=False, telegram_segredo_webhook="s3gr3d0"
    )
    twilio = Configuracoes(
        canal="twilio", twilio_dry_run=False, twilio_validar_assinatura=True
    )

    assert telegram.telegram_segredo_webhook == "s3gr3d0"
    assert twilio.twilio_validar_assinatura is True


def test_dry_run_nao_exige_segredo() -> None:
    """Rodar local sem credencial continua funcionando: em dry-run nada sai."""
    assert Configuracoes(canal="telegram", telegram_dry_run=True).canal == "telegram"
    assert Configuracoes(canal="twilio", twilio_dry_run=True).canal == "twilio"


# --- Teto de mensagens por remetente -----------------------------------------


def test_limite_descarta_o_excesso_do_mesmo_remetente(montar_cliente) -> None:
    """Cada mensagem custa duas chamadas de API; o teto é o que segura o volume."""
    client, servico, mensageria = montar_cliente(
        limite=LimitePorRemetente(maximo=3, janela_segundos=60.0)
    )

    for indice in range(10):
        client.post(
            "/webhook/whatsapp",
            data={
                "From": "whatsapp:+5511999999999",
                "Body": f"pergunta {indice}",
                "MessageSid": f"SM{indice}",
            },
        )

    assert len(servico.perguntas) == 3
    assert len(mensageria.enviados) == 3


def test_limite_nao_contamina_outro_remetente(montar_cliente) -> None:
    """O teto é por remetente: quem abusa não derruba o atendimento dos outros."""
    client, servico, _ = montar_cliente(
        limite=LimitePorRemetente(maximo=2, janela_segundos=60.0)
    )

    for indice in range(5):
        client.post(
            "/webhook/whatsapp",
            data={
                "From": "whatsapp:+5511000000000",
                "Body": f"spam {indice}",
                "MessageSid": f"SMa{indice}",
            },
        )
    client.post(
        "/webhook/whatsapp",
        data={
            "From": "whatsapp:+5511999999999",
            "Body": "cliente legítimo",
            "MessageSid": "SMb1",
        },
    )

    assert servico.perguntas == ["spam 0", "spam 1", "cliente legítimo"]


def test_limite_confirma_o_webhook_mesmo_descartando(montar_cliente) -> None:
    """O canal não pode saber do teto: um erro faria o Telegram reenviar em loop."""
    client, _, _ = montar_cliente(limite=LimitePorRemetente(maximo=1))

    client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+55119", "Body": "1", "MessageSid": "SM1"},
    )
    resposta = client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+55119", "Body": "2", "MessageSid": "SM2"},
    )

    assert resposta.status_code == 200


def test_limite_libera_ao_passar_a_janela() -> None:
    limite = LimitePorRemetente(maximo=2, janela_segundos=0.05)

    assert [limite.permitir("a") for _ in range(3)] == [True, True, False]
    time.sleep(0.06)
    assert limite.permitir("a") is True


def test_limite_zero_desliga_o_teto() -> None:
    limite = LimitePorRemetente(maximo=0)

    assert all(limite.permitir("a") for _ in range(50))


def test_limite_esquece_remetentes_antigos_sem_crescer_sem_fim() -> None:
    """Sem teto de capacidade, remetentes sempre novos viravam outro vetor de abuso."""
    limite = LimitePorRemetente(maximo=5, capacidade=3)

    for indice in range(10):
        limite.permitir(f"remetente-{indice}")

    assert limite.remetentes_rastreados <= 3


def test_reentrega_nao_gasta_a_cota_do_remetente(montar_cliente) -> None:
    """A rajada de reentregas é culpa do canal, não do cliente."""
    client, servico, _ = montar_cliente(limite=LimitePorRemetente(maximo=2))

    for _ in range(5):
        client.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+55119", "Body": "oi", "MessageSid": "SM_igual"},
        )
    client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+55119", "Body": "outra", "MessageSid": "SM_nova"},
    )

    # A reentrega foi descartada antes do teto, então a segunda vaga sobrou.
    assert servico.perguntas == ["oi", "outra"]


# --- Dado pessoal fora do log ------------------------------------------------


def test_log_nao_registra_o_texto_nem_o_remetente(montar_cliente, caplog) -> None:
    """Os logs do Render alcançam todo o workspace; não são lugar para PII."""
    client, _, _ = montar_cliente()

    with caplog.at_level(logging.INFO, logger="app.main"):
        client.post(
            "/webhook/whatsapp",
            data={
                "From": "whatsapp:+5511987654321",
                "Body": "meu CPF é 123.456.789-00",
                "MessageSid": "SM1",
            },
        )

    registrado = caplog.text
    assert "123.456.789-00" not in registrado
    assert "+5511987654321" not in registrado
    # O que sobra é o suficiente para seguir a conversa: apelido e tamanho.
    assert pseudonimo("whatsapp:+5511987654321") in registrado
    assert "24 caracteres" in registrado


def test_pseudonimo_e_estavel_e_distingue_remetentes() -> None:
    assert pseudonimo("whatsapp:+55119") == pseudonimo("whatsapp:+55119")
    assert pseudonimo("whatsapp:+55119") != pseudonimo("whatsapp:+55118")
    assert pseudonimo(None) == "?"
    assert "+55119" not in pseudonimo("whatsapp:+55119")
