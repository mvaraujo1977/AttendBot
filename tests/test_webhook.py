"""Testes do webhook HTTP, com serviço e canal substituídos por dublês.

O ``TestClient`` só executa o lifespan quando usado como context manager, então
aqui nada de ChromaDB ou OpenAI é carregado.
"""

import pytest
from fastapi.testclient import TestClient

from app.atendimento import RespostaAtendimento
from app.config import Configuracoes
from app.main import app, obter_config_app, obter_mensageria, obter_servico
from app.whatsapp.base import MensagemRecebida, ProvedorMensageria
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

    def _montar(mensageria=None, config=None, resposta=None):
        servico = ServicoFalso(resposta or resposta_padrao)
        mensageria = mensageria or MensageriaFalsa()
        config = config or Configuracoes()
        app.dependency_overrides[obter_servico] = lambda: servico
        app.dependency_overrides[obter_mensageria] = lambda: mensageria
        app.dependency_overrides[obter_config_app] = lambda: config
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
    client, _, _ = montar_cliente()

    resposta = client.post("/api/mensagem", json={"pergunta": "quando chega?"})

    assert resposta.status_code == 200
    assert resposta.json() == {
        "resposta": "Seu pedido chega em até 7 dias úteis.",
        "transbordo": False,
        "similaridade": 0.91,
        "motivo": None,
    }
