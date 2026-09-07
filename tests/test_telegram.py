"""Testes do canal Telegram: parsing do update, envio e a rota do webhook."""

import pytest
from fastapi.testclient import TestClient

from app.atendimento import RespostaAtendimento
from app.config import Configuracoes
from app.main import (
    RegistroDeUpdates,
    app,
    obter_config_app,
    obter_mensageria,
    obter_servico,
    obter_updates,
)
from app.whatsapp.telegram_client import CABECALHO_SEGREDO, ProvedorTelegram


def update(texto: str = "quando chega meu pedido?", chat_id: int = 42) -> dict:
    """Update mínimo no formato que o Telegram entrega no webhook."""
    return {
        "update_id": 1,
        "message": {
            "message_id": 7,
            "chat": {"id": chat_id, "type": "private"},
            "text": texto,
        },
    }


class ServicoFalso:
    def __init__(self, resposta: RespostaAtendimento) -> None:
        self.resposta = resposta
        self.perguntas: list[str] = []

    def responder(self, pergunta: str) -> RespostaAtendimento:
        self.perguntas.append(pergunta)
        return self.resposta


class TelegramEspiao(ProvedorTelegram):
    """Provedor real (parsing e validação de verdade) com o envio capturado."""

    def __init__(self, **kwargs) -> None:
        super().__init__(token=None, **kwargs)
        self.enviados: list[tuple[str, str]] = []

    def enviar(self, destinatario: str, texto: str) -> str | None:
        self.enviados.append((destinatario, texto))
        return "7"


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

    def _montar(mensageria=None, config=None):
        servico = ServicoFalso(resposta_padrao)
        mensageria = mensageria or TelegramEspiao()
        # Explícito para o teste não herdar o .env do desenvolvedor: com um
        # segredo configurado lá, a rota passaria a exigir o cabeçalho.
        config = config or Configuracoes(
            canal="telegram", telegram_segredo_webhook=None
        )
        # Registro novo por teste: `app` é um singleton de módulo, e todos os
        # updates daqui usam o mesmo message_id — sem isto o segundo teste
        # seria descartado como reentrega.
        updates = RegistroDeUpdates()
        app.dependency_overrides[obter_servico] = lambda: servico
        app.dependency_overrides[obter_mensageria] = lambda: mensageria
        app.dependency_overrides[obter_config_app] = lambda: config
        app.dependency_overrides[obter_updates] = lambda: updates
        return TestClient(app), servico, mensageria

    yield _montar
    app.dependency_overrides.clear()


# --- Parsing do update -------------------------------------------------------


def test_extrai_chat_id_texto_e_id_da_mensagem() -> None:
    mensagem = ProvedorTelegram(token=None).extrair_mensagem(update())

    assert mensagem.remetente == "42"
    assert mensagem.texto == "quando chega meu pedido?"
    assert mensagem.id_externo == "7"


def test_aceita_mensagem_editada() -> None:
    bruto = {"update_id": 2, "edited_message": update()["message"]}

    mensagem = ProvedorTelegram(token=None).extrair_mensagem(bruto)

    assert mensagem.remetente == "42"


def test_update_sem_mensagem_e_recusado() -> None:
    with pytest.raises(ValueError):
        ProvedorTelegram(token=None).extrair_mensagem({"update_id": 3})


def test_update_sem_chat_id_e_recusado() -> None:
    with pytest.raises(ValueError):
        ProvedorTelegram(token=None).extrair_mensagem(
            {"message": {"message_id": 7, "text": "oi"}}
        )


def test_mensagem_sem_texto_vira_string_vazia() -> None:
    bruto = {"message": {"message_id": 7, "chat": {"id": 42}}}

    assert ProvedorTelegram(token=None).extrair_mensagem(bruto).texto == ""


# --- Envio -------------------------------------------------------------------


def test_dry_run_nao_envia_e_devolve_none() -> None:
    assert ProvedorTelegram(token="t", dry_run=True).enviar("42", "oi") is None


def test_envio_real_chama_a_bot_api(monkeypatch) -> None:
    import httpx

    chamadas = []

    class RespostaFalsa:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"ok": True, "result": {"message_id": 99}}

    def post_falso(url, json, timeout):
        chamadas.append((url, json))
        return RespostaFalsa()

    monkeypatch.setattr(httpx, "post", post_falso)

    id_msg = ProvedorTelegram(token="123:ABC", dry_run=False).enviar("42", "olá")

    assert id_msg == "99"
    url, corpo = chamadas[0]
    assert url.endswith("/bot123:ABC/sendMessage")
    assert corpo == {"chat_id": "42", "text": "olá"}


def test_envio_sem_token_falha_com_mensagem_util() -> None:
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        ProvedorTelegram(token=None, dry_run=False).enviar("42", "oi")


def test_erro_da_bot_api_vira_runtime_error(monkeypatch) -> None:
    import httpx

    class RespostaFalsa:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"ok": False, "description": "chat not found"}

    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: RespostaFalsa())

    with pytest.raises(RuntimeError, match="chat not found"):
        ProvedorTelegram(token="t", dry_run=False).enviar("42", "oi")


def test_trunca_texto_acima_do_limite_do_telegram() -> None:
    truncado = ProvedorTelegram._truncar("a" * 5000)

    assert len(truncado) == 4000
    assert truncado.endswith("...")


# --- Validação do webhook ----------------------------------------------------


def test_segredo_correto_valida() -> None:
    provedor = ProvedorTelegram(token=None, segredo_webhook="s3gr3d0")

    assert provedor.validar_requisicao("", {}, {CABECALHO_SEGREDO: "s3gr3d0"})


def test_segredo_errado_nao_valida() -> None:
    provedor = ProvedorTelegram(token=None, segredo_webhook="s3gr3d0")

    assert not provedor.validar_requisicao("", {}, {CABECALHO_SEGREDO: "outro"})


def test_sem_segredo_configurado_nao_valida() -> None:
    assert not ProvedorTelegram(token=None).validar_requisicao("", {}, {})


# --- Rota --------------------------------------------------------------------


def test_webhook_responde_e_envia_pelo_canal(montar_cliente) -> None:
    client, servico, mensageria = montar_cliente()

    resposta = client.post("/webhook/telegram", json=update())

    assert resposta.status_code == 200
    assert resposta.json() == {"ok": True}
    assert servico.perguntas == ["quando chega meu pedido?"]
    assert mensageria.enviados == [("42", "Seu pedido chega em até 7 dias úteis.")]


def test_webhook_confirma_e_ignora_update_que_nao_e_mensagem(montar_cliente) -> None:
    client, servico, mensageria = montar_cliente()

    resposta = client.post("/webhook/telegram", json={"update_id": 9})

    # 200 de propósito: qualquer outra coisa faz o Telegram reenviar o update.
    assert resposta.status_code == 200
    assert resposta.json() == {"ok": True, "ignorado": True}
    assert servico.perguntas == []
    assert mensageria.enviados == []


def test_webhook_recusa_segredo_invalido(montar_cliente) -> None:
    config = Configuracoes(canal="telegram", telegram_segredo_webhook="s3gr3d0")
    client, _, mensageria = montar_cliente(
        mensageria=TelegramEspiao(segredo_webhook="s3gr3d0"), config=config
    )

    resposta = client.post(
        "/webhook/telegram",
        json=update(),
        headers={CABECALHO_SEGREDO: "errado"},
    )

    assert resposta.status_code == 403
    assert mensageria.enviados == []


def test_webhook_aceita_segredo_valido(montar_cliente) -> None:
    config = Configuracoes(canal="telegram", telegram_segredo_webhook="s3gr3d0")
    client, _, mensageria = montar_cliente(
        mensageria=TelegramEspiao(segredo_webhook="s3gr3d0"), config=config
    )

    resposta = client.post(
        "/webhook/telegram",
        json=update(),
        headers={CABECALHO_SEGREDO: "s3gr3d0"},
    )

    assert resposta.status_code == 200
    assert len(mensageria.enviados) == 1


def test_webhook_rejeita_corpo_que_nao_e_json(montar_cliente) -> None:
    client, _, _ = montar_cliente()

    resposta = client.post(
        "/webhook/telegram",
        content=b"nao sou json",
        headers={"content-type": "application/json"},
    )

    assert resposta.status_code == 400


def test_reentrega_do_mesmo_update_nao_responde_duas_vezes(montar_cliente) -> None:
    """O webhook confirma antes de processar, então o canal pode reenviar.

    É o que acontece ao acordar de uma hibernação: o Telegram já tentou
    entregar o update algumas vezes enquanto a instância subia, e todas as
    cópias chegam juntas. Responder a cada uma custaria uma chamada de LLM e
    entregaria a mesma mensagem várias vezes ao cliente.
    """
    client, servico, mensageria = montar_cliente()

    for _ in range(3):
        assert client.post("/webhook/telegram", json=update()).status_code == 200

    assert len(servico.perguntas) == 1
    assert len(mensageria.enviados) == 1


def test_mesma_message_id_em_conversas_diferentes_e_processada(
    montar_cliente,
) -> None:
    """O message_id do Telegram é sequencial por conversa, não global.

    Sem incluir o chat na chave, a primeira mensagem de um cliente descartaria
    a de outro — um bug bem pior do que o que o dedup resolve.
    """
    client, servico, _ = montar_cliente()

    client.post("/webhook/telegram", json=update(texto="oi", chat_id=42))
    client.post("/webhook/telegram", json=update(texto="olá", chat_id=99))

    assert servico.perguntas == ["oi", "olá"]
