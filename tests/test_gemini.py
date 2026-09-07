"""Testes do provedor de LLM do Gemini e do transporte HTTP compartilhado.

O transporte é testado com o ``httpx.post`` substituído; o provedor, com o
transporte substituído. Assim cada um responde por uma coisa: retry e
autenticação de um lado, formato do payload e leitura da resposta do outro.
"""

import pytest

from app import gemini_api
from app.llm.base import ErroGeracao
from app.llm.gemini_client import ProvedorGemini


class RespostaFalsa:
    def __init__(self, status_code: int, corpo: dict | None = None,
                 texto: str = "") -> None:
        self.status_code = status_code
        self._corpo = corpo or {}
        self.text = texto

    def json(self) -> dict:
        return self._corpo


def _quota(retry_delay: str | None = None) -> RespostaFalsa:
    """429 no formato real do Google, com ou sem o RetryInfo."""
    detalhes: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerMinutePerProject"}],
        }
    ]
    if retry_delay is not None:
        detalhes.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    return RespostaFalsa(
        429, {"error": {"code": 429, "message": "quota", "details": detalhes}}, "quota"
    )


@pytest.fixture(autouse=True)
def esperas(monkeypatch):
    """Registra as esperas em vez de dormir de verdade."""
    registradas: list[float] = []
    monkeypatch.setattr(gemini_api.time, "sleep", registradas.append)
    return registradas


# --- transporte --------------------------------------------------------------


def _capturar(monkeypatch, respostas: list):
    chamadas: list[dict] = []
    restantes = list(respostas)

    def post_falso(url, json, headers, timeout):
        chamadas.append({"url": url, "json": json, "headers": headers})
        resultado = restantes.pop(0)
        if isinstance(resultado, Exception):
            raise resultado
        return resultado

    import httpx

    monkeypatch.setattr(httpx, "post", post_falso)
    return chamadas


def test_transporte_autentica_pelo_cabecalho_e_nao_pela_url(monkeypatch) -> None:
    """A chave no cabeçalho não vaza em log de acesso nem em histórico."""
    chamadas = _capturar(monkeypatch, [RespostaFalsa(200, {"ok": True})])

    gemini_api.chamar("models/x:generateContent", {"a": 1}, "segredo")

    assert chamadas[0]["headers"][gemini_api.CABECALHO_CHAVE] == "segredo"
    assert "segredo" not in chamadas[0]["url"]
    assert chamadas[0]["url"].endswith("/models/x:generateContent")


def test_transporte_repete_apos_429(monkeypatch) -> None:
    """O tier gratuito estoura cota por minuto com facilidade."""
    chamadas = _capturar(
        monkeypatch,
        [RespostaFalsa(429, texto="quota"), RespostaFalsa(200, {"ok": True})],
    )

    assert gemini_api.chamar("m:x", {}, "k") == {"ok": True}
    assert len(chamadas) == 2


def test_transporte_repete_apos_falha_de_rede(monkeypatch) -> None:
    chamadas = _capturar(
        monkeypatch, [OSError("conexão caiu"), RespostaFalsa(200, {"ok": True})]
    )

    assert gemini_api.chamar("m:x", {}, "k") == {"ok": True}
    assert len(chamadas) == 2


def test_transporte_desiste_na_hora_em_erro_de_credencial(monkeypatch) -> None:
    """400/403 não melhoram com insistência — e insistir atrasaria a subida."""
    chamadas = _capturar(monkeypatch, [RespostaFalsa(403, texto="API key invalid")])

    with pytest.raises(RuntimeError, match="API key invalid"):
        gemini_api.chamar("m:x", {}, "k")

    assert len(chamadas) == 1


def test_transporte_desiste_apos_o_limite_de_tentativas(monkeypatch) -> None:
    chamadas = _capturar(
        monkeypatch, [RespostaFalsa(503, texto="indisponível")] * gemini_api.TENTATIVAS
    )

    with pytest.raises(RuntimeError, match="503"):
        gemini_api.chamar("m:x", {}, "k")

    assert len(chamadas) == gemini_api.TENTATIVAS


# --- provedor de LLM ---------------------------------------------------------


def _fixar(monkeypatch, resposta, registro: list | None = None):
    def chamar_falso(caminho, corpo, api_key, timeout=30.0):
        if registro is not None:
            registro.append({"caminho": caminho, "corpo": corpo})
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    monkeypatch.setattr("app.llm.gemini_client.chamar", chamar_falso)


def _texto(valor: str, finish: str = "STOP") -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"text": valor}]}, "finishReason": finish}
        ]
    }


def test_llm_exige_chave() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        ProvedorGemini(api_key=None)


def test_llm_devolve_o_texto_do_candidato(monkeypatch) -> None:
    _fixar(monkeypatch, _texto("  Chega em 7 dias úteis.  "))

    assert ProvedorGemini("k").gerar("sistema", "usuário") == "Chega em 7 dias úteis."


def test_llm_junta_partes_multiplas(monkeypatch) -> None:
    _fixar(
        monkeypatch,
        {"candidates": [{"content": {"parts": [{"text": "Chega em "},
                                               {"text": "7 dias."}]}}]},
    )

    assert ProvedorGemini("k").gerar("s", "u") == "Chega em 7 dias."


def test_llm_manda_instrucao_de_sistema_e_temperatura(monkeypatch) -> None:
    registro: list = []
    _fixar(monkeypatch, _texto("ok"), registro)

    ProvedorGemini("k", modelo="gemini-3.5-flash-lite", temperatura=0.3).gerar(
        "sis", "usu"
    )

    corpo = registro[0]["corpo"]
    assert registro[0]["caminho"] == "models/gemini-3.5-flash-lite:generateContent"
    assert corpo["systemInstruction"]["parts"][0]["text"] == "sis"
    assert corpo["contents"][0]["parts"][0]["text"] == "usu"
    assert corpo["generationConfig"]["temperature"] == 0.3


def test_llm_omite_thinking_config_por_padrao(monkeypatch) -> None:
    """Mandar o campo sempre quebrava: os modelos 3.5+ recusam com 400."""
    registro: list = []
    _fixar(monkeypatch, _texto("ok"), registro)

    ProvedorGemini("k").gerar("sis", "usu")

    assert "thinkingConfig" not in registro[0]["corpo"]["generationConfig"]


def test_llm_manda_thinking_config_quando_configurado(monkeypatch) -> None:
    """A alavanca continua disponível para modelos que aceitam o campo."""
    registro: list = []
    _fixar(monkeypatch, _texto("ok"), registro)

    ProvedorGemini("k", modelo="gemini-3.1-flash-lite", orcamento_raciocinio=0).gerar(
        "sis", "usu"
    )

    config = registro[0]["corpo"]["generationConfig"]
    assert config["thinkingConfig"]["thinkingBudget"] == 0


def test_llm_transborda_quando_o_prompt_e_bloqueado(monkeypatch) -> None:
    """Bloqueio de segurança vira ErroGeracao — que o serviço já transborda."""
    _fixar(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(ErroGeracao, match="SAFETY"):
        ProvedorGemini("k").gerar("s", "u")


def test_llm_recusa_resposta_vazia_informando_o_motivo(monkeypatch) -> None:
    _fixar(monkeypatch, _texto("", finish="MAX_TOKENS"))

    with pytest.raises(ErroGeracao, match="MAX_TOKENS"):
        ProvedorGemini("k").gerar("s", "u")


def test_llm_recusa_resposta_sem_candidatos(monkeypatch) -> None:
    _fixar(monkeypatch, {})

    with pytest.raises(ErroGeracao, match="candidato"):
        ProvedorGemini("k").gerar("s", "u")


def test_llm_traduz_falha_de_transporte(monkeypatch) -> None:
    _fixar(monkeypatch, RuntimeError("HTTP 503: indisponível"))

    with pytest.raises(ErroGeracao, match="indisponível"):
        ProvedorGemini("k").gerar("s", "u")


# --- backoff em 429 ----------------------------------------------------------
#
# O tier gratuito do Gemini limita por MINUTO, e cada subida do serviço no
# Render reindexa o FAQ. Um backoff de 1s+2s nunca atravessa essa janela: a
# subida falharia, e num wake da hibernação o serviço simplesmente não voltaria.
# Quem sabe o tamanho da janela é a API, no RetryInfo.


def test_espera_o_que_a_api_pediu_no_429(monkeypatch, esperas) -> None:
    _capturar(monkeypatch, [_quota("27s"), RespostaFalsa(200, {"ok": True})])

    assert gemini_api.chamar("m:x", {}, "k") == {"ok": True}
    assert esperas == [27.0]


def test_limita_a_espera_pedida_a_um_teto(monkeypatch, esperas) -> None:
    """Esperar é melhor que falhar, mas não a ponto de estourar o deploy."""
    _capturar(monkeypatch, [_quota("600s"), RespostaFalsa(200, {"ok": True})])

    gemini_api.chamar("m:x", {}, "k")

    assert esperas == [gemini_api.ESPERA_MAXIMA_SEGUNDOS]


def test_cai_no_backoff_exponencial_sem_retry_info(monkeypatch, esperas) -> None:
    _capturar(
        monkeypatch,
        [RespostaFalsa(503, texto="oops"), RespostaFalsa(503, texto="oops"),
         RespostaFalsa(200, {"ok": True})],
    )

    gemini_api.chamar("m:x", {}, "k")

    assert esperas == [1.0, 2.0]


def test_erro_de_cota_nao_e_truncado_antes_do_diagnostico(monkeypatch) -> None:
    """O `quotaId` vem depois da mensagem genérica; cortá-lo cega o log."""
    corpo = _quota("5s").json()
    longo = RespostaFalsa(429, corpo, texto=__import__("json").dumps(corpo))
    _capturar(monkeypatch, [longo] * gemini_api.TENTATIVAS)

    with pytest.raises(RuntimeError, match="GenerateRequestsPerMinutePerProject"):
        gemini_api.chamar("m:x", {}, "k")
