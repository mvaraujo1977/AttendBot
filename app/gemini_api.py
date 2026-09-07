"""Transporte HTTP compartilhado pelos dois provedores Gemini (LLM e embedding).

Existe para não duplicar autenticação, timeout e retry em dois arquivos. É
deliberadamente uma função só, sobre o ``httpx`` que o projeto já usa para a
Bot API do Telegram: o SDK oficial do Google arrastaria ``grpcio`` e
``protobuf`` para uma imagem que precisa caber em 512 MB, para chamar dois
endpoints REST.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
CABECALHO_CHAVE = "x-goog-api-key"

# Códigos que valem uma nova tentativa: cota por minuto estourada (429) e
# indisponibilidade momentânea (5xx). O tier gratuito estoura 429 com
# facilidade, e no startup do Render isso seria uma falha de subida.
STATUS_RETENTAVEIS = frozenset({429, 500, 502, 503, 504})
TENTATIVAS = 3
ESPERA_BASE_SEGUNDOS = 1.0
# Teto para a espera que a API pede em 429. O tier gratuito limita por minuto,
# então o `retryDelay` costuma vir na casa das dezenas de segundos — e vale a
# pena esperar: na subida do Render, desistir significa deploy falho, e num
# wake da hibernação significa o serviço não voltar.
ESPERA_MAXIMA_SEGUNDOS = 45.0


def chamar(
    caminho: str,
    corpo: dict[str, Any],
    api_key: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST autenticado na API do Gemini. Devolve o JSON já decodificado.

    ``caminho`` é a parte depois da versão, por exemplo
    ``models/gemini-embedding-001:batchEmbedContents``.

    Levanta ``RuntimeError`` em qualquer falha; quem chama traduz para o erro
    da sua camada (``ErroGeracao`` ou ``ErroEmbedding``), porque é isso que o
    ``ServicoAtendimento`` sabe transbordar.
    """
    import httpx

    url = f"{API_BASE}/{caminho}"
    ultima_falha = ""

    for tentativa in range(1, TENTATIVAS + 1):
        pedida: float | None = None
        try:
            resposta = httpx.post(
                url,
                json=corpo,
                headers={
                    CABECALHO_CHAVE: api_key,
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
        except Exception as erro:  # noqa: BLE001 - rede: sempre externo
            ultima_falha = f"falha de rede: {erro}"
        else:
            if resposta.status_code < 400:
                return resposta.json()
            # A mensagem útil do Google vem no corpo, inclusive nos 4xx.
            ultima_falha = f"HTTP {resposta.status_code}: {_resumir(resposta.text)}"
            pedida = _espera_pedida(resposta)
            if resposta.status_code not in STATUS_RETENTAVEIS:
                break

        if tentativa < TENTATIVAS:
            # Backoff exponencial só quando a API não disse quanto esperar. Ela
            # sabe melhor: contra uma cota por minuto, 1s e 2s nunca bastam.
            espera = min(
                pedida
                if pedida is not None
                else ESPERA_BASE_SEGUNDOS * (2 ** (tentativa - 1)),
                ESPERA_MAXIMA_SEGUNDOS,
            )
            logger.warning(
                "Gemini falhou (%s). Tentativa %d/%d em %.0fs.",
                ultima_falha,
                tentativa + 1,
                TENTATIVAS,
                espera,
            )
            time.sleep(espera)

    raise RuntimeError(ultima_falha)


def _espera_pedida(resposta) -> float | None:
    """Segundos que a própria API mandou esperar, se ela disse.

    Nos 429 o Google devolve um ``RetryInfo`` com ``retryDelay`` (ex.: "27s").
    Respeitá-lo é a diferença entre voltar do rate limit e desistir: o valor
    reflete a janela real da cota, que o nosso backoff não tem como adivinhar.
    """
    try:
        detalhes = resposta.json()["error"]["details"]
    except Exception:  # noqa: BLE001 - corpo de erro é sempre imprevisível
        return None

    for detalhe in detalhes:
        if not str(detalhe.get("@type", "")).endswith("RetryInfo"):
            continue
        bruto = str(detalhe.get("retryDelay", "")).strip()
        if bruto.endswith("s"):
            try:
                return float(bruto[:-1])
            except ValueError:
                return None
    return None


def _resumir(texto: str, limite: int = 600) -> str:
    """Corta o corpo do erro: o Google devolve JSON longo em algumas falhas.

    O limite é generoso de propósito: num 429 a parte que diz *qual* cota
    estourou vem depois da mensagem genérica, e cortá-la deixa o log inútil
    justamente no erro que mais precisa de diagnóstico.
    """
    texto = " ".join(texto.split())
    return texto if len(texto) <= limite else texto[:limite] + "..."
