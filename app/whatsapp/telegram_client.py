"""Implementação do canal de mensageria usando a Bot API do Telegram.

Existe por um motivo prático: o webhook do WhatsApp exige conta Twilio paga
(ver README), enquanto o Telegram entrega bot, token e webhook de graça. Como
toda a lógica de negócio conversa só com ``ProvedorMensageria``, trocar de
canal não toca no RAG nem no serviço de atendimento.
"""

import hmac
import logging
from collections.abc import Mapping
from typing import Any

from app.whatsapp.base import MensagemRecebida, ProvedorMensageria, pseudonimo

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
# O Telegram aceita no máximo 4096 caracteres por mensagem.
LIMITE_CARACTERES = 4000
TIMEOUT_SEGUNDOS = 10.0
# Cabeçalho que o Telegram repete em todo update quando o webhook é registrado
# com `secret_token`. É o análogo da assinatura do Twilio.
CABECALHO_SEGREDO = "X-Telegram-Bot-Api-Secret-Token"


class ProvedorTelegram(ProvedorMensageria):
    """Envia e recebe mensagens via Bot API do Telegram.

    Com ``dry_run=True`` nada é enviado de verdade: a resposta só vai para o
    log. Isso permite rodar o projeto inteiro sem credenciais configuradas.
    """

    def __init__(
        self,
        token: str | None,
        dry_run: bool = True,
        segredo_webhook: str | None = None,
    ) -> None:
        self._token = token
        self._dry_run = dry_run
        self._segredo_webhook = segredo_webhook

    # -- API do canal ---------------------------------------------------------

    def extrair_mensagem(self, payload: Mapping[str, Any]) -> MensagemRecebida:
        """Converte um Update do Telegram em ``MensagemRecebida``.

        O Telegram manda vários tipos de update pelo mesmo webhook (edições,
        entradas em grupo, callbacks de botão). Só mensagens interessam aqui;
        o resto vira ``ValueError`` e é descartado pela rota.
        """
        mensagem = payload.get("message") or payload.get("edited_message")
        if not isinstance(mensagem, Mapping):
            raise ValueError("Update do Telegram não contém 'message'.")

        chat = mensagem.get("chat")
        if not isinstance(chat, Mapping) or chat.get("id") is None:
            raise ValueError("Update do Telegram não contém 'message.chat.id'.")

        id_mensagem = mensagem.get("message_id")
        return MensagemRecebida(
            # O chat_id é o endereço de resposta — é ele que volta em `enviar`.
            remetente=str(chat["id"]),
            texto=str(mensagem.get("text") or "").strip(),
            id_externo=str(id_mensagem) if id_mensagem is not None else None,
        )

    def enviar(self, destinatario: str, texto: str) -> str | None:
        texto = self._truncar(texto)

        if self._dry_run:
            logger.info(
                "[DRY-RUN] resposta para %s: %s", pseudonimo(destinatario), texto
            )
            return None

        if not self._token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN é obrigatório quando TELEGRAM_DRY_RUN=false."
            )

        import httpx

        resposta = httpx.post(
            f"{API_BASE}/bot{self._token}/sendMessage",
            json={"chat_id": destinatario, "text": texto},
            timeout=TIMEOUT_SEGUNDOS,
        )
        # O Telegram devolve o motivo em `description`, inclusive nos 4xx — por
        # isso lemos o corpo em vez de usar `raise_for_status`.
        if resposta.status_code >= 400:
            raise RuntimeError(
                f"Telegram recusou o envio ({resposta.status_code}): {resposta.text}"
            )

        corpo = resposta.json()
        if not corpo.get("ok"):
            raise RuntimeError(
                f"Telegram recusou o envio: {corpo.get('description')}"
            )

        id_mensagem = str(corpo["result"]["message_id"])
        logger.info(
            "Mensagem %s enviada para %s", id_mensagem, pseudonimo(destinatario)
        )
        return id_mensagem

    def validar_requisicao(
        self, url: str, payload: Mapping[str, Any], cabecalhos: Mapping[str, str]
    ) -> bool:
        """Confere o ``X-Telegram-Bot-Api-Secret-Token`` enviado pelo Telegram.

        Diferente do Twilio, o Telegram não assina a URL: ele apenas repete o
        segredo combinado no `setWebhook`. Por isso `url` não é usada.
        """
        if not self._segredo_webhook:
            logger.warning("Validação ativa sem TELEGRAM_SEGREDO_WEBHOOK.")
            return False
        return hmac.compare_digest(
            cabecalhos.get(CABECALHO_SEGREDO, ""), self._segredo_webhook
        )

    # -- internos -------------------------------------------------------------

    @staticmethod
    def _truncar(texto: str) -> str:
        if len(texto) <= LIMITE_CARACTERES:
            return texto
        return texto[: LIMITE_CARACTERES - 3] + "..."
