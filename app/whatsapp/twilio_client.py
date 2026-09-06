"""Implementação do canal de mensageria usando o Twilio Sandbox para WhatsApp."""

import logging
from collections.abc import Mapping

from app.whatsapp.base import MensagemRecebida, ProvedorMensageria

logger = logging.getLogger(__name__)

# O Twilio aceita no máximo 1600 caracteres por mensagem de WhatsApp.
LIMITE_CARACTERES = 1500
PREFIXO_WHATSAPP = "whatsapp:"


class ProvedorTwilio(ProvedorMensageria):
    """Envia e recebe mensagens via Twilio.

    Com ``dry_run=True`` nada é enviado de verdade: a resposta só vai para o
    log. Isso permite rodar o projeto inteiro sem credenciais configuradas.
    """

    def __init__(
        self,
        account_sid: str | None,
        auth_token: str | None,
        numero_origem: str,
        dry_run: bool = True,
    ) -> None:
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._numero_origem = self._normalizar(numero_origem)
        self._dry_run = dry_run
        self._cliente = None

    # -- API do canal ---------------------------------------------------------

    def extrair_mensagem(self, payload: Mapping[str, str]) -> MensagemRecebida:
        remetente = (payload.get("From") or "").strip()
        if not remetente:
            raise ValueError("Payload do Twilio não contém o campo 'From'.")
        return MensagemRecebida(
            remetente=remetente,
            texto=(payload.get("Body") or "").strip(),
            id_externo=payload.get("MessageSid"),
        )

    def enviar(self, destinatario: str, texto: str) -> str | None:
        destinatario = self._normalizar(destinatario)
        texto = self._truncar(texto)

        if self._dry_run:
            logger.info("[DRY-RUN] resposta para %s: %s", destinatario, texto)
            return None

        mensagem = self._twilio.messages.create(
            from_=self._numero_origem, to=destinatario, body=texto
        )
        logger.info("Mensagem %s enviada para %s", mensagem.sid, destinatario)
        return mensagem.sid

    def validar_requisicao(
        self, url: str, payload: Mapping[str, str], cabecalhos: Mapping[str, str]
    ) -> bool:
        """Confere a assinatura ``X-Twilio-Signature`` enviada pelo Twilio."""
        from twilio.request_validator import RequestValidator

        if not self._auth_token:
            logger.warning("Validação de assinatura ativa sem TWILIO_AUTH_TOKEN.")
            return False
        assinatura = cabecalhos.get("X-Twilio-Signature", "")
        return RequestValidator(self._auth_token).validate(
            url, dict(payload), assinatura
        )

    # -- internos -------------------------------------------------------------

    @property
    def _twilio(self):
        """Cliente Twilio criado sob demanda (não exige credencial em dry-run)."""
        if self._cliente is None:
            if not (self._account_sid and self._auth_token):
                raise RuntimeError(
                    "TWILIO_ACCOUNT_SID e TWILIO_AUTH_TOKEN são obrigatórios "
                    "quando TWILIO_DRY_RUN=false."
                )
            from twilio.rest import Client

            self._cliente = Client(self._account_sid, self._auth_token)
        return self._cliente

    @staticmethod
    def _normalizar(numero: str) -> str:
        numero = (numero or "").strip()
        if numero and not numero.startswith(PREFIXO_WHATSAPP):
            return f"{PREFIXO_WHATSAPP}{numero}"
        return numero

    @staticmethod
    def _truncar(texto: str) -> str:
        if len(texto) <= LIMITE_CARACTERES:
            return texto
        return texto[: LIMITE_CARACTERES - 3] + "..."
