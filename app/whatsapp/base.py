"""Contrato do canal de mensageria.

Toda a lógica de negócio conversa apenas com esta interface. Trocar o Twilio
pela API oficial do WhatsApp Business (ou por Telegram) significa escrever uma
nova implementação aqui, sem tocar no RAG nem no webhook.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MensagemRecebida:
    """Mensagem do cliente, já normalizada e independente do canal."""

    remetente: str
    texto: str
    id_externo: str | None = None


class ProvedorMensageria(ABC):
    """Interface de entrada e saída de mensagens."""

    @abstractmethod
    def extrair_mensagem(self, payload: Mapping[str, Any]) -> MensagemRecebida:
        """Converte o payload bruto do webhook em uma ``MensagemRecebida``.

        Deve levantar ``ValueError`` se o payload não tiver os campos mínimos.
        """

    @abstractmethod
    def enviar(self, destinatario: str, texto: str) -> str | None:
        """Envia o texto ao destinatário e devolve o id da mensagem no canal.

        Retorna ``None`` quando o envio real não aconteceu (modo dry-run).
        """

    def validar_requisicao(
        self, url: str, payload: Mapping[str, Any], cabecalhos: Mapping[str, str]
    ) -> bool:
        """Valida a autenticidade do webhook. Por padrão, aceita tudo."""
        return True
