"""Configurações da aplicação, carregadas de variáveis de ambiente ou do .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent.parent


class Configuracoes(BaseSettings):
    """Todos os parâmetros ajustáveis do bot em um único lugar.

    O nome da variável de ambiente é o nome do campo em maiúsculas
    (ex.: ``limiar_similaridade`` -> ``LIMIAR_SIMILARIDADE``).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Identidade do bot ---------------------------------------------------
    nome_empresa: str = "Loja Exemplo"
    mensagem_transbordo: str = (
        "Não encontrei essa informação por aqui, mas já estou encaminhando você "
        "para um de nossos atendentes. Em instantes alguém fala com você."
    )

    # Vazio usa o texto padrão de app/atendimento.py (MENSAGEM_BOAS_VINDAS).
    mensagem_boas_vindas: str | None = None

    # --- RAG -----------------------------------------------------------------
    caminho_faq: Path = RAIZ / "app" / "data" / "faq_dataset.json"
    diretorio_chroma: Path = RAIZ / "chroma_db"
    colecao_chroma: str = "faq"
    # Modelo local de embeddings (~2,2 GB no primeiro uso). Acerta 10/10 no
    # retrieval do dataset de exemplo; o MiniLM multilíngue, bem mais leve,
    # acerta 6/10 em português acentuado. Ver a seção de calibração no README.
    modelo_embedding: str = "intfloat/multilingual-e5-large"
    top_k: int = 3
    # Primeira das duas barreiras de transbordo: descarta o que nem chega perto
    # da base. Fica deliberadamente baixo porque o e5 comprime as similaridades
    # numa faixa estreita — quem decide os casos de fronteira é a segunda
    # barreira, no LLM (ver app/rag/generator.py). Trocar de modelo de
    # embedding exige recalibrar este valor.
    limiar_similaridade: float = 0.80

    # --- LLM -----------------------------------------------------------------
    provedor_llm: str = "openai"  # "openai" | "demo"
    openai_api_key: str | None = None
    openai_modelo: str = "gpt-4o-mini"
    openai_temperatura: float = 0.3

    # --- Canal de mensageria -------------------------------------------------
    # "twilio" = WhatsApp (exige conta paga para configurar o webhook)
    # "telegram" = canal gratuito, usado na demonstração. Ver README.
    canal: str = "twilio"

    # --- WhatsApp / Twilio ---------------------------------------------------
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    # Placeholder: o número do Sandbox varia por conta, então este default
    # não serve para enviar nada. Defina TWILIO_NUMERO_ORIGEM no .env com o
    # número que o console do Twilio mostrou para a sua conta.
    twilio_numero_origem: str = "whatsapp:+15550000000"
    # Em dry-run o bot não chama a API do Twilio: apenas loga o que enviaria.
    twilio_dry_run: bool = True
    twilio_validar_assinatura: bool = False
    url_publica: str | None = None  # usada na validação de assinatura

    # --- Telegram ------------------------------------------------------------
    telegram_bot_token: str | None = None
    # Em dry-run o bot não chama a API do Telegram: apenas loga o que enviaria.
    telegram_dry_run: bool = True
    # Segredo combinado no setWebhook. Vazio desliga a validação do webhook.
    telegram_segredo_webhook: str | None = None


@lru_cache
def obter_configuracoes() -> Configuracoes:
    """Instância única de configuração (cacheada)."""
    return Configuracoes()
