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

    # --- RAG -----------------------------------------------------------------
    caminho_faq: Path = RAIZ / "app" / "data" / "faq_dataset.json"
    diretorio_chroma: Path = RAIZ / "chroma_db"
    colecao_chroma: str = "faq"
    # Modelo local de embeddings. O e5-large tem a melhor qualidade em
    # português, mas baixa ~2,2 GB no primeiro uso. Alternativa bem mais leve:
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (~470 MB).
    modelo_embedding: str = "intfloat/multilingual-e5-large"
    top_k: int = 3
    # Similaridade mínima (0 a 1) para o bot responder sozinho.
    # Abaixo disso, transborda para atendimento humano.
    limiar_similaridade: float = 0.60

    # --- LLM -----------------------------------------------------------------
    provedor_llm: str = "openai"  # "openai" | "demo"
    openai_api_key: str | None = None
    openai_modelo: str = "gpt-4o-mini"
    openai_temperatura: float = 0.3

    # --- WhatsApp / Twilio ---------------------------------------------------
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_numero_origem: str = "whatsapp:+14155238886"  # número do Sandbox
    # Em dry-run o bot não chama a API do Twilio: apenas loga o que enviaria.
    twilio_dry_run: bool = True
    twilio_validar_assinatura: bool = False
    url_publica: str | None = None  # usada na validação de assinatura


@lru_cache
def obter_configuracoes() -> Configuracoes:
    """Instância única de configuração (cacheada)."""
    return Configuracoes()
