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
    # Modelo local de embeddings (~470 MB no primeiro uso).
    #
    # O intfloat/multilingual-e5-large ranqueia um pouco melhor, mas NÃO serve
    # aqui: ele comprime todas as similaridades numa faixa estreita (~0.73 a
    # 0.90), então perguntas fora da base pontuam tão alto quanto as de dentro
    # e nenhum limiar consegue separá-las. Como o transbordo depende do valor
    # absoluto da similaridade, o MiniLM — que espalha de ~0.04 a ~0.84 — é a
    # escolha certa. Ver a seção de calibração no README.
    modelo_embedding: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 3
    # Similaridade mínima (0 a 1) para o bot responder sozinho. Abaixo disso,
    # transborda para atendimento humano. Calibrado para o modelo acima; trocar
    # de modelo de embedding exige recalibrar este valor.
    limiar_similaridade: float = 0.45

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
