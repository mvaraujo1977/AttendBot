"""Configurações da aplicação, carregadas de variáveis de ambiente ou do .env."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _vazio_e_nulo(valor):
    """Trata ``VARIAVEL=`` no .env como "não definida".

    Sem isto, um campo numérico opcional deixado em branco — que é como o
    .env.example pede o limiar — quebraria a subida com um erro de parsing em
    vez de cair no default.
    """
    return None if isinstance(valor, str) and not valor.strip() else valor


NumeroOpcional = Annotated[float | None, BeforeValidator(_vazio_e_nulo)]
InteiroOpcional = Annotated[int | None, BeforeValidator(_vazio_e_nulo)]

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

    # --- Contenção de abuso --------------------------------------------------
    # Mensagens que um mesmo remetente pode gastar por minuto. 0 desliga.
    #
    # Cada mensagem custa duas chamadas de API (o embedding da consulta e a
    # geração), cada uma com até 3 tentativas em 429 — e a cota do tier gratuito
    # é compartilhada entre todos os clientes. Sem um teto, uma única conversa
    # em volume derruba o atendimento de todo mundo. 20/min é folgado para
    # alguém digitando e apertado para um script.
    limite_mensagens_por_minuto: int = 20

    # Corte da pergunta antes de ela virar embedding e prompt. O custo em
    # tokens é linear no tamanho, e o Telegram já aceita 4096 caracteres —
    # uma dúvida de atendimento real não chega perto disto.
    limite_caracteres_pergunta: int = 1000

    # --- RAG -----------------------------------------------------------------
    caminho_faq: Path = RAIZ / "app" / "data" / "faq_dataset.json"
    diretorio_chroma: Path = RAIZ / "chroma_db"
    colecao_chroma: str = "faq"
    top_k: int = 3
    # Primeira das duas barreiras de transbordo: descarta o que nem chega perto
    # da base. Quem decide os casos de fronteira é a segunda barreira, no LLM
    # (ver app/rag/generator.py).
    #
    # Vazio usa o limiar calibrado do provedor de embedding ativo. É o default
    # certo: limiar e modelo de embedding são uma coisa só, e cada modelo
    # distribui as similaridades numa faixa própria. Fixe um valor aqui só
    # depois de medir (ver `scripts/calibrar_limiar.py`).
    limiar_similaridade: NumeroOpcional = None

    # --- Embeddings ----------------------------------------------------------
    # "local"  = modelo offline via sentence-transformers. Melhor retrieval,
    #            sem custo de API, mas exige torch (~2,2 GB) — não cabe nos
    #            512 MB do plano Free do Render.
    # "gemini" = embedding via API. É o de produção. Ver README.
    provedor_embedding: str = "local"
    # Modelo do provedor local (~2,2 GB no primeiro uso). Acerta 10/10 no
    # retrieval do dataset de exemplo; o MiniLM multilíngue, bem mais leve,
    # acerta 6/10 em português acentuado. Ver a seção de calibração no README.
    modelo_embedding: str = "intfloat/multilingual-e5-large"

    # --- Gemini (usado como provedor de LLM e/ou de embedding) ---------------
    gemini_api_key: str | None = None
    # Verificado contra a API: o gemini-2.5-flash ainda aparece na listagem de
    # modelos, mas o generateContent o recusa com 404 para chaves novas.
    gemini_modelo: str = "gemini-3.5-flash-lite"
    gemini_temperatura: float = 0.3
    # Vazio omite o campo, que é o que o modelo padrão exige. Só defina (0
    # desliga o raciocínio) em modelos que aceitam — os 3.5+ recusam com 400.
    gemini_orcamento_raciocinio: InteiroOpcional = None
    gemini_modelo_embedding: str = "gemini-embedding-001"
    # Dimensões da saída do embedding (Matryoshka: 3072, 1536 ou 768).
    gemini_dimensoes_embedding: int = 768

    # --- LLM -----------------------------------------------------------------
    provedor_llm: str = "openai"  # "openai" | "gemini" | "demo"
    openai_api_key: str | None = None
    openai_modelo: str = "gpt-4o-mini"
    openai_temperatura: float = 0.3

    # --- Deploy --------------------------------------------------------------
    # O Render tem filesystem efêmero: o chroma_db some a cada deploy, restart
    # ou hibernação. Reindexar na subida troca um volume persistente (que o
    # plano Free não tem) por alguns segundos de startup — o dataset é pequeno.
    # A ingestão é idempotente, então localmente isso é um no-op depois da
    # primeira vez. Ver `garantir_indice` em app/rag/ingest.py.
    reindexar_no_startup: bool = True

    # Liga o que existe para desenvolver e não tem função em produção: o
    # `/api/mensagem` e a documentação interativa (`/docs`, `/redoc`,
    # `/openapi.json`).
    #
    # Default `false` porque as duas coisas são caras quando a URL é pública. O
    # `/api/mensagem` roda o RAG inteiro sem passar por assinatura, segredo de
    # webhook ou deduplicação — é o caminho mais barato para alguém queimar a
    # cota do Gemini —, e o `/openapi.json` entrega o mapa das rotas de graça,
    # incluindo o schema dele.
    expor_ferramentas_de_teste: bool = False

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

    @model_validator(mode="after")
    def _exigir_autenticacao_de_webhook(self) -> "Configuracoes":
        """Recusa a subida de um canal que envia de verdade sem autenticar quem chama.

        As duas checagens de webhook são condicionais à própria configuração
        (ver as rotas em app/main.py): com o segredo vazio ou a assinatura
        desligada, o bloco de validação é *pulado* — não há requisição inválida
        para registrar, então nem o aviso do provedor chega ao log. Um deploy
        com o webhook aberto fica indistinguível de um deploy correto.

        Por isso a checagem mora aqui e derruba a subida: no Render os segredos
        entram no dashboard (`sync: false` no render.yaml), e deixar um campo em
        branco é um clique. Falhar o healthcheck com o motivo no log é melhor
        que servir para a internet inteira — o bot fora do ar é um incidente
        visível, o bot aberto não é.

        Só vale quando o canal envia de verdade: em dry-run nada sai, e exigir
        segredo ali só atrapalharia quem está rodando o projeto localmente.
        """
        canal = self.canal.strip().lower()

        if (
            canal == "telegram"
            and not self.telegram_dry_run
            and not self.telegram_segredo_webhook
        ):
            raise ValueError(
                "TELEGRAM_SEGREDO_WEBHOOK é obrigatório quando "
                "TELEGRAM_DRY_RUN=false: sem ele o webhook aceita qualquer "
                "POST. Gere um segredo, use o mesmo valor no `setWebhook` do "
                "Telegram e defina a variável."
            )

        if (
            canal == "twilio"
            and not self.twilio_dry_run
            and not self.twilio_validar_assinatura
        ):
            raise ValueError(
                "TWILIO_VALIDAR_ASSINATURA=true é obrigatório quando "
                "TWILIO_DRY_RUN=false: sem isso o webhook aceita qualquer POST. "
                "Defina também URL_PUBLICA com exatamente a URL configurada no "
                "console do Twilio — a assinatura é calculada sobre ela."
            )

        return self


@lru_cache
def obter_configuracoes() -> Configuracoes:
    """Instância única de configuração (cacheada)."""
    return Configuracoes()
