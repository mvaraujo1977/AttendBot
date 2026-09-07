"""Composition root: monta os objetos concretos a partir da configuração.

Concentrar a montagem aqui é o que mantém o resto do código dependendo só de
interfaces — e o que faz o webhook e os testes serem intercambiáveis. São três
fronteiras trocáveis por variável de ambiente: o canal, o LLM e o embedding.
"""

from app.atendimento import ServicoAtendimento
from app.config import Configuracoes
from app.llm.base import ProvedorLLM
from app.rag.embedding.base import ProvedorEmbedding
from app.rag.generator import Generator
from app.rag.retriever import Retriever
from app.whatsapp.base import ProvedorMensageria
from app.whatsapp.twilio_client import ProvedorTwilio


def criar_llm(config: Configuracoes) -> ProvedorLLM:
    """Escolhe a implementação de LLM conforme ``PROVEDOR_LLM``."""
    provedor = config.provedor_llm.strip().lower()

    if provedor == "openai":
        from app.llm.openai_client import ProvedorOpenAI

        return ProvedorOpenAI(
            api_key=config.openai_api_key,
            modelo=config.openai_modelo,
            temperatura=config.openai_temperatura,
        )

    if provedor == "gemini":
        from app.llm.gemini_client import ProvedorGemini

        return ProvedorGemini(
            api_key=config.gemini_api_key,
            modelo=config.gemini_modelo,
            temperatura=config.gemini_temperatura,
            orcamento_raciocinio=config.gemini_orcamento_raciocinio,
        )

    if provedor == "demo":
        from app.llm.demo_client import ProvedorDemo

        return ProvedorDemo()

    raise ValueError(
        f"PROVEDOR_LLM='{config.provedor_llm}' desconhecido. "
        "Use 'openai', 'gemini' ou 'demo'."
    )


def criar_embedding(config: Configuracoes) -> ProvedorEmbedding:
    """Escolhe a implementação de embedding conforme ``PROVEDOR_EMBEDDING``.

    Os imports são locais porque as duas implementações têm dependências
    pesadas e mutuamente exclusivas: em produção o ``sentence-transformers``
    nem está instalado, e importá-lo no topo derrubaria a subida do serviço.
    """
    provedor = config.provedor_embedding.strip().lower()

    if provedor == "local":
        from app.rag.embedding.local_client import ProvedorEmbeddingLocal

        return ProvedorEmbeddingLocal(modelo=config.modelo_embedding)

    if provedor == "gemini":
        from app.rag.embedding.gemini_client import ProvedorEmbeddingGemini

        return ProvedorEmbeddingGemini(
            api_key=config.gemini_api_key,
            modelo=config.gemini_modelo_embedding,
            dimensoes=config.gemini_dimensoes_embedding,
        )

    raise ValueError(
        f"PROVEDOR_EMBEDDING='{config.provedor_embedding}' desconhecido. "
        "Use 'local' ou 'gemini'."
    )


def resolver_limiar(config: Configuracoes, embedding: ProvedorEmbedding) -> float:
    """Limiar da 1a barreira: o do ``.env``, ou o calibrado para o modelo ativo.

    Sem esse fallback, trocar ``PROVEDOR_EMBEDDING`` e esquecer de recalibrar
    aplicaria um número medido para outro modelo — o transbordo continuaria
    "funcionando" e decidindo errado, que é o pior tipo de falha aqui.
    """
    if config.limiar_similaridade is not None:
        return config.limiar_similaridade
    return embedding.limiar_calibrado


def criar_servico(
    config: Configuracoes,
    vector_store=None,
    embedding: ProvedorEmbedding | None = None,
) -> ServicoAtendimento:
    """Monta o serviço de atendimento completo.

    ``vector_store`` e ``embedding`` são parâmetros para que a subida da app
    possa reaproveitar os que já criou para indexar o FAQ, em vez de abrir uma
    segunda conexão com o Chroma e carregar o modelo duas vezes.
    """
    from app.rag.vector_store import criar_vector_store

    embedding = embedding or criar_embedding(config)
    if vector_store is None:
        vector_store = criar_vector_store(config, embedding)

    retriever = Retriever(vector_store, top_k=config.top_k)
    generator = Generator(criar_llm(config), nome_empresa=config.nome_empresa)
    return ServicoAtendimento(
        retriever=retriever,
        generator=generator,
        limiar_similaridade=resolver_limiar(config, embedding),
        mensagem_transbordo=config.mensagem_transbordo,
        mensagem_boas_vindas=config.mensagem_boas_vindas,
        limite_caracteres=config.limite_caracteres_pergunta,
    )


def criar_mensageria(config: Configuracoes) -> ProvedorMensageria:
    """Escolhe a implementação de canal conforme ``CANAL``."""
    canal = config.canal.strip().lower()

    if canal == "twilio":
        return ProvedorTwilio(
            account_sid=config.twilio_account_sid,
            auth_token=config.twilio_auth_token,
            numero_origem=config.twilio_numero_origem,
            dry_run=config.twilio_dry_run,
        )

    if canal == "telegram":
        from app.whatsapp.telegram_client import ProvedorTelegram

        return ProvedorTelegram(
            token=config.telegram_bot_token,
            dry_run=config.telegram_dry_run,
            segredo_webhook=config.telegram_segredo_webhook,
        )

    raise ValueError(
        f"CANAL='{config.canal}' desconhecido. Use 'twilio' ou 'telegram'."
    )
