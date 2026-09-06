"""Composition root: monta os objetos concretos a partir da configuração.

Concentrar a montagem aqui é o que mantém o resto do código dependendo só de
interfaces — e o que faz o webhook e os testes serem intercambiáveis.
"""

from app.atendimento import ServicoAtendimento
from app.config import Configuracoes
from app.llm.base import ProvedorLLM
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

    if provedor == "demo":
        from app.llm.demo_client import ProvedorDemo

        return ProvedorDemo()

    raise ValueError(
        f"PROVEDOR_LLM='{config.provedor_llm}' desconhecido. Use 'openai' ou 'demo'."
    )


def criar_servico(config: Configuracoes) -> ServicoAtendimento:
    """Monta o serviço de atendimento completo (carrega o vector store)."""
    from app.rag.vector_store import criar_vector_store

    retriever = Retriever(criar_vector_store(config), top_k=config.top_k)
    generator = Generator(criar_llm(config), nome_empresa=config.nome_empresa)
    return ServicoAtendimento(
        retriever=retriever,
        generator=generator,
        limiar_similaridade=config.limiar_similaridade,
        mensagem_transbordo=config.mensagem_transbordo,
        mensagem_boas_vindas=config.mensagem_boas_vindas,
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
