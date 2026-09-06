"""Aplicação FastAPI: webhook do WhatsApp e endpoint de teste local."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.atendimento import RespostaAtendimento, ServicoAtendimento
from app.config import Configuracoes, obter_configuracoes
from app.dependencias import criar_mensageria, criar_servico
from app.whatsapp.base import ProvedorMensageria

logger = logging.getLogger(__name__)

# O Twilio espera uma resposta TwiML. Respondemos vazio porque a mensagem sai
# pela API (provider), e não pelo corpo da resposta HTTP — assim a lógica de
# envio fica no mesmo lugar para qualquer canal.
TWIML_VAZIO = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega configuração, vector store e canal uma única vez na subida."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    config = obter_configuracoes()
    logger.info(
        "Iniciando bot da %s (LLM=%s, limiar=%.2f, dry_run=%s)",
        config.nome_empresa,
        config.provedor_llm,
        config.limiar_similaridade,
        config.twilio_dry_run,
    )
    app.state.config = config
    app.state.servico = criar_servico(config)
    app.state.mensageria = criar_mensageria(config)
    yield


app = FastAPI(
    title="AttendBot",
    description="Bot de atendimento no WhatsApp com RAG (busca vetorial + LLM).",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Dependências (sobrescritas nos testes) ----------------------------------


def obter_servico(request: Request) -> ServicoAtendimento:
    return request.app.state.servico


def obter_mensageria(request: Request) -> ProvedorMensageria:
    return request.app.state.mensageria


def obter_config_app(request: Request) -> Configuracoes:
    return request.app.state.config


# --- Modelos do endpoint de teste --------------------------------------------


class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., description="Mensagem do cliente")


class RespostaResponse(BaseModel):
    resposta: str
    transbordo: bool
    similaridade: float
    motivo: str | None = None

    @classmethod
    def de_atendimento(cls, resposta: RespostaAtendimento) -> "RespostaResponse":
        return cls(
            resposta=resposta.texto,
            transbordo=resposta.transbordo,
            similaridade=round(resposta.similaridade, 4),
            motivo=resposta.motivo,
        )


# --- Rotas -------------------------------------------------------------------


@app.get("/health", summary="Healthcheck")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/whatsapp", summary="Webhook de mensagens do Twilio")
async def webhook_whatsapp(
    request: Request,
    servico: ServicoAtendimento = Depends(obter_servico),
    mensageria: ProvedorMensageria = Depends(obter_mensageria),
    config: Configuracoes = Depends(obter_config_app),
) -> Response:
    formulario = await request.form()
    payload = {chave: str(valor) for chave, valor in formulario.items()}

    if config.twilio_validar_assinatura:
        url = config.url_publica or str(request.url)
        if not mensageria.validar_requisicao(url, payload, request.headers):
            logger.warning("Webhook recusado: assinatura inválida.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Assinatura inválida."
            )

    try:
        mensagem = mensageria.extrair_mensagem(payload)
    except ValueError as erro:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(erro)
        ) from erro

    logger.info("Mensagem recebida de %s: %s", mensagem.remetente, mensagem.texto)

    # Busca vetorial e chamada ao LLM são bloqueantes: fora do event loop.
    resposta = await run_in_threadpool(servico.responder, mensagem.texto)
    await run_in_threadpool(mensageria.enviar, mensagem.remetente, resposta.texto)

    return Response(content=TWIML_VAZIO, media_type="application/xml")


@app.post(
    "/api/mensagem",
    response_model=RespostaResponse,
    summary="Testa o fluxo de RAG sem passar pelo WhatsApp",
)
async def api_mensagem(
    corpo: PerguntaRequest,
    servico: ServicoAtendimento = Depends(obter_servico),
) -> RespostaResponse:
    resposta = await run_in_threadpool(servico.responder, corpo.pergunta)
    return RespostaResponse.de_atendimento(resposta)
