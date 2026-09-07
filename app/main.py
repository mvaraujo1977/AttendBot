"""Aplicação FastAPI: webhooks dos canais e endpoint de teste local.

Há uma rota por canal porque o formato do webhook é diferente: o Twilio manda
formulário e espera TwiML de volta; o Telegram manda JSON e espera 200. O que
vem depois — RAG, transbordo, envio — é o mesmo para os dois.

As duas rotas confirmam o recebimento na hora e processam a mensagem em
segundo plano. Isso não é otimização: é o que mantém a resposta HTTP dentro do
orçamento de tempo do canal (60s no Telegram) mesmo quando o RAG e o LLM
demoram — o caso do primeiro acesso depois de uma hibernação no Render. Ver
"Hibernação no plano Free" no README.
"""

import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.atendimento import RespostaAtendimento, ServicoAtendimento
from app.config import Configuracoes, obter_configuracoes
from app.dependencias import criar_embedding, criar_mensageria, criar_servico
from app.rag.ingest import garantir_indice
from app.rag.vector_store import criar_vector_store
from app.whatsapp.base import MensagemRecebida, ProvedorMensageria

logger = logging.getLogger(__name__)

# O Twilio espera uma resposta TwiML. Respondemos vazio porque a mensagem sai
# pela API (provider), e não pelo corpo da resposta HTTP — assim a lógica de
# envio fica no mesmo lugar para qualquer canal.
TWIML_VAZIO = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


class RegistroDeUpdates:
    """Lembra as mensagens já processadas, para não responder duas vezes.

    Necessário porque agora confirmamos o webhook antes de responder. Quando o
    serviço estava hibernando, o canal já pode ter reenviado o mesmo update
    algumas vezes antes de a instância subir — e todas essas cópias chegam
    juntas. Sem isto o cliente receberia a mesma resposta várias vezes, cada
    uma gastando uma chamada de LLM.

    Memória do processo, de propósito: uma cópia por instância resolve o
    problema real (uma rajada de reentregas do mesmo update) e não justifica um
    Redis num serviço de instância única.
    """

    def __init__(self, capacidade: int = 512) -> None:
        self._capacidade = capacidade
        self._vistos: OrderedDict[str, None] = OrderedDict()
        # As rotas são async, mas as tarefas de fundo rodam em threads.
        self._trava = threading.Lock()

    def registrar(self, chave: str | None) -> bool:
        """Marca a mensagem como vista. ``False`` se ela já tinha sido vista."""
        if chave is None:
            return True
        with self._trava:
            if chave in self._vistos:
                return False
            self._vistos[chave] = None
            if len(self._vistos) > self._capacidade:
                self._vistos.popitem(last=False)
            return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega configuração, índice, vector store e canal uma vez na subida."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    config = obter_configuracoes()
    embedding = criar_embedding(config)
    logger.info(
        "Iniciando bot da %s (canal=%s, LLM=%s, embedding=%s)",
        config.nome_empresa,
        config.canal,
        config.provedor_llm,
        embedding.descricao,
    )

    vector_store = criar_vector_store(config, embedding)
    if config.reindexar_no_startup:
        # Bloqueia a subida de propósito: servir com o índice vazio faria todo
        # cliente cair em transbordo por `sem_resultados`.
        inicio = time.perf_counter()
        indexados = await run_in_threadpool(garantir_indice, config, vector_store)
        if indexados:
            logger.info(
                "FAQ indexado na subida: %d documentos em %.1fs.",
                indexados,
                time.perf_counter() - inicio,
            )

    app.state.config = config
    app.state.servico = criar_servico(
        config, vector_store=vector_store, embedding=embedding
    )
    app.state.mensageria = criar_mensageria(config)
    app.state.updates = RegistroDeUpdates()
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


def obter_updates(request: Request) -> RegistroDeUpdates:
    # Criado sob demanda para as rotas funcionarem nos testes, que substituem
    # as outras dependências e não executam o lifespan.
    registro = getattr(request.app.state, "updates", None)
    if registro is None:
        registro = RegistroDeUpdates()
        request.app.state.updates = registro
    return registro


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


# --- Núcleo comum aos canais -------------------------------------------------


def _processar(
    mensagem: MensagemRecebida,
    servico: ServicoAtendimento,
    mensageria: ProvedorMensageria,
) -> None:
    """Roda o RAG e devolve a resposta pelo mesmo canal que trouxe a mensagem.

    Executa depois de o webhook já ter respondido, então não há para quem
    propagar exceção: ou a falha é registrada aqui, ou some sem deixar rastro.
    """
    logger.info("Mensagem recebida de %s: %s", mensagem.remetente, mensagem.texto)
    try:
        resposta = servico.responder(mensagem.texto)
        mensageria.enviar(mensagem.remetente, resposta.texto)
    except Exception:  # noqa: BLE001 - último ponto antes de a tarefa sumir
        logger.exception("Falha ao processar a mensagem de %s.", mensagem.remetente)


def _agendar(
    tarefas: BackgroundTasks,
    mensagem: MensagemRecebida,
    servico: ServicoAtendimento,
    mensageria: ProvedorMensageria,
    updates: RegistroDeUpdates,
) -> None:
    """Enfileira o processamento, ignorando reentregas da mesma mensagem.

    A chave junta remetente e id da mensagem porque o ``message_id`` do
    Telegram é sequencial por conversa, não global: sozinho, ele faria a
    mensagem de um cliente descartar a de outro.
    """
    chave = (
        f"{mensagem.remetente}:{mensagem.id_externo}"
        if mensagem.id_externo is not None
        else None
    )
    if not updates.registrar(chave):
        logger.info("Reentrega ignorada (%s).", chave)
        return
    tarefas.add_task(_processar, mensagem, servico, mensageria)


# --- Rotas -------------------------------------------------------------------


@app.get("/health", summary="Healthcheck")
async def health() -> dict[str, str]:
    """Resposta barata, sem tocar no RAG.

    É o alvo do healthcheck do Render e do ping de keep-alive que evita a
    hibernação do plano Free (ver README).
    """
    return {"status": "ok"}


@app.post("/webhook/whatsapp", summary="Webhook de mensagens do Twilio")
async def webhook_whatsapp(
    request: Request,
    tarefas: BackgroundTasks,
    servico: ServicoAtendimento = Depends(obter_servico),
    mensageria: ProvedorMensageria = Depends(obter_mensageria),
    config: Configuracoes = Depends(obter_config_app),
    updates: RegistroDeUpdates = Depends(obter_updates),
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

    _agendar(tarefas, mensagem, servico, mensageria, updates)

    return Response(content=TWIML_VAZIO, media_type="application/xml")


@app.post("/webhook/telegram", summary="Webhook de mensagens do Telegram")
async def webhook_telegram(
    request: Request,
    tarefas: BackgroundTasks,
    servico: ServicoAtendimento = Depends(obter_servico),
    mensageria: ProvedorMensageria = Depends(obter_mensageria),
    config: Configuracoes = Depends(obter_config_app),
    updates: RegistroDeUpdates = Depends(obter_updates),
) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError as erro:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Corpo não é JSON."
        ) from erro

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Update inválido."
        )

    # Sem segredo configurado não há o que validar: o webhook fica aberto, o
    # que só é aceitável em demonstração local (ver README).
    if config.telegram_segredo_webhook:
        if not mensageria.validar_requisicao(
            str(request.url), payload, request.headers
        ):
            logger.warning("Webhook recusado: segredo inválido.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Segredo inválido."
            )

    try:
        mensagem = mensageria.extrair_mensagem(payload)
    except ValueError as erro:
        # O Telegram reenvia o update em qualquer resposta != 2xx. Updates que
        # não são mensagem (edição de canal, callback de botão) são normais:
        # confirmamos o recebimento e descartamos.
        logger.info("Update do Telegram ignorado: %s", erro)
        return JSONResponse({"ok": True, "ignorado": True})

    _agendar(tarefas, mensagem, servico, mensageria, updates)

    return JSONResponse({"ok": True})


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
