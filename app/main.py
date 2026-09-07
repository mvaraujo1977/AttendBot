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
from collections import OrderedDict, deque
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
from app.whatsapp.base import MensagemRecebida, ProvedorMensageria, pseudonimo

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


class LimitePorRemetente:
    """Teto de mensagens por remetente numa janela de tempo.

    Existe por causa do custo: cada mensagem que passa a 1a barreira vira duas
    chamadas de API (embedding da consulta e geração), cada uma com até 3
    tentativas em 429 — e a cota do tier gratuito é uma só para todos os
    clientes. Sem teto, uma conversa em volume esgota a cota e derruba o
    atendimento de todo mundo; a deduplicação de updates não ajuda, porque ela
    só reconhece reentregas da *mesma* mensagem.

    Memória do processo, pela mesma razão do ``RegistroDeUpdates``: o alvo é
    conter abuso de volume numa instância única, não contabilizar com precisão
    entre réplicas. O custo de errar é uma mensagem legítima descartada num
    restart, contra um Redis para um serviço que não tem nem disco.

    A capacidade limita os remetentes rastreados: sem ela, mensagens de
    remetentes sempre novos fariam o dicionário crescer sem fim — trocando um
    vetor de abuso por outro.
    """

    def __init__(
        self,
        maximo: int = 20,
        janela_segundos: float = 60.0,
        capacidade: int = 1024,
    ) -> None:
        self._maximo = maximo
        self._janela = janela_segundos
        self._capacidade = capacidade
        self._historico: OrderedDict[str, deque[float]] = OrderedDict()
        self._trava = threading.Lock()

    @property
    def remetentes_rastreados(self) -> int:
        """Quantos remetentes estão na memória. Exposto porque o teto de
        capacidade é uma garantia do componente, não um detalhe: sem ele, uma
        rajada de remetentes novos faria o dicionário crescer sem fim."""
        with self._trava:
            return len(self._historico)

    def permitir(self, remetente: str) -> bool:
        """Registra a mensagem e diz se ela cabe no teto do remetente."""
        if self._maximo <= 0:  # 0 desliga o limite.
            return True

        agora = time.monotonic()
        with self._trava:
            marcas = self._historico.get(remetente)
            if marcas is None:
                marcas = deque()
                self._historico[remetente] = marcas
            self._historico.move_to_end(remetente)

            while marcas and agora - marcas[0] > self._janela:
                marcas.popleft()

            # O remetente que estourou o teto continua no fim da fila de
            # descarte: quem está martelando é justamente quem não pode ser
            # esquecido, ou o limite se reinicia sozinho.
            if len(marcas) >= self._maximo:
                return False

            marcas.append(agora)
            while len(self._historico) > self._capacidade:
                self._historico.popitem(last=False)
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
    app.state.limite = LimitePorRemetente(
        maximo=config.limite_mensagens_por_minuto
    )
    yield


# A documentação interativa é decidida na construção da app, antes do lifespan,
# então ela lê a configuração aqui em vez de receber por dependência. Ligá-la
# publica o mapa das rotas — inclusive o `/api/mensagem` e o schema do corpo
# dele —, o que só faz sentido em desenvolvimento.
_ferramentas_de_teste = obter_configuracoes().expor_ferramentas_de_teste

app = FastAPI(
    title="AttendBot",
    description="Bot de atendimento no WhatsApp com RAG (busca vetorial + LLM).",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _ferramentas_de_teste else None,
    redoc_url="/redoc" if _ferramentas_de_teste else None,
    openapi_url="/openapi.json" if _ferramentas_de_teste else None,
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


def obter_limite(request: Request) -> LimitePorRemetente:
    # Sob demanda pelo mesmo motivo de `obter_updates`.
    limite = getattr(request.app.state, "limite", None)
    if limite is None:
        limite = LimitePorRemetente()
        request.app.state.limite = limite
    return limite


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


def _exigir_canal(config: Configuracoes, esperado: str) -> None:
    """Recusa a rota do canal que não é o desta instância.

    As duas rotas de webhook existem sempre, mas cada uma valida com o guard do
    *seu* canal (assinatura do Twilio, segredo do Telegram) enquanto todas
    chamam o mesmo ``mensageria`` — o do canal configurado. Isso separa o guard
    do extrator: rodando com ``CANAL=twilio``, um POST em ``/webhook/telegram``
    não encontra segredo nenhum para conferir, cai no extrator do Twilio (que
    só quer ``From`` e ``Body``) e dispara um envio real. A assinatura vira
    opcional para quem trocar de rota.

    Fechar a rota do canal inativo é o que amarra as duas pontas: cada webhook
    passa a ser alcançável só na configuração em que o seu próprio guard vale.
    404 e não 403 porque, neste deploy, a rota realmente não existe.
    """
    if config.canal.strip().lower() != esperado:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


def _processar(
    mensagem: MensagemRecebida,
    servico: ServicoAtendimento,
    mensageria: ProvedorMensageria,
) -> None:
    """Roda o RAG e devolve a resposta pelo mesmo canal que trouxe a mensagem.

    Executa depois de o webhook já ter respondido, então não há para quem
    propagar exceção: ou a falha é registrada aqui, ou some sem deixar rastro.
    """
    logger.info(
        "Mensagem recebida de %s (%d caracteres).",
        pseudonimo(mensagem.remetente),
        len(mensagem.texto),
    )
    logger.debug("Texto de %s: %s", pseudonimo(mensagem.remetente), mensagem.texto)
    try:
        resposta = servico.responder(mensagem.texto)
        mensageria.enviar(mensagem.remetente, resposta.texto)
    except Exception:  # noqa: BLE001 - último ponto antes de a tarefa sumir
        logger.exception(
            "Falha ao processar a mensagem de %s.", pseudonimo(mensagem.remetente)
        )


def _agendar(
    tarefas: BackgroundTasks,
    mensagem: MensagemRecebida,
    servico: ServicoAtendimento,
    mensageria: ProvedorMensageria,
    updates: RegistroDeUpdates,
    limite: LimitePorRemetente,
) -> None:
    """Enfileira o processamento, descartando reentregas e excesso de volume.

    A chave de deduplicação junta remetente e id da mensagem porque o
    ``message_id`` do Telegram é sequencial por conversa, não global: sozinho,
    ele faria a mensagem de um cliente descartar a de outro.

    O teto por remetente vem depois da deduplicação, de propósito: uma rajada
    de reentregas do mesmo update é culpa do canal, não do cliente, e não deve
    consumir a cota dele.
    """
    chave = (
        f"{mensagem.remetente}:{mensagem.id_externo}"
        if mensagem.id_externo is not None
        else None
    )
    if not updates.registrar(chave):
        logger.info("Reentrega ignorada (%s).", pseudonimo(chave))
        return

    if not limite.permitir(mensagem.remetente):
        # Descarte silencioso: responder "você excedeu o limite" ensinaria o
        # limite a quem está sondando, e gastaria um envio por mensagem
        # descartada — exatamente o que o teto existe para evitar.
        logger.warning(
            "Mensagem descartada: %s excedeu o limite por minuto.",
            pseudonimo(mensagem.remetente),
        )
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
    limite: LimitePorRemetente = Depends(obter_limite),
) -> Response:
    _exigir_canal(config, "twilio")

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

    _agendar(tarefas, mensagem, servico, mensageria, updates, limite)

    return Response(content=TWIML_VAZIO, media_type="application/xml")


@app.post("/webhook/telegram", summary="Webhook de mensagens do Telegram")
async def webhook_telegram(
    request: Request,
    tarefas: BackgroundTasks,
    servico: ServicoAtendimento = Depends(obter_servico),
    mensageria: ProvedorMensageria = Depends(obter_mensageria),
    config: Configuracoes = Depends(obter_config_app),
    updates: RegistroDeUpdates = Depends(obter_updates),
    limite: LimitePorRemetente = Depends(obter_limite),
) -> JSONResponse:
    _exigir_canal(config, "telegram")

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

    _agendar(tarefas, mensagem, servico, mensageria, updates, limite)

    return JSONResponse({"ok": True})


@app.post(
    "/api/mensagem",
    response_model=RespostaResponse,
    summary="Testa o fluxo de RAG sem passar pelo WhatsApp",
)
async def api_mensagem(
    corpo: PerguntaRequest,
    servico: ServicoAtendimento = Depends(obter_servico),
    config: Configuracoes = Depends(obter_config_app),
) -> RespostaResponse:
    """Atalho de desenvolvimento: o RAG sem o canal no caminho.

    Desligado por padrão (``EXPOR_FERRAMENTAS_DE_TESTE``). Ele não passa por
    assinatura, segredo de webhook nem deduplicação de updates, então numa URL
    pública seria o jeito mais direto de esgotar a cota do LLM — e ainda
    devolve similaridade e motivo, que são diagnóstico interno.
    """
    if not config.expor_ferramentas_de_teste:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    resposta = await run_in_threadpool(servico.responder, corpo.pergunta)
    return RespostaResponse.de_atendimento(resposta)
