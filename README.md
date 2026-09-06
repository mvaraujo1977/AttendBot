# AttendBot 🤖

Bot de atendimento automatizado via WhatsApp com IA, usando **RAG
(Retrieval-Augmented Generation)** para responder perguntas com base em uma
FAQ, e transbordo automático para atendimento humano quando a IA não tem
confiança suficiente na resposta.

Projeto de portfólio — arquitetura pensada para ser genérica o suficiente
para se tornar a base de um produto real ou de uma entrega para cliente.

## Por que este projeto

A maioria dos bots de "IA para WhatsApp" trava em dois pontos: dependem de
um único provedor de mensageria/LLM amarrado no código, e não sabem dizer
"não sei" — inventam respostas (alucinam) em vez de encaminhar para um
humano. O AttendBot resolve os dois:

- **Arquitetura desacoplada**: trocar o canal (WhatsApp → Telegram, Twilio →
  API oficial) ou o provedor de LLM (OpenAI → outro) é escrever uma classe
  nova, não reescrever o sistema.
- **Transbordo consciente**: a IA só responde quando a similaridade entre a
  pergunta do usuário e a base de conhecimento está acima de um limiar
  configurável. Abaixo disso — ou se o LLM falhar — a conversa é encaminhada
  para um humano, com o motivo registrado (`sem_resultados`,
  `baixa_similaridade`, `erro_geracao`).
- **100% executável offline**: dá para rodar o fluxo inteiro — embeddings,
  busca vetorial, geração de resposta e "envio" pelo canal — sem nenhuma
  credencial de API, usando modos de demonstração. Isso facilita tanto
  para quem for avaliar o projeto quanto para desenvolvimento local.

## Como funciona

```
Mensagem recebida
       │
       ▼
 Gera embedding da pergunta (multilingual-e5-large, local)
       │
       ▼
 Busca no ChromaDB pelas perguntas mais similares da FAQ
       │
       ▼
 Calcula similaridade (0–1) a partir da distância de cosseno
       │
       ├── Abaixo do limiar ──► Transbordo para humano (motivo registrado)
       │
       ▼ Acima do limiar
 Monta prompt com o contexto recuperado
       │
       ▼
 LLM gera a resposta final (ou falha ──► transbordo por erro_geracao)
       │
       ▼
 Resposta enviada pelo canal (WhatsApp via Twilio)
```

## Stack técnica

- **Python 3.11** (necessário — `chromadb` e `sentence-transformers` ainda
  não têm wheels para Python 3.14; use o `.venv` do projeto)
- **FastAPI** — webhook e endpoint de mensagens
- **ChromaDB** — banco vetorial
- **sentence-transformers** com `intfloat/multilingual-e5-large` — embeddings
  locais, sem custo de API, com boa cobertura de português
- **OpenAI** — provedor de LLM padrão (trocável)
- **Twilio** — provedor de mensageria padrão para WhatsApp (trocável)
- **Pytest** — 40 testes cobrindo o fluxo RAG, o transbordo e as duas
  camadas de provedores

### Duas fronteiras trocáveis (o ponto arquitetural do projeto)

| Interface | Implementação padrão | Trocar por |
|---|---|---|
| `ProvedorMensageria` (`whatsapp/base.py`) | Twilio | Qualquer canal — a lógica de negócio não sabe que WhatsApp existe |
| `ProvedorLLM` (`llm/base.py`) | OpenAI | Qualquer LLM — nova classe + uma linha em `dependencias.py` |

## Rodando localmente

### 1. Ambiente

```bash
# use Python 3.11 — o projeto não roda em 3.14 (sem wheels de chromadb/sentence-transformers)
python3.11 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> ⚠️ A instalação completa baixa `torch` e o modelo de embedding
> (~2,2 GB no primeiro uso). Para uma alternativa mais leve, veja a seção
> **Configuração** abaixo.

### 2. Indexar a base de FAQ

```bash
python -m app.rag.ingest
```

### 3. Rodar em modo demonstração (sem nenhuma credencial)

```bash
# .env
TWILIO_DRY_RUN=true
PROVEDOR_LLM=demo
```

```bash
python scripts/simular_conversa.py
```

Isso conversa com o bot diretamente pelo terminal — sem precisar de conta
Twilio ou chave de API.

### 4. Testar via API

```bash
uvicorn app.main:app --reload
```

`POST /api/mensagem` devolve, além da resposta, a **similaridade** e o
**motivo** de eventual transbordo — use isso para calibrar o limiar.

## Configuração

| Variável | Descrição | Padrão |
|---|---|---|
| `TWILIO_DRY_RUN` | Se `true`, loga a mensagem em vez de enviar de verdade | `true` |
| `PROVEDOR_LLM` | `openai` ou `demo` (responde sem API, para testes) | `openai` |
| `LIMIAR_SIMILARIDADE` | Limiar (0–1) abaixo do qual ocorre transbordo | `0.60` |
| `MODELO_EMBEDDING` | Modelo local usado para gerar embeddings | `intfloat/multilingual-e5-large` |

Para reduzir o download inicial, é possível trocar o modelo de embedding por
um mais leve (ex. MiniLM multilíngue, ~470 MB) alterando `MODELO_EMBEDDING` —
ver comentário em `rag/embeddings.py`.

## Testes

```bash
pytest
```

40 testes cobrindo o fluxo RAG ponta a ponta, a lógica de transbordo e as
duas camadas de provedor (mensageria e LLM).

## Roadmap / possíveis evoluções

- Dashboard de métricas de uso (perguntas mais frequentes, taxa de
  transbordo)
- Suporte à API oficial do WhatsApp Business (além do Twilio)
- Busca híbrida (vetorial + palavra-chave) para bases de FAQ maiores

---

Projeto desenvolvido por [Marcelo Araújo](https://github.com/mvaraujo1977)
como peça de portfólio, aplicando RAG a um caso de uso real de atendimento
automatizado.
