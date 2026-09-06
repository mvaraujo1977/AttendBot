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
       ├── 1a BARREIRA: abaixo do limiar ──► transbordo (baixa_similaridade)
       │
       ▼ Acima do limiar
 Monta prompt com o contexto recuperado
       │
       ▼
 LLM lê o contexto e julga se ele responde à pergunta
       │
       ├── 2a BARREIRA: não responde ──► transbordo (contexto_insuficiente)
       ├── falha na chamada ───────────► transbordo (erro_geracao)
       │
       ▼ Responde
 Resposta enviada pelo canal (WhatsApp via Twilio)
```

## Stack técnica

- **Python 3.11** (necessário — `chromadb` e `sentence-transformers` ainda
  não têm wheels para Python 3.14; use o `.venv` do projeto)
- **FastAPI** — webhook e endpoint de mensagens
- **ChromaDB** — banco vetorial
- **sentence-transformers** com `intfloat/multilingual-e5-large` — embeddings
  locais, sem custo de API (escolha medida na seção **Calibrando o transbordo**)
- **OpenAI** — provedor de LLM padrão (trocável)
- **Twilio** — provedor de mensageria padrão para WhatsApp (trocável)
- **Pytest** — 47 testes unitários (rápidos, com dublês) e 23 de integração
  (dataset e embeddings reais)

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

> ⚠️ A instalação completa baixa `torch`, e a primeira indexação baixa o
> modelo de embedding (~2,2 GB).

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
| `LIMIAR_SIMILARIDADE` | Primeira barreira: similaridade mínima (0–1) | `0.80` |
| `MODELO_EMBEDDING` | Modelo local usado para gerar embeddings | `intfloat/multilingual-e5-large` |

### Calibrando o transbordo

Foi a parte mais interessante do projeto, e a que mais mudou de rumo com a
medição.

A ideia inicial era simples: um limiar de similaridade decide se o bot responde.
Para escolher o modelo de embedding, medi 20 perguntas — 10 parafraseadas a
partir da FAQ, 10 completamente fora dela:

| Modelo | Acerto do retrieval | Faixa das corretas | Faixa das de fora | Sobreposição |
|---|---|---|---|---|
| `multilingual-e5-large` | **10/10** | 0.839 – 0.891 | 0.735 – 0.845 | 0.006 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 6/10 | 0.390 – 0.839 | 0.039 – 0.524 | 0.134 |

Duas conclusões, e a segunda só apareceu depois de um erro de método.

**1. O e5 recupera muito melhor.** Acerta as 10; o MiniLM erra 4 — manda "o
frete é grátis?" para a garantia e "posso pagar com pix?" para o reembolso.

**2. Nenhum dos dois separa as faixas por limiar.** O e5 é treinado com
negativos in-batch e temperatura, o que comprime todas as similaridades numa
faixa alta e estreita: "quanto é 2 + 2?" pontua 0.832, praticamente o mesmo que
a pergunta legítima mais fraca (0.839). Ele **ordena** muito bem, mas o valor
absoluto quase não discrimina.

> O erro de método vale registrar: na primeira medição escrevi as perguntas sem
> acentuação, e o MiniLM pareceu separar as faixas com folga. Com acentuação —
> que é como clientes de verdade escrevem — o resultado se inverte: "o frete é
> grátis?" cai de 0.586 para 0.454 e passa a casar com a entrada errada. Uma
> medição descuidada quase fixou o modelo pior no projeto.

Daí o desenho atual, com **duas barreiras**:

| | Barreira | Custo | Pega o quê |
|---|---|---|---|
| 1ª | Limiar de similaridade (`0.80`) | zero | O que nem chega perto da base |
| 2ª | O LLM julga se o contexto responde | nenhuma chamada extra | Os casos de fronteira |

A segunda barreira aproveita que o LLM **já está lendo o contexto** para
responder: o prompt de sistema instrui que, se o contexto não responder à
pergunta, ele devolva apenas `TRANSBORDO`. O `ServicoAtendimento` reconhece o
sinal e encaminha para um humano. Não custa uma chamada a mais, e resolve
justamente a faixa que a similaridade não consegue separar.

O limiar fica baixo de propósito: ele é um filtro barato, não o juiz. Quem
decide "vocês vendem passagem aérea?" (0.845 de similaridade, mas sem resposta
na FAQ) é a segunda barreira.

**Se trocar `MODELO_EMBEDDING`, recalibre**: mande ~20 perguntas em
`POST /api/mensagem` (metade cobertas pela FAQ, metade não, todas com
acentuação), confira se o retrieval acerta a entrada certa e ajuste o limiar
abaixo da faixa das corretas. Reindexe com `--recriar`, já que vetores de
modelos diferentes não são comparáveis.

## Testes

```bash
pytest                  # 47 testes unitários, ~1s, sem tocar em modelo ou API
pytest -m integracao    # 23 testes com dataset e embeddings reais, ~25s
```

Os unitários usam dublês nas fronteiras (vector store, LLM, canal) e cobrem a
conversão de distância em similaridade, as quatro causas de transbordo, o
reconhecimento do sinal do LLM, a validação e o upsert do dataset, e o webhook
HTTP.

Os de integração indexam o `faq_dataset.json` de verdade e verificam que o
retrieval recupera a entrada certa e que as faixas de similaridade continuam
onde a calibração assumiu — **é o teste que pegou o erro de calibração descrito
acima**, e que nenhum teste com dublê pegaria. Os que exigem a segunda barreira
precisam de `OPENAI_API_KEY` e são pulados sem ela.

## Roadmap / possíveis evoluções

- Dashboard de métricas de uso (perguntas mais frequentes, taxa de
  transbordo)
- Suporte à API oficial do WhatsApp Business (além do Twilio)
- Busca híbrida (vetorial + palavra-chave) para bases de FAQ maiores

---

Projeto desenvolvido por [Marcelo Araújo](https://github.com/mvaraujo1977)
como peça de portfólio, aplicando RAG a um caso de uso real de atendimento
automatizado.
