# AttendBot 🤖

Bot de atendimento automatizado via **WhatsApp ou Telegram** com IA, usando
**RAG (Retrieval-Augmented Generation)** para responder perguntas com base em
uma FAQ, e transbordo automático para atendimento humano quando a IA não tem
confiança suficiente na resposta.

> **Para avaliar rodando:** use o Telegram (`CANAL=telegram`). O webhook do
> WhatsApp exige conta Twilio paga — ver **Canais de mensageria** abaixo.

Projeto de portfólio — arquitetura pensada para ser genérica o suficiente
para se tornar a base de um produto real ou de uma entrega para cliente.

## Por que este projeto

A maioria dos bots de "IA para WhatsApp" trava em dois pontos: dependem de
um único provedor de mensageria/LLM amarrado no código, e não sabem dizer
"não sei" — inventam respostas (alucinam) em vez de encaminhar para um
humano. O AttendBot resolve os dois:

- **Arquitetura desacoplada**: trocar o canal ou o provedor de LLM é escrever
  uma classe nova, não reescrever o sistema. Isso deixou de ser promessa: o
  Telegram entrou como segundo canal sem uma linha alterada no RAG, no
  serviço de atendimento ou nos testes existentes — só uma implementação de
  `ProvedorMensageria` e uma rota.
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
       ├── /start, /help, /ajuda ──► boas-vindas (texto fixo, sem RAG)
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
 Resposta enviada pelo canal (Telegram ou WhatsApp via Twilio)
```

## Stack técnica

- **Python 3.11** (necessário — `chromadb` e `sentence-transformers` ainda
  não têm wheels para Python 3.14; use o `.venv` do projeto)
- **FastAPI** — webhook e endpoint de mensagens
- **ChromaDB** — banco vetorial
- **sentence-transformers** com `intfloat/multilingual-e5-large` — embeddings
  locais, sem custo de API (escolha medida na seção **Calibrando o transbordo**)
- **OpenAI** — provedor de LLM padrão (trocável)
- **Twilio** (WhatsApp) e **Telegram Bot API** — dois canais sobre a mesma
  interface, escolhidos por variável de ambiente
- **Pytest** — 82 testes unitários (rápidos, com dublês) e 23 de integração
  (dataset e embeddings reais)

### Duas fronteiras trocáveis (o ponto arquitetural do projeto)

| Interface | Implementação padrão | Trocar por |
|---|---|---|
| `ProvedorMensageria` (`whatsapp/base.py`) | Twilio (WhatsApp) e Telegram | Qualquer canal — a lógica de negócio não sabe que WhatsApp existe |
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
CANAL=telegram
TELEGRAM_DRY_RUN=true
PROVEDOR_LLM=demo
```

```bash
python scripts/simular_conversa.py
```

Isso conversa com o bot diretamente pelo terminal — sem precisar de conta em
canal nenhum ou chave de API.

### 4. Testar via API

```bash
uvicorn app.main:app --reload
```

`POST /api/mensagem` devolve, além da resposta, a **similaridade** e o
**motivo** de eventual transbordo — use isso para calibrar o limiar.

## Canais de mensageria

Um canal é uma implementação de `ProvedorMensageria` (`app/whatsapp/base.py`)
— três métodos: extrair a mensagem do payload, enviar a resposta, validar a
autenticidade do webhook. Nada acima dessa interface sabe que WhatsApp ou
Telegram existem.

Foi o que a entrada do Telegram exercitou. O que precisou ser escrito:

- `app/whatsapp/telegram_client.py` — a classe nova
- um `if` em `criar_mensageria` (`app/dependencias.py`), o composition root
- a rota `POST /webhook/telegram`

O que **não** foi tocado: `ServicoAtendimento`, retriever, generator, prompt,
calibração do transbordo e todos os testes que já existiam. É a diferença
entre um canal acoplado e uma fronteira de verdade — e a razão de o bloqueio
comercial da Twilio, descrito abaixo, ter custado uma tarde em vez de um
reprojeto.

O canal ativo é escolhido por `CANAL` no `.env`, do mesmo jeito que o LLM é
escolhido por `PROVEDOR_LLM`. Existe uma rota por canal porque o formato do
webhook é diferente — o Twilio manda formulário e espera TwiML de volta, o
Telegram manda JSON e espera `200` — mas o que vem depois (RAG, transbordo,
envio) é exatamente o mesmo código.

| Canal | `CANAL` | Rota do webhook | Enviar | Receber (webhook) |
|---|---|---|---|---|
| **Telegram** | `telegram` | `POST /webhook/telegram` | grátis | grátis |
| **WhatsApp (Twilio)** | `twilio` | `POST /webhook/whatsapp` | funciona no trial | **exige conta paga** |

### Comandos

`/start`, `/help` e `/ajuda` são respondidos com uma mensagem fixa de
boas-vindas, sem passar pelo RAG. Não é economia de chamada: `/start` é o que o
Telegram envia quando alguém abre a conversa pela primeira vez, e ele não é
pergunta de cliente nenhuma — medido, pontuava 0.771 de similaridade e chamava
um atendente humano para responder a um clique.

O tratamento fica no `ServicoAtendimento`, não no `ProvedorTelegram`: a resposta
é a mesma em qualquer canal e não depende de transporte. O parser aceita as
formas que o Telegram usa na prática — `/start@nome_do_bot` em grupos e
`/start <payload>` em links de convite. Comando desconhecido segue o fluxo
normal e acaba em transbordo, que é o certo: quem inventou um comando quer
falar com alguém.

Personalize o texto com `MENSAGEM_BOAS_VINDAS` no `.env`.

### Por que o WhatsApp não é demonstrável de graça

No trial da Twilio dá para **enviar** mensagens de WhatsApp normalmente. O que
o trial não libera é **registrar a URL do webhook** — que é justamente o que
faz o bot *receber* a pergunta do cliente:

- **Console novo**: *Messaging → WhatsApp → Try out WhatsApp → aba Inbound →
  Auto-Reply: Custom*. O campo de webhook fica atrás do upgrade.
- **Console legado**: *Sandbox settings → "When a message comes in"* redireciona
  para a página de upgrade.

Sem esse campo o Twilio nunca chama `/webhook/whatsapp`: o bot fica correto e
inalcançável. Por isso a implementação Twilio segue no código, testada e pronta
para uma conta paga — e o canal usado na demonstração é o Telegram, gratuito de
ponta a ponta.

### Ligando o Telegram (~5 minutos, sem cartão)

**1. Criar o bot.** No Telegram, fale com o [@BotFather](https://t.me/BotFather),
mande `/newbot` e guarde o token (formato `123456789:AA...`).

**2. Configurar o `.env`:**

```bash
CANAL=telegram
TELEGRAM_BOT_TOKEN=123456789:AA...
TELEGRAM_DRY_RUN=false
TELEGRAM_SEGREDO_WEBHOOK=um-segredo-qualquer
```

**3. Subir o app e um túnel HTTPS** (o Telegram só aceita webhook em HTTPS):

```bash
uvicorn app.main:app
cloudflared tunnel --url http://localhost:8000    # em outro terminal
```

**4. Registrar o webhook** na URL que o túnel imprimiu:

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://SEU-TUNEL.trycloudflare.com/webhook/telegram","secret_token":"um-segredo-qualquer"}'
```

O `secret_token` precisa ser idêntico ao `TELEGRAM_SEGREDO_WEBHOOK`: o Telegram
o repete no cabeçalho `X-Telegram-Bot-Api-Secret-Token` de todo update, e o bot
recusa com `403` quem não souber o segredo. Deixar `TELEGRAM_SEGREDO_WEBHOOK`
vazio desliga a checagem — aceitável só em teste local.

**5. Conferir e conversar.** `getWebhookInfo` mostra a URL registrada e o último
erro de entrega, que é o primeiro lugar para olhar quando nada chega:

```bash
curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"
```

Agora é só mandar uma mensagem para o bot no Telegram.

> Túnel gratuito do `cloudflared` gera um hostname novo a cada reinício — depois
> de reiniciar, rode o `setWebhook` de novo com a URL nova.

## Configuração

| Variável | Descrição | Padrão |
|---|---|---|
| `CANAL` | `telegram` ou `twilio` | `twilio` |
| `MENSAGEM_BOAS_VINDAS` | Resposta fixa de `/start` e `/help` | texto padrão |
| `TELEGRAM_BOT_TOKEN` | Token do bot criado no @BotFather | — |
| `TELEGRAM_DRY_RUN` | Se `true`, loga a mensagem em vez de enviar de verdade | `true` |
| `TELEGRAM_SEGREDO_WEBHOOK` | Segredo do `setWebhook`; vazio desliga a checagem | — |
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

### Variações e abreviações no FAQ

Cliente não escreve como a FAQ está escrita. O primeiro teste pelo Telegram
pegou o caso extremo:

| Mensagem | Similaridade | Resultado |
|---|---|---|
| `tem nota fiscal` | 0.887 | responde |
| `tem NF` | **0.799** | transbordo — perdeu por 0.001 |

O embedding não sabe que "NF" é nota fiscal, e as `tags` não ajudam: elas são
metadado, não entram no vetor. Havia duas saídas ruins e uma boa:

- **Baixar o limiar** resolveria o `tem NF` e deixaria passar todo o resto que
  está entre 0.77 e 0.80 — inclusive `/start`, que pontuava 0.771.
- **Confiar na 2a barreira** custaria uma chamada de LLM para cada abreviação e,
  pior, o contexto recuperado seria o errado.
- **Enriquecer o índice**, que é o que foi feito: o problema não era o limiar
  estar alto demais, era a base não conter a forma como as pessoas perguntam.

Cada entrada do `faq_dataset.json` ganhou um campo `variacoes` — abreviações,
gíria e escrita sem acento:

```json
{
  "pergunta": "Vocês emitem nota fiscal?",
  "resposta": "Sim, a nota fiscal eletrônica é emitida junto com o envio...",
  "categoria": "pedidos",
  "tags": ["nota fiscal", "nfe", "documento"],
  "variacoes": ["tem NF?", "vocês mandam NF?", "emitem NF-e?", "cadê minha nota fiscal?"]
}
```

Cada variação vira um **documento com embedding próprio**, carregando os
metadados da entrada canônica: quem casa com a busca é o texto da variação,
quem responde continua sendo a pergunta original. São 10 perguntas e 58
variações — 68 documentos indexados.

O efeito, medido depois da reindexação:

| Mensagem | Antes | Depois |
|---|---|---|
| `tem NF` | 0.799 (transbordo) | **0.885** (responde) |
| `tem NF?` | — | 0.929 |
| `cod de rastreio` | — | 0.929 |
| `qual o horario de atendimento` | — | 0.911 |

E o controle, que é o que importa: perguntas fora da base **continuam
transbordando**. `vocês vendem passagem aérea?` (0.861), `quanto é 2 + 2?`
(0.832) e `tem vaga de emprego?` (0.857) passam da 1a barreira, como sempre
passaram, e são barradas pela 2a com `contexto_insuficiente`. Enriquecer o
índice aproximou as paráfrases legítimas sem aproximar o que não tem resposta.

> Efeito colateral que precisou de tratamento: uma pergunta e suas variações são
> vetores diferentes com a mesma resposta, então `tem NF` recuperava a mesma
> entrada duas vezes e gastava duas das três vagas do `top_k`. O `Retriever`
> agora colapsa resultados repetidos, mantendo a ocorrência de maior
> similaridade — o LLM recebe três contextos distintos, como antes.

Ao adicionar variações, **reindexe** (`python -m app.rag.ingest`). Não precisa
de `--recriar`: os ids são estáveis e a carga faz upsert. Variações removidas do
JSON, porém, ficam órfãs na base — aí sim vale um `--recriar`.

## Testes

```bash
pytest                  # 82 testes unitários, ~1s, sem tocar em modelo ou API
pytest -m integracao    # 23 testes com dataset e embeddings reais, ~25s
```

Os unitários usam dublês nas fronteiras (vector store, LLM, canal) e cobrem a
conversão de distância em similaridade, as quatro causas de transbordo, o
reconhecimento do sinal do LLM, a validação e o upsert do dataset, e os dois
webhooks HTTP — incluindo o parsing do update do Telegram, a checagem do
segredo, o descarte de updates que não são mensagem, o curto-circuito dos
comandos (provado pelo dublê: nem busca nem LLM são chamados) e a expansão das
variações na ingestão.

Os de integração indexam o `faq_dataset.json` de verdade e verificam que o
retrieval recupera a entrada certa e que as faixas de similaridade continuam
onde a calibração assumiu — **é o teste que pegou o erro de calibração descrito
acima**, e que nenhum teste com dublê pegaria. Os que exigem a segunda barreira
precisam de `OPENAI_API_KEY` e são pulados sem ela.

## Roadmap / possíveis evoluções

- Dashboard de métricas de uso (perguntas mais frequentes, taxa de
  transbordo)
- Suporte à API oficial do WhatsApp Business (além do Twilio) — hoje o
  WhatsApp depende de conta Twilio paga para o webhook
- Busca híbrida (vetorial + palavra-chave) para bases de FAQ maiores

---

Projeto desenvolvido por [Marcelo Araújo](https://github.com/mvaraujo1977)
como peça de portfólio, aplicando RAG a um caso de uso real de atendimento
automatizado.
