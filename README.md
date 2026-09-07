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

- **Arquitetura desacoplada**: trocar o canal, o LLM ou o modelo de embedding
  é escrever uma classe nova, não reescrever o sistema. Isso deixou de ser
  promessa duas vezes. O Telegram entrou como segundo canal sem uma linha
  alterada no RAG, no serviço de atendimento ou nos testes existentes. Depois,
  o deploy no Render exigiu tirar o `torch` da memória: o embedding virou uma
  terceira fronteira e passou a rodar via API — de novo sem tocar em retriever,
  prompt ou handoff.
- **Transbordo consciente**: a IA só responde quando a similaridade entre a
  pergunta do usuário e a base de conhecimento está acima de um limiar
  configurável. Abaixo disso — ou se o LLM falhar — a conversa é encaminhada
  para um humano, com o motivo registrado (`sem_resultados`,
  `baixa_similaridade`, `erro_geracao`).
- **100% executável offline**: dá para rodar o fluxo inteiro — embeddings,
  busca vetorial, geração de resposta e "envio" pelo canal — sem nenhuma
  credencial de API, usando modos de demonstração. Isso facilita tanto
  para quem for avaliar o projeto quanto para desenvolvimento local.
- **Roda de graça em produção**: `Dockerfile` e `render.yaml` prontos para o
  plano Free do Render, com embedding e LLM no tier gratuito do Gemini. As
  limitações reais desse plano — e o que dá e o que não dá para mitigar —
  estão em **Deploy no Render**.

## Como funciona

```
Mensagem recebida
       │
       ├── /start, /help, /ajuda ──► boas-vindas (texto fixo, sem RAG)
       │
       ▼
 Gera embedding da pergunta (modelo local OU API do Gemini)
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
  não têm wheels para Python 3.14). O `Dockerfile` fixa a versão, então em
  produção isso deixou de ser algo a lembrar
- **FastAPI** — webhook e endpoint de mensagens
- **ChromaDB** — banco vetorial
- **Embeddings trocáveis** — `intfloat/multilingual-e5-large` local via
  sentence-transformers, ou a API do Gemini (escolha medida na seção
  **Calibrando o transbordo**)
- **LLM trocável** — OpenAI, Gemini, ou o modo demonstração sem API
- **Twilio** (WhatsApp) e **Telegram Bot API** — dois canais sobre a mesma
  interface, escolhidos por variável de ambiente
- **Docker + Render** — deploy no plano Free, sem custo recorrente
- **Pytest** — 158 testes unitários (rápidos, com dublês) e 38 de integração
  (dataset e embeddings reais)

### Três fronteiras trocáveis (o ponto arquitetural do projeto)

| Interface | Implementações | Variável |
|---|---|---|
| `ProvedorMensageria` (`whatsapp/base.py`) | Twilio (WhatsApp), Telegram | `CANAL` |
| `ProvedorLLM` (`llm/base.py`) | OpenAI, Gemini, demo | `PROVEDOR_LLM` |
| `ProvedorEmbedding` (`rag/embedding/base.py`) | local (sentence-transformers), Gemini | `PROVEDOR_EMBEDDING` |

Cada uma é uma interface pequena com uma implementação por provedor e um `if`
no composition root (`app/dependencias.py`). Nada acima delas sabe qual está
ativa: o `ServicoAtendimento` não sabe que WhatsApp existe, e o `Retriever` não
sabe se o vetor veio de um modelo em RAM ou de uma chamada HTTP.

A terceira fronteira é a mais recente e a que teve a justificativa mais
concreta: o `multilingual-e5-large` tem 560M de parâmetros e ocupa ~2,2 GB em
memória quando carregado — sozinho, quatro vezes o limite de 512 MB do plano
Free do Render. Sem `torch` no processo, o serviço cabe.

## Rodando localmente

Há duas formas de rodar, e elas diferem em **uma variável**:

| | `PROVEDOR_EMBEDDING=local` | `PROVEDOR_EMBEDDING=gemini` |
|---|---|---|
| Dependências | `requirements.txt` + `requirements-local.txt` | só `requirements.txt` |
| Peso em disco | ~875 MB de libs + ~2,2 GB de pesos | ~185 MB de libs |
| RAM | ~2,2 GB só para o modelo | dezenas de MB |
| Chave de API | nenhuma | `GEMINI_API_KEY` |
| Rede | só no primeiro download | uma chamada por busca |
| Onde usar | desenvolvimento, medição, CI offline | produção (Render Free) |

O resto do sistema é idêntico nos dois casos. Comece pela primeira se quiser
ver o projeto rodando sem criar conta em lugar nenhum.

### 1. Ambiente

```bash
# use Python 3.11 — o projeto não roda em 3.14 (sem wheels de chromadb/sentence-transformers)
python3.11 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2a. Opção offline: modelo de embedding local

```bash
pip install -r requirements-local.txt   # torch + sentence-transformers
```

```bash
# .env
PROVEDOR_EMBEDDING=local
MODELO_EMBEDDING=intfloat/multilingual-e5-large
PROVEDOR_LLM=demo
CANAL=telegram
TELEGRAM_DRY_RUN=true
```

> ⚠️ São ~2,5 GB de instalação, e a primeira indexação baixa mais ~2,2 GB de
> pesos. Depois disso roda offline, sem chave de API nenhuma.

### 2b. Opção leve: embedding via API do Gemini

Sem `torch`, sem download de modelo. Pegue uma chave gratuita em
[aistudio.google.com/apikey](https://aistudio.google.com/apikey):

```bash
# .env
PROVEDOR_EMBEDDING=gemini
PROVEDOR_LLM=gemini
GEMINI_API_KEY=...
CANAL=telegram
TELEGRAM_DRY_RUN=true
```

A mesma chave serve para o embedding e para o LLM. É a configuração que roda
em produção — ver **Deploy no Render**.

### 3. Indexar a base de FAQ

```bash
python -m app.rag.ingest
```

Não é obrigatório: a aplicação indexa sozinha na subida se o índice ainda não
refletir o `faq_dataset.json` (`REINDEXAR_NO_STARTUP=true`, o padrão). O
comando existe para reindexar sem reiniciar o serviço, e aceita `--recriar`.

> Vetores de modelos diferentes não são comparáveis. Ao trocar
> `PROVEDOR_EMBEDDING`, rode `python -m app.rag.ingest --recriar` **e**
> recalibre o limiar (ver **Calibrando o transbordo**).

### 4. Conversar pelo terminal (sem canal e sem servidor)

```bash
python scripts/simular_conversa.py
```

Com `PROVEDOR_LLM=demo` isso roda o fluxo inteiro — busca vetorial, transbordo
e "envio" — sem chave de API nenhuma.

### 5. Testar via API

```bash
uvicorn app.main:app --reload
```

`POST /api/mensagem` devolve, além da resposta, a **similaridade** e o
**motivo** de eventual transbordo — use isso para calibrar o limiar. Ele exige
`EXPOR_FERRAMENTAS_DE_TESTE=true` (como no `.env.example`), que também liga a
documentação interativa em `/docs`; sem a variável, os dois respondem 404.

O default é ficar desligado porque este endpoint roda o RAG inteiro **sem**
passar por assinatura, segredo de webhook ou deduplicação de updates: numa URL
pública, é o jeito mais direto de esgotar a cota do LLM.

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

## Deploy no Render

O repositório traz `Dockerfile` e `render.yaml`. No Render: **New → Blueprint**,
aponte para o repositório, e ele lê o `render.yaml`. Os três segredos
(`GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_SEGREDO_WEBHOOK`) estão
marcados com `sync: false`, então o Render os pede no dashboard em vez de
esperá-los versionados.

Depois do primeiro deploy, registre o webhook na URL que o Render deu
(`https://<seu-servico>.onrender.com/webhook/telegram`) com o mesmo
`setWebhook` da seção anterior.

### Verificado localmente antes de subir

```bash
docker build -t attendbot .
docker run --rm --memory 512m -e PORT=10000 -p 10000:10000   -e PROVEDOR_EMBEDDING=gemini -e PROVEDOR_LLM=gemini -e GEMINI_API_KEY=...   attendbot
```

| Medida | Valor |
|---|---|
| Tamanho da imagem | **190 MB** (`python:3.11-slim`, sem build tools) |
| RAM em uso, sob `--memory 512m` | **97 MiB** (19% do limite) |
| Reindexação do FAQ na subida | **~2s**, 68 documentos, uma chamada de batch |

O `chromadb` 1.x instala de wheel no `slim` — não foi preciso `gcc` nem trocar
para a imagem completa. A imagem não leva `.env` (está no `.dockerignore`):
toda a configuração vem de variáveis de ambiente, que é como o Render entrega.

### Variáveis de ambiente

O `render.yaml` já traz as não-secretas; a tabela existe para quem criar o
serviço à mão, sem o Blueprint.

| Variável | Valor em produção | Por quê |
|---|---|---|
| `PROVEDOR_EMBEDDING` | `gemini` | O modelo local não cabe em 512 MB |
| `PROVEDOR_LLM` | `gemini` | Tier gratuito, sem custo recorrente |
| `CANAL` | `telegram` | O webhook do WhatsApp exige conta Twilio paga |
| `REINDEXAR_NO_STARTUP` | `true` | Filesystem efêmero: o índice some a cada subida |
| `TELEGRAM_DRY_RUN` | `false` | Sem isto o bot só loga a resposta, não a envia |
| `NOME_EMPRESA` | o seu | Aparece no prompt de sistema |
| `GEMINI_API_KEY` | 🔒 segredo | Serve para o embedding **e** para o LLM |
| `TELEGRAM_BOT_TOKEN` | 🔒 segredo | Token do @BotFather |
| `TELEGRAM_SEGREDO_WEBHOOK` | 🔒 segredo | Conferido em todo update; **obrigatório** fora de dry-run |

`EXPOR_FERRAMENTAS_DE_TESTE` fica **fora** desta lista de propósito: o default
já é `false`, e é o valor certo em produção.

Com `TELEGRAM_DRY_RUN=false`, a app **recusa subir** sem
`TELEGRAM_SEGREDO_WEBHOOK` (o equivalente no Twilio é `TWILIO_VALIDAR_ASSINATURA`).
Falhar o deploy é intencional: a validação do webhook é condicional à própria
variável, então um campo em branco no dashboard não gerava erro nem aviso — só
um webhook que aceitava POST de qualquer origem, indistinguível de um deploy
correto. Um bot fora do ar é um incidente visível; um bot aberto não é.

Não defina `PORT` — o Render injeta, e o `CMD` do `Dockerfile` a lê
(verificado: com `PORT=10000` sobe em 10000, sem ela cai no 8000).
`LIMIAR_SIMILARIDADE` também fica de fora de propósito: vazio, ele usa o limiar
calibrado do provedor de embedding ativo (`0.77` para o Gemini). Ver
**Calibrando o transbordo**.

> **Ao escolher `GEMINI_MODELO`, teste com uma chamada real.** O
> `gemini-2.5-flash` — o default original deste projeto — continua listado em
> `GET /v1beta/models`, mas o `generateContent` devolve `404 "no longer
> available to new users"` para chaves criadas depois da retirada. Só o
> smoke test do container pegou isso; a listagem de modelos não é sinal de
> disponibilidade. O default hoje é `gemini-3.5-flash-lite`, medido em ~1,15s
> de mediana no prompt real.
>
> `GEMINI_ORCAMENTO_RACIOCINIO` existe pelo motivo oposto: desligar o
> raciocínio é uma alavanca de latência real (o `gemini-flash-latest` levou
> 19s no mesmo prompt), mas os modelos 3.5+ **recusam** o campo com 400. Por
> isso ele é opcional e vem vazio.

### O que o plano Free impõe, e o que o projeto faz a respeito

| Restrição do Free | Consequência | O que foi feito |
|---|---|---|
| 512 MB de RAM | O `e5-large` (~2,2 GB carregado) não entra | `PROVEDOR_EMBEDDING=gemini`; `torch` fora do `requirements.txt` |
| Filesystem efêmero, sem disco persistente | O `chroma_db` some a cada deploy, restart ou hibernação | O índice é reconstruído na subida a partir do `faq_dataset.json` |
| A porta vem em `$PORT` | Escutar em 8000 fixo dá "no open ports detected" | O `CMD` do `Dockerfile` expande `${PORT:-8000}` |
| Hiberna após 15 min sem tráfego | O acesso seguinte espera a instância subir | Ver abaixo — é a limitação que **não** dá para eliminar |

Reindexar na subida em vez de usar volume é uma troca consciente: 68 documentos
numa chamada de batch custam poucos segundos, e o índice sempre nasce coerente
com o dataset que subiu junto. Não escalaria para uma base grande — aí o certo
seria um vector store gerenciado, não um disco.

### Hibernação: a limitação que fica

Documentando com os números reais, porque é o tipo de coisa que só aparece em
produção:

- O Render desliga a instância **após 15 minutos sem tráfego de entrada**, e
  religá-la leva **cerca de um minuto** (números da documentação do Render).
- O webhook do Telegram tem um orçamento de **60 segundos** por entrega.
- O healthcheck do Render (`healthCheckPath`) roda só durante o deploy: ele
  **não** conta como tráfego e não evita a hibernação.

Ou seja: a primeira mensagem depois de uma hibernação chega enquanto o
contêiner ainda está subindo, e o Render segura a requisição até ele responder.
Isso encosta no limite dos 60s.

**O que o Telegram faz sozinho.** A documentação da Bot API diz que, numa
entrega malsucedida, "repetimos a requisição e desistimos depois de um número
razoável de tentativas", e que os updates ficam guardados no servidor por até
24 horas. Então a primeira mensagem depois da hibernação normalmente é
**atrasada, não perdida** — o Telegram reenvia, e aí a instância já está de pé.

**O que o projeto faz.** Duas mudanças no webhook, ambas por causa disto:

1. **Responde 200 antes de processar.** As rotas confirmam o recebimento na
   hora e rodam RAG, LLM e envio numa tarefa de fundo. Assim o tempo do
   atendimento em si deixa de contar contra os 60s do canal — sobra o tempo de
   subida, que é o que não está nas nossas mãos.
2. **Descarta reentregas.** Consequência direta da mudança acima: se o Telegram
   já tinha reenviado o update duas ou três vezes enquanto a instância subia,
   todas as cópias chegam juntas. O `RegistroDeUpdates` (`app/main.py`) as
   colapsa pela chave `chat + message_id`, então o cliente recebe uma resposta
   só e o tier gratuito do Gemini não paga por três.

**O que sobra, e como evitar.** Nada disso impede a hibernação. Para um bot que
precise responder rápido a qualquer hora, a saída é manter o serviço acordado
com um ping externo (cron-job.org, UptimeRobot) em `GET /health` a cada ~10
minutos — o endpoint responde sem tocar no RAG justamente para isso. A conta a
fazer antes: o Free dá **750 horas de instância por mês por workspace**, e um
mês tem 720–744 horas. Manter um serviço acordado 24/7 cabe, mas consome quase
toda a cota — não sobra para um segundo serviço gratuito no mesmo workspace.

Para uma demonstração de portfólio, aceitar a hibernação e avisar que a
primeira mensagem demora ~1 minuto é a escolha mais honesta. Para um cliente
real, o plano pago do Render resolve, e o resto do projeto não muda.

## Configuração

| Variável | Descrição | Padrão |
|---|---|---|
| `CANAL` | `telegram` ou `twilio` | `twilio` |
| `MENSAGEM_BOAS_VINDAS` | Resposta fixa de `/start` e `/help` | texto padrão |
| `TELEGRAM_BOT_TOKEN` | Token do bot criado no @BotFather | — |
| `TELEGRAM_DRY_RUN` | Se `true`, loga a mensagem em vez de enviar de verdade | `true` |
| `TELEGRAM_SEGREDO_WEBHOOK` | Segredo do `setWebhook`; obrigatório fora de dry-run | — |
| `TWILIO_DRY_RUN` | Se `true`, loga a mensagem em vez de enviar de verdade | `true` |
| `TWILIO_VALIDAR_ASSINATURA` | Confere o `X-Twilio-Signature`; obrigatório fora de dry-run | `false` |
| `EXPOR_FERRAMENTAS_DE_TESTE` | Liga `/api/mensagem` e `/docs`. Deixe `false` em produção | `false` |
| `LIMITE_MENSAGENS_POR_MINUTO` | Teto por remetente; `0` desliga | `20` |
| `LIMITE_CARACTERES_PERGUNTA` | Corte da pergunta antes do embedding e do prompt | `1000` |
| `PROVEDOR_LLM` | `openai`, `gemini` ou `demo` (responde sem API, para testes) | `openai` |
| `PROVEDOR_EMBEDDING` | `local` (sentence-transformers) ou `gemini` | `local` |
| `MODELO_EMBEDDING` | Modelo do provedor **local** | `intfloat/multilingual-e5-large` |
| `GEMINI_API_KEY` | Chave única, serve para o LLM e para o embedding | — |
| `GEMINI_MODELO` | Modelo de LLM do Gemini | `gemini-3.5-flash-lite` |
| `GEMINI_ORCAMENTO_RACIOCINIO` | `thinkingBudget`; vazio omite o campo | vazio |
| `GEMINI_MODELO_EMBEDDING` | Modelo de embedding do Gemini | `gemini-embedding-001` |
| `GEMINI_DIMENSOES_EMBEDDING` | Dimensões da saída (3072, 1536 ou 768) | `768` |
| `LIMIAR_SIMILARIDADE` | Primeira barreira: similaridade mínima (0–1). **Vazio usa o limiar calibrado do provedor de embedding ativo** | vazio |
| `REINDEXAR_NO_STARTUP` | Reconstrói o índice na subida se ele não refletir o FAQ | `true` |

### Calibrando o transbordo

Foi a parte mais interessante do projeto, e a que mais mudou de rumo com a
medição.

A ideia inicial era simples: um limiar de similaridade decide se o bot responde.
Para escolher o modelo de embedding, medi cada candidato com perguntas
acentuadas — metade parafraseadas a partir da FAQ, metade completamente fora
dela — conferindo primeiro se o retrieval traz a entrada certa e depois onde
ficam as faixas de similaridade.

| Modelo | Acerto do retrieval | Faixa das corretas | Faixa das de fora | Sobreposição | Limiar |
|---|---|---|---|---|---|
| `multilingual-e5-large` (local) | **10/10** | 0.839 – 0.891 | 0.735 – 0.845 | 0.006 | `0.80` |
| `paraphrase-multilingual-MiniLM-L12-v2` (local) | 6/10 | 0.390 – 0.839 | 0.039 – 0.524 | 0.134 | descartado |
| `gemini-embedding-001` @768d (API) | **10/10** | 0.798 – 0.860 | 0.584 – 0.705 | **0.000** | `0.77` |

> As duas primeiras linhas são da medição original, com 20 perguntas (10
> cobertas, 10 fora). A do Gemini usou o corpus de 16 versionado em
> `app/rag/calibracao.py` — 10 cobertas e as 6 de fora que ficaram como
> regressão. As colunas de "fora" não são estritamente comparáveis entre as
> duas medições; as conclusões abaixo não dependem dessa diferença.

**1. Retrieval é o primeiro filtro, e ele elimina candidato.** O e5 e o Gemini
acertam as 10. O MiniLM erra 4 — manda "o frete é grátis?" para a garantia e
"posso pagar com pix?" para o reembolso. Limiar nenhum conserta um modelo que
busca a resposta errada, então ele saiu na primeira coluna.

> O erro de método vale registrar: na primeira medição escrevi as perguntas sem
> acentuação, e o MiniLM pareceu separar as faixas com folga. Com acentuação —
> que é como clientes de verdade escrevem — o resultado se inverte: "o frete é
> grátis?" cai de 0.586 para 0.454 e passa a casar com a entrada errada. Uma
> medição descuidada quase fixou o modelo pior no projeto.

**2. Separar por limiar depende do modelo — e essa conclusão foi revisada.**
A versão anterior deste README afirmava que *nenhum* modelo separa as faixas, e
tratava isso como propriedade do problema. Era propriedade do e5.

O e5 é treinado com negativos in-batch e temperatura, o que comprime as
similaridades numa faixa alta e estreita: "quanto é 2 + 2?" pontua 0.832,
praticamente o mesmo que a pergunta legítima mais fraca (0.839). Ele **ordena**
muito bem, mas o valor absoluto quase não discrimina — sobreposição de 0.006.

O `gemini-embedding-001` recupera igualmente bem **e** separa: sobreposição
0.000, com um intervalo livre de 0.093 entre a pergunta de fora mais alta
(0.705, "vocês têm vaga de emprego?") e a legítima mais fraca (0.798). As
quatro perguntas de fronteira que o e5 empurrava para a segunda barreira —
inclusive "vocês vendem passagem aérea?" — morrem na primeira, sem gastar uma
chamada de LLM.

**3. A segunda barreira fica — como defesa em profundidade.** É a mudança de
justificativa que a medição do Gemini forçou:

| | Antes (só o e5 medido) | Agora |
|---|---|---|
| Por que a 2ª barreira existe | A 1ª **falha**: as faixas se sobrepõem e o limiar não separa | A 1ª **pode falhar**: separa neste corpus, mas 6 perguntas não provam cobertura |
| O que ela é | Correção necessária | Camada redundante, de custo zero |

Ela continua não custando chamada extra — o LLM já está lendo o contexto para
responder, e o prompt de sistema manda devolver só `TRANSBORDO` quando o
contexto não serve. Manter uma defesa que não custa nada e cobre o caso que a
medição não viu é barato; remover porque 6 perguntas deram certo seria confiar
demais na amostra.

O que mudou é **quanto trabalho cada barreira faz**, e isso agora está explícito
nos testes: `test_pergunta_de_fronteira_nunca_e_respondida` exige o invariante
que vale nos dois regimes (pergunta fora do domínio não é respondida, seja qual
for a barreira que a pare), e o teste específico da 2ª se pula quando a 1ª já
barrou tudo.

> O achado veio de um teste, não de leitura de código. O
> `test_a_primeira_barreira_sozinha_nao_basta` original afirmava que ao menos
> uma pergunta de fronteira passa do limiar, com o comentário "se algum dia a
> similaridade passar a separar essas perguntas sozinha, este teste falha". Ele
> falhou na primeira execução com o Gemini. O teste estava certo em falhar; o
> que estava errado era tratar uma propriedade do e5 como propriedade do
> sistema.

#### Trocar de LLM também mexe na segunda barreira

Menos óbvio que trocar o embedding, e sem número para calibrar: a 2ª barreira é
um julgamento do modelo, e modelos julgam com estritez diferente.

Rodando a suíte de integração com o Gemini, `vocês atendem no domingo?` passou
a transbordar — com o retrieval correto e 0.860 de similaridade. A resposta do
FAQ lista "de segunda a sexta" e "aos sábados", sem citar domingo, e o
`gemini-3.5-flash-lite` leu isso como informação ausente. Determinístico: 5/5
transbordos, contra 0/5 do `gpt-4o-mini`, que respondia "não atendemos aos
domingos".

Os dois estavam obedecendo o prompt, que mandava usar **apenas** o contexto e
transbordar quando ele respondesse "só parcialmente". O prompt é que estava
incompleto: responder "não" a partir de uma lista completa é ler o contexto,
não inventar. A regra entrou no `PROMPT_SISTEMA`, com a ressalva de que listas
parciais ou exemplos continuam valendo transbordo.

O que **não** resolveria: adicionar `vocês atendem domingo?` às `variacoes` do
FAQ. Variações atuam no retrieval, e o retrieval já estava certo — o problema
morava depois da busca. É uma distinção fácil de errar.

Fica um resíduo honesto: `aceitam cheque?` sobre a resposta "Aceitamos Pix,
boleto bancário e cartão de crédito" ainda transborda no Gemini e é respondida
pela OpenAI. A lista não diz "apenas", então tratá-la como fechada é um passo
que um modelo dá e o outro recusa — e a recusa é a leitura mais segura das
duas. É ambiguidade do dataset, não do prompt.

**Se trocar de modelo de embedding, recalibre.** O protocolo acima virou
script, porque a troca deixou de ser hipotética quando o deploy exigiu sair do
modelo local:

```bash
python -m scripts.calibrar_limiar
```

Ele indexa o FAQ num ChromaDB descartável com o `PROVEDOR_EMBEDDING` ativo,
roda o corpus de calibração (`app/rag/calibracao.py` — 16 perguntas acentuadas:
10 cobertas pela FAQ e 6 fora), e imprime as faixas, a sobreposição, o limiar
sugerido e a linha pronta para a tabela acima. Depois: coloque o número no
provedor correspondente em `app/rag/embedding/`, reindexe com `--recriar` — os
vetores dos dois modelos não são comparáveis — e confirme com
`pytest -m integracao`.

O limiar **não** fica só no `.env`. Cada provedor de embedding carrega o valor
medido para ele (`limiar_calibrado`), e `LIMIAR_SIMILARIDADE` vazio usa esse
valor. Sem isso, o modo de falha óbvio seria trocar `PROVEDOR_EMBEDDING` e
herdar em silêncio um limiar calibrado para outro modelo — com o transbordo
continuando a "funcionar", só que decidindo errado.

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
pytest                  # 158 testes unitários, ~1s, sem tocar em modelo ou API
pytest -m integracao    # 38 testes com dataset e embeddings reais
```

Os de integração rodam contra o `PROVEDOR_EMBEDDING` configurado, e os dois
provedores passam:

| Provedor | Limiar | Resultado |
|---|---|---|
| `local` (e5) | `0.80` | 38 passam — a 2ª barreira pega as 4 perguntas de fronteira |
| `gemini` | `0.77` | 37 passam, 1 pula — a 1ª barreira já barrou toda a fronteira |

Os unitários usam dublês nas fronteiras (vector store, LLM, embedding, canal) e
cobrem a conversão de distância em similaridade, as quatro causas de
transbordo, o reconhecimento do sinal do LLM, a validação e o upsert do
dataset, e os dois webhooks HTTP — incluindo o parsing do update do Telegram, a
checagem do segredo, o descarte de updates que não são mensagem, o
curto-circuito dos comandos (provado pelo dublê: nem busca nem LLM são
chamados) e a expansão das variações na ingestão.

O que as fronteiras novas acrescentaram, tudo com o transporte HTTP
substituído — nenhum teste chama a API do Gemini:

- **`test_embedding.py`** — que documento e consulta são marcados de forma
  diferente (`passage:`/`query:` no e5, `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY`
  no Gemini) e que a saída é normalizada. Os dois são erros que não quebram
  nada: o sistema continua devolvendo vetores, só que busca pior e o limiar
  passa a comparar números sem escala.
- **`test_gemini.py`** — retry em 429 e falha de rede, desistência imediata em
  credencial inválida, a chave no cabeçalho e não na URL, e a tradução de
  resposta bloqueada ou vazia para `ErroGeracao` (que o serviço já transborda).
- **`test_dependencias.py`** — que as três fronteiras são mesmo escolhidas por
  variável, e que um limiar vazio herda o do provedor ativo.
- **`test_ingest.py`** — a decisão de reindexar na subida: indexa com a base
  vazia, não gasta chamada nenhuma quando o índice já reflete o FAQ, e
  reindexa quando o dataset ganhou pergunta.
- **`test_telegram.py`** — que três entregas do mesmo update produzem uma
  resposta só, e que o mesmo `message_id` em conversas diferentes **não** é
  confundido.

Os de integração indexam o `faq_dataset.json` de verdade e verificam que o
retrieval recupera a entrada certa e que as faixas de similaridade continuam
onde a calibração assumiu — **é o teste que pegou o erro de calibração descrito
acima**, e foi também o que mostrou que o embedding do Gemini separa as faixas
sozinho. Nenhum teste com dublê pegaria nem um nem outro.

Os que exigem a segunda barreira precisam de um LLM real (`PROVEDOR_LLM=openai`
ou `gemini`, com a chave correspondente) e são pulados sem ele.

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
