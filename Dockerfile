# Python 3.11 fixado. Não é preferência: chromadb e sentence-transformers ainda
# não publicam wheels para o 3.14, e fora do container o projeto já dependia de
# lembrar disso. Aqui a versão certa vem junto com o código.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# As dependências entram antes do código para que editar o app não invalide a
# camada de instalação — o que encurta bastante o build a cada deploy.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# requirements-local.txt (torch + sentence-transformers) fica de fora de
# propósito: são ~2,5 GB, e o plano Free do Render tem 512 MB de RAM. Em
# produção o embedding vem da API do Gemini (PROVEDOR_EMBEDDING=gemini).

COPY app ./app

# Usuário sem privilégios. O chroma_db é criado aqui, e não montado: o
# filesystem do Render é efêmero e o índice é reconstruído na subida a partir
# do faq_dataset.json que veio na imagem (ver garantir_indice).
RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /app/chroma_db \
    && chown -R bot:bot /app
USER bot

# Só documental: quem manda é o $PORT abaixo.
EXPOSE 8000

# O Render injeta PORT e espera que o processo escute nela. `sh -c` para
# expandir a variável, `exec` para o uvicorn virar o PID 1 e receber o SIGTERM
# do shutdown em vez de ser morto no timeout.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
