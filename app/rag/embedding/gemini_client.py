"""Provedor de embeddings via API do Gemini.

É o provedor de produção. Sem ``torch`` e sem pesos em disco, o serviço passa a
caber nos 512 MB do plano Free do Render — a razão de esta fronteira existir.
Em troca, cada busca vira uma chamada de rede, e o índice precisa ser
reconstruído quando o sistema de arquivos efêmero apaga o Chroma.
"""

import logging
from collections.abc import Sequence

from app.gemini_api import chamar
from app.rag.embedding.base import ErroEmbedding, ProvedorEmbedding, normalizar

logger = logging.getLogger(__name__)

MODELO_PADRAO = "gemini-embedding-001"

# O gemini-embedding-001 usa Matryoshka: dá para truncar a saída (3072, 1536,
# 768) trocando um pouco de qualidade por payload menor. 768 mantém o índice e
# as respostas leves — o que importa quando cada cold start reindexa a base.
DIMENSOES_PADRAO = 768

# Limiar da 1a barreira, medido para ESTE modelo (setembro/2026,
# `python -m scripts.calibrar_limiar`). Não é o 0.80 do e5: cada modelo
# distribui as similaridades numa faixa própria.
#
#   retrieval  10/10
#   cobertas   0.798 - 0.860
#   fora       0.584 - 0.705   <- sem sobreposição, ao contrário do e5
#
# O intervalo livre entre 0.705 e 0.798 é largo, então aqui a 1a barreira
# discrimina de verdade, e não só filtra o obviamente distante. Ver a seção
# "Calibrando o transbordo" no README.
LIMIAR_CALIBRADO = 0.77

# A assimetria documento/consulta do e5 (prefixos `passage:`/`query:`) aparece
# aqui como `taskType`: é a mesma ideia, dita no vocabulário do Gemini.
TAREFA_DOCUMENTO = "RETRIEVAL_DOCUMENT"
TAREFA_CONSULTA = "RETRIEVAL_QUERY"

# A API aceita até 100 requisições por chamada de batch.
TAMANHO_LOTE = 100
TIMEOUT_SEGUNDOS = 60.0


class ProvedorEmbeddingGemini(ProvedorEmbedding):
    """Chama ``embedContent`` / ``batchEmbedContents`` da API do Gemini."""

    def __init__(
        self,
        api_key: str | None,
        modelo: str = MODELO_PADRAO,
        dimensoes: int = DIMENSOES_PADRAO,
        limiar: float = LIMIAR_CALIBRADO,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não configurada. Defina a chave no .env ou use "
                "PROVEDOR_EMBEDDING=local para rodar o modelo offline."
            )
        self._api_key = api_key
        self._modelo = modelo if modelo.startswith("models/") else f"models/{modelo}"
        self._dimensoes = dimensoes
        self._limiar = limiar

    @property
    def limiar_calibrado(self) -> float:
        return self._limiar

    @property
    def descricao(self) -> str:
        return f"gemini:{self._modelo.removeprefix('models/')}@{self._dimensoes}d"

    def vetorizar_documentos(self, textos: Sequence[str]) -> list[list[float]]:
        vetores: list[list[float]] = []
        for inicio in range(0, len(textos), TAMANHO_LOTE):
            lote = textos[inicio : inicio + TAMANHO_LOTE]
            vetores.extend(self._batch(lote))
        return vetores

    def vetorizar_consulta(self, texto: str) -> list[float]:
        resposta = self._chamar(
            "embedContent", self._requisicao(texto, TAREFA_CONSULTA)
        )
        return self._extrair(resposta.get("embedding"))

    # -- internos -------------------------------------------------------------

    def _batch(self, textos: Sequence[str]) -> list[list[float]]:
        resposta = self._chamar(
            "batchEmbedContents",
            {
                "requests": [
                    self._requisicao(texto, TAREFA_DOCUMENTO) for texto in textos
                ]
            },
        )
        embeddings = resposta.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(textos):
            raise ErroEmbedding(
                f"O Gemini devolveu {len(embeddings or [])} embeddings para "
                f"{len(textos)} textos."
            )
        return [self._extrair(item) for item in embeddings]

    def _requisicao(self, texto: str, tarefa: str) -> dict:
        return {
            "model": self._modelo,
            "content": {"parts": [{"text": texto}]},
            "taskType": tarefa,
            "outputDimensionality": self._dimensoes,
        }

    def _chamar(self, metodo: str, corpo: dict) -> dict:
        try:
            return chamar(
                f"{self._modelo}:{metodo}",
                corpo,
                self._api_key,
                timeout=TIMEOUT_SEGUNDOS,
            )
        except RuntimeError as erro:
            raise ErroEmbedding(f"Falha ao chamar o Gemini: {erro}") from erro

    def _extrair(self, embedding) -> list[float]:
        valores = (embedding or {}).get("values")
        if not valores:
            raise ErroEmbedding("O Gemini devolveu um embedding vazio.")
        # Só a saída de 3072 dimensões vem normalizada; as truncadas do
        # Matryoshka, não. Sem normalizar, a conversão `1 - distância` do
        # Retriever deixa de produzir uma similaridade de 0 a 1 e o limiar
        # passa a comparar números sem escala.
        return normalizar([float(valor) for valor in valores])
