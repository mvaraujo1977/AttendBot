"""Provedor de embeddings local, rodando o modelo na própria máquina.

É o padrão de desenvolvimento: nenhuma chave de API, nenhuma chamada de rede
depois do primeiro download, e o retrieval medido no README (10/10). O preço é
tamanho — ``torch`` mais ~2,2 GB de pesos —, que é exatamente o que não cabe
no plano Free do Render. Por isso ele não vem no ``requirements.txt`` de
produção: está em ``requirements-local.txt``.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache

from app.rag.embedding.base import ErroEmbedding, ProvedorEmbedding

logger = logging.getLogger(__name__)

MODELO_PADRAO = "intfloat/multilingual-e5-large"

# Calibrado com o protocolo do README (20 perguntas acentuadas, 10 cobertas
# pela FAQ e 10 fora). Fica deliberadamente baixo: o e5 comprime as
# similaridades numa faixa alta e estreita, então este limiar é um filtro
# barato, não o juiz. Quem decide os casos de fronteira é a 2a barreira, no LLM.
LIMIAR_CALIBRADO = 0.80

INSTRUCAO_FALTA_DEPENDENCIA = (
    "O provedor de embedding 'local' exige sentence-transformers e torch, que "
    "ficam fora do requirements.txt de produção. Instale com "
    "`pip install -r requirements-local.txt`, ou use PROVEDOR_EMBEDDING=gemini."
)


class ProvedorEmbeddingLocal(ProvedorEmbedding):
    """Roda um modelo do ``sentence-transformers`` no processo.

    Aplica os prefixos que a família E5 espera: ``query:`` nas consultas e
    ``passage:`` nos documentos indexados. Sem eles a qualidade da busca cai
    bastante, e é um detalhe fácil de esquecer porque o modelo continua
    devolvendo vetores plausíveis.
    """

    def __init__(
        self, modelo: str = MODELO_PADRAO, limiar: float = LIMIAR_CALIBRADO
    ) -> None:
        self._modelo = modelo
        self._limiar = limiar
        self._usa_prefixos_e5 = "e5" in modelo.lower()

    @property
    def limiar_calibrado(self) -> float:
        return self._limiar

    @property
    def descricao(self) -> str:
        return f"local:{self._modelo}"

    def vetorizar_documentos(self, textos: Sequence[str]) -> list[list[float]]:
        return self._codificar(
            [self._prefixar(texto, "passage") for texto in textos]
        )

    def vetorizar_consulta(self, texto: str) -> list[float]:
        return self._codificar([self._prefixar(texto, "query")])[0]

    # -- internos -------------------------------------------------------------

    def _prefixar(self, texto: str, papel: str) -> str:
        return f"{papel}: {texto}" if self._usa_prefixos_e5 else texto

    def _codificar(self, textos: list[str]) -> list[list[float]]:
        vetores = _carregar_modelo(self._modelo).encode(
            textos, normalize_embeddings=True
        )
        return [[float(valor) for valor in vetor] for vetor in vetores]


@lru_cache
def _carregar_modelo(nome: str):
    """Carrega o modelo uma vez por processo (o download é caro, e o load também)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as erro:  # pragma: no cover - depende do ambiente
        raise ErroEmbedding(INSTRUCAO_FALTA_DEPENDENCIA) from erro

    logger.info("Carregando modelo de embedding local: %s", nome)
    return SentenceTransformer(nome)
