"""Contrato do provedor de embeddings.

Terceira fronteira trocável do projeto, ao lado de ``ProvedorMensageria`` e
``ProvedorLLM``. Ela nasceu de uma restrição concreta de deploy: o modelo local
carrega ~2,2 GB de pesos e o ``torch`` junto, o que não cabe nos 512 MB do
plano Free do Render. Atrás desta interface, trocar embedding local por
embedding de API é uma variável de ambiente — o retriever, o handoff e a
ingestão não sabem qual dos dois está rodando.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings


class ErroEmbedding(RuntimeError):
    """Falha ao gerar embeddings (rede, cota, credencial inválida...)."""


class ProvedorEmbedding(ABC):
    """Converte texto em vetores, para indexar e para buscar.

    Documento e consulta têm métodos separados de propósito. Os modelos usados
    aqui tratam os dois lados de forma diferente — o e5 espera os prefixos
    ``passage:``/``query:``, o Gemini espera ``taskType`` ``RETRIEVAL_DOCUMENT``
    ou ``RETRIEVAL_QUERY`` — e uma interface com um método só esconderia essa
    assimetria. O modelo continuaria "funcionando", só que buscando pior.
    """

    @abstractmethod
    def vetorizar_documentos(self, textos: Sequence[str]) -> list[list[float]]:
        """Vetoriza os textos que vão para o índice."""

    @abstractmethod
    def vetorizar_consulta(self, texto: str) -> list[float]:
        """Vetoriza a pergunta do cliente."""

    @property
    @abstractmethod
    def limiar_calibrado(self) -> float:
        """Limiar da 1a barreira de transbordo medido para este modelo.

        Mora no provedor, e não só no ``.env``, porque limiar e modelo de
        embedding são uma coisa só: cada modelo distribui as similaridades numa
        faixa própria, então um número calibrado para o e5 não significa nada
        no Gemini. Deixar o default junto do modelo evita o modo de falha mais
        provável desta troca — mudar ``PROVEDOR_EMBEDDING`` e esquecer o
        limiar, quebrando o transbordo em silêncio. ``LIMIAR_SIMILARIDADE`` no
        ``.env`` continua tendo a palavra final.
        """

    @property
    def descricao(self) -> str:
        """Identificação do modelo ativo, para o log de inicialização."""
        return type(self).__name__


class AdaptadorLangChain(Embeddings):
    """Expõe um ``ProvedorEmbedding`` na interface que o Chroma consome.

    O ChromaDB (via ``langchain-chroma``) fala com a interface ``Embeddings``
    do LangChain. Em vez de fazer cada provedor herdar dela — o que amarraria a
    nossa fronteira ao vocabulário de uma biblioteca de terceiros — o
    acoplamento fica confinado neste adaptador.
    """

    def __init__(self, provedor: ProvedorEmbedding) -> None:
        self._provedor = provedor

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._provedor.vetorizar_documentos(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._provedor.vetorizar_consulta(text)


def normalizar(vetor: Sequence[float]) -> list[float]:
    """Devolve o vetor com norma 1.

    Não é detalhe de implementação: a coleção do Chroma usa distância de
    cosseno e o ``Retriever`` converte distância em similaridade com
    ``1 - distância``. Essa conta só cai na faixa 0–1 se os vetores forem
    unitários — com vetores não normalizados o limiar de transbordo passa a
    comparar números sem escala fixa.
    """
    norma = math.sqrt(sum(valor * valor for valor in vetor))
    if norma == 0:
        return list(vetor)
    return [valor / norma for valor in vetor]
