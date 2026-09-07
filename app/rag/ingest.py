"""Carga do dataset de FAQ no vector store.

O id de cada documento é derivado da pergunta, então reindexar o mesmo dataset
faz *upsert* (atualiza) em vez de duplicar. Na prática: editar uma resposta e
rodar a ingestão de novo já atualiza a base — não existe re-treino de modelo.
"""

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.config import Configuracoes, obter_configuracoes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemFAQ:
    """Uma entrada do FAQ, no formato padrão do ``faq_dataset.json``."""

    pergunta: str
    resposta: str
    categoria: str = "geral"
    tags: tuple[str, ...] = ()
    # Outras formas de perguntar a mesma coisa: abreviações ("NF"), gíria e
    # escrita sem acento. Cada uma vira um documento próprio apontando para a
    # MESMA resposta — ver `indexar`. Tags não resolveriam isso: são metadado,
    # não entram no embedding.
    variacoes: tuple[str, ...] = ()

    @staticmethod
    def _hash(chave: str) -> str:
        return hashlib.sha1(chave.strip().lower().encode("utf-8")).hexdigest()[:16]

    @property
    def id(self) -> str:
        """Id estável derivado da pergunta canônica (é o que permite o upsert)."""
        return self._hash(self.pergunta)

    def id_da_variacao(self, variacao: str) -> str:
        """Id estável da variação, derivado do par (pergunta, variação).

        Amarrar à pergunta evita que a mesma variação escrita em duas entradas
        colida no upsert e apague uma delas.
        """
        return self._hash(f"{self.pergunta}|{variacao}")

    def metadados(self) -> dict[str, str]:
        """Metadados do documento. O Chroma só aceita valores escalares."""
        return {
            "pergunta": self.pergunta,
            "resposta": self.resposta,
            "categoria": self.categoria,
            "tags": ", ".join(self.tags),
        }


def carregar_faq(caminho: Path) -> list[ItemFAQ]:
    """Lê e valida o dataset de FAQ em JSON."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if not isinstance(dados, list):
        raise ValueError("O dataset de FAQ deve ser uma lista de objetos JSON.")

    itens: list[ItemFAQ] = []
    for indice, bruto in enumerate(dados):
        if not isinstance(bruto, dict):
            raise ValueError(f"Item {indice} do FAQ não é um objeto JSON.")

        pergunta = str(bruto.get("pergunta") or "").strip()
        resposta = str(bruto.get("resposta") or "").strip()
        if not pergunta or not resposta:
            raise ValueError(
                f"Item {indice} do FAQ precisa dos campos 'pergunta' e 'resposta'."
            )

        tags = tuple(
            str(tag).strip() for tag in (bruto.get("tags") or []) if str(tag).strip()
        )
        variacoes = tuple(
            str(variacao).strip()
            for variacao in (bruto.get("variacoes") or [])
            if str(variacao).strip()
        )
        itens.append(
            ItemFAQ(
                pergunta=pergunta,
                resposta=resposta,
                categoria=str(bruto.get("categoria") or "geral").strip(),
                tags=tags,
                variacoes=variacoes,
            )
        )

    if not itens:
        raise ValueError("O dataset de FAQ está vazio.")
    return itens


def indexar(itens: Sequence[ItemFAQ], vector_store) -> int:
    """Grava (ou atualiza) os itens no vector store. Devolve quantos documentos.

    Cada variação vira um documento com embedding próprio, mas carregando os
    metadados da entrada canônica: quem casa com a busca é o texto da variação,
    quem responde continua sendo a pergunta original.
    """
    textos: list[str] = []
    metadados: list[dict[str, str]] = []
    ids: list[str] = []

    for item in itens:
        textos.append(item.pergunta)
        metadados.append(item.metadados())
        ids.append(item.id)
        for variacao in item.variacoes:
            textos.append(variacao)
            metadados.append(item.metadados())
            ids.append(item.id_da_variacao(variacao))

    if not textos:
        return 0

    vector_store.add_texts(texts=textos, metadatas=metadados, ids=ids)
    return len(textos)


def ids_do_dataset(itens: Sequence[ItemFAQ]) -> set[str]:
    """Todos os ids que o dataset deveria ter no índice (perguntas + variações)."""
    ids = set()
    for item in itens:
        ids.add(item.id)
        ids.update(item.id_da_variacao(variacao) for variacao in item.variacoes)
    return ids


def garantir_indice(config: Configuracoes, vector_store) -> int:
    """Indexa o FAQ se o índice ainda não o refletir. Devolve quantos gravou.

    Chamada na subida da aplicação. É o que substitui um volume persistente no
    Render, cujo plano Free não tem disco e apaga o ``chroma_db`` a cada deploy,
    restart ou hibernação — indexar 68 documentos custa segundos, e o índice
    sempre nasce coerente com o ``faq_dataset.json`` que subiu junto.

    Compara ids em vez de contar documentos: assim uma pergunta nova no JSON é
    detectada, e reiniciar o serviço com o índice já pronto (o caso local) não
    gasta uma chamada de embedding sequer.
    """
    itens = carregar_faq(config.caminho_faq)
    esperados = ids_do_dataset(itens)
    existentes = set(vector_store.get(include=[])["ids"])

    if esperados <= existentes:
        logger.info(
            "Índice já contém os %d documentos do FAQ; ingestão dispensada.",
            len(esperados),
        )
        return 0

    logger.info(
        "Índice incompleto (%d de %d documentos): indexando o FAQ.",
        len(esperados & existentes),
        len(esperados),
    )
    return indexar(itens, vector_store)


def executar_ingestao(
    config: Configuracoes, recriar: bool = False, embedding=None
) -> int:
    """Fluxo completo: lê o JSON, gera os embeddings e grava no Chroma."""
    from app.dependencias import criar_embedding
    from app.rag.vector_store import criar_vector_store

    if recriar and config.diretorio_chroma.exists():
        logger.info("Removendo base vetorial anterior em %s", config.diretorio_chroma)
        shutil.rmtree(config.diretorio_chroma)

    itens = carregar_faq(config.caminho_faq)
    vector_store = criar_vector_store(config, embedding or criar_embedding(config))
    total = indexar(itens, vector_store)
    logger.info(
        "%d documentos indexados na coleção %s (%d perguntas + %d variações).",
        total,
        config.colecao_chroma,
        len(itens),
        total - len(itens),
    )
    return total


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Indexa o FAQ no ChromaDB.")
    parser.add_argument(
        "--recriar",
        action="store_true",
        help="apaga a base vetorial antes de indexar (carga completa)",
    )
    parser.add_argument(
        "--faq", type=Path, default=None, help="caminho alternativo do faq_dataset.json"
    )
    argumentos = parser.parse_args()

    config = obter_configuracoes()
    if argumentos.faq:
        config = config.model_copy(update={"caminho_faq": argumentos.faq})

    total = executar_ingestao(config, recriar=argumentos.recriar)
    print(f"{total} documentos indexados em {config.diretorio_chroma}")


if __name__ == "__main__":
    main()
