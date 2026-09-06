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

    @property
    def id(self) -> str:
        """Id estável derivado da pergunta canônica (é o que permite o upsert)."""
        chave = self.pergunta.strip().lower().encode("utf-8")
        return hashlib.sha1(chave).hexdigest()[:16]

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
        itens.append(
            ItemFAQ(
                pergunta=pergunta,
                resposta=resposta,
                categoria=str(bruto.get("categoria") or "geral").strip(),
                tags=tags,
            )
        )

    if not itens:
        raise ValueError("O dataset de FAQ está vazio.")
    return itens


def indexar(itens: Sequence[ItemFAQ], vector_store) -> int:
    """Grava (ou atualiza) os itens no vector store. Devolve quantos foram."""
    if not itens:
        return 0

    vector_store.add_texts(
        texts=[item.pergunta for item in itens],
        metadatas=[item.metadados() for item in itens],
        ids=[item.id for item in itens],
    )
    return len(itens)


def executar_ingestao(config: Configuracoes, recriar: bool = False) -> int:
    """Fluxo completo: lê o JSON, gera os embeddings e grava no Chroma."""
    from app.rag.vector_store import criar_vector_store

    if recriar and config.diretorio_chroma.exists():
        logger.info("Removendo base vetorial anterior em %s", config.diretorio_chroma)
        shutil.rmtree(config.diretorio_chroma)

    itens = carregar_faq(config.caminho_faq)
    total = indexar(itens, criar_vector_store(config))
    logger.info("%d perguntas indexadas na coleção %s.", total, config.colecao_chroma)
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
    print(f"{total} perguntas indexadas em {config.diretorio_chroma}")


if __name__ == "__main__":
    main()
