"""Mede o limiar de transbordo para o provedor de embedding ativo.

Rode isto ao trocar ``PROVEDOR_EMBEDDING`` ou ``MODELO_EMBEDDING``. Ele indexa
o FAQ real num ChromaDB temporário, roda o corpus de calibração (16 perguntas
com acentuação: 10 cobertas pela FAQ e 6 fora) e imprime as faixas de
similaridade mais o limiar sugerido.

    python -m scripts.calibrar_limiar

O número impresso é ponto de partida, não veredito: coloque-o em
``LIMIAR_SIMILARIDADE`` (ou no ``limiar`` do provedor, em
``app/rag/embedding/``) e confirme com ``pytest -m integracao``.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import obter_configuracoes  # noqa: E402
from app.dependencias import criar_embedding, resolver_limiar  # noqa: E402
from app.rag.calibracao import Medicao, medir  # noqa: E402
from app.rag.ingest import executar_ingestao  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.vector_store import criar_vector_store  # noqa: E402


def _tabela(medicao: Medicao) -> str:
    linhas = ["", "  Perguntas COBERTAS pela FAQ (devem passar do limiar):", ""]
    for amostra in sorted(medicao.cobertas, key=lambda a: a.similaridade):
        marca = "ok " if amostra.acertou else "ERRO"
        linhas.append(f"    {marca} {amostra.similaridade:.3f}  {amostra.pergunta}")
        if not amostra.acertou:
            linhas.append(f"           recuperou: {amostra.recuperada}")
            linhas.append(f"           esperava:  {amostra.esperada}")

    linhas += ["", "  Perguntas FORA da FAQ (devem transbordar):", ""]
    for amostra in sorted(medicao.fora, key=lambda a: -a.similaridade):
        linhas.append(f"         {amostra.similaridade:.3f}  {amostra.pergunta}")
    return "\n".join(linhas)


def _resumo(medicao: Medicao, descricao: str, limiar_atual: float) -> str:
    minima, maxima = medicao.faixa_cobertas
    minima_fora, maxima_fora = medicao.faixa_fora
    sugerido = medicao.limiar_sugerido

    linhas = [
        "",
        "  " + "-" * 68,
        f"  Modelo              {descricao}",
        f"  Acerto do retrieval {medicao.acertos}/{len(medicao.cobertas)}",
        f"  Faixa das corretas  {minima:.3f} - {maxima:.3f}",
        f"  Faixa das de fora   {minima_fora:.3f} - {maxima_fora:.3f}",
        f"  Sobreposição        {medicao.sobreposicao:.3f}",
        f"  Limiar atual        {limiar_atual:.2f}",
        f"  Limiar SUGERIDO     {sugerido:.2f}",
        "  " + "-" * 68,
        "",
        "  Linha para o README:",
        f"  | {descricao} | {medicao.acertos}/{len(medicao.cobertas)} | "
        f"{minima:.3f} – {maxima:.3f} | {minima_fora:.3f} – {maxima_fora:.3f} | "
        f"{medicao.sobreposicao:.3f} |",
        "",
    ]

    if medicao.acertos < len(medicao.cobertas):
        linhas += [
            "  ATENÇÃO: o retrieval errou alguma entrada. Limiar nenhum conserta",
            "  isso — o modelo está buscando a resposta errada. Descarte-o.",
            "",
        ]
    if medicao.sobreposicao > 0:
        linhas += [
            "  As faixas se sobrepõem: nenhum limiar separa cobertas de não",
            "  cobertas sozinho. É o esperado, e é para isso que existe a 2a",
            "  barreira (o LLM julgando o contexto). Ver README.",
            "",
        ]
    return "\n".join(linhas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verboso", action="store_true", help="mostra os logs da ingestão"
    )
    argumentos = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if argumentos.verboso else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = obter_configuracoes()
    embedding = criar_embedding(config)

    print(f"\nCalibrando: {embedding.descricao}")
    print("Indexando o FAQ num ChromaDB temporário...")

    # Índice descartável: vetores de modelos diferentes não são comparáveis, e
    # medir dentro do chroma_db de trabalho misturaria os dois.
    # ignore_cleanup_errors: no Windows o Chroma segura o handle do índice
    # depois do uso, e a limpeza do diretório falharia com o resultado na tela.
    with tempfile.TemporaryDirectory(
        prefix="calibracao-", ignore_cleanup_errors=True
    ) as diretorio:
        temporaria = config.model_copy(update={"diretorio_chroma": Path(diretorio)})
        executar_ingestao(temporaria, embedding=embedding)
        retriever = Retriever(
            criar_vector_store(temporaria, embedding), top_k=config.top_k
        )
        medicao = medir(retriever)

    print(_tabela(medicao))
    print(_resumo(medicao, embedding.descricao, resolver_limiar(config, embedding)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
