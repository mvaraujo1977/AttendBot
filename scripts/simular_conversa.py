"""Conversa com o bot pelo terminal, sem Twilio e sem servidor HTTP.

Uso:
    python -m scripts.simular_conversa                 # modo interativo
    python -m scripts.simular_conversa "qual o frete?" # pergunta única
"""

import logging
import sys

from app.config import obter_configuracoes
from app.dependencias import criar_servico


def _imprimir(servico, pergunta: str) -> None:
    resposta = servico.responder(pergunta)
    marcador = "TRANSBORDO" if resposta.transbordo else "BOT"
    detalhe = f"similaridade={resposta.similaridade:.3f}"
    if resposta.motivo:
        detalhe += f", motivo={resposta.motivo}"
    print(f"\n[{marcador}] ({detalhe})\n{resposta.texto}\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    config = obter_configuracoes()
    print(f"Carregando o bot da {config.nome_empresa} (pode demorar no 1o uso)...")
    servico = criar_servico(config)

    if len(sys.argv) > 1:
        _imprimir(servico, " ".join(sys.argv[1:]))
        return

    print(f"Pronto. Limiar de transbordo: {servico.limiar:.2f}")
    print("Digite sua pergunta (Ctrl+C para sair).")
    while True:
        try:
            pergunta = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAté mais!")
            return
        if pergunta:
            _imprimir(servico, pergunta)


if __name__ == "__main__":
    main()
