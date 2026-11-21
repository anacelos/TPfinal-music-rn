"""
Script de visualização das curvas de treinamento da rede ModelMusic.

Este módulo lê o arquivo JSON de histórico gerado durante o treinamento
(contendo as perdas de treino e validação por época), calcula a perplexidade
correspondente (exp(perda)) e salva uma figura com os dois gráficos lado a lado.

Fluxo do script:
1. Leitura dos argumentos de linha de comando (caminho do histórico e saída).
2. Carregamento do arquivo history.json.
3. Cálculo da perplexidade de treino e validação por época.
4. Plotagem e salvamento da figura com as curvas de perda e perplexidade.
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt


def read_args():
    """
    Lê e interpreta os argumentos de linha de comando para a plotagem.

    Returns:
        Objeto argparse.Namespace contendo o caminho do arquivo de histórico
        (.json) e o caminho opcional da imagem de saída (.png).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plota as curvas de perda e perplexidade por época a partir de "
            "um arquivo history.json."
        )
    )

    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Caminho para o arquivo history.json gerado no treinamento.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Caminho do arquivo de imagem (.png) de saída. "
            "Se omitido, salva como curvas.png na pasta do --history."
        ),
    )

    args = parser.parse_args()
    return args


def main():
    """
    Ponto de entrada do script de plotagem.

    Carrega o histórico de perdas, calcula a perplexidade por época, gera os
    gráficos comparando treino e validação e salva a figura em disco.
    """
    args = read_args()

    with open(args.history, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Define o caminho de saída: se não informado, salva na pasta do history.
    saida = args.output
    if saida is None:
        pasta = os.path.dirname(args.history)
        saida = os.path.join(pasta, "curvas.png")

    epochs_range = range(1, len(history["train"]) + 1)
    ppl_treino = [math.exp(loss) for loss in history["train"]]
    ppl_val = [math.exp(loss) for loss in history["val"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico 1: Perda (Cross-Entropy Loss)
    axes[0].plot(epochs_range, history["train"], marker="o", label="Treino")
    axes[0].plot(epochs_range, history["val"], marker="o", label="Validação")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda")
    axes[0].set_title("Perda por época")
    axes[0].legend()

    # Gráfico 2: Perplexidade (exp(Perda))
    axes[1].plot(epochs_range, ppl_treino, marker="o", label="Treino")
    axes[1].plot(epochs_range, ppl_val, marker="o", label="Validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Perplexidade")
    axes[1].set_title("Perplexidade por época")
    axes[1].legend()

    plt.tight_layout()

    # Garante que a pasta de destino exista antes de salvar a imagem.
    os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)
    plt.savefig(saida, dpi=300)
    plt.close(fig)

    print(f"Gráfico salvo em: {saida}")


if __name__ == "__main__":
    main()