"""
Avalia um modelo treinado no conjunto de teste.

Carrega o checkpoint salvo (best.pt) e calcula a perda (loss)
e a perplexidade no conjunto de teste.
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import create_pytorch_datasets
from model import ModelMusic


def read_args() -> argparse.Namespace:
    """
    Lê e faz o parse dos argumentos de linha de comando.

    Returns:
        argparse.Namespace: Objeto contendo os argumentos fornecidos.
    """

    parser = argparse.ArgumentParser(
        description="Avalia o modelo treinado no conjunto de teste."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Caminho para o arquivo de checkpoint (ex.: best.pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Diretório com os dados processados (processed/<modo>).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch para avaliação (padrão: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução (ex.: 'cpu' ou 'cuda'). Padrão: None.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Carrega o checkpoint, avalia o modelo no teste e salva as métricas.
    """

    args = read_args()

    # Define o dispositivo
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]

    datasets, vocab_info = create_pytorch_datasets(
        args.data_dir,
        config["seq_len"],
        config["stride"],
    )
    pad_id = vocab_info["token_to_id"]["PAD"]

    assert len(vocab_info["token_to_id"]) == config["vocab_size"], (
        f"Tamanho do vocabulário nos dados ({len(vocab_info['token_to_id'])}) "
        f"difere do vocab_size do checkpoint ({config['vocab_size']})."
    )

    model = ModelMusic(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        padding_idx=pad_id,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # Cria o DataLoader de teste sem embaralhar
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Janelas no conjunto de teste: {len(datasets['test'])}")
    print(f"PAD id: {pad_id}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, config["vocab_size"]),
                y.reshape(-1),
            )
            total_loss += loss.item()
            total_tokens += (y != pad_id).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"Total de tokens avaliados: {total_tokens}")
    print(f"Perda média (teste): {avg_loss:.4f}")
    print(f"Perplexidade (teste): {perplexity:.4f}")

    metrics = {
        "test_loss": avg_loss,
        "test_ppl": perplexity,
        "n_tokens": total_tokens,
        "checkpoint": args.checkpoint,
    }

    output_dir = os.path.dirname(args.checkpoint)
    output_path = os.path.join(output_dir, "test_metrics.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas salvas em: {output_path}")

if __name__ == "__main__":
    main()

