"""
Script de treinamento da rede neural para geração de tokens musicais.

Este módulo lê os hiperparâmetros via linha de comando, carrega os datasets
de treino e validação, executa o laço de treinamento autoregressivo da
arquitetura ModelMusic e salva o melhor checkpoint junto ao histórico de
perdas.

Fluxo de Execução:
1. Leitura de argumentos e configuração de semente/dispositivo (CPU ou GPU)
2. Carregamento dos datasets, vocabulário (V e pad_id) e DataLoaders
3. Instanciação do modelo, função de perda (CrossEntropyLoss) e otimizador
4. Laço de treino (com gradient clipping) e validação por época
5. Salvamento do melhor checkpoint (best.pt) e do histórico (history.json)

Notas:
- Os valores de vocab_size (V) e pad_id são lidos dinamicamente de
      vocab_info para evitar dependência de constantes fixas.
- O diretório de saída padrão ("outputs") é ignorado pelo .gitignore.

Referências:
- PyTorch Optimization Tutorial:
      https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch DataLoader Documentation:
      https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import create_pytorch_datasets
from model import ModelMusic


def read_args():
    """
    Lê e interpreta os hiperparâmetros fornecidos pela linha de comando.

    Returns:
        argparse.Namespace contendo os caminhos de entrada/saída e todos os
        hiperparâmetros de arquitetura, treinamento e reprodutibilidade.
    """

    parser = argparse.ArgumentParser(
        description="Treinamento da RNN autoregressiva ModelMusic."
    )

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default="teste")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)

    args = parser.parse_args()
    return args


def main():
    """
    Executa o pipeline completo de treinamento e validação do modelo.

    Configura a reprodutibilidade e o dispositivo, valida o formato dos dados
    carregados, treina o ModelMusic por `args.epochs` épocas e persiste o
    melhor checkpoint (`best.pt`) e o histórico de perdas (`history.json`) em
    `args.out_dir/args.run_name`.
    """
    args = read_args()

    # Reprodutibilidade: fixa a semente global na CPU e nas GPUs disponíveis.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Seleção do dispositivo de execução (GPU/acelerador ou CPU)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = (
            torch.accelerator.current_accelerator()
            if torch.accelerator.is_available()
            else torch.device("cpu")
        )

    print("Args:", args)
    print("Device:", device)

    # Carregamento dos datasets e informações do vocabulário
    datasets, vocab_info = create_pytorch_datasets(
        args.data_dir, args.seq_len, args.stride
    )
    V = len(vocab_info["token_to_id"])
    pad_id = vocab_info["token_to_id"]["PAD"]

    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=args.batch_size, shuffle=False
    )

    print(
        f"V = {V}, pad_id = {pad_id}, "
        f"{len(datasets['train'])} janelas no treino e "
        f"{len(datasets['val'])} na validação"
    )

    # Conferências de integridade dos dados antes de iniciar o treino.
    assert len(datasets["train"]) > 0, "Dataset de treino está vazio."
    assert len(datasets["val"]) > 0, "Dataset de validação está vazio."

    x_sample, _ = datasets["train"][0]
    assert x_sample.shape == (args.seq_len,), (
        f"Esperado shape ({args.seq_len},), mas veio {x_sample.shape}."
    )
    assert x_sample.dtype == torch.long, (
        f"Esperado dtype torch.long, mas veio {x_sample.dtype}."
    )
    assert x_sample.max().item() < V, (
        f"Maior id em x ({x_sample.max().item()}) é >= V ({V})."
    )

    # Instanciação do modelo, função de perda e otimizador
    model = ModelMusic(
        vocab_size=V,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        padding_idx=pad_id,
        dropout=args.dropout,
    ).to(device)

    # ignore_index impede que tokens de padding contribuam para a perda
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Preparação do diretório de saída e do controle de histórico/checkpoint
    out_path = os.path.join(args.out_dir, args.run_name)
    os.makedirs(out_path, exist_ok=True)
    history = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, V), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, V), y.reshape(-1))
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)

        print(
            f"Época {epoch:02d}/{args.epochs} | "
            f"Perda treino: {train_loss:.4f} | "
            f"Perda val: {val_loss:.4f}"
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            config = {
                **vars(args),
                "vocab_size": V,
                "pad_id": pad_id,
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                os.path.join(out_path, "best.pt"),
            )
            print("  -> checkpoint salvo")

        with open(
            os.path.join(out_path, "history.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()