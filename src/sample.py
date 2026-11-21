"""
Script de geração de tokens musicais a partir de um modelo treinado.

Este módulo carrega os pesos e as configurações de um checkpoint salvo da rede
ModelMusic, instancia o tokenizador correspondente, gera novas sequências
musicais de forma autorregressiva (com temperatura e filtro top-k) e salva as
amostras decodificadas em formato JSON Lines (.jsonl).

Fluxo do script:
1. Leitura dos argumentos de linha de comando.
2. Seleção do dispositivo de execução e carregamento do checkpoint.
3. Reconstrução da arquitetura ModelMusic em modo de avaliação (eval).
4. Validação de compatibilidade entre o vocabulário do tokenizador e o modelo.
5. Geração autorregressiva das sequências musicais (temperatura e top-k).
6. Decodificação e salvamento das amostras em arquivo .jsonl.

Referências:
- Geração de sequências com RNNs no tutorial oficial do PyTorch:
      https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from model import ModelMusic
from tokenizer import Pico8Tokenizer


def read_args():
    """
    Lê e interpreta os argumentos de linha de comando para a geração musical.

    Returns:
        Objeto argparse.Namespace contendo o caminho do checkpoint, dispositivo,
        modo do tokenizador, hiperparâmetros de amostragem (temperatura, top-k,
        tamanho máximo, quantidade de amostras, semente) e caminho de saída.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Gera sequências de tokens musicais usando um checkpoint "
            "treinado da rede ModelMusic."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Caminho para o arquivo de checkpoint (.pt) do modelo treinado.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução (ex: 'cpu', 'cuda'). Padrão: automático.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="nlm",
        help="Modo de tokenização do Pico8Tokenizer (deve bater com o treino).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Caminho do arquivo de saída (.jsonl) para salvar as amostras.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Número de sequências musicais a serem geradas.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1828,
        help="Comprimento máximo (em tokens) de cada sequência gerada.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura aplicada aos logits antes do softmax.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Número de tokens mais prováveis mantidos no filtro top-k (0 desativa).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade da amostragem.",
    )

    args = parser.parse_args()
    return args


def gerar(modelo, bos_id, eos_id, max_len, temperatura, top_k, gerador,
          device):
    """
    Gera uma sequência de tokens musicais de forma autorregressiva.

    A partir do token inicial (BOS), o modelo prevê iterativamente o próximo
    token, reaproveitando o estado oculto da LSTM a cada passo, até gerar o
    token de parada (EOS) ou atingir o limite máximo de passos.

    Args:
        modelo: Instância treinada de ModelMusic em modo de avaliação (eval).
        bos_id: Inteiro representando o índice do token de início (BOS).
        eos_id: Inteiro representando o índice do token de fim (EOS).
        max_len: Inteiro definindo o número máximo de novos tokens a gerar.
        temperatura: Float maior que zero usado para escalar os logits antes
            do softmax.
        top_k: Inteiro indicando quantos tokens mais prováveis manter na
            amostragem. Se menor ou igual a 0, o filtro top-k é desativado.
        gerador: Instância de torch.Generator para controlar a aleatoriedade
            local da amostragem.
        device: Dispositivo (torch.device) onde o modelo e os tensores estão
            alocados.

    Returns:
        Lista de inteiros Python contendo a sequência gerada de IDs de tokens,
        iniciando por bos_id e terminando em eos_id (caso sorteado dentro de
        max_len passos).
    """
    ids = [bos_id]

    entrada = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    estado = None

    with torch.no_grad():
        for _ in range(max_len):
            logits, estado = modelo(entrada, estado)

            z = logits[0, -1] / temperatura

            if top_k > 0:
                maiores, _ = torch.topk(z, k=top_k)
                limite = maiores[-1]
                z = z.masked_fill(z < limite, float("-inf"))

            p = F.softmax(z, dim=-1)
            prox = torch.multinomial(
                p, num_samples=1, generator=gerador
            ).item()

            ids.append(prox)
            if prox == eos_id:
                break

            entrada = torch.tensor([[prox]], dtype=torch.long, device=device)

    return ids


def main():
    """
    Ponto de entrada do script de geração.

    Carrega o checkpoint informado via CLI, reconstrói o modelo em modo de
    avaliação no dispositivo selecionado, valida o vocabulário do tokenizador,
    gera as amostras musicais e salva o resultado em formato JSON Lines.
    """
    args = read_args()

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

    # Carrega o checkpoint na CPU para evitar erros caso o treino
    # tenha sido feito em uma GPU indisponível.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=torch.device("cpu"),
    )
    config = checkpoint["config"]

    loaded_model = ModelMusic(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        padding_idx=config["pad_id"],
        dropout=config["dropout"],
    )

    loaded_model.load_state_dict(checkpoint["model_state"])
    loaded_model.to(device)
    loaded_model.eval()

    # Validação para ver se o tokenizador gera o mesmo vocabulário.
    tokenizador = Pico8Tokenizer(mode=args.mode)
    assert len(tokenizador.token_to_id) == config["vocab_size"], (
        f"Tamanho do vocabulário incompatível ({len(tokenizador.token_to_id)}"
        f" != {config['vocab_size']}): "
        f"o --mode provavelmente não é o mesmo usado no treino."
    )
    print(
        f"Epoch: {checkpoint['epoch']} | Val Loss: "
        f"{checkpoint['val_loss']} | BOS: {tokenizador.bos_id} | "
        f"EOS: {tokenizador.eos_id}"
    )

    # Cria um gerador local no mesmo device para garantir reprodutibilidade
    # sem alterar o estado aleatório global do PyTorch.
    gerador = torch.Generator(device=device).manual_seed(args.seed)

    # Define o caminho do arquivo de saída (.jsonl). Se não for informado via
    # CLI, salva na mesma pasta do checkpoint.
    saida = args.out
    if saida is None:
        pasta = os.path.dirname(args.checkpoint)
        saida = os.path.join(pasta, "samples.jsonl")

    # Garante que a pasta de destino exista (usa "." caso o caminho não tenha
    # subdiretórios, ex: --out samples.jsonl).
    os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)

    # Gera as amostras e salva uma música por linha no formato JSON Lines.
    with open(saida, "w", encoding="utf-8") as f:
        for i in range(args.n_samples):
            ids = gerar(
                loaded_model,
                tokenizador.bos_id,
                tokenizador.eos_id,
                args.max_len,
                args.temperature,
                args.top_k,
                gerador,
                device,
            )

            # Decodifica os IDs para o dicionário no formato do songs.json.
            musica = tokenizador.decode_ids(ids)
            musica["tid"] = f"lstm_seed{args.seed}"
            musica["song_id"] = i
            musica["token_ids"] = ids

            f.write(json.dumps(musica) + "\n")

            terminou_eos = ids[-1] == tokenizador.eos_id
            print(
                f"Amostra {i} | Tamanho: {len(ids)} | "
                f"Terminou com EOS: {terminou_eos}"
            )

    print(f"Amostras salvas em: {saida}")


if __name__ == "__main__":
    main()