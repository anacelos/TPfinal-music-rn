"""
Definição da estrutura da rede neural para geração de tokens musicais.

Este módulo implementa uma arquitetura de rede neural que recebe uma sequência
de tokens musicais e retorna logits (pontuações). Esses logits determinam a
probabilidade de cada um dos V tokens possíveis no vocabulário ser o próximo
da sequência.

Arquitetura da Rede:
    1. Embedding
    2. LSTM (Long Short-Term Memory)
    3. Linear (Fully Connected)

Notas:
    - O modelo aqui definido contém apenas a arquitetura com inicialização
      aleatória. O processo de treinamento e geração ainda não está sendo
      aplicado neste escopo.

Referências:
    - Construção baseada no tutorial oficial do PyTorch:
      https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""
import torch.nn as nn

# Definindo a classe
# nn.Module: classe-base do modelo (usada em model.py).
class ModelMusic(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 padding_idx, dropout=0.0):
        """
        Inicializa a arquitetura do modelo de geração musical.

        Args:
            vocab_size: Inteiro indicando o tamanho do vocabulário. Lido do
                vocab.json
            embedding_dim: Inteiro especificando a dimensão do vetor denso
                usado para representar cada token na camada de embedding.
            hidden_dim: Inteiro definindo o número de features no estado
                oculto da rede LSTM.
            num_layers: Inteiro que determina o número de camadas recorrentes
                empilhadas na rede LSTM.
            padding_idx: Inteiro representando o índice do token de
                preenchimento (padding). O embedding deste índice será
                inicializado com zeros e não será atualizado durante o treino.
                Id do PAD no vocab.json
            dropout: Valor indicando a probabilidade de zerar elementos
                nas conexões entre as camadas da LSTM (se num_layers > 1) para
                regularização.
        """

        super(ModelMusic, self).__init__()

        # nn.Embedding: troca cada id por um vetor.
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=padding_idx)

        # nn.LSTM: a rede recorrente. Leia com calma as seções Inputs e
        # Outputs, porque elas explicam os shapes de output, h_n e c_n.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)

        # nn.Linear: transforma os hidden_dim núm. da LSTM em vocab_size
        # logits, uma pontuação para cada token possível ser o próximo.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Executa forward pass do modelo, processando os dados de entrada
        através de uma camada de embedding, por uma LSTM e uma camada
        linear.

        Args:
            x: Tensor com dados de entrada a serem processados.
            hidden: Estado oculto inicial a ser passado pela camada LSTM.

        Returns:
            Uma tupla (output, hidden)
            onde output é o tensor resultante que passou pela camada
            linear e hidden é o estado oculto atualizado após o
            processamento da LSTM.
        """

        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    