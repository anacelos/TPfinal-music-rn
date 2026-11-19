# Usa a rede já treinada para compor: sorteia um token a partir das probabi-
# lidades, anexa à sequência, pergunta de novo, e repete até sair o EOS.
# https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

# nn.Module: classe-base do modelo (usada em model.py). Na mesma página estão 
# train(), eval(), state_dict() e parameters(), que você vai usar nos outros 
# arquivos.

# torch.tensor: transforma o prompt (lista de ids) em tensor. Lembre do 
# dtype=torch.long.

# F.softmax: converte logits em probabilidades. A temperatura entra antes 
# dele.

# torch.topk: o filtro top-k.

# torch.sort e torch.cumsum: o filtro top-p, que ordena as probabilidades 
# e soma até passar de p.

# Tensor.masked_fill: “apaga” os tokens filtrados.
# torch.multinomial: sorteia o próximo token. Repare no parâmetro generator.

# torch.Generator: semente local da amostragem, sem mexer na semente global.

# torch.argmax: o greedy, para o teste “top-k = 1 dá o mesmo resultado que 
# greedy”.

# Tensor.tolist e Tensor.item: voltam de tensor para lista ou número, porque 
# o contrato pede que a função devolva uma lista de ids.

# A página do nn.LSTM, de novo, na parte de (h_0, c_0): é como passar o estado
# adiante na geração, sem reprocessar o prefixo inteiro.