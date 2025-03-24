import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_in, d_out, context_length, n_heads, qkv_bias=False, dropout=0.1
    ):
        super().__init__()

        assert d_out % n_heads == 0, "d_out deve ser divisível por num_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.d_head = d_out // n_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        n_batch, n_tokens, _ = x.size()

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(n_batch, n_tokens, self.n_heads, self.d_head)
        keys = keys.view(n_batch, n_tokens, self.n_heads, self.d_head)
        values = values.view(n_batch, n_tokens, self.n_heads, self.d_head)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens], -torch.inf
        )

        attention_weights = torch.softmax(
            attention_scores / keys.size(-1) ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(n_batch, n_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector

class LayerNorm(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.eps = 1e-6  # Epsilon para evitar divisão por zero
        self.scale = nn.Parameter(
            torch.ones(d_emb)
        )  # Gamma - Inicializando com 1 para não alterar a escala
        self.shift = nn.Parameter(
            torch.zeros(d_emb)
        )  # Beta - Inicializando com 0 para não alterar o deslocamento

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift  # Gamma * x_norm + Beta

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

class FeedForward(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4),
            GeLU(),
            nn.Linear(d_emb * 4, d_emb),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_emb, n_heads, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_in=d_emb,
            d_out=d_emb,
            context_length=context_length,
            n_heads=n_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedForward(d_emb)
        self.norm1 = LayerNorm(d_emb)
        self.norm2 = LayerNorm(d_emb)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)  # 1
        x = self.mha(x)  # 2
        x = self.drop_shortcut(x)  # 3
        x = x + shortcut  # 4

        shortcut = x
        x = self.norm2(x)  # 5
        x = self.ff(x)  # 6
        x = self.drop_shortcut(x)  # 7
        x = x + shortcut  # 8
        return x

class GPTModel(nn.Module):
    def __init__(
        self, d_vocab, d_emb, context_length, n_layers, n_heads, dropout, qkv_bias
    ):
        super().__init__()
        self.context_length = context_length

        # Embedding de tokens e de posição
        self.tok_emb = nn.Embedding(d_vocab, d_emb)
        self.pos_emb = nn.Embedding(context_length, d_emb)
        self.drop_emb = nn.Dropout(dropout)

        # Sequência de Blocos Transformers
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(d_emb, n_heads, context_length, dropout, qkv_bias)
                for _ in range(n_layers)
            ]
        )

        # Normalização e projeção para o vocabulário
        self.final_norm = LayerNorm(d_emb)
        self.out_head = nn.Linear(d_emb, d_vocab, bias=False)

        # Inicialização dos pesos
        self.apply(self._init_weights)

        # Weight tying
        self.out_head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Embedding de tokens e de posição
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Sequência de Blocos Transformers
        x = self.trf_blocks(x)

        # Normalização e projeção para o vocabulário
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    # Função para gerar texto
    def generate(self, input, max=200, temperature=1.0, top_k=0, top_p=1.0):
        for _ in tqdm(range(max), desc="Gerando Tokens..."):
            input = input[
                :, -self.context_length :
            ]  # Garantindo que o seja no máximo do tamanho do contexto.
            logits = self(input)
            logits = logits[:, -1, :]

            # Aplicando temperature
            logits = logits / temperature

            # Aplicando top_k se especificado
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(
                    logits, min(top_k, logits.size(-1))
                )
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, top_k_indices, top_k_values)

            # Aplicando top_p (nucleus sampling) se menor que 1.0
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens com probabilidade cumulativa acima do threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift para manter pelo menos um token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[..., indices_to_remove] = float("-inf")

            # Convertendo para probabilidades e amostrando
            prob = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
            input = torch.cat([input, next_token], dim=-1)
        return input
 