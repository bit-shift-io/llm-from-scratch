# docs/02-the-transformer.md

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 65       # character-level: 65 unique chars in Shakespeare
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension



# Attention gathers information from the entire sequence, but each token can only attend to previous tokens (causal attention).
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # c_attn produces Q/K/V for all 6 heads via a single linear layer instead of
        # 18 separate small ones (3 x 6 heads). Its output size is 3 * n_embd (1152),
        # i.e. 3x the embedding dimension — not 18x — because n_embd (384) already
        # spans all 6 heads combined (head_dim = 384/6 = 64). One matmul is more
        # efficient on GPUs than 18 tiny ones.
        #
        # Q, K, V are named for the roles the fixed math in scaled_dot_product_attention
        # gives them: two of the three slices get dot-producted together to form
        # attention weights (query/key), and the third gets weighted-summed by those
        # weights (value). The math defines that structure; training is what shapes
        # c_attn's weights to actually fill those roles usefully.
        # The math sets up the game; training is what actually plays it.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q, K, V projections

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # output projection
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape for multi-head: (B, T, C) → (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # attention with causal mask (each token can only attend to previous tokens)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    

# The "thinking" or processing part of the transformer block, a simple feed-forward network.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)       # project up: 384 → 1536
        x = self.gelu(x)       # non-linearity
        return self.c_proj(x)  # project back down: 1536 → 384
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connections (x = x + sublayer(x)): The input is added back to the output.
        # This lets gradients flow directly through the network during backpropagation, 
        # making deep networks trainable. Without residuals, a 6-layer network would be much harder to train.

        x = x + self.attn(self.ln_1(x))   # attention with residual connection
        x = x + self.mlp(self.ln_2(x))    # MLP with residual connection
        return x
    

# The GPT model consists of a stack of transformer blocks, with token and position embeddings, and a final linear layer for language modeling.
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Tokens are normally around up to 4 chars in length such as "un" + "bel" + "iev" + "able".
        # This is a tade off between vocabulary size and sequence length. 
        # Both exteremes are bad. Sequence length blows out compute. Vocab size blows out memory.
        # For this project, doe to the limited vocab size, we can use a character-level model. This is simpler and works well for small datasets like Shakespeare.

        # Two embedding tables:
        # wte (word token embedding): maps each token ID to a learned vector. Size: [65, 384]
        # wpe (word position embedding): maps each position (0 to 255) to a learned vector. Size: [256, 384]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the output projection shares weights with the token embeddings
        self.transformer.wte.weight = self.lm_head.weight


    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd)
        x = tok_emb + pos_emb                  # (B, T, n_embd) — broadcasting adds position info

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
    


config = GPTConfig()
model = GPT(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params / 1e6:.1f}M")  # ~10.8M