from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65       # character-level: 65 unique chars in Shakespeare
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6           # number of transformer blocks
    n_head: int = 6            # number of attention heads
    n_embd: int = 384          # embedding dimension