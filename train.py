# docs/03-training-loop.md

import os
import torch
import math
import json
from tqdm import tqdm
from model import GPT, GPTConfig
from generate import generate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")

    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))

        # x (input): characters from position i to i + block_size (input)
        # y (expected output): characters from position i+1 to i + block_size + 1 (target — shifted by one)

        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val, vocab_size, stoi, itos


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")     # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")    # NVIDIA GPU
    return torch.device("cpu")


# Warmup (first ~100 steps) - large updates early to explore.
# cosine decay (remaining steps) smoothly decrease the learning rate - small updates later to refine.
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(data_path, max_steps=5000, batch_size=64,
          n_layer=6, n_head=6, n_embd=384, block_size=256):
    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    model = GPT(config).to(device)
    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log = {"steps": [], "train": [], "val": []}

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        # --- validation loss ---
        # Every 100 steps, evaluate on held-out data. If train loss goes down but val loss goes up, you're overfitting.
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x, y = get_val_batch()
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")
            model.train()

        # --- update learning rate ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- training step ---
        # Gradient clipping (clip_grad_norm_): Caps the total gradient magnitude at 1.0. Prevents occasional large gradients from blowing up the weights.
        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # --- log loss ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)

        # --- generate sample ---
        # Every 100 steps, generate text so you can watch the model learn. You'll see it go from random characters → random words → Shakespeare-like text.
        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate(model, "To be or not", stoi, itos,
                            max_new_tokens=100, temperature=0.8)
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        # --- save checkpoint ---
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": config,
                "stoi": stoi,
                "itos": itos,
            }, f"checkpoint_{step}.pt")

    # --- save final checkpoint and loss log ---
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }, "checkpoint_final.pt")

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos


if __name__ == "__main__":
    # TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories/tree/main), is the next size up. You may want to switch to BPE tokenization.
    # https://huggingface.co/datasets/next-token/clean-PD-16000-books3, another data set might be worth trying.
    data_path = os.path.join(SCRIPT_DIR, "data", "shakespeare.txt")

    # # 6L/6H/384D
    # train(data_path, max_steps=5000, batch_size=64,
    #       n_layer=6, n_head=6, n_embd=384, block_size=256)
    
    # # 4L/4H/256D
    # train(data_path, max_steps=5000, batch_size=64,
    #       n_layer=4, n_head=4, n_embd=256, block_size=256)
    
    # 2L/2H/128D
    train(data_path, max_steps=5000, batch_size=64,
          n_layer=2, n_head=2, n_embd=128, block_size=256)