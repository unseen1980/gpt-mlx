import mlx.core as mx
import mlx.nn as nn
import numpy as np


class GPTConfig:
    vocab_size = 5000
    max_seq_len = 128
    n_embd = 256
    n_layer = 6
    n_head = 8


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = mx.array(pe)

    def __call__(self, x):
        return x + self.pe[:x.shape[1]]


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.out = nn.Linear(n_embd, n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, self.n_head, 3 * self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = np.split(qkv, [self.head_dim, 2 * self.head_dim], axis=-1)
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        mask = np.tril(np.ones((T, T)))
        att = att * mask - 1e10 * (1 - mask)
        att = nn.softmax(att, axis=-1)
        out = att @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = PositionalEncoding(cfg.max_seq_len, cfg.n_embd)
        self.blocks = [TransformerBlock(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def __call__(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def generate_text(model: GPT, prompt: list[int], num_tokens: int = 20, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> list[int]:
    def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
        indices_to_remove = logits < np.partition(logits, -k)[-k]
        logits[indices_to_remove] = -np.inf
        return logits

    def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs /= np.sum(probs)
        cumulative_probs = np.cumsum(probs)
        cutoff = cumulative_probs > p
        if np.any(cutoff):
            cutoff_index = np.argmax(cutoff)
            sorted_logits[cutoff_index + 1:] = -np.inf
        filtered = -np.inf * np.ones_like(logits)
        filtered[sorted_indices] = sorted_logits
        return filtered

    model.eval()
    tokens = prompt.copy()

    for _ in range(num_tokens):
        input_tokens = tokens[-model.cfg.max_seq_len:]
        x = mx.array([input_tokens], dtype=mx.int32)
        logits = model(x)[:, -1, :] / temperature
        logits_np = np.array(logits[0])

        if top_p > 0.0:
            logits_np = apply_top_p(logits_np, p=top_p)
        elif top_k > 0:
            logits_np = apply_top_k(logits_np, k=top_k)

        probs = nn.softmax(mx.array(logits_np[None, :]), axis=-1)
        next_token = int(mx.random.categorical(probs[0]))
        tokens.append(next_token)

    return tokens
