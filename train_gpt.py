import os
import csv
import requests
import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from gpt_mini import GPT, GPTConfig, generate_text
from sp_tokenizer import SentencePieceTokenizer


def download_dataset(path="tiny_shakespeare.txt"):
    if not os.path.exists(path):
        print("📥 Downloading dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data = requests.get(url).text
        with open(path, "w") as f:
            f.write(data)
        print("✅ Dataset downloaded.")
    else:
        print("📄 Dataset already available.")
    return open(path, "r").read()


def create_batches(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return mx.array(x), mx.array(y)


def save_loss_plot(loss_log, out_path="loss_plot.png"):
    steps, losses = zip(*loss_log)
    plt.plot(steps, losses, marker='o', linestyle='-')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.savefig(out_path)
    print(f"📈 Loss plot saved as {out_path}")


def generate_sample(model, tokenizer, prompt="The king", step=0):
    prompt_ids = tokenizer.encode(prompt)
    output_ids = generate_text(model, prompt_ids, num_tokens=100, temperature=0.7, top_k=20)
    text = tokenizer.decode(output_ids)
    with open(f"sample_step{step}.txt", "w") as f:
        f.write(text)
    print(f"📝 Sample generated and saved to sample_step{step}.txt")


def flatten_params(params):
    flat = {}

    def recurse(prefix, d):
        for k, v in d.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                recurse(name, v)
            elif isinstance(v, mx.array):
                flat[name] = v

    recurse("", params)
    return flat


def collect_arrays(params_dict):
    """
    Recursively yields all mx.array values from a nested parameters dictionary.
    """
    for v in params_dict.values():
        if isinstance(v, dict):
            yield from collect_arrays(v)
        elif isinstance(v, mx.array):
            yield v


def train(resume_checkpoint=None):
    print("🔧 Initializing training...")
    raw_text = download_dataset()
    tokenizer = SentencePieceTokenizer("spm.model")
    encoded = tokenizer.encode(raw_text)

    cfg = GPTConfig()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = 128
    cfg.n_embd = 256
    cfg.n_layer = 6
    cfg.n_head = 4

    model = GPT(cfg)
    optimizer = optim.Adam(learning_rate=3e-4, betas=(0.9, 0.95))

    start_step = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_checkpoint}")
        model.update(mx.load(resume_checkpoint))
        start_step = int(resume_checkpoint.split("step")[-1].split(".")[0])

    batch_size = 64
    total_steps = 3000
    checkpoint_interval = 500

    loss_log = []
    loss_csv_path = "loss_log.csv"
    csv_mode = "a" if start_step > 0 else "w"

    with open(loss_csv_path, mode=csv_mode, newline="") as f:
        writer = csv.writer(f)
        if start_step == 0:
            writer.writerow(["step", "loss"])

        for step in range(start_step, total_steps):
            x, y = create_batches(encoded, cfg.max_seq_len, batch_size)

            def loss_fn(dummy_model):
                logits = model(x)
                logits = logits.reshape(-1, cfg.vocab_size)
                y_flat = y.reshape(-1)
                ce_loss = nn.losses.cross_entropy(logits, y_flat).mean()

                # ✅ Manual L2 regularization
                weight_decay = 1e-2
                l2_loss = sum([(param**2).sum() for param in collect_arrays(model.parameters())])
                return ce_loss + weight_decay * l2_loss

            loss, grads = mx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

            if step % 10 == 0 or step == total_steps - 1:
                loss_value = loss.item()
                print(f"🔁 Step {step:04d} | Loss: {loss_value:.4f}")
                writer.writerow([step, loss_value])
                loss_log.append((step, loss_value))

            if step % checkpoint_interval == 0 or step == total_steps - 1:
                checkpoint_name = f"gpt_checkpoint_step{step}.npz"
                mx.savez(checkpoint_name, **flatten_params(model.parameters()))
                print(f"💾 Saved checkpoint: {checkpoint_name}")
                generate_sample(model, tokenizer, step=step)

    mx.savez("gpt_model.npz", **flatten_params(model.parameters()))
    print("🔚 Training complete.")
    print("✅ Final model saved as gpt_model.npz")
    save_loss_plot(loss_log)


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_checkpoint=checkpoint)
