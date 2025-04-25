# Optional: create virtual environment

python3 -m venv gpt-mlx
source gpt-mlx/bin/activate

# Upgrade pip

pip install --upgrade pip

# Install MLX (Apple's deep learning library)

pip install mlx

# Install other required packages

pip install sentencepiece matplotlib requests

# Train SentencePiece Tokenizer

python train_tokenizer.py
This will generate:

spm.model – the tokenizer model

spm.vocab – the vocabulary

# Train the GPT Model

python train_gpt.py

Downloads tiny_shakespeare.txt if needed

Tokenizes it using SentencePiece

Trains the GPT model

Logs loss in loss_log.csv

Plots training loss in loss_plot.png

Saves model checkpoints (e.g. gpt_checkpoint_step100.npz)

Generates sample output (e.g. sample_step100.txt)

To resume from checkpoint: python train_gpt.py gpt_checkpoint_step100.npz
