import sentencepiece as spm
import os
import requests

if not os.path.exists("tiny_shakespeare.txt"):
    print("📥 Downloading tiny_shakespeare.txt for tokenizer...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data = requests.get(url).text
    with open("tiny_shakespeare.txt", "w") as f:
        f.write(data)
    print("✅ tiny_shakespeare.txt downloaded.")



def train_tokenizer(input_file="tiny_shakespeare.txt", model_prefix="spm", vocab_size=8000):
    """
    Trains a SentencePiece tokenizer on the given input text file.

    Args:
        input_file: Path to the training text file.
        model_prefix: Prefix for the output model and vocab files.
        vocab_size: Target vocabulary size for the tokenizer.
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print("✅ SentencePiece tokenizer trained and saved.")

if __name__ == "__main__":
    train_tokenizer()