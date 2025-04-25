import mlx.core as mx
from gpt_mini import GPT, GPTConfig, generate_text
from sp_tokenizer import SentencePieceTokenizer


def main():
    """
    Loads a trained GPT model and tokenizer, then generates text from a user prompt.
    Adds debugging and stability improvements to help troubleshoot generation quality.
    """
    print("🧠 Loading model and tokenizer...")
    tokenizer = SentencePieceTokenizer("spm.model")

    # Load GPT configuration and model weights
    cfg = GPTConfig()
    cfg.vocab_size = tokenizer.vocab_size
    model = GPT(cfg)
    model.update(mx.load("gpt_model.npz"))
    print("✅ Model and tokenizer loaded.")

    # Prompt input
    prompt_text = input("💬 Enter a prompt: ").strip()
    if not prompt_text:
        print("⚠️  Prompt is empty. Exiting.")
        return

    prompt_ids = tokenizer.encode(prompt_text)

    print("🔮 Generating...")
    output_ids = generate_text(model, prompt_ids, num_tokens=100, temperature=0.7, top_k=20)

    print("🔢 Output token IDs:", output_ids)

    # Remove special tokens like <unk> if needed
    decoded_ids = [t for t in output_ids if t != tokenizer.sp.unk_id()]

    try:
        result = tokenizer.decode(decoded_ids)
    except Exception as e:
        result = f"(Error decoding tokens: {e})"

    print("\n📝 Generated Output:\n" + "-" * 50)
    print(result)
    print("-" * 50)


if __name__ == "__main__":
    main()
