import sentencepiece as spm

class SentencePieceTokenizer:
    """
    A wrapper class around SentencePieceProcessor to simplify encode/decode usage
    and make it plug-and-play with training and generation scripts.
    """
    def __init__(self, model_file: str = "spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)