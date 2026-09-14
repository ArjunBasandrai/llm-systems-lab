from pathlib import Path

from bpe.bpe import BPETokenizer


DATA_PATH = Path("../data/wikitext-10mb.txt")
VOCAB_SIZE = 8000


def load_training_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main():
    text = load_training_text(DATA_PATH)

    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)

    tokenizer.train(text)

    tokenizer.save(f"models/wikitext-bpe-{VOCAB_SIZE}.json")

if __name__ == "__main__":
    main()