from pathlib import Path

from bpe.bpe import BPETokenizer

DATASET_NAME = "fineweb"
VOCAB_SIZE = 32_768
DATA_PATH = Path(f"../data/{DATASET_NAME}-2gb.txt")
MODEL_PATH = Path(f"models/{DATASET_NAME}-bpe-{VOCAB_SIZE}.json")

def main() -> None:
    tokenizer = BPETokenizer(
        vocab_size=VOCAB_SIZE,
        encode_cache_size=100_000,
    )

    tokenizer.train_file(DATA_PATH)
    tokenizer.save(MODEL_PATH)

    print()
    print(f"Saved tokenizer to: {MODEL_PATH}")
    print(f"Vocabulary size:    {len(tokenizer.vocab):,}")
    print(f"Learned merges:     {len(tokenizer.merges):,}")

    if tokenizer.training_time is not None:
        print(
            f"Training time:      "
            f"{tokenizer.training_time / 60:.2f} min"
        )


if __name__ == "__main__":
    main()