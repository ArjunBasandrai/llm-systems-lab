from pathlib import Path
import time

from bpe.bpe import BPETokenizer

DATA_PATH = Path("../data/wikitext-10mb.txt")

def load_training_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def main():
    tokenizer = BPETokenizer.load("models/wikitext-bpe-500.json")

    text = load_training_text(DATA_PATH)

    start_time = time.perf_counter()
    ids = tokenizer.encode(text)
    end_time = time.perf_counter()
    encoding_time_s = end_time - start_time

    start_time = time.perf_counter()
    decoded = tokenizer.decode(ids)
    end_time = time.perf_counter()
    decoding_time_s = end_time - start_time


    print("n_characters:", len(text))
    print("n_words:", len(text.split()))
    print("n_tokens:", len(ids))
    print(f"Encoding time: {encoding_time_s:.4f} seconds")
    print(f"Encoding speed: {len(ids) / encoding_time_s:.2f} tokens/second")
    print(f"Decoding time: {decoding_time_s:.4f} seconds")

    assert decoded == text

if __name__ == "__main__":
    main()