from pathlib import Path
from time import perf_counter

from bpe.bpe import BPETokenizer


MODEL_PATH = Path("models/fineweb-bpe-32768.json")


TEST_CASES = {
    "simple word": "hello",
    "leading space": " hello",
    "subword word": "unbelievable",
    "integer": "123456789",
    "decimal": "3.14159265",
    "python": "def get_user_by_id(user_id):",
    "repeated characters": "aaaaaaaaaaaaaaaaaaaa",
    "hindi": "नमस्ते",
    "japanese": "こんにちは世界",
    "emoji": "😊",
    "complex emoji": "👨‍👩‍👧‍👦",
    "url": "https://example.com/users?id=123456789",
    "json": '{"user_id":123456789,"active":true}',
}


def display_token(tokenizer: BPETokenizer, token_id: int) -> str:
    token_bytes = tokenizer.vocab[token_id]

    text = token_bytes.decode(
        "utf-8",
        errors="backslashreplace",
    )

    return (
        f"id={token_id:<6} "
        f"bytes={token_bytes.hex(' '):<30} "
        f"text={text!r}"
    )


def test_case(
    tokenizer: BPETokenizer,
    name: str,
    text: str,
) -> None:
    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    print(f"Input:      {text!r}")
    print(f"Characters: {len(text)}")
    print(f"UTF-8 bytes:{len(text.encode('utf-8'))}")

    encode_start = perf_counter()
    ids = tokenizer.encode(text)
    encode_time = perf_counter() - encode_start

    decode_start = perf_counter()
    decoded = tokenizer.decode(ids)
    decode_time = perf_counter() - decode_start

    print(f"Token count:{len(ids)}")
    print(f"Token IDs:  {ids}")

    if len(text) > 0:
        print(
            f"Tokens/char:{len(ids) / len(text):.3f}"
        )

    byte_count = len(text.encode("utf-8"))

    if byte_count > 0:
        print(
            f"Tokens/byte:{len(ids) / byte_count:.3f}"
        )

    print(
        f"Encode time:{encode_time * 1000:.3f} ms"
    )
    print(
        f"Decode time:{decode_time * 1000:.3f} ms"
    )

    print("\nTokens:")

    for position, token_id in enumerate(ids):
        print(
            f"{position:>3}: "
            f"{display_token(tokenizer, token_id)}"
        )

    print(f"\nDecoded:    {decoded!r}")

    assert decoded == text, (
        f"Round-trip failed for {name!r}: "
        f"{text!r} != {decoded!r}"
    )

    print("Round-trip: PASS")


def main() -> None:
    print(f"Loading tokenizer: {MODEL_PATH}")

    tokenizer = BPETokenizer.load(MODEL_PATH)

    print(f"Vocabulary size: {len(tokenizer.vocab):,}")
    print(f"Merge rules:     {len(tokenizer.merges):,}")
    print(
        "Training bytes:  "
        f"{tokenizer.training_text_bytes:,}"
        if tokenizer.training_text_bytes
        else "Training bytes:  unknown"
    )

    if tokenizer.training_time is not None:
        print(
            "Training time:   "
            f"{tokenizer.training_time:.2f} s"
        )

    for name, text in TEST_CASES.items():
        test_case(
            tokenizer,
            name,
            text,
        )

    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()