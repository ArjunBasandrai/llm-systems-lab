from bpe.bpe import BPETokenizer

def main():
    tokenizer = BPETokenizer.load("models/wikitext-bpe-500.json")

    text = "Hello from the café 😊"

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Text:", repr(text))
    print("IDs:", ids)
    print("n_characters:", len(text))
    print("n_words:", len(text.split()))
    print("n_tokens:", len(ids))
    print("Tokens:", [tokenizer.vocab[i] for i in ids])
    print("Decoded:", repr(decoded))

    assert decoded == text

if __name__ == "__main__":
    main()