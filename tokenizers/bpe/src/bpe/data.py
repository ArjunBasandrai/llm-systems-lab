from datasets import load_dataset
from pathlib import Path


DATA_TARGET_MB = 256
OUTPUT_PATH = Path(f"../data/wikitext-{DATA_TARGET_MB}mb.txt")
TARGET_BYTES = DATA_TARGET_MB * 1024 * 1024


def main():
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with OUTPUT_PATH.open("wb") as f:
        for row in ds:
            text = row["text"]

            if not text:
                continue

            encoded = (text + "\n").encode("utf-8")

            remaining = TARGET_BYTES - written

            if len(encoded) > remaining:
                chunk = encoded[:remaining]

                while chunk:
                    try:
                        chunk.decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        chunk = chunk[:-1]

                f.write(chunk)
                written += len(chunk)
                break

            f.write(encoded)
            written += len(encoded)

    print(f"Wrote {written:,} bytes")
    print(f"Wrote {written / (1024 ** 2):.2f} MiB")

if __name__ == "__main__":
    main()