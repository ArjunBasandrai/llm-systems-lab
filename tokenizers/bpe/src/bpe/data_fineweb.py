from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


OUTPUT_PATH = Path("../data/fineweb-2gb.txt")
TARGET_BYTES = 2 * 1024 * 1024 * 1024


def truncate_utf8(data: bytes, max_bytes: int) -> bytes:
    chunk = data[:max_bytes]

    while chunk:
        try:
            chunk.decode("utf-8")
            return chunk
        except UnicodeDecodeError:
            chunk = chunk[:-1]

    return b""


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    written = 0
    documents = 0

    with OUTPUT_PATH.open("wb") as f, tqdm(
        total=TARGET_BYTES,
        desc="Downloading FineWeb",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for row in dataset:
            text = row["text"]

            if not text:
                continue

            encoded = (
                text.rstrip() + "\n\n"
            ).encode("utf-8")

            remaining = TARGET_BYTES - written

            if len(encoded) > remaining:
                chunk = truncate_utf8(
                    encoded,
                    remaining,
                )

                f.write(chunk)

                written += len(chunk)
                pbar.update(len(chunk))

                break

            f.write(encoded)

            written += len(encoded)
            documents += 1

            pbar.update(len(encoded))

            if written >= TARGET_BYTES:
                break

    print()
    print(f"Output:     {OUTPUT_PATH}")
    print(
        f"Size:       "
        f"{written / (1024 ** 3):.3f} GiB"
    )
    print(f"Documents:  {documents:,}")


if __name__ == "__main__":
    main()