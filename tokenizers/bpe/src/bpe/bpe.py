from collections import Counter
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone

from tqdm import tqdm


class BPETokenizer:
    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.vocab: dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }
        self.merges: dict[tuple[int, int], int] = {}

        self.training_time: float | None = None
        self.training_text_bytes: int | None = None

    def __pre_tokenize(self, text: str) -> list[bytes]:
        lines = text.splitlines(keepends=True)
        subsequences = []

        for line in lines:
            parts = re.findall(r'[^.;:!?]*[.;:!?]+|[^.;:!?]+', line)
            subsequences.extend(parts)

        return [s.encode("utf-8") for s in subsequences]

    def __count_pairs(self, sequences: list[list[int]]) -> Counter:
        counter = Counter()

        for seq in sequences:
            counter.update(zip(seq, seq[1:]))

        return counter

    def __merge_pair(
        self,
        sequence: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        merged_sequence = []
        i = 0

        while i < len(sequence):
            if (
                i < len(sequence) - 1
                and (sequence[i], sequence[i + 1]) == pair
            ):
                merged_sequence.append(new_token_id)
                i += 2
            else:
                merged_sequence.append(sequence[i])
                i += 1

        return merged_sequence

    def train(self, text: str) -> None:
        start_time = time.perf_counter()

        byte_sequences = self.__pre_tokenize(text)
        sequences = [list(seq) for seq in byte_sequences]

        self.training_text_bytes = len(text.encode("utf-8"))

        with tqdm(
            total=self.vocab_size - len(self.vocab),
            desc="Training BPE",
            unit="tokens",
        ) as pbar:
            while len(self.vocab) < self.vocab_size:
                pair_counts = self.__count_pairs(sequences)

                if not pair_counts:
                    break

                most_frequent_pair = pair_counts.most_common(1)[0][0]

                new_token_id = len(self.vocab)

                self.vocab[new_token_id] = (
                    self.vocab[most_frequent_pair[0]]
                    + self.vocab[most_frequent_pair[1]]
                )

                self.merges[most_frequent_pair] = new_token_id

                for i in range(len(sequences)):
                    sequences[i] = self.__merge_pair(
                        sequences[i],
                        most_frequent_pair,
                        new_token_id,
                    )

                pbar.update(1)

        self.training_time = time.perf_counter() - start_time

    def encode(self, text: str) -> list[int]:
        byte_sequences = self.__pre_tokenize(text)
        sequences = [list(seq) for seq in byte_sequences]

        for pair, new_token_id in self.merges.items():
            for i in range(len(sequences)):
                sequences[i] = self.__merge_pair(
                    sequences[i],
                    pair,
                    new_token_id,
                )

        return [
            token_id
            for seq in sequences
            for token_id in seq
        ]

    def decode(self, ids: list[int]) -> str:
        bytes_seq = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )
        return bytes_seq.decode("utf-8")

    def save(self, path: str | Path) -> None:
        path = Path(path)

        data = {
            "vocab_size": self.vocab_size,

            "vocab": {
                str(token_id): token_bytes.hex()
                for token_id, token_bytes in self.vocab.items()
            },

            "merges": [
                [left_id, right_id, new_token_id]
                for (left_id, right_id), new_token_id
                in self.merges.items()
            ],

            "metadata": {
                "training_time_seconds": self.training_time,
                "training_text_bytes": self.training_text_bytes,
                "num_merges": len(self.merges),
                "actual_vocab_size": len(self.vocab),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "format_version": 1,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])

        tokenizer.vocab = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in data["vocab"].items()
        }

        tokenizer.merges = {
            (left_id, right_id): new_token_id
            for left_id, right_id, new_token_id in data["merges"]
        }

        return tokenizer
