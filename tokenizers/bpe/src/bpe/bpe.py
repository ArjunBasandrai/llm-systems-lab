from collections import Counter, OrderedDict, defaultdict
import heapq
import json
from pathlib import Path
import time
from datetime import datetime, timezone

from tqdm import tqdm
import regex


class BPETokenizer:
    DEFAULT_PRE_TOKEN_PATTERN = (
        r"(?i:['’](?:s|t|re|ve|m|ll|d))"
        r"| ?\p{L}+"
        r"| ?\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    def __init__(
        self,
        vocab_size: int = 500,
        encode_cache_size: int = 100_000,
        pre_token_pattern: str = DEFAULT_PRE_TOKEN_PATTERN,
    ):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        self.vocab_size = vocab_size

        self.vocab: dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }

        self.merges: dict[tuple[int, int], int] = {}
        self.merge_ranks: dict[tuple[int, int], int] = {}

        self.encode_cache_size = encode_cache_size
        self.encode_cache: OrderedDict[
            bytes, tuple[int, ...]
        ] = OrderedDict()

        self.pre_token_pattern = pre_token_pattern
        self._pattern = regex.compile(pre_token_pattern)

        self.training_time: float | None = None
        self.training_text_bytes: int | None = None

    def __pre_tokenize(self, text: str) -> list[bytes]:
        return [
            match.group(0).encode("utf-8")
            for match in self._pattern.finditer(text)
        ]

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

    def __invalidate_encode_cache(self) -> None:
        self.encode_cache.clear()

    def __rebuild_merge_ranks(self) -> None:
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.__invalidate_encode_cache()

    def __reset_model(self) -> None:
        self.vocab = {
            i: bytes([i]) for i in range(256)
        }

        self.merges.clear()
        self.merge_ranks.clear()
        self.__invalidate_encode_cache()

    def __cache_get(
        self,
        chunk: bytes,
    ) -> tuple[int, ...] | None:
        cached = self.encode_cache.get(chunk)

        if cached is None:
            return None

        self.encode_cache.move_to_end(chunk)

        return cached

    def __cache_put(
        self,
        chunk: bytes,
        ids: tuple[int, ...],
    ) -> None:
        if self.encode_cache_size <= 0:
            return

        self.encode_cache[chunk] = ids
        self.encode_cache.move_to_end(chunk)

        if len(self.encode_cache) > self.encode_cache_size:
            self.encode_cache.popitem(last=False)

    def __encode_chunk(
        self,
        chunk: bytes,
    ) -> tuple[int, ...]:
        cached = self.__cache_get(chunk)

        if cached is not None:
            return cached

        sequence = list(chunk)

        while len(sequence) >= 2:
            best_pair = None
            best_rank = None

            for pair in zip(sequence, sequence[1:]):
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            sequence = self.__merge_pair(
                sequence,
                best_pair,
                self.merges[best_pair],
            )

        result = tuple(sequence)

        self.__cache_put(chunk, result)

        return result

    def __build_pair_stats(
        self,
        sequences: list[list[int]],
        frequencies: list[int],
        pbar: tqdm,
        progress_units: float,
    ) -> tuple[
        Counter,
        dict[tuple[int, int], set[int]],
    ]:
        pair_counts = Counter()
        pair_to_sequences = defaultdict(set)

        total = len(sequences)
        reported = 0.0

        for sequence_index, (sequence, frequency) in enumerate(
            zip(sequences, frequencies)
        ):
            local_counts = Counter(
                zip(sequence, sequence[1:])
            )

            for pair, local_count in local_counts.items():
                pair_counts[pair] += (
                    local_count * frequency
                )

                pair_to_sequences[pair].add(
                    sequence_index
                )

            if (
                total > 0
                and (
                    sequence_index % 1000 == 0
                    or sequence_index == total - 1
                )
            ):
                target = progress_units * (
                    (sequence_index + 1) / total
                )

                pbar.update(target - reported)
                reported = target

        return pair_counts, pair_to_sequences

    def __pop_best_pair(
        self,
        heap: list[tuple[int, tuple[int, int]]],
        pair_counts: Counter,
    ) -> tuple[tuple[int, int], int] | None:
        while heap:
            negative_count, pair = heapq.heappop(heap)
            count = -negative_count

            if (
                count > 0
                and pair_counts.get(pair, 0) == count
            ):
                return pair, count

        return None

    def __train_from_chunk_counts(
        self,
        chunk_counts: Counter,
        pbar: tqdm,
        pair_stats_progress: float = 5.0,
        merge_progress: float = 90.0,
    ) -> None:
        self.__reset_model()

        items = list(chunk_counts.items())

        sequences = [
            list(chunk)
            for chunk, _ in items
        ]

        frequencies = [
            frequency
            for _, frequency in items
        ]

        pbar.set_postfix_str("building pair statistics")

        pair_counts, pair_to_sequences = (
            self.__build_pair_stats(
                sequences,
                frequencies,
                pbar,
                pair_stats_progress,
            )
        )

        heap = [
            (-count, pair)
            for pair, count in pair_counts.items()
            if count > 0
        ]

        heapq.heapify(heap)

        total_merges = max(
            self.vocab_size - len(self.vocab),
            0,
        )

        completed_merges = 0

        progress_per_merge = (
            merge_progress / total_merges
            if total_merges > 0
            else 0.0
        )

        pbar.set_postfix_str(
            f"learning merges 0/{total_merges:,}"
        )

        while len(self.vocab) < self.vocab_size:
            best = self.__pop_best_pair(
                heap,
                pair_counts,
            )

            if best is None:
                break

            best_pair, _ = best

            new_token_id = len(self.vocab)

            left_id, right_id = best_pair

            self.vocab[new_token_id] = (
                self.vocab[left_id]
                + self.vocab[right_id]
            )

            self.merges[best_pair] = new_token_id

            affected_sequences = list(
                pair_to_sequences.get(
                    best_pair,
                    (),
                )
            )

            changed_pairs = set()

            for sequence_index in affected_sequences:
                old_sequence = sequences[sequence_index]

                old_local_counts = Counter(
                    zip(
                        old_sequence,
                        old_sequence[1:],
                    )
                )

                new_sequence = self.__merge_pair(
                    old_sequence,
                    best_pair,
                    new_token_id,
                )

                new_local_counts = Counter(
                    zip(
                        new_sequence,
                        new_sequence[1:],
                    )
                )

                frequency = frequencies[sequence_index]

                local_pairs = (
                    old_local_counts.keys()
                    | new_local_counts.keys()
                )

                for pair in local_pairs:
                    old_count = old_local_counts.get(
                        pair,
                        0,
                    )

                    new_count = new_local_counts.get(
                        pair,
                        0,
                    )

                    if old_count != new_count:
                        delta = (
                            new_count - old_count
                        ) * frequency

                        updated_count = (
                            pair_counts.get(pair, 0)
                            + delta
                        )

                        if updated_count > 0:
                            pair_counts[pair] = (
                                updated_count
                            )
                        else:
                            pair_counts.pop(
                                pair,
                                None,
                            )

                        changed_pairs.add(pair)

                    if old_count > 0 and new_count == 0:
                        sequence_ids = (
                            pair_to_sequences.get(pair)
                        )

                        if sequence_ids is not None:
                            sequence_ids.discard(
                                sequence_index
                            )

                            if not sequence_ids:
                                pair_to_sequences.pop(
                                    pair,
                                    None,
                                )

                    elif old_count == 0 and new_count > 0:
                        pair_to_sequences[pair].add(
                            sequence_index
                        )

                sequences[sequence_index] = (
                    new_sequence
                )

            for pair in changed_pairs:
                count = pair_counts.get(pair, 0)

                if count > 0:
                    heapq.heappush(
                        heap,
                        (-count, pair),
                    )

            completed_merges += 1
            pbar.update(progress_per_merge)

            if (
                completed_merges % 100 == 0
                or completed_merges == total_merges
            ):
                pbar.set_postfix_str(
                    f"learning merges "
                    f"{completed_merges:,}/{total_merges:,}"
                )

        remaining_progress = (
            merge_progress
            - completed_merges * progress_per_merge
        )

        if remaining_progress > 0:
            pbar.update(remaining_progress)

        self.__rebuild_merge_ranks()

    def train(self, text: str) -> None:
        start_time = time.perf_counter()

        pretoken_progress = 5.0
        pair_stats_progress = 5.0
        merge_progress = 90.0

        with tqdm(
            total=100.0,
            desc="Training tokenizer",
            unit="%",
            dynamic_ncols=True,
            smoothing=0.1,
        ) as pbar:
            pbar.set_postfix_str("pre-tokenizing")

            chunk_counts = Counter()

            total_chars = max(len(text), 1)
            reported = 0.0

            for match_index, match in enumerate(
                self._pattern.finditer(text)
            ):
                chunk = match.group(0).encode("utf-8")
                chunk_counts[chunk] += 1

                if match_index % 10_000 == 0:
                    target = pretoken_progress * (
                        match.end() / total_chars
                    )

                    if target > reported:
                        pbar.update(target - reported)
                        reported = target

            if reported < pretoken_progress:
                pbar.update(
                    pretoken_progress - reported
                )

            self.training_text_bytes = len(
                text.encode("utf-8")
            )

            self.__train_from_chunk_counts(
                chunk_counts,
                pbar,
                pair_stats_progress,
                merge_progress,
            )

            if pbar.n < pbar.total:
                pbar.update(
                    pbar.total - pbar.n
                )

            pbar.set_postfix_str("complete")

        self.training_time = (
            time.perf_counter() - start_time
        )

    def train_file(
        self,
        path: str | Path,
    ) -> None:
        start_time = time.perf_counter()

        path = Path(path)

        pretoken_progress = 5.0
        pair_stats_progress = 5.0
        merge_progress = 90.0

        file_size = max(
            path.stat().st_size,
            1,
        )

        chunk_counts = Counter()
        training_text_bytes = 0

        with tqdm(
            total=100.0,
            desc="Training tokenizer",
            unit="%",
            dynamic_ncols=True,
            smoothing=0.1,
        ) as pbar:
            pbar.set_postfix_str("pre-tokenizing")

            reported = 0.0

            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                for line in f:
                    line_bytes = len(
                        line.encode("utf-8")
                    )

                    training_text_bytes += line_bytes

                    chunk_counts.update(
                        match.group(0).encode("utf-8")
                        for match in self._pattern.finditer(line)
                    )

                    target = pretoken_progress * min(
                        training_text_bytes / file_size,
                        1.0,
                    )

                    if target - reported >= 0.01:
                        pbar.update(
                            target - reported
                        )
                        reported = target

            if reported < pretoken_progress:
                pbar.update(
                    pretoken_progress - reported
                )

            self.training_text_bytes = (
                training_text_bytes
            )

            self.__train_from_chunk_counts(
                chunk_counts,
                pbar,
                pair_stats_progress,
                merge_progress,
            )

            if pbar.n < pbar.total:
                pbar.update(
                    pbar.total - pbar.n
                )

            pbar.set_postfix_str("complete")

        self.training_time = (
            time.perf_counter() - start_time
        )

    def encode(self, text: str) -> list[int]:
        byte_sequences = self.__pre_tokenize(text)

        encoded: list[int] = []

        for chunk in tqdm(
            byte_sequences,
            total=len(byte_sequences),
            desc="Encoding text",
            unit="chunks",
        ):
            encoded.extend(
                self.__encode_chunk(chunk)
            )

        return encoded

    def decode(self, ids: list[int]) -> str:
        bytes_seq = b"".join(
            self.vocab[token_id]
            for token_id in tqdm(
                ids,
                total=len(ids),
                desc="Decoding tokens",
                unit="tokens",
            )
        )

        return bytes_seq.decode("utf-8")

    def save(self, path: str | Path) -> None:
        path = Path(path)

        data = {
            "vocab_size": self.vocab_size,

            "vocab": {
                str(token_id): token_bytes.hex()
                for token_id, token_bytes
                in self.vocab.items()
            },

            "merges": [
                [
                    left_id,
                    right_id,
                    new_token_id,
                ]
                for (
                    left_id,
                    right_id,
                ), new_token_id
                in self.merges.items()
            ],

            "pre_token_pattern": (
                self.pre_token_pattern
            ),

            "encode_cache_size": (
                self.encode_cache_size
            ),

            "metadata": {
                "training_time_seconds": (
                    self.training_time
                ),
                "training_text_bytes": (
                    self.training_text_bytes
                ),
                "num_merges": len(self.merges),
                "actual_vocab_size": len(self.vocab),
                "saved_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "format_version": 2,
            },
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "BPETokenizer":
        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            encode_cache_size=data.get(
                "encode_cache_size",
                100_000,
            ),
            pre_token_pattern=data.get(
                "pre_token_pattern",
                cls.DEFAULT_PRE_TOKEN_PATTERN,
            ),
        )

        tokenizer.vocab = {
            int(token_id): bytes.fromhex(
                token_hex
            )
            for token_id, token_hex
            in data["vocab"].items()
        }

        tokenizer.merges = {
            (
                left_id,
                right_id,
            ): new_token_id
            for (
                left_id,
                right_id,
                new_token_id,
            ) in data["merges"]
        }

        tokenizer.__rebuild_merge_ranks()

        metadata = data.get(
            "metadata",
            {},
        )

        tokenizer.training_time = metadata.get(
            "training_time_seconds"
        )

        tokenizer.training_text_bytes = metadata.get(
            "training_text_bytes"
        )

        return tokenizer