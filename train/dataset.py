"""PyTorch Dataset for Goldbach binary files (v1 and v2)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_GENERATOR_PYTHON = Path(__file__).resolve().parents[1] / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import (  # noqa: E402
    CANDIDATE_PRIME_COUNT,
    HEADER_SIZE,
    MIN_TRAINING_EVEN,
    VALID_MASK_BYTES,
    VERSION_V2,
)
from format_io import sample_size_for_version, verify_header  # noqa: E402


def even_to_bits(even_number: int) -> np.ndarray:
    """Expand even_number into 64 bit features (bit 0 = LSB)."""
    return ((even_number >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)


def decode_valid_mask_array(
    mask_bytes: bytes | np.ndarray,
    bit_count: int = CANDIDATE_PRIME_COUNT,
) -> np.ndarray:
    """Decode little-endian bitset bytes into float32 multi-hot mask."""
    if isinstance(mask_bytes, (bytes, bytearray)):
        array = np.frombuffer(mask_bytes, dtype=np.uint8)
    else:
        array = mask_bytes.astype(np.uint8)
    bits = np.unpackbits(array, bitorder="little")
    return bits[:bit_count].astype(np.float32)


class GoldbachBinDataset(Dataset):
    """Memory-mapped Goldbach dataset supporting v1 (p1_label) and v2 (valid_mask)."""

    def __init__(
        self,
        path: Path | str,
        min_even: int = MIN_TRAINING_EVEN,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with self.path.open("rb") as handle:
            header = verify_header(handle.read(HEADER_SIZE))

        self.version = header.version
        self.record_size = sample_size_for_version(header.version)
        self.raw_sample_count = int(header.sample_count)
        self.min_even = min_even
        self.candidate_count = CANDIDATE_PRIME_COUNT if self.version == VERSION_V2 else None

        self._mmap = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=HEADER_SIZE,
            shape=(self.raw_sample_count, self.record_size),
        )
        self._indices = self._build_indices(min_even)

    def _build_indices(self, min_even: int) -> list[int] | None:
        if min_even <= 4:
            return None
        indices: list[int] = []
        for index in range(self.raw_sample_count):
            even_number = int.from_bytes(self._mmap[index, 0:8].tobytes(), "little")
            if even_number >= min_even:
                indices.append(index)
        return indices

    def __len__(self) -> int:
        if self._indices is None:
            return self.raw_sample_count
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"index out of range: {index}")
        row_index = self._indices[index] if self._indices is not None else index

        row = self._mmap[row_index]
        even_number = int.from_bytes(row[0:8].tobytes(), "little")
        mods = row[8:18].astype(np.int64)
        density = struct.unpack("<f", row[18:22].tobytes())[0]

        item: dict[str, torch.Tensor] = {
            "even": torch.tensor(even_number, dtype=torch.int64),
            "bits": torch.from_numpy(even_to_bits(even_number)),
            "mods": torch.from_numpy(mods),
            "density": torch.tensor(density, dtype=torch.float32),
        }

        if self.version == VERSION_V2:
            mask_bytes = row[22 : 22 + VALID_MASK_BYTES]
            valid_mask = decode_valid_mask_array(mask_bytes, self.candidate_count)
            item["valid_mask"] = torch.from_numpy(valid_mask)
        else:
            p1_label = int(struct.unpack("<I", row[22:26].tobytes())[0])
            item["p1_label"] = torch.tensor(p1_label, dtype=torch.int64)

        return item
