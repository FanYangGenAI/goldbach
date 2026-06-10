"""PyTorch Dataset for Goldbach binary files (v1 and v2).

Goldbach 二进制训练集的 PyTorch Dataset，支持 v1（p1_label）和 v2（valid_mask）。
Uses memory-mapped I/O for large files without loading everything into RAM.
使用 mmap 按需读取，避免将整个数据集载入内存。
"""

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
    """Expand even_number into 64 bit features (bit 0 = LSB).

    将偶数 N 展开为 64 维 bit 特征（bit 0 为最低位 LSB）。
    """
    return ((even_number >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)


def decode_valid_mask_array(
    mask_bytes: bytes | np.ndarray,
    bit_count: int = CANDIDATE_PRIME_COUNT,
) -> np.ndarray:
    """Decode little-endian bitset bytes into float32 multi-hot mask.

    将 little-endian 位集字节解码为 float32 多热掩码（K 维）。
    """
    if isinstance(mask_bytes, (bytes, bytearray)):
        array = np.frombuffer(mask_bytes, dtype=np.uint8)
    else:
        array = mask_bytes.astype(np.uint8)
    bits = np.unpackbits(array, bitorder="little")
    return bits[:bit_count].astype(np.float32)


class GoldbachBinDataset(Dataset):
    """Memory-mapped Goldbach dataset supporting v1 (p1_label) and v2 (valid_mask).

    内存映射的 Goldbach 数据集：
    - v1: 单标签 p1_label（第一个有效候选的索引）
    - v2: 多标签 valid_mask（K 维 bitset，标记所有有效候选）
    """

    def __init__(
        self,
        path: Path | str,
        min_even: int = MIN_TRAINING_EVEN,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        # --- Step 1: read and validate file header ---
        # --- 步骤 1：读取并校验文件头 ---
        with self.path.open("rb") as handle:
            header = verify_header(handle.read(HEADER_SIZE))

        self.version = header.version
        self.record_size = sample_size_for_version(header.version)
        self.raw_sample_count = int(header.sample_count)
        self.min_even = min_even
        self.candidate_count = CANDIDATE_PRIME_COUNT if self.version == VERSION_V2 else None

        # --- Step 2: mmap payload region (skip header) ---
        # --- 步骤 2：对数据区做内存映射（跳过文件头） ---
        self._mmap = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=HEADER_SIZE,
            shape=(self.raw_sample_count, self.record_size),
        )

        # --- Step 3: build index list excluding N < min_even ---
        # --- 步骤 3：构建索引，过滤 N < min_even 的样本 ---
        self._indices = self._build_indices(min_even)

    def _build_indices(self, min_even: int) -> list[int] | None:
        """Return row indices with even_number >= min_even, or None if no filter.

        返回满足 N >= min_even 的行索引；若 min_even <= 4 则不过滤，返回 None。
        """
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
        """Load one sample and decode all fields into tensors.

        加载一条样本，解码为 PyTorch 张量字典。

        v2 record layout (85 bytes):
            [0:8]   even_number (uint64 LE)
            [8:18]  mods[10]     (uint8, N mod small primes)
            [18:22] density      (float32, 1/log N)
            [22:85] valid_mask   (63 bytes bitset -> 500 bits)
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"index out of range: {index}")
        row_index = self._indices[index] if self._indices is not None else index

        row = self._mmap[row_index]

        # --- Decode fixed fields shared by v1 and v2 ---
        # --- 解码 v1/v2 共用的固定字段 ---
        even_number = int.from_bytes(row[0:8].tobytes(), "little")
        mods = row[8:18].astype(np.int64)
        density = struct.unpack("<f", row[18:22].tobytes())[0]

        item: dict[str, torch.Tensor] = {
            "even": torch.tensor(even_number, dtype=torch.int64),
            "bits": torch.from_numpy(even_to_bits(even_number)),
            "mods": torch.from_numpy(mods),
            "density": torch.tensor(density, dtype=torch.float32),
        }

        # --- Version-specific label field ---
        # --- 版本相关的标签字段 ---
        if self.version == VERSION_V2:
            mask_bytes = row[22 : 22 + VALID_MASK_BYTES]
            valid_mask = decode_valid_mask_array(mask_bytes, self.candidate_count)
            item["valid_mask"] = torch.from_numpy(valid_mask)
        else:
            p1_label = int(struct.unpack("<I", row[22:26].tobytes())[0])
            item["p1_label"] = torch.tensor(p1_label, dtype=torch.int64)

        return item
