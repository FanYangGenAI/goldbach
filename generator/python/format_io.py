"""Binary dataset header and sample serialization."""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from constants import (
    CANDIDATE_PRIME_COUNT,
    HEADER_SIZE,
    MAGIC,
    SAMPLE_SIZE,
    SAMPLE_SIZE_V2,
    VALID_MASK_BYTES,
    VERSION,
    VERSION_V2,
)
from labeling import compute_valid_mask_default
from primes import compute_modulo_features


@dataclass(frozen=True)
class GoldbachSample:
    even_number: int
    modulo_features: tuple[int, ...]
    prime_density: float
    p1_label: int

    def to_bytes(self) -> bytes:
        if len(self.modulo_features) != 10:
            raise ValueError("modulo_features must contain 10 residues")
        payload = struct.pack("<Q", self.even_number)
        payload += bytes(self.modulo_features)
        payload += struct.pack("<f", self.prime_density)
        payload += struct.pack("<I", self.p1_label)
        if len(payload) != SAMPLE_SIZE:
            raise RuntimeError(f"Unexpected sample size: {len(payload)}")
        return payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "GoldbachSample":
        if len(data) != SAMPLE_SIZE:
            raise ValueError(f"Expected {SAMPLE_SIZE} bytes, got {len(data)}")
        even_number = struct.unpack("<Q", data[0:8])[0]
        modulo_features = tuple(data[8:18])
        prime_density = struct.unpack("<f", data[18:22])[0]
        p1_label = struct.unpack("<I", data[22:26])[0]
        return cls(
            even_number=even_number,
            modulo_features=modulo_features,
            prime_density=prime_density,
            p1_label=p1_label,
        )


def encode_valid_mask(
    mask: Sequence[int],
    bit_count: int = CANDIDATE_PRIME_COUNT,
) -> bytes:
    """Pack multi-hot mask into little-endian bitset; bit i -> byte[i//8], shift i%8."""
    if len(mask) != bit_count:
        raise ValueError(f"mask length must be {bit_count}, got {len(mask)}")
    buf = bytearray(VALID_MASK_BYTES)
    for index, value in enumerate(mask):
        if value not in (0, 1):
            raise ValueError(f"mask values must be 0 or 1, got {value} at index {index}")
        if value:
            buf[index // 8] |= 1 << (index % 8)
    return bytes(buf)


def decode_valid_mask(
    data: bytes,
    bit_count: int = CANDIDATE_PRIME_COUNT,
) -> tuple[int, ...]:
    """Unpack little-endian bitset into multi-hot mask."""
    if len(data) != VALID_MASK_BYTES:
        raise ValueError(f"Expected {VALID_MASK_BYTES} mask bytes, got {len(data)}")
    mask: list[int] = []
    for index in range(bit_count):
        mask.append((data[index // 8] >> (index % 8)) & 1)
    return tuple(mask)


@dataclass(frozen=True)
class GoldbachSampleV2:
    even_number: int
    modulo_features: tuple[int, ...]
    prime_density: float
    valid_mask: tuple[int, ...]

    def to_bytes(self) -> bytes:
        if len(self.modulo_features) != 10:
            raise ValueError("modulo_features must contain 10 residues")
        payload = struct.pack("<Q", self.even_number)
        payload += bytes(self.modulo_features)
        payload += struct.pack("<f", self.prime_density)
        payload += encode_valid_mask(self.valid_mask)
        if len(payload) != SAMPLE_SIZE_V2:
            raise RuntimeError(f"Unexpected sample size: {len(payload)}")
        return payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "GoldbachSampleV2":
        if len(data) != SAMPLE_SIZE_V2:
            raise ValueError(f"Expected {SAMPLE_SIZE_V2} bytes, got {len(data)}")
        even_number = struct.unpack("<Q", data[0:8])[0]
        modulo_features = tuple(data[8:18])
        prime_density = struct.unpack("<f", data[18:22])[0]
        valid_mask = decode_valid_mask(data[22:22 + VALID_MASK_BYTES])
        return cls(
            even_number=even_number,
            modulo_features=modulo_features,
            prime_density=prime_density,
            valid_mask=valid_mask,
        )


@dataclass(frozen=True)
class DatasetHeader:
    magic: int
    version: int
    sample_count: int
    feature_flags: int
    header_crc32: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "DatasetHeader":
        if len(data) < 28:
            raise ValueError("Header is too short")
        magic, version, sample_count, feature_flags, header_crc32 = struct.unpack(
            "<IIQQI", data[:28]
        )
        return cls(
            magic=magic,
            version=version,
            sample_count=sample_count,
            feature_flags=feature_flags,
            header_crc32=header_crc32,
        )


SUPPORTED_VERSIONS = (VERSION, VERSION_V2)


def sample_size_for_version(version: int) -> int:
    if version == VERSION:
        return SAMPLE_SIZE
    if version == VERSION_V2:
        return SAMPLE_SIZE_V2
    raise ValueError(f"Unsupported dataset version: {version}")


def build_header(
    sample_count: int,
    feature_flags: int = 0,
    version: int = VERSION,
) -> bytes:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported dataset version: {version}")
    body = struct.pack("<IIQQ", MAGIC, version, sample_count, feature_flags)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    header = body + struct.pack("<I", crc) + bytes(36)
    if len(header) != HEADER_SIZE:
        raise RuntimeError(f"Unexpected header size: {len(header)}")
    return header


def verify_header(header: bytes) -> DatasetHeader:
    if len(header) < HEADER_SIZE:
        raise ValueError("Header is too short")
    parsed = DatasetHeader.from_bytes(header)
    if parsed.magic != MAGIC:
        raise ValueError(f"Invalid magic: {parsed.magic:#x}")
    if parsed.version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version: {parsed.version}")
    body = header[:24]
    expected_crc = zlib.crc32(body) & 0xFFFFFFFF
    if parsed.header_crc32 != expected_crc:
        raise ValueError(
            f"Header CRC mismatch: expected {expected_crc:#x}, got {parsed.header_crc32:#x}"
        )
    return parsed


def make_sample(even_number: int, p1_label: int) -> GoldbachSample:
    return GoldbachSample(
        even_number=even_number,
        modulo_features=compute_modulo_features(even_number),
        prime_density=1.0 / math.log(even_number),
        p1_label=p1_label,
    )


def make_sample_v2(
    even_number: int,
    valid_mask: Sequence[int] | None = None,
    *,
    candidate_count: int = CANDIDATE_PRIME_COUNT,
) -> GoldbachSampleV2:
    mask = (
        tuple(valid_mask)
        if valid_mask is not None
        else compute_valid_mask_default(even_number, candidate_count)
    )
    if len(mask) != candidate_count:
        raise ValueError(
            f"valid_mask length must be {candidate_count}, got {len(mask)}"
        )
    return GoldbachSampleV2(
        even_number=even_number,
        modulo_features=compute_modulo_features(even_number),
        prime_density=1.0 / math.log(even_number),
        valid_mask=mask,
    )
