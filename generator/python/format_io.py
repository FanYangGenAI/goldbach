"""Binary dataset header and sample serialization."""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass

from constants import HEADER_SIZE, MAGIC, SAMPLE_SIZE, VERSION
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


def build_header(sample_count: int, feature_flags: int = 0) -> bytes:
    body = struct.pack("<IIQQ", MAGIC, VERSION, sample_count, feature_flags)
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
    if parsed.version != VERSION:
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
