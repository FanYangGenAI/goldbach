"""Multiprocessing worker helpers."""

from __future__ import annotations

from format_io import make_sample
from labeling import find_min_p1


def label_even_chunk(evens: list[int]) -> tuple[bytes, list[str]]:
    payload = bytearray()
    errors: list[str] = []
    for even_number in evens:
        try:
            p1 = find_min_p1(even_number)
            payload.extend(make_sample(even_number, p1).to_bytes())
        except RuntimeError as exc:
            errors.append(f"even={even_number}: {exc}")
    return bytes(payload), errors
