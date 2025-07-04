from __future__ import annotations

from gzip import GzipFile
from pathlib import Path
from typing import IO, TYPE_CHECKING, BinaryIO

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def absolute_path(filename: str) -> Path:
    return Path(__file__).parent / filename


def open_file_gz(name: str, mode: str = "rb") -> Iterator[IO]:
    with GzipFile(absolute_path(name), mode) as fh:
        yield fh


@pytest.fixture
def cramfs() -> Iterator[BinaryIO]:
    yield from open_file_gz("_data/cramfs.img.gz")


@pytest.fixture
def webcramfs() -> Iterator[BinaryIO]:
    yield from open_file_gz("_data/webcramfs.img.gz")


@pytest.fixture
def holecramfs() -> Iterator[BinaryIO]:
    yield from open_file_gz("_data/holecramfs.img.gz")
