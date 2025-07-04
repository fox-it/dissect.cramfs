from __future__ import annotations

from typing import BinaryIO

from dissect.cramfs.cramfs import CramFS, c_cramfs


def test_cramfs(cramfs: BinaryIO) -> None:
    cramfs = CramFS(cramfs)
    assert cramfs.sb.magic == c_cramfs.CRAMFS_MAGIC
    assert cramfs.sb.size == 69632
    assert cramfs.sb.flags == 0x3
    assert cramfs.sb.signature == b"Compressed ROMFS"
    assert cramfs.root.is_dir()
    assert sorted(cramfs.root.listdir().keys()) == ["etc", "home", "usr", "var"]

    file = cramfs.get("/home/user/.bashrc")
    assert file.is_file()
    assert file.size == 10
    assert file.open().read() == b"PS1='UwU'\n"

    file2 = cramfs.get("/var/log/access.log")
    assert file2.is_file()
    assert file2.size == 5
    assert file2.open().read() == b"test\n"


def test_webcramfs(webcramfs: BinaryIO) -> None:
    cramfs = CramFS(webcramfs)
    assert cramfs.sb.magic == c_cramfs.CRAMFS_MAGIC
    assert cramfs.sb.size == 3088384
    assert cramfs.sb.flags == 0x3
    assert cramfs.sb.signature == b"Compressed ROMFS"
    assert sorted(cramfs.root.listdir().keys()) == [
        "bin",
        "boot",
        "dev",
        "etc",
        "home",
        "lib",
        "linuxrc",
        "mnt",
        "opt",
        "proc",
        "root",
        "sbin",
        "share",
        "slv",
        "sys",
        "tmp",
        "usr",
        "utils",
        "var",
    ]

    file = cramfs.get("/bin/busybox")
    assert file.is_file()
    assert file.size == 330256
    assert len(file.open().read()) == file.size

    symlink = cramfs.get("/bin/macGuarder")
    assert symlink.is_symlink()
    assert symlink.link == "./dvrbox"


def test_holecramfs(holecramfs: BinaryIO) -> None:
    cramfs = CramFS(holecramfs)
    assert cramfs.sb.magic == c_cramfs.CRAMFS_MAGIC
    assert cramfs.sb.size == 4096
    assert cramfs.sb.flags == 0x103
    assert cramfs.sb.signature == b"Compressed ROMFS"
    assert sorted(cramfs.root.listdir().keys()) == [
        "dev",
        "empty.txt",
        "folder",
        "muchnull.txt",
        "somenull.txt",
        "test.txt",
    ]

    # test empty file
    empty_file = cramfs.get("empty.txt")
    assert empty_file.is_file()
    assert empty_file.size == 0
    assert empty_file.offset == 0
    assert empty_file.open().read() == b""

    # test complete sparse file
    hole_file = cramfs.get("muchnull.txt")
    assert hole_file.is_file()
    assert hole_file.size == 69420
    assert hole_file.offset == 292
    assert hole_file.open().read() == b"\x00" * 69420

    # test partial sparse file
    hole_file = cramfs.get("somenull.txt")
    assert hole_file.is_file()
    assert hole_file.size == 3308
    assert hole_file.offset == 360
    assert hole_file.open().read() == b"\x00" * 1234 + b"\x69" * 420 + b"\x00" * 1234 + b"\x69" * 420

    # test device files
    dev_file = cramfs.get("/dev/blocky")
    assert dev_file.is_block_device()
    assert dev_file.size == 0
    assert dev_file.major == 13
    assert dev_file.minor == 37
    assert dev_file.offset == 0

    dev_file = cramfs.get("/dev/chary")
    assert dev_file.is_character_device()
    assert dev_file.size == 0
    assert dev_file.major == 69
    assert dev_file.minor == 69
    assert dev_file.offset == 0
