from __future__ import annotations

import stat
import zlib
from bisect import bisect_right
from functools import cache, cached_property
from typing import TYPE_CHECKING, BinaryIO

from dissect.util.stream import AlignedStream, RangeStream

from dissect.cramfs.c_cramfs import (
    BLOCK_SIZE,
    DIRECT_POINTER_FLAG,
    UNCOMPRESSED_BLOCK_FLAG,
    c_cramfs,
)
from dissect.cramfs.exceptions import (
    FileNotFoundError,
    NotADirectoryError,
    NotAFileError,
    NotASymlinkError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class CramFS:
    """CramFS filesystem implementation.

    Args:
        fh: A file-like object of the volume containing the filesystem.
    """

    def __init__(self, fh: BinaryIO):
        fh.seek(0)
        self.fh = fh
        self.sb = c_cramfs.cramfs_super_block(fh)

        if self.sb.magic != c_cramfs.CRAMFS_MAGIC:
            raise ValueError("Invalid CramFS filesystem")

        self.root = INode(self, (self.sb.root.offset << 2) - 12)

    def get(self, path: str, node: INode | None = None) -> INode:
        """Return an inode object for the given path or inode number.

        Args:
            path: The path of the inode.
            node: An optional inode object to relatively resolve the path from.
        """
        node = node or self.root
        parts = path.split("/")

        for part in filter(None, parts):
            for entry in node.iterdir():
                if entry.name == part:
                    node = entry
                    break
            else:
                raise FileNotFoundError(f"File not found: {path}")
        return node


class INode:
    def __init__(
        self,
        fs: CramFS,
        offset: int,
        name: bytes | None = None,
    ):
        self.fs = fs
        self.fh = self.fs.fh
        self._name = name
        self._offset = offset

        self.header = self.inode
        self.listdir = cache(self.listdir)

    def __repr__(self) -> str:
        return f"<INode {self.name!r} ({self.offset})>"

    @cached_property
    def inode(self) -> c_cramfs.cramfs_inode:
        """Return the inode header."""
        self.fh.seek(self._offset)
        return c_cramfs.cramfs_inode(self.fh)

    @property
    def mode(self) -> int:
        """Return the inode mode."""
        return self.header.mode

    @property
    def uid(self) -> int:
        """Return the user ID of the inode."""
        return self.header.uid

    @property
    def major(self) -> int:
        """Return the major device ID for block and character devices."""
        if not self.is_device():
            raise NotAFileError(f"{self!r} is not character- or block device")
        return (self.header.size >> 8) & 0xFF

    @property
    def minor(self) -> int:
        """Return the minor device ID for block and character devices."""
        if not self.is_device():
            raise NotAFileError(f"{self!r} is not character- or block device")
        return self.header.size & 0xFF

    @property
    def size(self) -> int:
        """Return the file size."""
        if self.is_device():
            return 0
        return self.header.size

    @property
    def gid(self) -> int:
        """Return the group ID of the inode."""
        return self.header.gid

    @property
    def namelen(self) -> int:
        """Return the length of the name in bytes."""
        # namelen is stored divided by 4
        return self.header.namelen * 4

    @property
    def offset(self) -> int:
        """Offset to the start of the data block or ``INode``.

        * For files: this is the offset to the first data block.
        * For directories: this is the offset to the first ``INode``.
        * For symlinks: this is the offset the data block holding the target name.
        """
        return self.header.offset << 2

    @property
    def name(self) -> str:
        """Return the file name."""
        if self._name:
            return self._name.decode().strip("\x00")
        return self.header.name.decode().strip("\x00")

    @cached_property
    def link(self) -> str:
        """Return the symlink target."""
        if not self.is_symlink():
            raise NotASymlinkError(f"{self!r} is not a symlink")
        return self.open().read().decode().strip("\x00")

    @cached_property
    def numblocks(self) -> int:
        """Return the number of blocks in the file."""
        return ((self.size + BLOCK_SIZE) - 1) // BLOCK_SIZE

    @cached_property
    def dataruns(self) -> list[tuple[int, int]]:
        """Return the start and end offset of each data block."""
        self.fs.fh.seek(self.offset)

        start = self.offset + self.numblocks * 4
        blocks = []
        for _ in range(self.numblocks):
            end = c_cramfs.uint32(self.fs.fh)
            blocks.append((start, end))
            start = end
        return blocks

    def is_dir(self) -> bool:
        """Return whether this inode is a directory."""
        return stat.S_ISDIR(self.mode)

    def is_file(self) -> bool:
        """Return whether this inode is a file."""
        return stat.S_ISREG(self.mode)

    def is_symlink(self) -> bool:
        """Return whether this inode is a symlink."""
        return stat.S_ISLNK(self.mode)

    def is_block_device(self) -> bool:
        """Return whether this inode is a block device."""
        return stat.S_ISBLK(self.mode)

    def is_character_device(self) -> bool:
        """Return whether this inode is a character device."""
        return stat.S_ISCHR(self.mode)

    def is_device(self) -> bool:
        """Return whether this inode is a device file."""
        return self.is_block_device() or self.is_character_device()

    def is_fifo(self) -> bool:
        """Return whether this inode is a FIFO (named pipe)."""
        return stat.S_ISFIFO(self.mode)

    def is_socket(self) -> bool:
        """Return whether this inode is a socket."""
        return stat.S_ISSOCK(self.mode)

    def is_ipc(self) -> bool:
        """Return whether this inode is an IPC object (FIFO or socket)."""
        return self.is_fifo() or self.is_socket()

    def listdir(self) -> dict[str, INode]:
        """Return a directory listing."""
        return {inode.name: inode for inode in self.iterdir()}

    dirlist = listdir

    def iterdir(self) -> Iterator[INode]:
        """Iterate directory contents."""
        if not self.is_dir():
            raise NotADirectoryError(f"{self!r} is not a directory")

        self.fh.seek(self.offset)
        end = self.offset + self.size
        while (offset := self.fh.tell()) != end:
            yield INode(self.fs, offset)

    def open(self) -> FileStream | RangeStream:
        """Return a file-like object for reading."""
        if self.is_dir():
            return RangeStream(self.fs.fh, self.offset, self.size)
        return FileStream(self)


class FileStream(AlignedStream):
    def __init__(self, inode: INode):
        super().__init__(inode.size, BLOCK_SIZE)
        self._fh = inode.fs.fh
        self.offset = inode.offset

        self.inode = inode
        self.dataruns = self.inode.dataruns

    def _read(self, offset: int, length: int) -> bytes:
        result = []

        offset = offset // BLOCK_SIZE
        run_idx = bisect_right(list(range(1, self.inode.numblocks)), offset)
        run_len = len(self.inode.dataruns)

        while length > 0:
            if run_idx >= run_len:
                break

            start, end = self.dataruns[run_idx]
            read_len = end - start

            data = self._read_block(start, read_len)
            result.append(data)

            offset += read_len
            length -= BLOCK_SIZE
            run_idx += 1

        return b"".join(result)

    def _read_block(self, offset: int, size: int) -> bytes:
        """Read a block of data from the filesystem.

        Args:
            offset: The offset of the block.
            size: The size of the block to read.
        """
        uncompressed = offset & UNCOMPRESSED_BLOCK_FLAG
        direct = offset & DIRECT_POINTER_FLAG

        if direct:
            raise NotImplementedError("Direct pointers are not supported yet.")

        offset = offset & ~(UNCOMPRESSED_BLOCK_FLAG | DIRECT_POINTER_FLAG)

        self._fh.seek(offset)
        data = self._fh.read(size)
        if not data:
            # sparse block aka hole
            return b"\x00" * BLOCK_SIZE

        if uncompressed:
            return data

        return zlib.decompress(data)
