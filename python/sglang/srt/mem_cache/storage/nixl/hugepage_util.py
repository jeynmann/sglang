from __future__ import annotations

import ctypes
import logging
from typing import Any, Optional, Tuple

from sglang.srt.mem_cache.storage.nixl.cuda_host_register_util import (
    CudaHostRegisterUtil,
)

logger = logging.getLogger(__name__)

_PROT_READ = 0x1
_PROT_WRITE = 0x2

_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_HUGETLB = 0x40000
_MAP_FAILED = ctypes.c_void_p(-1).value

# linux/mman.h: log2(huge page size in bytes) << 26 for explicit 2 MiB pages.
_MAP_HUGE_SHIFT = 26
_MAP_HUGE_2MB = 21 << _MAP_HUGE_SHIFT

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_size_t,
]
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIBC.munmap.restype = ctypes.c_int


class HugepageBufferHolder:
    """Owns a hugetlb ``mmap`` mapping; unregisters and unmaps on destruction."""

    __slots__ = ("addr", "length", "cuda_registered")

    def __init__(self, addr: int, length: int) -> None:
        self.addr = addr
        self.length = length
        self.cuda_registered = False

    def cuda_register(self) -> None:
        """Register the full mmap with CUDA (idempotent)."""
        if self.cuda_registered:
            return
        CudaHostRegisterUtil.register(self.addr, self.length, 0)
        self.cuda_registered = True

    def cuda_unregister(self) -> None:
        """Unregister from CUDA if previously registered (idempotent)."""
        if not self.cuda_registered:
            return
        CudaHostRegisterUtil.unregister(self.addr)
        self.cuda_registered = False

    def munmap(self) -> None:
        """Release the hugetlb mapping (idempotent)."""
        if self.addr == 0:
            return
        try:
            _LIBC.munmap(ctypes.c_void_p(self.addr), ctypes.c_size_t(self.length))
        except Exception:
            pass
        self.addr = 0

    def release(self) -> None:
        """Unregister from CUDA and unmap."""
        try:
            self.cuda_unregister()
        except Exception:
            logger.debug(
                "Failed to unregister hugepage buffer from CUDA during release",
                exc_info=True,
            )
        self.munmap()

    def __del__(self) -> None:
        self.release()


class HugepageUtil:
    """Linux hugepage cache/validation/mmap helpers."""

    _hugepages_info: Optional[Tuple[int, int, int]] = None
    _hugepages_cached: bool = False

    @classmethod
    def _read_meminfo_from_proc(cls) -> Optional[Tuple[int, int, int]]:
        try:
            vals: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    for key in ("HugePages_Total", "HugePages_Free", "Hugepagesize"):
                        if line.startswith(key):
                            vals[key] = int(line.split()[1])
            if len(vals) == 3:
                return (
                    vals["HugePages_Total"],
                    vals["HugePages_Free"],
                    vals["Hugepagesize"],
                )
        except OSError:
            pass
        return None

    @classmethod
    def read_meminfo(cls, refresh: bool = False) -> Optional[Tuple[int, int, int]]:
        if refresh or not cls._hugepages_cached:
            cls._hugepages_info = cls._read_meminfo_from_proc()
            cls._hugepages_cached = True
        return cls._hugepages_info

    @classmethod
    def hugepage_size_bytes(cls) -> int:
        info = cls.read_meminfo()
        if info is None:
            return 2 * 1024 * 1024
        _, _, size_kb = info
        return int(size_kb) * 1024

    @staticmethod
    def align_up(size: int, alignment: int) -> int:
        return (size + alignment - 1) // alignment * alignment

    @classmethod
    def mmap_buffer(cls, num_bytes: int) -> Tuple[Any, HugepageBufferHolder]:
        """Map hugetlb-backed memory.

        Returns ``(ctypes_byte_array, holder)``.
        """
        page_size = cls.hugepage_size_bytes()
        mapped_len = cls.align_up(num_bytes, page_size)

        flags = _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB
        if page_size == 2 * 1024 * 1024:
            flags |= _MAP_HUGE_2MB
        addr = _LIBC.mmap(
            None,
            ctypes.c_size_t(mapped_len),
            _PROT_READ | _PROT_WRITE,
            flags,
            -1,
            0,
        )
        addr_u = ctypes.cast(addr, ctypes.c_void_p).value or 0
        if addr_u == _MAP_FAILED or addr_u == 0:
            errno = ctypes.get_errno()
            diag = cls.read_meminfo(refresh=True)
            if diag is not None:
                total, free, page_kb = diag
                need_pages = (mapped_len + page_size - 1) // page_size
                raise RuntimeError(
                    "mmap(MAP_HUGETLB) failed "
                    f"(errno={errno}, requested_bytes={num_bytes}, mapped_len={mapped_len}, "
                    f"total={total}, free={free}, hugepage_size={page_kb} kB, need={need_pages}). "
                    "Reserve pages with `sysctl vm.nr_hugepages=<N>`."
                )
            raise RuntimeError(
                f"mmap(MAP_HUGETLB) failed (errno={errno}); /proc/meminfo unavailable."
            )

        array_type = (ctypes.c_byte * mapped_len).from_address(addr_u)
        holder = HugepageBufferHolder(addr_u, mapped_len)
        return array_type, holder

    @classmethod
    def validate_meminfo(cls) -> bool:
        return cls.read_meminfo() is not None

    @classmethod
    def require_free_bytes(cls, num_bytes: int, label: str = "") -> None:
        info = cls.read_meminfo(refresh=True)
        if info is None:
            raise RuntimeError(
                f"{label}Could not read huge page stats from /proc/meminfo.".strip()
            )
        total, free, page_kb = info
        page_size = int(page_kb) * 1024
        need = (num_bytes + page_size - 1) // page_size
        if free < need:
            raise RuntimeError(
                f"{label}Not enough free huge pages: need {need} pages ({num_bytes} bytes), "
                f"have {free} free (total={total}, hugepage_size={page_kb} kB).".strip()
            )
