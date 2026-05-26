from __future__ import annotations

from typing import Optional, Tuple


class HugepageUtil:
    """Linux hugetlb pool checks via /proc/meminfo (allocation uses mmap_allocator)."""

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
