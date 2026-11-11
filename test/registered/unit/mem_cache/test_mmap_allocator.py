import tempfile

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import mmap_allocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import mmap
import os
import unittest

import torch

from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm


class TestMmapAllocator(unittest.TestCase):
    def test_alloc_mmap(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor = alloc_mmap(dims, dtype)
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        # Verify it has mapped memory address
        self.assertGreater(tensor.data_ptr(), 0)

    def test_alloc_mmap_hugepage_modes(self):
        import ctypes
        from unittest.mock import patch

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
            MEM_BACKEND_HUGEPAGE,
            MEM_BACKEND_MMAP,
            tensor_mem_backend,
        )

        with envs.SGLANG_HUGEPAGE_MODE.override("off"):
            with envs.SGLANG_HUGEPAGE_SIZE.override(""):
                buf = mmap_allocator.alloc_mmap((4,), torch.float32)
                self.assertEqual(tensor_mem_backend(buf), MEM_BACKEND_MMAP)

        with envs.SGLANG_HUGEPAGE_MODE.override("prefer"):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(mmap_allocator, "_alloc_hugepage") as mock_hp:
                    mock_hp.return_value = (ctypes.c_uint8 * 16)()
                    buf = mmap_allocator.alloc_mmap((4,), torch.float32)
                self.assertEqual(tensor_mem_backend(buf), MEM_BACKEND_HUGEPAGE)

        with envs.SGLANG_HUGEPAGE_MODE.override("prefer"):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(
                    mmap_allocator,
                    "_alloc_hugepage",
                    side_effect=OSError("no hugepages"),
                ):
                    with patch.object(mmap_allocator, "_libc", object()):
                        buf = mmap_allocator.alloc_mmap((4,), torch.float32)
                self.assertEqual(tensor_mem_backend(buf), MEM_BACKEND_MMAP)

        with envs.SGLANG_HUGEPAGE_MODE.override("required"):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(
                    mmap_allocator,
                    "_alloc_hugepage",
                    side_effect=OSError("no hugepages"),
                ):
                    with patch.object(mmap_allocator, "_libc", object()):
                        with patch.object(mmap_allocator.mmap, "mmap") as plain_mmap:
                            with self.assertRaisesRegex(RuntimeError, "no hugepages"):
                                mmap_allocator.alloc_mmap((4,), torch.float32)
                            plain_mmap.assert_not_called()

        with envs.SGLANG_HUGEPAGE_MODE.override("required"):
            with envs.SGLANG_HUGEPAGE_SIZE.override(""):
                with self.assertRaisesRegex(ValueError, "SGLANG_HUGEPAGE_SIZE"):
                    mmap_allocator.alloc_mmap((4,), torch.float32)

    def test_memory_available_bytes_hugepage_modes(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
            HICACHE_HOST_MEMORY_RESERVE_BYTES,
            HUGEPAGE_BYTES_2MB,
            memory_available_bytes,
        )

        normal_usable_100mb = 100 * 1024 * 1024
        normal_ram_with_reserve = (
            HICACHE_HOST_MEMORY_RESERVE_BYTES + normal_usable_100mb
        )
        hugetlb_200mb = 100 * HUGEPAGE_BYTES_2MB
        low_ram = MagicMock(available=normal_ram_with_reserve)
        with patch.object(
            mmap_allocator.psutil, "virtual_memory", return_value=low_ram
        ):
            with envs.SGLANG_HUGEPAGE_MODE.override("off"):
                with envs.SGLANG_HUGEPAGE_SIZE.override(""):
                    self.assertEqual(memory_available_bytes(), normal_usable_100mb)
            with envs.SGLANG_HUGEPAGE_MODE.override("prefer"):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(
                        mmap_allocator, "hugepage_available_bytes", return_value=0
                    ):
                        self.assertEqual(memory_available_bytes(), normal_usable_100mb)
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(
                        mmap_allocator,
                        "hugepage_available_bytes",
                        return_value=hugetlb_200mb,
                    ):
                        self.assertEqual(memory_available_bytes(), hugetlb_200mb)

        hugetlb_bytes = 3 * HUGEPAGE_BYTES_2MB
        with patch.object(
            mmap_allocator.psutil,
            "virtual_memory",
            return_value=MagicMock(available=1 << 50),
        ):
            with envs.SGLANG_HUGEPAGE_MODE.override("required"):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(
                        mmap_allocator,
                        "hugepage_available_bytes",
                        return_value=hugetlb_bytes,
                    ):
                        self.assertEqual(memory_available_bytes(), hugetlb_bytes)

    def test_alloc_shm(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertGreater(tensor.data_ptr(), 0)
        self.assertGreaterEqual(fd, 0)
        self.assertIsInstance(mm, mmap.mmap)

        # Check that we can write to the tensor
        tensor[0, 0] = 42.0
        self.assertEqual(tensor[0, 0].item(), 42.0)

        # Check that the FD is open and valid
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            self.fail("FD is not valid or closed")

        # Cleanup
        mm.close()
        os.close(fd)

    def test_shm_host_tensor_allocator(self):
        allocator = ShmHostTensorAllocator()
        dims = (2, 512)
        dtype = torch.int32

        tensor = allocator.allocate(dims, dtype, "cpu")
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertIsNotNone(allocator.fd)
        self.assertGreaterEqual(allocator.fd, 0)

        # Write data and check
        tensor[1, 1] = 99
        self.assertEqual(tensor[1, 1].item(), 99)

        # Test destructor cleans up fd
        fd = allocator.fd
        # Trigger GC / deletion
        del allocator

        # Verify fd is closed
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_alloc_shm_unlinked(self):
        dims = (4, 256)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        # On Linux, the path of an unlinked fd shows up in /proc/self/fd/
        # with a ' (deleted)' suffix.
        fd_path = f"/proc/self/fd/{fd}"
        try:
            resolved_path = os.readlink(fd_path)
            self.assertIn("sglang_host_pool_", resolved_path)
            self.assertTrue(resolved_path.endswith(" (deleted)"))
        except OSError:
            # If procfs is not available or readlink fails, fallback to direct path existence check
            self.assertFalse(os.path.exists(f"/dev/shm/sglang_host_pool_"))

        # Cleanup
        mm.close()
        os.close(fd)

    def test_alloc_shm_hugepage_warning(self):
        from sglang.srt.environ import envs

        envs.SGLANG_HUGEPAGE_SIZE.override("2MB")
        try:
            # Should succeed by falling back to plain page size mapping
            dims = (2, 2)
            tensor, fd, mm = alloc_shm(dims, torch.float32)
            self.assertEqual(tensor.shape, dims)
            mm.close()
            os.close(fd)
        finally:
            envs.SGLANG_HUGEPAGE_SIZE.override(None)

    def test_shm_host_tensor_allocator_invalid_device(self):
        allocator = ShmHostTensorAllocator()
        with self.assertRaises(AssertionError) as ctx:
            allocator.allocate((2, 2), torch.float32, device="cuda")
        self.assertIn("only supports CPU allocations", str(ctx.exception))


class TestHugetlbPool(unittest.TestCase):
    """What the host-pool preflight may credit from the kernel's hugetlb pool."""

    @staticmethod
    def _sysfs(root, free_2mb=None, free_1gb=None, resv_2mb=0, resv_1gb=0):
        for name, free, resv in (
            ("hugepages-2048kB", free_2mb, resv_2mb),
            ("hugepages-1048576kB", free_1gb, resv_1gb),
        ):
            if free is None:
                continue
            os.makedirs(os.path.join(root, name))
            for counter, value in (("free_hugepages", free), ("resv_hugepages", resv)):
                with open(os.path.join(root, name, counter), "w") as f:
                    f.write(f"{value}\n")

    def test_hugepage_env_parsing(self):
        for raw, expected in {"": 0, "2MB": 2 * 1024**2, " 1gb ": 1024**3}.items():
            with self.subTest(raw=raw), envs.SGLANG_HUGEPAGE_SIZE.override(raw):
                self.assertEqual(mmap_allocator.hugepage_size_requested(), expected)
        with (
            envs.SGLANG_HUGEPAGE_SIZE.override("4MB"),
            self.assertLogs(mmap_allocator.logger, "WARNING"),
        ):
            self.assertEqual(mmap_allocator.hugepage_size_requested(), 0)

        for raw_mode, size, expected in (
            ("", 0, mmap_allocator.HUGEPAGE_MODE_OFF),
            ("", 2 * 1024**2, mmap_allocator.HUGEPAGE_MODE_PREFER),
            (" Required ", 2 * 1024**2, mmap_allocator.HUGEPAGE_MODE_REQUIRED),
        ):
            with (
                self.subTest(raw_mode=raw_mode, size=size),
                envs.SGLANG_HUGEPAGE_MODE.override(raw_mode),
            ):
                self.assertEqual(mmap_allocator.hugepage_mode(size), expected)

    def test_hugetlb_pool_free_bytes(self):
        # sysfs keeps one pool per page size, and MemAvailable excludes all of
        # them. A mapping reserves its pages at mmap time and faults them in
        # afterwards; until then sysfs still counts them as free, and
        # co-located ranks populate their pools concurrently, so only
        # free - resv can back a new mapping.
        pool_2mb = (6144 - 1000) * 2 * 1024**2
        with tempfile.TemporaryDirectory() as root:
            self._sysfs(root, free_2mb=6144, free_1gb=3, resv_2mb=1000)
            with unittest.mock.patch.object(
                mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root
            ):
                # The requested page size selects its pool, net of reservations.
                with (
                    self.subTest(why="2MB pool"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), pool_2mb)
                with (
                    self.subTest(why="1GB pool"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
                ):
                    self.assertEqual(
                        mmap_allocator.hugetlb_pool_free_bytes(), 3 * 1024**3
                    )
                # No credit where a new mapping would not take MAP_HUGETLB.
                with (
                    self.subTest(why="not requested"),
                    envs.SGLANG_HUGEPAGE_SIZE.override(""),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                with (
                    self.subTest(why="mode off"),
                    envs.SGLANG_HUGEPAGE_MODE.override("off"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                with (
                    self.subTest(why="libc unavailable"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                    unittest.mock.patch.object(mmap_allocator, "_libc", None),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                # A cgroup limit below the pool bounds the credit.
                for cgroup, expected in [
                    (None, pool_2mb),  # no container limit: the sysfs pool as-is
                    (pool_2mb + 1, pool_2mb),  # room above the pool: the pool bounds
                    (1024**3, 1024**3),  # in-container: the cgroup bounds
                ]:
                    with (
                        self.subTest(why="cgroup bound", cgroup=cgroup),
                        envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                        unittest.mock.patch.object(
                            mmap_allocator,
                            "cgroup_hugetlb_headroom_bytes",
                            return_value=cgroup,
                        ),
                    ):
                        self.assertEqual(
                            mmap_allocator.hugetlb_pool_free_bytes(), expected
                        )

        # No pool of the requested size: warn and report zero.
        with tempfile.TemporaryDirectory() as empty_root:
            with (
                self.subTest(why="no pool of that size"),
                unittest.mock.patch.object(
                    mmap_allocator, "_HUGEPAGE_SYSFS_DIR", empty_root
                ),
                envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
                self.assertLogs(mmap_allocator.logger, "WARNING"),
            ):
                self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
