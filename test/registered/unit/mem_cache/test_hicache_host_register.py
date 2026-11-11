import ctypes
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_register,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.storage.mmap import mmap_allocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeBuffer:
    def __init__(self, base, size):
        self.base = base
        self.size = size

    def data_ptr(self):
        return self.base

    def numel(self):
        return self.size

    def element_size(self):
        return 1


class _FakeCudart:
    def __init__(self, status=0):
        self.status = status
        self.registrations = []
        self.unregistrations = []

    def cudaHostRegister(self, ptr, size, flags):
        self.registrations.append((ptr, size, flags))
        return self.status

    def cudaHostUnregister(self, ptr):
        self.unregistrations.append(ptr)
        return 0

    def cudaGetErrorString(self, status):
        return "injected error"


class TestHiCacheHostRegister(unittest.TestCase):
    def test_large_plain_buffer_uses_one_registration(self):
        # Match 1.0.0-3 even above the removed 1 GiB chunk threshold.
        buffer = _FakeBuffer(0x10000000, 3 * 1024**3 + 16)
        cudart = _FakeCudart()
        with mock.patch.object(torch.cuda, "cudart", return_value=cudart):
            _cuda_host_register(buffer)
            _cuda_host_unregister(buffer)
        self.assertEqual(cudart.registrations, [(buffer.base, buffer.size, 0)])
        self.assertEqual(cudart.unregistrations, [buffer.base])

    def test_registration_failure_is_reported(self):
        buffer = _FakeBuffer(0x10000000, 16)
        cudart = _FakeCudart(status=1)
        with mock.patch.object(torch.cuda, "cudart", return_value=cudart):
            with self.assertRaisesRegex(RuntimeError, "cudaHostRegister failed"):
                _cuda_host_register(buffer)
        self.assertEqual(len(cudart.registrations), 1)

    def test_hugepage_buffer_registers_the_rounded_mapping(self):
        cudart = _FakeCudart()
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("required"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            mock.patch.object(
                mmap_allocator, "_alloc_hugepage", return_value=(ctypes.c_uint8 * 16)()
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            buffer = mmap_allocator.alloc_mmap((4,), torch.float32)
            _cuda_host_register(buffer)
            _cuda_host_unregister(buffer)
        self.assertEqual(cudart.registrations, [(buffer.data_ptr(), 2 * 1024**2, 0)])
        self.assertEqual(cudart.unregistrations, [buffer.data_ptr()])

    def test_mha_and_asymmetric_pool_expose_actual_backing_buffers(self):
        for cls in (MHATokenToKVPoolHost, AsymmetricMHATokenToKVPoolHost):
            for buffers in (torch.empty(4), (torch.empty(4), torch.empty(8))):
                with self.subTest(cls=cls.__name__, split=isinstance(buffers, tuple)):
                    pool = cls.__new__(cls)
                    pool.kv_buffer = buffers
                    expected = (
                        list(buffers) if isinstance(buffers, tuple) else [buffers]
                    )
                    actual = pool.get_hybrid_pool_buffer()
                    self.assertEqual(len(actual), len(expected))
                    for a, b in zip(actual, expected):
                        self.assertIs(a, b)

    def test_layer_first_v4_pool_checks_per_mapping_hugetlb_rounding(self):
        alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))
        device_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(3)]
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
            base.host_memory_budget_scope((0, 2 * 1024**3)),
            mock.patch.object(base, "device_uses_allocator", return_value=True),
            mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}),
        ):
            with self.assertRaises(ValueError):
                DeepSeekV4PagedHostPool(
                    pool_name="test",
                    device_buffers=device_buffers,
                    item_bytes=11,
                    num_host_pages=4,
                    slot_page_size=2,
                    layout="layer_first",
                )
            DeepSeekV4PagedHostPool(
                pool_name="test",
                device_buffers=device_buffers,
                item_bytes=11,
                num_host_pages=4,
                slot_page_size=2,
                layout="page_first",
            )


if __name__ == "__main__":
    unittest.main()
