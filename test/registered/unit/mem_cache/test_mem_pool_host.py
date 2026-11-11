"""Unit tests for host-pool allocation and free-list bookkeeping."""

import threading
import unittest
import unittest.mock
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import memory_pool_host as mamba_pool_host
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DSAIndexerPoolHost,
    MambaPoolHost,
)
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHostKVCache(CustomTestCase):
    def setUp(self):
        self.page_size = 2
        # Small device pool is enough to construct the host pool.
        self.device_pool = MHATokenToKVPool(
            size=self.page_size * 2,
            page_size=self.page_size,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def test_double_alloc(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        # Mimic bookkeeping corruption: push an already-used slot back to the
        # head of free_slots so the next alloc would hand out an in-use slot.
        leak = torch.tensor([int(indices[0])])
        self.host_pool.free_slots = torch.cat([leak, self.host_pool.free_slots])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.alloc(4)
        msg = str(ctx.exception)
        self.assertIn("Double-alloc", msg)
        self.assertIn(f"[{int(leak[0])}]", msg)

    def test_double_free(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        self.host_pool.free(indices[:2])
        # indices[1] is double freed.
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices[1:])
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[1])}]", msg)

    def test_free_unallocated(self):
        indices = torch.tensor([1])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[0])}]", msg)

    def test_free_after_clear(self):
        indices = self.host_pool.alloc(4)
        self.host_pool.clear()
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(str(indices.tolist()), msg)

    def test_shm_allocator(self):
        shm_host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="shm",
        )
        self.assertIsNotNone(shm_host_pool.fd)
        self.assertGreaterEqual(shm_host_pool.fd, 0)

        indices = shm_host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        shm_host_pool.free(indices)

    def test_empty_free_keeps_release_list_empty(self):
        self.assertEqual(self.host_pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(self.host_pool.num_release_slots, 0)
        self.assertEqual(self.host_pool.release_slots, [])


class TestLazyHostPoolRelease(CustomTestCase):
    @staticmethod
    def _make_mamba_pool():
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_deepseek_v4_pool():
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    def _assert_lazy_release(self, pool):
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

        allocated = pool.alloc(6)
        free_slots_before = pool.free_slots

        pool.free(allocated[:2])

        # free() should keep the primary free-list untouched and only record
        # the released chunk for a later merge.
        self.assertIs(pool.free_slots, free_slots_before)
        self.assertEqual(pool.num_release_slots, 2)
        self.assertEqual(len(pool.release_slots), 1)
        self.assertEqual(pool.available_size(), 4)

        # Consume the primary free-list first without merging pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([6, 7])))
        self.assertEqual(pool.num_release_slots, 2)

        # Once the primary free-list is exhausted, alloc() merges and reuses
        # the pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([0, 1])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 0)

        pool.free(torch.tensor([0, 1]))
        pool.clear()
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 8)

        # Exercise the general merge path with multiple released chunks.
        allocated = pool.alloc(8)
        pool.free(allocated[:2])
        pool.free(allocated[2:4])
        self.assertEqual(len(pool.release_slots), 2)
        self.assertTrue(torch.equal(pool.alloc(4), torch.tensor([0, 1, 2, 3])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

    def test_mamba_pool_lazy_release(self):
        self._assert_lazy_release(self._make_mamba_pool())

    def test_deepseek_v4_pool_lazy_release(self):
        pool = self._make_deepseek_v4_pool()
        self._assert_lazy_release(pool)

        # Preserve the pool's page-aligned allocation behavior.
        pool.clear()
        self.assertEqual(len(pool.alloc(1)), 2)


class TestHostMappingLengths(CustomTestCase):
    def test_conv_only_mamba_constructor_uses_release_device_pool(self):
        device_pool = SimpleNamespace(
            num_mamba_layers=2,
            size=4,
            device="cpu",
            mamba_cache=SimpleNamespace(
                temporal=torch.empty((2, 4, 0), dtype=torch.float16),
                conv=[torch.empty((2, 4, 3), dtype=torch.float16)],
            ),
        )
        with unittest.mock.patch.object(
            mamba_pool_host, "host_memory_preflight", return_value=(60, 1024)
        ) as preflight:
            pool = MambaPoolHost(
                device_pool, 1, 0, pin_memory=False, layout="page_first"
            )
        self.addCleanup(pool.destroy)
        self.assertEqual(pool.get_mapping_lengths(), [0, 60])
        preflight.assert_called_once_with([0, 60], pool.allocator, "cpu")
        self.assertEqual(pool.alloc(1).numel(), 1)
        self.assertEqual(pool.temporal_buffer.numel(), 0)
        self.assertEqual(pool.conv_buffer[0].numel() * 2, 60)

    def test_dsa_guard_page_is_included_in_mapping(self):
        pool = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        pool.indexer_page_num = 5
        pool.layer_num = 2
        pool.indexer_page_stride_size = 132
        pool.indexer_dtype = torch.uint8
        self.assertEqual(pool.get_mapping_lengths(), [1320])

    def test_mla_split_reports_each_mapping(self):
        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = "page_first_kv_split"
        pool.layer_num = 2
        pool.dtype = torch.float16
        pool.kv_lora_rank = 8
        pool.qk_rope_head_dim = 4
        pool.page_num = 4
        pool.page_size = 2
        pool.dsa_kv_cache_store_fp8 = False
        pool.device_pool = SimpleNamespace(
            num_indexer_layers=1, index_head_dim=3, index_k_scale_buffer=object()
        )
        self.assertEqual(pool.get_mapping_lengths(), [256, 128, 96])


class TestHostMemoryBudget(CustomTestCase):
    # Pinned so the two budget reads below see identical free memory; the real
    # psutil value drifts between calls and would flake the equality checks.
    _AVAILABLE = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3)

    def _budget_with_ranks(self, ranks):
        # Deliberate single-accessor stub: isolates the budget math from the
        # topology derivation, which the ranks_per_host case below covers.
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=ranks),
            unittest.mock.patch.object(
                base, "available_host_memory_bytes", return_value=self._AVAILABLE
            ),
            envs.SGLANG_HUGEPAGE_MODE.override("off"),
        ):
            return base.host_memory_preflight().available_bytes

    def test_budget_is_split_across_co_located_ranks(self):
        solo = self._budget_with_ranks(1)
        self.assertEqual(self._budget_with_ranks(4), solo // 4)

    def test_reserve_is_taken_before_the_split(self):
        # Each rank must not get its own copy of the reserve.
        budget = self._budget_with_ranks(8)
        self.assertLessEqual(
            budget * 8, self._AVAILABLE - base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        )

    def test_ranks_per_host_divides_world_size_by_nodes(self):
        with (
            unittest.mock.patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
            unittest.mock.patch.object(
                torch.distributed, "get_world_size", return_value=16
            ),
            unittest.mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=SimpleNamespace(nnodes=2),
            ),
        ):
            self.assertEqual(base.ranks_per_host(), 8)

    @staticmethod
    def _hugetlb_allocator(free_bytes):
        return unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=free_bytes),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )

    def _budget_for(
        self, allocator, device, available, ranks=8, mode="prefer", size="2MB"
    ):
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=ranks),
            unittest.mock.patch.object(
                base, "available_host_memory_bytes", return_value=available
            ),
            unittest.mock.patch.dict(
                "os.environ",
                {"SGLANG_HUGEPAGE_MODE": mode, "SGLANG_HUGEPAGE_SIZE": size},
            ),
        ):
            return base.host_memory_preflight(
                allocator=allocator, device=device
            ).available_bytes

    def test_budget_is_the_larger_of_plain_ram_and_the_hugetlb_pool(self):
        # One mapping is served entirely by the hugetlb pool or entirely by
        # plain pages, so the larger of the two is the budget; the reserve is
        # for the OS and applies to plain RAM only.
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        for hugetlb_free, expected in (
            (96 * gib, 96 * gib // 8),  # the hugetlb pool bounds
            (16 * gib, 64 * gib // 8),  # plain RAM wins
        ):
            with self.subTest(hugetlb_free=hugetlb_free):
                budget = self._budget_for(
                    self._hugetlb_allocator(hugetlb_free),
                    "cuda",
                    available=reserve + 64 * gib,
                )
                self.assertEqual(budget, expected)

    def test_hugetlb_credit_is_gated_off(self):
        # No credit where a mapping cannot take MAP_HUGETLB: npu/musa allocate
        # with torch.empty(pin_memory=True) and never see the allocator, and
        # mode=off opts out entirely.
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = self._hugetlb_allocator(96 * gib)
        for device, mode in (
            ("npu", "prefer"),
            ("npu:0", "prefer"),
            ("cuda", "off"),
        ):
            with self.subTest(device=repr(device), mode=mode):
                budget = self._budget_for(
                    allocator, device, available=reserve + 64 * gib, mode=mode
                )
                self.assertEqual(budget, 64 * gib // 8)
                allocator.free_hugetlb_bytes.assert_not_called()

    def test_plain_budget_page_alignment(self):
        mib = 1024**2
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        # Only the pool component rounds down to whole pages; plain RAM is
        # handed out unaligned.
        for hugetlb_free, plain, expected in (
            (13 * mib + mib // 2, 4 * mib, 12 * mib),
            (0, 13 * mib + mib // 2, 13 * mib + mib // 2),
        ):
            with self.subTest(hugetlb_free=hugetlb_free):
                budget = self._budget_for(
                    self._hugetlb_allocator(hugetlb_free),
                    "cuda",
                    available=reserve + plain,
                    ranks=1,
                )
                self.assertEqual(budget, expected)
        # A booked plain charge is measured unaligned as well.
        allocator = self._hugetlb_allocator(0)
        with (
            base.host_memory_budget_scope(13 * mib + mib // 2),
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
        ):
            preflight = base.host_memory_preflight([13 * mib], allocator, "cuda")
            self.assertEqual(preflight.allocation_bytes, 13 * mib)
            self.assertEqual(preflight.available_bytes, 13 * mib + mib // 2)

    def test_allocation_bytes_rounds_each_mapping_on_the_hugetlb_path(self):
        # Two 1.1 GiB mappings reserve 4 pages of a 1 GiB-page pool, not the
        # 3 pages their plain sum suggests: each MAP_HUGETLB mapping rounds up
        # on its own.
        gib = 1024**3
        mapping_lengths = [int(1.1 * gib), int(1.1 * gib)]
        allocator = self._hugetlb_allocator(4 * gib)

        def allocation_bytes():
            return base.host_memory_preflight(
                mapping_lengths, allocator, "cuda"
            ).allocation_bytes

        with (
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
        ):
            self.assertEqual(allocation_bytes(), 4 * gib)
            # With only 3 GiB of pool left the rounded batch (4 GiB) no longer
            # fits and falls back to its plain sum.
            allocator.free_hugetlb_bytes.return_value = 3 * gib
            self.assertEqual(allocation_bytes(), 2 * int(1.1 * gib))
            # npu pins through torch and never maps MAP_HUGETLB: plain sum.
            self.assertEqual(
                base.host_memory_preflight(
                    mapping_lengths, allocator, "npu"
                ).allocation_bytes,
                2 * int(1.1 * gib),
            )
            # The bookable boundary is this rank's pool share, not the
            # host-wide pool: 2 ranks, an 8 GiB pool and a 6 GiB batch bill
            # plain, which fits the 7 GiB per-rank plain budget.
            allocator.free_hugetlb_bytes.return_value = 8 * gib
            with (
                unittest.mock.patch.object(base, "ranks_per_host", return_value=2),
                unittest.mock.patch.object(
                    base,
                    "available_host_memory_bytes",
                    return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 14 * gib,
                ),
            ):
                preflight = base.host_memory_preflight([6 * gib], allocator, "cuda")
                self.assertEqual(preflight.allocation_bytes, 6 * gib)
                self.assertEqual(preflight.available_bytes, 7 * gib)
        # Without SGLANG_HUGEPAGE_SIZE there is no page size to round to.
        with envs.SGLANG_HUGEPAGE_SIZE.override(""):
            self.assertEqual(allocation_bytes(), 2 * int(1.1 * gib))
        # mode=off opts out of rounding even with a page size set.
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("off"),
            envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
        ):
            self.assertEqual(allocation_bytes(), 2 * int(1.1 * gib))

    def test_scope_books_plain_and_hugetlb_buckets_separately(self):
        # Inside a scope a prefer-mode charge books the pool bucket while the
        # rounded batch fits it, and plain RAM once it does not. One mapping
        # draws from one side only, so available_bytes is the larger bucket.
        gib = 1024**3
        half = gib + gib // 2
        allocator = self._hugetlb_allocator(0)  # the live pool is unused here
        with (
            base.host_memory_budget_scope((8 * gib, 3 * gib)),
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
        ):
            preflight = base.host_memory_preflight([half], allocator, "cuda")
            self.assertEqual(preflight.allocation_bytes, 2 * gib)
            self.assertEqual(preflight.available_bytes, 8 * gib)
            self.assertEqual(
                base.host_memory_preflight(
                    allocator=allocator, device="cuda"
                ).available_bytes,
                8 * gib,
            )
            # The 4 GiB rounded batch exceeds the 1 GiB left in the pool
            # bucket, so it falls back to plain RAM; booking it also closes
            # the pool bucket.
            preflight = base.host_memory_preflight([half, half], allocator, "cuda")
            self.assertEqual(preflight.allocation_bytes, 2 * half)
            self.assertEqual(
                base.host_memory_preflight(
                    allocator=allocator, device="cuda"
                ).available_bytes,
                5 * gib,
            )
            # Plain RAM is the binding bucket for this oversized charge.
            overflow = base.host_memory_preflight([7 * gib], allocator, "cuda")
            self.assertLess(overflow.available_bytes, overflow.allocation_bytes)

    def test_plain_budget_is_reported_as_is_without_a_hugetlb_pool(self):
        # A host below the reserve keeps its negative budget, and the failure
        # message that shows it, when there is no hugetlb pool to credit.
        gib = 1024**3
        budget = self._budget_for(self._hugetlb_allocator(0), "cuda", available=4 * gib)
        self.assertEqual(
            budget, (4 * gib - base.HICACHE_HOST_MEMORY_RESERVE_BYTES) // 8
        )

    def test_required_mode_uses_only_hugetlb(self):
        gib = 1024**3
        budget = self._budget_for(
            self._hugetlb_allocator(24 * gib),
            "cuda",
            available=1 << 50,
            mode="required",
        )
        self.assertEqual(budget, 24 * gib // 8)
        # 9 MiB of hugetlb across 4 ranks is 2.25 MiB each, but a mapping only
        # ever takes whole 2 MiB pages, so each rank may claim just one page.
        mib = 1024**2
        budget = self._budget_for(
            self._hugetlb_allocator(9 * mib),
            "cuda",
            available=0,
            ranks=4,
            mode="required",
        )
        self.assertEqual(budget, 2 * mib)
        # ... and without a page size it is a configuration error.
        with self.assertRaisesRegex(ValueError, "SGLANG_HUGEPAGE_SIZE"):
            self._budget_for(
                unittest.mock.Mock(
                    supports_hugetlb=unittest.mock.Mock(return_value=True)
                ),
                "cuda",
                available=1 << 50,
                mode="required",
                size="",
            )
        # ... inside a budget scope too, which never samples host availability.
        with (
            base.host_memory_budget_scope(1 << 40),
            envs.SGLANG_HUGEPAGE_MODE.override("required"),
            envs.SGLANG_HUGEPAGE_SIZE.override(""),
        ):
            with self.assertRaisesRegex(ValueError, "SGLANG_HUGEPAGE_SIZE"):
                base.host_memory_preflight([mib], self._hugetlb_allocator(0), "cuda")

    def test_required_mode_is_ignored_by_unsupported_allocation_paths(self):
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=96 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=False),
        )
        with unittest.mock.patch.object(base, "hugepage_size_requested") as parser:
            budget = self._budget_for(
                allocator, "cuda", available=reserve + 64 * gib, mode="required"
            )
            self.assertEqual(budget, 64 * gib // 8)
            parser.assert_not_called()
        allocator.free_hugetlb_bytes.assert_not_called()

    def test_guard_passes_its_allocator_and_device(self):
        device_pool = MHATokenToKVPool(
            size=4,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )

        def plain_alloc(dims, dtype, device, pin_memory, allocator, **kwargs):
            return torch.empty(dims, dtype=dtype)

        with (
            unittest.mock.patch.object(
                base, "host_memory_preflight", return_value=(0, 1024**3)
            ) as budget,
            unittest.mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": plain_alloc}),
        ):
            pool = MHATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=2,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                allocator_type="default",
            )
        budget.assert_called_once_with(
            pool.get_mapping_lengths(), pool.allocator, device_pool.device
        )


if __name__ == "__main__":
    unittest.main()
