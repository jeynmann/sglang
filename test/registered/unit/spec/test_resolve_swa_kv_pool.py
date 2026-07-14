"""Unit tests for attention-backend SWA KV pool resolution."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-large-amd")

_RESOLVERS = (
    ("trtllm_mha", TRTLLMHAAttnBackend._resolve_swa_kv_pool, SWAKVPool),
    ("flashinfer", FlashInferAttnBackend._resolve_swa_kv_pool, BaseSWAKVPool),
)


def _mock_runner(
    *,
    active_pool=None,
    is_draft_worker=False,
    spec_algorithm=SpeculativeAlgorithm.NONE,
    allocator_kvcache=None,
):
    runner = MagicMock()
    runner.token_to_kv_pool = active_pool
    runner.is_draft_worker = is_draft_worker
    runner.spec_algorithm = spec_algorithm
    runner.token_to_kv_pool_allocator.get_kvcache.return_value = allocator_kvcache
    return runner


class TestResolveSwaKvPool(CustomTestCase):
    def test_active_pool_is_swa_returns_it(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(active_pool=swa)
                self.assertIs(resolve(runner), swa)

    def test_non_swa_active_pool_falls_through_to_allocator(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(active_pool=MagicMock(), allocator_kvcache=swa)
                self.assertIs(resolve(runner), swa)

    def test_allocator_kvcache_not_swa_returns_none(self):
        for name, resolve, _ in _RESOLVERS:
            with self.subTest(backend=name):
                runner = _mock_runner(
                    active_pool=MagicMock(), allocator_kvcache=MagicMock()
                )
                self.assertIsNone(resolve(runner))

    def test_draft_worker_non_frozen_kv_returns_none(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                runner = _mock_runner(
                    active_pool=MagicMock(),
                    is_draft_worker=True,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    allocator_kvcache=MagicMock(spec=pool_type),
                )
                self.assertIsNone(resolve(runner))

    def test_draft_worker_frozen_kv_mtp_returns_allocator_swa(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(
                    active_pool=MagicMock(),
                    is_draft_worker=True,
                    spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
                    allocator_kvcache=swa,
                )
                self.assertIs(resolve(runner), swa)

    def test_non_draft_worker_ignores_spec_algorithm(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(
                    active_pool=MagicMock(),
                    is_draft_worker=False,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    allocator_kvcache=swa,
                )
                self.assertIs(resolve(runner), swa)

    @patch("sglang.srt.disaggregation.decode_kvcache_offload_manager.HiCacheController")
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.torch.distributed.get_world_size",
        return_value=1,
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.get_mha_host_pool_cls"
    )
    def test_decode_offload_unwraps_swa_full_kv_pool(
        self, get_host_pool_cls, _get_world_size, _controller
    ):
        full_kv_pool = MagicMock(spec=MHATokenToKVPool)
        swa_pool = MagicMock(spec=BaseSWAKVPool)
        swa_pool.full_kv_pool = full_kv_pool
        allocator = MagicMock()
        allocator.get_kvcache.return_value = swa_pool

        host_pool_cls = get_host_pool_cls.return_value
        server_args = MagicMock()
        server_args.page_size = 1
        server_args.hicache_ratio = 2.0
        server_args.hicache_size = 0
        server_args.hicache_mem_layout = "page_first"
        server_args.hicache_storage_backend_extra_config = None

        DecodeKVCacheOffloadManager(
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=allocator,
            tp_group=MagicMock(),
            tree_cache=MagicMock(),
            server_args=server_args,
        )

        get_host_pool_cls.assert_called_once_with(full_kv_pool)
        host_pool_cls.assert_called_once_with(
            full_kv_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            server_args.page_size,
            server_args.hicache_mem_layout,
        )


if __name__ == "__main__":
    unittest.main()
