#!/usr/bin/env python3

import hashlib
import os
import unittest
import warnings
from typing import List
from unittest.mock import MagicMock, patch

# HiCacheController pulls in torch._inductor, which imports torch.utils.mkldnn
# and emits a DeprecationWarning for torch.jit.script_method during collection.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.script_method` is deprecated",
    category=DeprecationWarning,
)

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlBackendConfig,
    NixlBackendSelection,
    NixlFileManager,
    NixlRegistration,
)


class TestNixlUnified(unittest.TestCase):
    """Unified test suite for all NIXL components."""

    def test_obj_reg_tuple_default(self):
        self.assertEqual(
            HiCacheNixl._obj_reg_tuple_default("key-a"), (0, 0, "key-a", "")
        )

    def test_obj_reg_tuple_doca_memos(self):
        key = "cache/key"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        meta = digest.hex()[:32]
        dev_id = int.from_bytes(digest[:8], "big")
        self.assertEqual(
            HiCacheNixl._obj_reg_tuple_doca_memos(key), (0, 0, dev_id, meta)
        )

    def test_format_key_doca_memos(self):
        k = "some/cache/key@suffix"
        formatted = hashlib.sha256(k.encode("utf-8")).hexdigest()[:32]
        self.assertEqual(HiCacheNixl._format_key_doca_memos(k), formatted)

    def _make_hicache(self, extra_config: dict) -> HiCacheNixl:
        def _stub_create_backend(selector_self, agent):
            selector_self.backend_name = selector_self.plugin
            selector_self.mem_type = (
                "OBJ"
                if selector_self.backend_name in NixlBackendSelection.OBJ_PLUGINS
                else "FILE"
            )
            return True

        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config=extra_config,
        )
        with (
            patch(
                "sglang.srt.mem_cache.storage.nixl.hicache_nixl.nixl_agent",
                return_value=MagicMock(),
            ),
            patch.object(NixlBackendSelection, "create_backend", _stub_create_backend),
        ):
            return HiCacheNixl(storage_config=storage_config, file_path=self.test_dir)

    def test_format_key_backend_routing(self):
        """HiCacheNixl.__init__ binds _format_key from backend_selector.backend_name."""
        key = "some/cache/key@suffix"
        cases = (
            (
                "OBJ",
                {"plugin": {"obj": {"active": True}}},
                HiCacheNixl._format_key_default,
                HiCacheNixl._obj_reg_tuple_default,
                False,
            ),
            (
                "DOCA_MEMOS",
                {"plugin": {"doca_memos": {"active": True}}},
                HiCacheNixl._format_key_doca_memos,
                HiCacheNixl._obj_reg_tuple_doca_memos,
                True,
            ),
        )
        for backend, extra, fmt_fn, reg_fn, exp_hugepages in cases:
            with self.subTest(backend=backend):
                hicache = self._make_hicache(extra)
                sel = hicache.backend_selector

                self.assertEqual(sel.backend_name, backend)
                self.assertEqual(sel.mem_type, "OBJ")
                self.assertEqual(hicache._format_key(key), fmt_fn(key))
                self.assertEqual(hicache._obj_reg_tuple(key), reg_fn(key))
                self.assertEqual(hicache._require_host_hugepages, exp_hugepages)

    def test_nixl_backend_config_use_host_hugepages(self):
        self.assertFalse(NixlBackendConfig({}).use_host_hugepages())
        self.assertTrue(
            NixlBackendConfig({"use_host_hugepages": True}).use_host_hugepages()
        )
        self.assertTrue(
            NixlBackendConfig({"use_host_hugepages": "true"}).use_host_hugepages()
        )
        self.assertFalse(
            NixlBackendConfig({"use_host_hugepages": False}).use_host_hugepages()
        )

    def test_parse_storage_batch_size(self):
        self.assertEqual(HiCacheController._parse_storage_batch_size({}), 128)
        extra = {"storage_batch_size": 64}
        self.assertEqual(HiCacheController._parse_storage_batch_size(extra), 64)
        extra = {"storage_batch_size": 0}
        self.assertRaises(
            ValueError, HiCacheController._parse_storage_batch_size, extra
        )

    def test_doca_memos_backend_requires_hugepages_and_meminfo(self):
        agent = MagicMock()
        agent.get_plugin_list.return_value = ["DOCA_MEMOS"]
        agent.get_backend_params.return_value = {}
        # use_host_hugepages = False
        selector = NixlBackendSelection(
            plugin="DOCA_MEMOS",
            nixlconfig=NixlBackendConfig({"use_host_hugepages": False}),
        )
        self.assertFalse(selector.create_backend(agent))
        agent.create_backend.assert_not_called()
        # use_host_hugepages = True, validate_meminfo = False
        selector = NixlBackendSelection(
            plugin="DOCA_MEMOS",
            nixlconfig=NixlBackendConfig({"use_host_hugepages": True}),
        )
        with patch(
            "sglang.srt.mem_cache.storage.nixl.hugepage_util.HugepageUtil.validate_meminfo",
            return_value=False,
        ):
            self.assertFalse(selector.create_backend(agent))
        agent.create_backend.assert_not_called()
        # use_host_hugepages = True, validate_meminfo = True
        selector = NixlBackendSelection(
            plugin="DOCA_MEMOS",
            nixlconfig=NixlBackendConfig({"use_host_hugepages": True}),
        )
        with patch(
            "sglang.srt.mem_cache.storage.nixl.hugepage_util.HugepageUtil.validate_meminfo",
            return_value=True,
        ):
            self.assertTrue(selector.create_backend(agent))
        agent.create_backend.assert_called()

    def test_hicache_nixl_requires_host_hugepages(self):
        host_pool = MagicMock()
        host_pool.use_host_hugepages = False
        fake = type("FakeHiCache", (), {"_require_host_hugepages": True})()
        # _require_host_hugepages = True, use_host_hugepages = False
        with self.assertRaises(RuntimeError):
            HiCacheNixl._assert_doca_memos_host_hugepages(fake, host_pool)
        # _require_host_hugepages = False, use_host_hugepages = False
        fake._require_host_hugepages = False
        HiCacheNixl._assert_doca_memos_host_hugepages(fake, host_pool)
        # _require_host_hugepages = True, use_host_hugepages = True
        fake._require_host_hugepages = True
        host_pool.use_host_hugepages = True
        HiCacheNixl._assert_doca_memos_host_hugepages(fake, host_pool)

    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = "/tmp/test_nixl_unified"
        os.makedirs(self.test_dir, exist_ok=True)
        os.environ["SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR"] = self.test_dir

        # Mock NIXL agent for registration tests
        self.mock_agent = MagicMock()
        self.mock_agent.get_reg_descs.return_value = "mock_reg_descs"
        self.mock_agent.register_memory.return_value = "mock_registered_memory"

        # Create instances
        self.file_manager = NixlFileManager(self.test_dir)
        self.registration = NixlRegistration(self.mock_agent)

        # Create storage config for testing
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test_model",
            extra_config={"plugin": {"posix": {"active": True}}},
        )

        self.hicache = None
        tests_skip_hicache_init = {
            "test_obj_reg_tuple_default",
            "test_obj_reg_tuple_doca_memos",
            "test_format_key_doca_memos",
            "test_format_key_backend_routing",
            "test_nixl_backend_config_use_host_hugepages",
            "test_parse_storage_batch_size",
            "test_doca_memos_backend_requires_hugepages_and_meminfo",
            "test_hicache_nixl_requires_host_hugepages",
            "test_basic_file_operations",
            "test_create_nixl_tuples",
            "test_error_handling",
        }
        if self._testMethodName in tests_skip_hicache_init:
            return

        try:
            self.hicache = HiCacheNixl(
                storage_config=self.storage_config,
                file_path=self.test_dir,
            )
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def tearDown(self):
        """Clean up test directories."""
        self.hicache = None
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir, ignore_errors=True)

    @staticmethod
    def _open_fds() -> int:
        return len(os.listdir("/proc/self/fd"))

    def delete_test_file(self, file_path: str) -> bool:
        """Helper method to delete a test file.

        Args:
            file_path: Path to the file to delete

        Returns:
            bool: True if file was deleted or didn't exist, False on error
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            return False

    def verify_tensors_equal(self, expected: torch.Tensor, actual: torch.Tensor):
        """Helper to verify tensor equality."""
        self.assertIsNotNone(actual, "Retrieved tensor is None")
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            f"Tensors not equal:\nExpected: {expected}\nActual: {actual}",
        )

    def verify_tensor_lists_equal(
        self, expected: List[torch.Tensor], actual: List[torch.Tensor]
    ):
        """Helper to verify lists of tensors are equal."""
        self.assertEqual(len(expected), len(actual), "Lists have different lengths")
        for exp, act in zip(expected, actual):
            self.verify_tensors_equal(exp, act)

    # ============================================================================
    # HiCache Integration Tests
    # ============================================================================

    def test_single_set_get(self):
        """Test single tensor set/get operations."""
        key = "test_key"
        value = torch.randn(10, 10, device="cpu")
        dst_tensor = torch.zeros_like(value, device="cpu")

        # Test set
        self.assertTrue(self.hicache.set(key, value))
        self.assertTrue(self.hicache.exists(key))

        # Test get
        retrieved = self.hicache.get(key, dst_tensor)
        self.verify_tensors_equal(value, dst_tensor)
        self.verify_tensors_equal(value, retrieved)

        # Same test in addr,len mode with another key and dst_tensor
        key2 = "test_key2"
        dst_tensor2 = torch.zeros_like(value, device="cpu")
        src_addr, src_len = value.data_ptr(), value.numel() * value.element_size()
        dst_addr, dst_len = (
            dst_tensor2.data_ptr(),
            dst_tensor2.numel() * dst_tensor2.element_size(),
        )

        # Test set
        self.assertTrue(self.hicache.set(key2, None, src_addr, src_len))
        self.assertTrue(self.hicache.exists(key2))

        # Test get
        retrieved2 = self.hicache.get(key2, dst_addr, dst_len)
        self.assertTrue(retrieved2 is None)
        self.verify_tensors_equal(value, dst_tensor2)

    def test_batch_set_get(self):
        """Test batch tensor set/get operations."""
        keys = ["key1", "key2", "key3"]
        values = [
            torch.randn(5, 5, device="cpu"),
            torch.randn(3, 3, device="cpu"),
            torch.randn(7, 7, device="cpu"),
        ]
        dst_tensors = [torch.zeros_like(v, device="cpu") for v in values]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys, values))
        self.assertTrue(all(self.hicache.exists(key) for key in keys))

        # Test batch get
        retrieved = self.hicache.batch_get(keys, dst_tensors)
        self.verify_tensor_lists_equal(values, retrieved)

        # Same test in addr,len mode with another key and dst_tensor
        keys2 = ["key4", "key5", "key6"]
        dst_tensors2 = [torch.zeros_like(v, device="cpu") for v in values]
        src_addrs = [v.data_ptr() for v in values]
        src_lens = [v.numel() * v.element_size() for v in values]
        dst_addrs = [dt.data_ptr() for dt in dst_tensors2]
        dst_lens = [dt.numel() * dt.element_size() for dt in dst_tensors2]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys2, None, src_addrs, src_lens))
        self.assertTrue(all(self.hicache.exists(key) for key in keys2))

        # Test batch get
        retrieved2 = self.hicache.batch_get(keys2, dst_addrs, dst_lens)
        self.assertTrue(all(ret is None for ret in retrieved2))
        self.verify_tensor_lists_equal(values, dst_tensors2)

    def test_mixed_operations(self):
        """Test mixing single and batch operations."""
        # Test interleaved set/get operations
        key1, key2 = "key1", "key2"
        value1 = torch.randn(4, 4, device="cpu")
        value2 = torch.randn(6, 6, device="cpu")
        dst1 = torch.zeros_like(value1)
        dst2 = torch.zeros_like(value2)

        # Single set/get; baseline after first set absorbs any one-time NIXL internals
        self.assertTrue(self.hicache.set(key1, value1))
        fds = self._open_fds()
        retrieved1 = self.hicache.get(key1, dst1)
        self.verify_tensors_equal(value1, retrieved1)
        self.assertEqual(self._open_fds(), fds, "fd leak after get")

        # Batch set/get
        self.assertTrue(self.hicache.batch_set([key2], [value2]))
        self.assertEqual(self._open_fds(), fds, "fd leak after batch_set")
        retrieved2 = self.hicache.batch_get([key2], [dst2])
        self.verify_tensors_equal(value2, retrieved2[0])
        self.assertEqual(self._open_fds(), fds, "fd leak after batch_get")

    def test_data_integrity(self):
        """Test data integrity across operations."""
        # Test with various tensor types and sizes
        test_cases = [
            ("float32", torch.randn(10, 10, dtype=torch.float32)),
            ("float64", torch.randn(5, 5, dtype=torch.float64)),
            ("int32", torch.randint(-100, 100, (8, 8), dtype=torch.int32)),
            ("int64", torch.randint(-100, 100, (6, 6), dtype=torch.int64)),
            ("bool", torch.randint(0, 2, (4, 4)).bool()),
        ]

        for name, tensor in test_cases:
            with self.subTest(tensor_type=name):
                key = f"test_{name}"
                dst_tensor = torch.zeros_like(tensor)

                # Set and immediately get
                self.assertTrue(self.hicache.set(key, tensor))
                retrieved1 = self.hicache.get(key, dst_tensor)
                self.verify_tensors_equal(tensor, retrieved1)

                # Get again to verify persistence
                dst_tensor.zero_()
                retrieved2 = self.hicache.get(key, dst_tensor)
                self.verify_tensors_equal(tensor, retrieved2)

    def test_basic_file_operations(self):
        """Test basic file operations."""
        test_file = os.path.join(self.test_dir, "test_file.bin")
        self.file_manager.create_file(test_file)
        self.assertTrue(os.path.exists(test_file))
        self.assertEqual(os.path.getsize(test_file), 0)  # Empty file

        # Test file deletion
        self.assertTrue(self.delete_test_file(test_file))
        self.assertFalse(os.path.exists(test_file))

    def test_create_nixl_tuples(self):
        """Test creation of NIXL tuples."""
        test_file = os.path.join(self.test_dir, "test_file.bin")
        self.file_manager.create_file(test_file)

        # Test tuple creation
        tuples = self.file_manager.files_to_nixl_tuples([test_file])
        self.assertIsNotNone(tuples)
        self.assertTrue(len(tuples) > 0)

    def test_error_handling(self):
        """Test error handling in file operations."""
        # Test non-existent file
        self.assertTrue(
            self.delete_test_file("nonexistent_file.bin")
        )  # Returns True if file doesn't exist

        # Test invalid file path
        self.assertFalse(self.file_manager.create_file(""))  # Empty path should fail

    def test_register_buffers(self):
        """Test registration of memory buffers."""
        # Create test tensor
        tensor = torch.randn(10, 10)

        # Test buffer registration
        self.assertIsNotNone(self.hicache.register_buffers(tensor))

        # Test batch registration
        tensors = [torch.randn(5, 5) for _ in range(3)]
        self.assertIsNotNone(self.hicache.register_buffers(tensors))

    def test_register_files(self):
        """Test registration of files with NIXL."""
        files = [os.path.join(self.test_dir, f"test_file_{i}.bin") for i in range(3)]
        for file in files:
            self.file_manager.create_file(file)

        result = self.hicache.register_files(files)
        self.assertIsNotNone(result)

    def test_batch_set_v1_skips_on_nonzero_mla_rank(self):
        """Test batch_set_v1 is a no-op on nonzero MLA backup ranks."""
        self.hicache.storage_config.is_mla_model = True
        self.hicache.storage_config.tp_rank = 1
        self.hicache.backup_skip = True
        self.hicache._batch_set_preprocess = MagicMock(
            side_effect=AssertionError("batch_set_v1 should have been skipped")
        )

        results = self.hicache.batch_set_v1(["key1", "key2"], torch.tensor([0, 1]))

        self.assertEqual(results, [True, True])
        self.hicache._batch_set_preprocess.assert_not_called()

    def test_batch_exists_zero_copy_mla_uses_single_key_denominator(self):
        """Test zero-copy MLA batch_exists counts one storage key per logical key."""
        self.hicache.is_zero_copy = True
        self.hicache.is_mla_model = True
        self.hicache.agent.query_memory = MagicMock(return_value=[object(), None])

        result = self.hicache.batch_exists(["key1", "key2"])

        self.assertEqual(result, 1)

    def test_batch_exists_zero_copy_mha_uses_two_key_denominator(self):
        """Test zero-copy MHA batch_exists counts k/v pairs per logical key."""
        self.hicache.is_zero_copy = True
        self.hicache.is_mla_model = False
        self.hicache.agent.query_memory = MagicMock(
            return_value=[object(), object(), None, None]
        )

        result = self.hicache.batch_exists(["key1", "key2"])

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
