import hashlib
import logging
import time
import uuid
from typing import Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.mmap_allocator import alloc_mmap

from .nixl_registry import NixlRegistry
from .nixl_utils import NixlBackendConfig, NixlBackendSelection, NixlFileManager

try:
    from nixl._api import nixl_agent, nixl_agent_config, nixlBind
except ImportError as e:
    raise ImportError(
        "Please install NIXL by following the instructions at "
        "https://github.com/ai-dynamo/nixl/blob/main/README.md "
        "to use HiCacheNixl storage backend."
    ) from e

try:
    from nixl._api import nixl_thread_sync_t as _nixl_thread_sync_t

    _NIXL_SYNC_MODE_SUPPORTED = True
    _NIXL_SYNC_MODE_DEFAULT = _nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT
except ImportError:
    _nixl_thread_sync_t = None  # type: ignore[misc,assignment]
    _NIXL_SYNC_MODE_SUPPORTED = False
    _NIXL_SYNC_MODE_DEFAULT = None

logger = logging.getLogger(__name__)


class HiCacheNixl(HiCacheStorage):
    """HiCacheNixl provides high-performance storage using NIXL plugins."""

    @staticmethod
    def _parse_sync_mode(cfg: dict[str, Any]) -> Any:
        sync_mode = _NIXL_SYNC_MODE_DEFAULT
        sync_mode_str = cfg.get("nixl_sync_mode")
        if sync_mode_str is None:
            return sync_mode

        if not _NIXL_SYNC_MODE_SUPPORTED:
            raise ValueError(
                "nixl_sync_mode is set in --hicache-storage-backend-extra-config but "
                "this NIXL build does not support it (requires ai-dynamo/nixl#1501). "
                "Remove nixl_sync_mode from config or upgrade NIXL."
            )

        attr_name = f"NIXL_THREAD_SYNC_{str(sync_mode_str).upper()}"
        if not hasattr(_nixl_thread_sync_t, attr_name):
            raise ValueError(
                f"Invalid nixl_sync_mode '{sync_mode_str}'. "
                "Use a suffix of NIXL_THREAD_SYNC_* from nixl_thread_sync_t."
            )
        return getattr(_nixl_thread_sync_t, attr_name)

    @staticmethod
    def _parse_agent_config(extra: Optional[dict]) -> Any:
        """Build ``nixl_agent_config`` with optional ``nixl_enable_prog_thread`` / ``nixl_sync_mode``."""
        cfg = extra or {}
        kwargs: dict[str, Any] = {"backends": []}

        enable_prog = NixlBackendConfig.is_truthy(
            cfg.get("nixl_enable_prog_thread", True)
        )
        kwargs["enable_prog_thread"] = enable_prog

        sync_mode = HiCacheNixl._parse_sync_mode(cfg)
        if sync_mode is not None:
            kwargs["sync_mode"] = sync_mode

        return nixl_agent_config(**kwargs)

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_storage",
    ):
        """Initialize NIXL storage connector."""

        # create nixlconfig from the --hicache-storage-backend-extra-config
        nixlconfig = NixlBackendConfig(storage_config.extra_config)

        # select the NIXL backend plugin from extra_config or environment variable
        plugin = nixlconfig.get_specified_plugin()

        use_direct_io = nixlconfig.get_use_direct_io()

        # Might be better to be unified across HiCache backends and moved to HiCacheController
        file_path = envs.SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR.get() or file_path
        self.file_manager = (
            NixlFileManager(file_path, use_direct_io=use_direct_io)
            if plugin not in NixlBackendSelection.OBJ_PLUGINS
            else None
        )

        tp_rank, tp_size, model_name = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
        )

        self.is_mla_model = storage_config.is_mla_model
        self.is_zero_copy = False
        self.storage_config = storage_config
        self.backup_skip = self.is_mla_model and storage_config.tp_rank != 0

        model_name = "-".join(model_name.split("/")) if model_name else ""

        if self.is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        agent_config = self._parse_agent_config(storage_config.extra_config)
        sync_mode = getattr(agent_config, "sync_mode", None)
        if sync_mode is None:
            sync_mode = getattr(
                nixlBind, "NIXL_THREAD_SYNC_RW", nixlBind.NIXL_THREAD_SYNC_STRICT
            )
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"
        self.agent = nixl_agent(self.agent_name, agent_config)
        bind_cfg = nixlBind.nixlAgentConfig()
        bind_cfg.useProgThread = agent_config.enable_pthread
        bind_cfg.useListenThread = agent_config.enable_listen
        bind_cfg.listenPort = agent_config.port
        bind_cfg.syncMode = sync_mode
        bind_cfg.pthrDelay = 0
        bind_cfg.lthrDelay = 100000
        bind_cfg.captureTelemetry = agent_config.capture_telemetry
        self.agent.agent = nixlBind.nixlAgent(self.agent_name, bind_cfg)
        self.agent.plugin_list = self.agent.agent.getAvailPlugins()

        self.backend_selector = NixlBackendSelection(plugin, nixlconfig)
        if not self.backend_selector.create_backend(self.agent):
            raise RuntimeError("Failed to create NIXL backend")

        self._require_host_hugepages = (
            self.backend_selector.backend_name == "DOCA_MEMOS"
        )
        if self.backend_selector.backend_name == "DOCA_MEMOS":
            self._format_key = HiCacheNixl._format_key_doca_memos
            obj_reg_tuple_fn = HiCacheNixl._obj_reg_tuple_doca_memos
        else:
            self._format_key = HiCacheNixl._format_key_default
            obj_reg_tuple_fn = None

        self.registry = NixlRegistry(
            self.agent,
            self.backend_selector.mem_type,
            self.file_manager,
            obj_reg_tuple_fn=obj_reg_tuple_fn,
        )
        self.needs_page_alignment = use_direct_io and self.file_manager is not None
        if self.needs_page_alignment:
            logger.info(
                "HiCacheNixl: O_DIRECT is active with a file-based backend (%s). "
                "Page-aligned host buffers are required (needs_page_alignment=True).",
                self.backend_selector.backend_name,
            )
        self._host_regs: List[Any] = []
        self._bounce_set: Optional[torch.Tensor] = None
        self._bounce_get: Optional[torch.Tensor] = None
        self._bounce_page_bytes: Optional[int] = None

    def _assert_doca_memos_host_hugepages(self, mem_pool_host: HostKVCache) -> None:
        if not self._require_host_hugepages:
            return
        if getattr(mem_pool_host, "kv_buffer", None) is None:
            return
        if not mem_pool_host.use_host_hugepages:
            raise RuntimeError(
                "HiCache NIXL DOCA_MEMOS requires hugetlb-backed host memory. "
                "Enable use_host_hugepages in --hicache-storage-backend-extra-config "
                "and reserve Linux huge pages (vm.nr_hugepages)."
            )

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._assert_doca_memos_host_hugepages(host_pool)

    @staticmethod
    def _format_key_default(key: str) -> str:
        return key

    @staticmethod
    def _format_key_doca_memos(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _obj_reg_tuple_doca_memos(key: str, size: int, _dev_id: int = 0) -> tuple:
        key_hash = hashlib.sha256(key.encode("utf-8"))
        return (
            0,
            size,
            int.from_bytes(key_hash.digest()[:8], "big"),
            key_hash.hexdigest()[:32],
        )

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _create_query_tuple(self, key: str) -> tuple:
        """Build the NIXL query_memory tuple for a single key."""
        if self.backend_selector.mem_type == "FILE":
            return (0, 0, 0, self.file_manager.get_file_path(key))
        return (0, 0, 0, self._format_key(key))

    # ------------------------------------------------------------------
    # Transfer pipeline
    # ------------------------------------------------------------------

    def _xfer_and_wait(
        self,
        host_descs: Any,
        storage_descs: Any,
        direction: str,
    ) -> bool:
        """Initialize and poll a NIXL transfer to completion."""
        try:
            xfer_req = self.agent.initialize_xfer(
                direction, host_descs, storage_descs, self.agent_name
            )
        except Exception as e:
            logger.error(f"Failed to create transfer request: {e}")
            return False

        try:
            state = self.agent.transfer(xfer_req)
            while state != "DONE":
                state = self.agent.check_xfer_state(xfer_req)
                if state == "ERR":
                    logger.error("Transfer failed")
                    return False
                time.sleep(0.0001)  # Can be changed to os.sched_yield() or parametrized
            return True
        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            self.agent.release_xfer_handle(xfer_req)

    def _xfer_pre_registered(
        self,
        host_buffers: List[tuple],
        keys: List[str],
        direction: str,
    ) -> bool:
        """Run a transfer where the host side is already pre-registered.

        ``host_buffers`` is a list of ``(addr, size)`` tuples within the
        pre-registered host region (kv_buffer for zero-copy, bounce buffer
        otherwise). Only the storage side is registered per transfer.
        """
        if len(host_buffers) != len(keys):
            logger.error("Mismatch between number of host buffers and keys")
            return False

        host_descs = self.agent.get_xfer_descs(
            [(addr, size, 0) for (addr, size) in host_buffers], "DRAM"
        )
        if host_descs is None:
            logger.error("Failed to build host xfer descs")
            return False

        storage_keys = (
            [self._format_key(k) for k in keys]
            if self.backend_selector.mem_type == "OBJ"
            else keys
        )
        with self.registry.storage(
            host_buffers, storage_keys, direction
        ) as storage_descs:
            if storage_descs is None:
                return False
            return self._xfer_and_wait(host_descs, storage_descs, direction)

    # Deprecated; only batch_{get,set}_v1 are used.

    # Deprecated
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        pass

    # Deprecated
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        pass

    # Deprecated
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    # Deprecated
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        self._assert_doca_memos_host_hugepages(mem_pool_host)

        # enable zero-copy automatically if mem layout is page_first or page_first_direct
        self.is_zero_copy = self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]

        if self._require_host_hugepages and not self.is_zero_copy:
            raise RuntimeError(
                "HiCache NIXL DOCA_MEMOS requires page_first or page_first_direct "
                "host layout for L3 zero-copy I/O (got "
                f"{self.mem_pool_host.layout!r}). Set --hicache-mem-layout page_first "
                "or page_first_direct."
            )

        if self.needs_page_alignment and self.is_zero_copy:
            # Check that the kv_buffer base AND per-page strides are multiples of
            # the OS page size so every pointer passed to NIXL (base + p * stride)
            # is page-aligned. The base is whatever torch.empty() happened to give
            # us -- it is not guaranteed to be page-aligned. Fall back to copy mode
            # if either condition fails.
            # 4096: O_DIRECT alignment is FS-dependent (some allow 512 B); 4 KiB
            # is the safe lower bound all known FSes accept, and real page-sizes meet it.
            if not self.mem_pool_host.is_stride_page_aligned(4096):
                logger.warning(
                    "HiCacheNixl: O_DIRECT is active but the host kv_buffer is "
                    "not OS-page-aligned (base or per-page stride). Falling back "
                    "to copy mode for this pool."
                )
                self.is_zero_copy = False

        if self.is_zero_copy:
            kv = mem_pool_host.kv_buffer
            self._pre_register_host(
                kv.data_ptr(), kv.numel() * kv.element_size(), "kv_buffer"
            )
        else:
            # One bounce buffer per direction so set/get run lock-free across
            # the prefetch and backup threads. Sized from get_dummy_flat_data_page()
            # so each slot matches what the v1 path would otherwise allocate.
            sample = mem_pool_host.get_dummy_flat_data_page()
            page_numel = sample.numel()
            self._bounce_page_bytes = page_numel * sample.element_size()
            del sample
            pin_memory = bool(getattr(mem_pool_host, "pin_memory", False))
            self._bounce_set = self._alloc_registered(
                page_numel, mem_pool_host.dtype, pin_memory, "bounce_set"
            )
            self._bounce_get = self._alloc_registered(
                page_numel, mem_pool_host.dtype, pin_memory, "bounce_get"
            )

        logger.info(
            f"HiCacheNixl: pre-registered host regions for "
            f"layout={mem_pool_host.layout} zero_copy={self.is_zero_copy}"
        )

    def _alloc_registered(
        self,
        page_numel: int,
        dtype: torch.dtype,
        pin_memory: bool,
        kind: str,
    ) -> torch.Tensor:
        """Allocate a ``(STORAGE_BATCH_SIZE, page_numel)`` bounce buffer and
        pre-register it as a DRAM region with NIXL. Uses alloc_mmap so the
        buffer is page-aligned -- required when O_DIRECT is on for any
        file-based backend (POSIX/GDS/GDS_MT/3FS). pin_memory is currently
        unused (alloc_mmap does not support it)."""
        buf = alloc_mmap((STORAGE_BATCH_SIZE, page_numel), dtype)
        self._pre_register_host(buf.data_ptr(), buf.numel() * buf.element_size(), kind)
        return buf

    def _pre_register_host(self, base_addr: int, total_size: int, kind: str) -> None:
        """Register a single DRAM region up-front and remember the handle."""
        reg_descs = self.agent.get_reg_descs([(base_addr, total_size, 0, "")], "DRAM")
        if reg_descs is None:
            raise RuntimeError(f"Failed to build reg descs for host {kind}")
        try:
            self._host_regs.append(self.agent.register_memory(reg_descs))
        except Exception as e:
            raise RuntimeError(f"Failed to pre-register host {kind} with NIXL") from e

    def close(self):
        while self._host_regs:
            reg = self._host_regs.pop()
            try:
                self.agent.deregister_memory(reg)
            except Exception as e:
                logger.debug("deregister of pre-registered host region failed: %s", e)
        self._bounce_set = None
        self._bounce_get = None
        self._bounce_page_bytes = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        results = self.batch_exists([key])
        return results > 0

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if self.is_zero_copy:
            key_list = self._get_key_list_from_meta(keys)
            key_denominator = (
                1 if self.is_mla_model else 2
            )  # MLA: 1 key per page (_k only), non-MLA: 2 NIXL keys per page (_k + _v)
        else:
            key_list = [self._get_suffixed_key(key) for key in keys]
            key_denominator = 1

        tuples = [self._create_query_tuple(key) for key in key_list]

        query_res = self.agent.query_memory(
            tuples,
            self.backend_selector.backend_name,
            mem_type=self.backend_selector.mem_type,
        )

        for i in range(len(query_res)):
            if query_res[i] is None:
                return i // key_denominator
        return len(query_res) // key_denominator

    # ------------------------------------------------------------------
    # batch_*_v1 implementation (zero-copy + non-zero-copy)
    # ------------------------------------------------------------------

    def _get_key_list_from_meta(self, keys: List[str]) -> List[str]:
        # Each key maps to a `_k` entry, plus a `_v` entry on non-MLA models
        # (MLA stores k/v interleaved in a single buffer).
        key_list = []
        for key in keys:
            suffixed_key = self._get_suffixed_key(key)
            key_list.append(f"{suffixed_key}_k")
            if not self.is_mla_model:
                key_list.append(f"{suffixed_key}_v")
        return key_list

    def _get_location_and_size_list_from_meta(
        self, keys: List[str], host_indices: torch.Tensor
    ):
        # zero copy: mem_pool_host.get_data_page() does not work due to non-contiguous tensors, causing issues for NIXL transfer
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(
            host_indices
        )
        key_list = self._get_key_list_from_meta(keys)

        if len(key_list) != len(ptr_list):
            logger.error(
                f"HiCacheNixl: mismatch between number of keys and number of buffer meta entries, keys: {len(keys)}, key_list: {len(key_list)}, buffer meta entries: {len(ptr_list)}"
            )
            return [], [], []

        return key_list, ptr_list, element_size_list

    def _bounce_slot_buffers(self, buf: torch.Tensor, page_num: int) -> List[tuple]:
        """Return ``page_num`` ``(addr, size)`` tuples pointing at the first
        ``page_num`` slots of ``buf``.
        """
        base = buf.data_ptr()
        return [
            (base + i * self._bounce_page_bytes, self._bounce_page_bytes)
            for i in range(page_num)
        ]

    def _batch_preprocess(self, keys: List[str], host_indices: torch.Tensor, op: str):
        """Build (key_list, host_buffers) for the v1 path.

        For zero-copy: ``host_buffers`` are ``(addr, size)`` tuples inside the
        pre-registered ``kv_buffer``.
        For non-zero-copy: ``host_buffers`` are slots of the direction-specific
        pre-registered bounce buffer (``_bounce_set`` for set, ``_bounce_get``
        for get); for ``op == "set"`` we copy the host pages into those slots
        here so the subsequent transfer reads from the bounce buffer.
        Returns ``([], [])`` on validation failure.
        """
        page_size = self.mem_pool_host.page_size
        page_num = len(host_indices) // page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {page_size}"
            )
            return [], []

        if self.is_zero_copy:
            key_list, ptr_list, size_list = self._get_location_and_size_list_from_meta(
                keys, host_indices
            )
            host_buffers = list(zip(ptr_list, size_list))
            return key_list, host_buffers

        if page_num > STORAGE_BATCH_SIZE:
            logger.error(
                f"HiCacheNixl: batch size {page_num} exceeds bounce buffer capacity {STORAGE_BATCH_SIZE}"
            )
            return [], []

        bounce = self._bounce_set if op == "set" else self._bounce_get
        if op == "set":
            for i in range(page_num):
                src = self.mem_pool_host.get_data_page(
                    host_indices[i * page_size], flat=True
                )
                bounce[i].copy_(src)

        host_buffers = self._bounce_slot_buffers(bounce, page_num)
        key_list = [self._get_suffixed_key(key) for key in keys]
        return key_list, host_buffers

    def _batch_xfer(
        self,
        keys: List[str],
        key_strs: List[str],
        host_buffers: List[tuple],
        direction: str,
    ) -> List[bool]:
        """Run a batch READ or WRITE for the v1 path against the pre-registered
        host region (no per-transfer host registration).
        """
        if not key_strs or not host_buffers:
            return [False] * len(keys)

        if len(key_strs) != len(host_buffers):
            logger.error("Mismatch between number of key_strs and host_buffers")
            return [False] * len(keys)

        if self.backend_selector.mem_type == "FILE":
            file_paths = [self.file_manager.get_file_path(key) for key in key_strs]
            success = self._xfer_pre_registered(host_buffers, file_paths, direction)
        else:  # mem_type == "OBJ"
            success = self._xfer_pre_registered(host_buffers, key_strs, direction)

        # READ results are consumed by _batch_get_postprocess, which pairs
        # entries 2*i / 2*i+1 for non-MLA zero-copy: it needs one bool per
        # key_str (i.e. per `_k`/`_v` buffer). WRITE results map 1:1 to
        # pages, i.e. to `keys`.
        result_len = len(key_strs) if direction == "READ" else len(keys)
        return [success] * result_len

    def _batch_get_postprocess(
        self,
        host_indices: torch.Tensor,
        results: List[bool],
    ) -> List[bool]:
        page_size = self.mem_pool_host.page_size
        page_num = len(host_indices) // page_size

        if self.is_zero_copy:
            # zero copy: update final results based on the boolean results from NIXL transfer
            if self.is_mla_model:
                return results
            return [(results[2 * i] and results[2 * i + 1]) for i in range(page_num)]

        # non zero copy: copy data from the get-side bounce buffer to mem_pool_host
        for i in range(page_num):
            if not results[i]:
                break
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * page_size], self._bounce_get[i]
            )
        return results

    def _log_xfer_stats(
        self,
        op_name: str,
        num_keys: int,
        host_indices: torch.Tensor,
        buffer_sizes: List[int],
        elapsed_ms: float,
    ) -> None:
        total_bytes = sum(s for s in buffer_sizes if s is not None)
        bw = total_bytes / (elapsed_ms / 1000) / (1024 * 1024) if elapsed_ms else 0.0
        logger.debug(
            f"HiCacheNixl {op_name} transferred: {num_keys} keys (pages), "
            f"{host_indices.numel()} host_indices, {total_bytes} bytes, "
            f"total time: {elapsed_ms:.3f} ms, effective bandwidth: {bw:.2f} MB/s"
        )

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if not self._host_regs:
            logger.error(
                "HiCacheNixl batch_get_v1: register_mem_pool_host must be called first"
            )
            return [False] * len(keys)

        key_strs, host_buffers = self._batch_preprocess(keys, host_indices, "get")
        if not key_strs or not host_buffers:
            return [False] * len(keys)

        start_time = time.perf_counter()
        results = self._batch_xfer(keys, key_strs, host_buffers, "READ")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log_xfer_stats(
            "batch_get_v1",
            len(keys),
            host_indices,
            [s for _, s in host_buffers],
            elapsed_ms,
        )

        return self._batch_get_postprocess(host_indices, results)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # skip on MLA backup rank
        if self.backup_skip:
            return [True] * len(keys)

        if len(keys) == 0:
            return []

        if not self._host_regs:
            logger.error(
                "HiCacheNixl batch_set_v1: register_mem_pool_host must be called first"
            )
            return [False] * len(keys)

        key_strs, host_buffers = self._batch_preprocess(keys, host_indices, "set")
        if not key_strs or not host_buffers:
            return [False] * len(keys)

        start_time = time.perf_counter()
        results = self._batch_xfer(keys, key_strs, host_buffers, "WRITE")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log_xfer_stats(
            "batch_set_v1",
            len(keys),
            host_indices,
            [s for _, s in host_buffers],
            elapsed_ms,
        )

        return results
