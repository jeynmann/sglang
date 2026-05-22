"""HiCache host tensor allocator for NIXL storage (optional hugetlb-backed CPU buffers)."""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.memory_pool_host import HostTensorAllocator


class NixlHostTensorAllocator(HostTensorAllocator):
    """Allocates host tensors; uses Linux hugetlb when ``use_host_hugepages`` is true."""

    def __init__(self, use_host_hugepages: bool = False):
        super().__init__()
        self._use_host_hugepages = use_host_hugepages

    @property
    def use_host_hugepages(self) -> bool:
        return self._use_host_hugepages

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        self.dtype = dtype
        self.dims = dims
        if self._use_host_hugepages and str(device) == "cpu":
            from sglang.srt.mem_cache.storage.nixl.hugepage_util import HugepageUtil

            elem = int(torch.empty((), dtype=dtype).element_size())
            n_elem = 1
            for d in dims:
                n_elem *= int(d)
            num_bytes = n_elem * elem
            array_type, holder = HugepageUtil.mmap_buffer(num_bytes)
            tensor_u8 = torch.frombuffer(array_type, dtype=torch.uint8, count=num_bytes)
            buffer = tensor_u8.view(dtype).reshape(dims)
            # view/reshape drops arbitrary attrs; keep holder on the tensor HiCache uses.
            buffer._sglang_hugepage_holder = holder
            return buffer
        return super().allocate(dims, dtype=dtype, device=device)
