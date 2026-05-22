from __future__ import annotations

import ctypes
import logging

import torch

logger = logging.getLogger(__name__)


class CudaHostRegisterUtil:
    """cudaHostRegister / cudaHostUnregister via PyTorch cudart."""

    @staticmethod
    def register(ptr: int, size: int, flags: int) -> None:
        """Register host memory. No-op if CUDA is unavailable; raises on driver error."""
        err = torch.cuda.cudart().cudaHostRegister(ptr, size, flags)
        if err != 0:
            raise RuntimeError(
                f"cudaHostRegister failed (err={err}, size={size}, ptr={ptr}, flags={flags})"
            )

    @staticmethod
    def unregister(ptr: int) -> None:
        """Unregister host memory. No-op if CUDA is unavailable."""
        err = torch.cuda.cudart().cudaHostUnregister(ctypes.c_void_p(ptr))
        if err != 0:
            logger.warning("cudaHostUnregister failed (err=%s, ptr=%s)", err, ptr)
