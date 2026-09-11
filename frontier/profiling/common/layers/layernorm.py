"""Custom normalization layers for profiling."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from frontier.profiling.common.cuda_timer import CudaTimer

try:
    from vllm.model_executor.layers.layernorm import (
        GemmaRMSNorm as VllmGemmaRMSNorm,
        rms_norm as vllm_rms_norm,
        fused_add_rms_norm as vllm_fused_add_rms_norm,
    )

    HAS_VLLM_RMSNORM = True
except ImportError:
    # vLLM >= 0.27: RMSNorm became a CustomOp without module-level rms_norm /
    # fused_add_rms_norm functions. Fall back to dispatching through the
    # CustomOp instances under a default vLLM config context.
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.layernorm import (
            GemmaRMSNorm as VllmGemmaRMSNorm,
            RMSNorm as VllmRMSNormClass,
        )

        def _vllm_default_config_context():
            return set_current_vllm_config(VllmConfig())

        HAS_VLLM_RMSNORM = True

        class _VllmRmsNormShim:
            """Dispatches vLLM 0.27+ RMSNorm CustomOp as a plain function.

            Mirrors the pre-0.27 module-level function signatures:
              rms_norm(x, weight, eps)
              fused_add_rms_norm(x, residual, weight, eps) -> (out, residual)
            """

            def __init__(self, use_fused_add: bool):
                self._use_fused_add = use_fused_add
                with _vllm_default_config_context():
                    self._op = VllmRMSNormClass(1, eps=1e-6)

            def _bind(self, weight, eps):
                self._op.variance_epsilon = eps
                # Replace the weight slot so the CustomOp forward sees the
                # caller's weight (plain attribute bypasses Parameter checks).
                object.__setattr__(self._op, "weight", weight)

            def __call__(self, *args):
                if self._use_fused_add:
                    x, residual, weight, eps = args
                    self._bind(weight, eps)
                    return self._op(x, residual)
                x, weight, eps = args
                self._bind(weight, eps)
                return self._op(x)

        vllm_rms_norm = _VllmRmsNormShim(use_fused_add=False)
        vllm_fused_add_rms_norm = _VllmRmsNormShim(use_fused_add=True)
    except Exception:
        HAS_VLLM_RMSNORM = False
        VllmGemmaRMSNorm = None
        vllm_rms_norm = None
        vllm_fused_add_rms_norm = None


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using vLLM's fused RMSNorm CUDA kernel."""
        with self._norm_timer:
            if not HAS_VLLM_RMSNORM:
                raise ImportError(
                    "vLLM is required for RMSNorm profiling. "
                    "Install vllm or set PYTHONPATH to the vllm source tree."
                )
            if residual is None:
                return vllm_rms_norm(x, self.weight.data, self.variance_epsilon)
            return vllm_fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm wrapper with profiling timer."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)
        if not HAS_VLLM_RMSNORM or VllmGemmaRMSNorm is None:
            raise ImportError(
                "vLLM is required for GemmaRMSNorm profiling. "
                "Install vllm or set PYTHONPATH to the vllm source tree."
            )
        self._impl = VllmGemmaRMSNorm(hidden_size, eps=eps)

    @property
    def weight(self) -> nn.Parameter:
        return self._impl.weight

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with self._norm_timer:
            return self._impl(x, residual)
