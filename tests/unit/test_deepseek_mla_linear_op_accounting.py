"""Contracts for DeepSeek V2 MLA external-projection accounting.

Two rules are enforced by ``DeepseekV2MlaCausalSelfAttention`` and guarded here:

1. ``attn_pre_proj`` is the per-forward *sum* of ``q_proj``,
   ``kv_a_proj_with_mqa`` and ``kv_a_layernorm``.  Giving several distinct
   operators the same timer name would make ``TimerStatsStore`` report a median
   over interleaved samples of unrelated operations instead of a sum.
2. ``kv_a_layernorm`` is part of the real vLLM op sequence and must actually run.
"""

from __future__ import annotations

import pytest
import torch

from frontier.profiling.common.layers.layernorm import RMSNorm
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.linear_op.linear_op_impl import (
    DeepseekV2MlaCausalSelfAttention,
    build_linear_op_attention_module,
)
from frontier.profiling.utils.singleton import Singleton
from frontier.types import ActivationType, NormType

# The profiled linear layers allocate their weights on the current CUDA device,
# so these structural checks need a GPU host.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="profiled linear-op layers require CUDA weight allocation",
)


@pytest.fixture(autouse=True)
def _timer_stats_store():
    # ``CudaTimer`` resolves its configuration through the ``TimerStatsStore``
    # singleton, which is normally created by the profiling entry point.  Reset
    # and seed it here so the module can be constructed without running the
    # full profiler.
    Singleton._instances.pop(TimerStatsStore, None)  # pylint: disable=protected-access
    TimerStatsStore(profile_method="cuda_event")
    yield
    Singleton._instances.pop(TimerStatsStore, None)  # pylint: disable=protected-access


def _deepseek_v2_lite_config(**overrides) -> ModelConfig:
    kwargs = dict(
        name="DeepSeek/DeepSeekV2-Lite-Accounting-Unit",
        num_layers=27,
        num_q_heads=16,
        num_kv_heads=16,
        embedding_dim=2048,
        mlp_hidden_dim=1408,
        dense_mlp_hidden_dim=10944,
        max_position_embeddings=163840,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=102400,
        dtype="bfloat16",
        model_type="deepseek_v2",
        use_mla=True,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        # Left unset so the structural test does not depend on RoPE kernel
        # availability; the rope accounting rule is covered separately.
        rope_theta=None,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def test_pre_projection_sub_operators_use_distinct_timer_names() -> None:
    module = DeepseekV2MlaCausalSelfAttention(_deepseek_v2_lite_config(), world_size=1)

    # Distinct names are what makes the composite a sum.  A shared name would
    # make TimerStatsStore report a median over interleaved samples.
    assert module.q_proj._linear_timer.name == "vidur_attn_q_proj"
    assert module.kv_a_proj_with_mqa._linear_timer.name == "vidur_attn_kv_a_proj"
    assert (
        module.q_proj._linear_timer.name
        != module.kv_a_proj_with_mqa._linear_timer.name
    )


def test_composite_pre_projection_timer_owns_the_canonical_name() -> None:
    module = DeepseekV2MlaCausalSelfAttention(_deepseek_v2_lite_config(), world_size=1)

    assert module._attn_pre_proj_timer.name == "vidur_attn_pre_proj"
    # The quantization contract stays bound to the canonical composite name.
    assert module.q_proj._precision_op_name == "attn_pre_proj"
    assert module.kv_a_proj_with_mqa._precision_op_name == "attn_pre_proj"
    # The composite timer must not be disabled, otherwise the canonical metric
    # would disappear from the profile.
    assert module._attn_pre_proj_timer.disabled is False


def test_kv_a_layernorm_is_present_and_untimed() -> None:
    module = DeepseekV2MlaCausalSelfAttention(_deepseek_v2_lite_config(), world_size=1)

    assert isinstance(module.kv_a_layernorm, RMSNorm)
    assert module.kv_a_layernorm.weight.numel() == 512
    # Untimed on purpose: the cost is carried by the composite attn_pre_proj
    # timer, matching vLLM where the norm sits between the projections.
    assert module.kv_a_layernorm._norm_timer.disabled is True


def test_post_projection_timer_name_is_unchanged() -> None:
    module = DeepseekV2MlaCausalSelfAttention(_deepseek_v2_lite_config(), world_size=1)

    assert module.o_proj._linear_timer.name == "vidur_attn_post_proj"


def test_builder_selects_deepseek_mla_module_for_direct_q_proj_path() -> None:
    module = build_linear_op_attention_module(
        _deepseek_v2_lite_config(),
        world_size=1,
        enabled_ops=None,
        attn_sharded_enabled=True,
    )

    assert isinstance(module, DeepseekV2MlaCausalSelfAttention)


def test_builder_rejects_latent_q_projection_path() -> None:
    config = _deepseek_v2_lite_config(q_lora_rank=1536)

    with pytest.raises(NotImplementedError, match="q_lora_rank=None"):
        build_linear_op_attention_module(
            config,
            world_size=1,
            enabled_ops=None,
            attn_sharded_enabled=True,
        )
