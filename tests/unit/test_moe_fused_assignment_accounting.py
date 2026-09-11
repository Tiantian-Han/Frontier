"""Contracts for MoE token-assignment accounting.

``fused_experts()`` performs ``_prepare_expert_assignment`` →
``moe_align_block_size(...)`` internally before the grouped GEMM, and the modular
``TritonExperts`` runtime does the same.  Frontier models that alignment work
separately as ``moe_shuffling``, so when the grouped-GEMM term was measured
through the fused runtime kernel the separate term must be suppressed to avoid
double counting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frontier.config import ClusterType
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)


class _Predictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _Batch:
    num_prefill_tokens = 0
    requests = []
    total_num_tokens = 32

    def get_effective_total_tokens_rounded(self, _cluster_type: ClusterType) -> int:
        return 32


def _predictor(moe_input_file: Path | None) -> _Predictor:
    predictor = object.__new__(_Predictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._supports_operation = lambda _operation: True
    predictor._predictions = {"moe_shuffling": {(32,): 4.0}}
    if moe_input_file is not None:
        predictor._moe_input_file = str(moe_input_file)
    return predictor


def _write_moe_csv(
    path: Path,
    backend_values: list[str],
    assignment_values: list[str] | None = None,
) -> Path:
    if assignment_values is None:
        assignment_values = ["false"] * len(backend_values)
    assert len(assignment_values) == len(backend_values)
    rows = "\n".join(
        f"{backend},{includes}"
        for backend, includes in zip(backend_values, assignment_values)
    )
    path.write_text(
        "moe_grouped_gemm_backend,moe_grouped_gemm_includes_assignment\n"
        f"{rows}\n",
        encoding="utf-8",
    )
    return path


def test_fused_runtime_kernel_requires_explicit_assignment_provenance(
    tmp_path: Path,
) -> None:
    moe_csv = _write_moe_csv(
        tmp_path / "fused.csv",
        ["vllm_fused", "vllm_fused"],
        ["true", "true"],
    )

    predictor = _predictor(moe_csv)

    assert predictor._moe_grouped_gemm_includes_assignment() is True


def test_fused_runtime_kernel_suppresses_the_separate_shuffling_term(
    tmp_path: Path,
) -> None:
    moe_csv = _write_moe_csv(tmp_path / "fused.csv", ["vllm_fused"], ["true"])

    predictor = _predictor(moe_csv)

    # The grouped-GEMM term already contains moe_align_block_size, so adding the
    # separately measured local shuffling term would count it twice.
    assert predictor._get_moe_shuffling_time(_Batch()) == 0.0


def test_loop_backend_keeps_the_separate_shuffling_term(tmp_path: Path) -> None:
    moe_csv = _write_moe_csv(tmp_path / "loop.csv", ["frontier_loop"])

    predictor = _predictor(moe_csv)

    assert predictor._moe_grouped_gemm_includes_assignment() is False
    assert predictor._get_moe_shuffling_time(_Batch()) == pytest.approx(4.0)


def test_legacy_profile_without_provenance_keeps_the_separate_term(
    tmp_path: Path,
) -> None:
    legacy_csv = tmp_path / "legacy.csv"
    legacy_csv.write_text(
        "num_tokens,time_stats.moe_shuffling.median,moe_grouped_gemm_backend\n"
        "32,0.5,vllm_fused\n",
        encoding="utf-8",
    )

    predictor = _predictor(legacy_csv)

    # Profiles that predate the provenance column have unchanged grouped-GEMM
    # semantics, so the separate term must be preserved.
    assert predictor._moe_grouped_gemm_includes_assignment() is False
    assert predictor._get_moe_shuffling_time(_Batch()) == pytest.approx(4.0)


def test_missing_moe_profile_keeps_the_separate_term() -> None:
    predictor = _predictor(None)

    assert predictor._moe_grouped_gemm_includes_assignment() is False


def test_provenance_is_resolved_once_per_predictor(tmp_path: Path) -> None:
    moe_csv = _write_moe_csv(tmp_path / "fused.csv", ["vllm_fused"], ["true"])
    predictor = _predictor(moe_csv)

    assert predictor._moe_grouped_gemm_includes_assignment() is True

    # Removing the file must not change the already-resolved answer, otherwise a
    # long simulation could flip the accounting mid-run.
    moe_csv.unlink()

    assert predictor._moe_grouped_gemm_includes_assignment() is True
