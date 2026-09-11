"""Reproducible CPU-only smoke for the stage-1 DeepSeek V2 Lite profiles.

Never deletes prior outputs. Cache keys are validated by Frontier (including the
input dataframe hash). Set --cache-dir to reuse a previous compatible cache.
The 2-request audit is not a performance benchmark; --audit enables OP-TRACE.
Use the default 100 requests and RF settings for the six-cell smoke.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workloads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--tp', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--qps', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--timeout', type=int, default=2400)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.workloads = args.workloads.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = ROOT / 'data/profiling/compute/h100/DeepSeek/DeepSeekV2-Lite'
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                'profile_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in profile_dir.glob('*.csv')},
                'runs': []}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for tp in args.tp:
        if tp not in (1, 2, 4):
            raise ValueError('Only TP1/2/4 shapes are in this experiment contract')
        for qps in args.qps:
            directory = args.output / f'tp{tp}_ep{tp}_qps{qps}'
            directory.mkdir()
            source = args.workloads / f'fixed1024x128__qps{qps}__n100.csv'
            with source.open() as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                rows = list(reader)
            if not 0 < args.requests <= len(rows):
                raise ValueError('Invalid request count')
            trace = directory / 'trace.csv'
            with trace.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows[:args.requests])
            command = [sys.executable, '-m', 'frontier.main',
                '--simulation_mode', 'online', '--sys_arch', 'co-location', '--seed', '42',
                '--log_level', 'info' if args.audit else 'warning', '--time_limit', '600',
                '--cluster_scheduler_config_type', 'round_robin',
                '--cluster_config_num_replicas', str(4 // tp),
                '--replica_config_attn_tensor_parallel_size', str(tp),
                '--replica_config_moe_tensor_parallel_size', '1',
                '--replica_config_moe_expert_parallel_size', str(tp),
                '--replica_config_total_expert_num', '64',
                '--replica_config_local_expert_num', str(64 // tp),
                '--replica_config_router_topk', '6',
                '--replica_config_moe_routing_distribution_type', 'balanced',
                '--replica_config_moe_routing_seed', '42',
                '--replica_config_device', 'h100', '--replica_config_network_device', 'h100_dgx',
                '--replica_config_model_name', 'DeepSeek/DeepSeekV2-Lite',
                '--replica_scheduler_config_type', 'vllm_v1',
                '--vllm_v1_scheduler_config_block_size', '64',
                '--vllm_v1_scheduler_config_batch_size_cap', '256',
                '--vllm_v1_scheduler_config_max_tokens_in_batch', '2048',
                '--vllm_v1_scheduler_config_enable_chunked_prefill',
                '--no-vllm_v1_scheduler_config_enable_prefix_caching',
                '--decode_cuda_graph_mode', 'piecewise',
                '--execution_time_predictor_config_type', 'random_forrest',
                '--no-random_forrest_execution_time_predictor_config_enable_dummy_mode',
                '--random_forrest_execution_time_predictor_config_num_estimators', '250',
                '--random_forrest_execution_time_predictor_config_max_depth', '16',
                '--random_forrest_execution_time_predictor_config_min_samples_split', '2',
                '--random_forrest_execution_time_predictor_config_num_training_job_threads', '2',
                '--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling',
                '--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size', '4096',
                '--random_forrest_execution_time_predictor_config_prediction_max_batch_size', '256',
                '--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request', '67584',
                '--cc_backend_config_type', 'analytical',
                '--metrics_config_cache_dir', str(args.cache_dir),
                '--metrics_config_output_dir', str(directory),
                '--request_generator_config_type', 'trace_replay',
                '--trace_request_generator_config_trace_file', str(trace)]
            record = {'tp': tp, 'ep': tp, 'replicas': 4 // tp, 'qps': qps,
                      'requests': args.requests, 'command': command,
                      'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest()}
            (directory / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
            start = time.monotonic()
            with (directory / 'run.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    record['exit_code'] = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    record['exit_code'] = 124
            record['wall_seconds'] = time.monotonic() - start
            metrics = list(directory.rglob('request_metrics.csv'))
            record['metrics_files'] = [str(p) for p in metrics]
            record['completed_rows'] = 0
            if len(metrics) == 1:
                with metrics[0].open() as f:
                    record['completed_rows'] = sum(1 for _ in csv.DictReader(f))
            record['execution_pass'] = record['exit_code'] == 0 and record['completed_rows'] == args.requests
            manifest['runs'].append(record)
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps({k: record[k] for k in ('tp', 'qps', 'wall_seconds', 'exit_code', 'completed_rows', 'execution_pass')}), flush=True)
            if not record['execution_pass']:
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
