"""Replay saved training rollouts with a frozen teacher; never execute tool calls.

Example: python -m scripts.analyze_teacher_tail --run D:/opdistill/experiments/opd_06b_4b_validation_8735251 --teacher-run D:/opdistill/experiments/sft_4b_xlam_teacher_1e722fe --output D:/opdistill/experiments/teacher_tail_8735251
"""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer

from scripts.run_opd_pilot import hash_tree, load_base, sha256
from scripts.run_sft_baseline import git_state
from src.json_io import write_json_atomic
from src.teacher_tail import check_replay_log_probs, teacher_statistics, validate_trajectory

KS = (8, 32, 128, 512)
REPLAY_LOGP_ATOL = 0.03


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def mean(values):
    return sum(values) / len(values)


def aggregate(rows):
    """Report both action-weighted and equal-trajectory means; no independence claim."""
    n = sum(r['action_count'] for r in rows)
    result = {'trajectory_count': len(rows), 'action_count': n,
              'entropy_nats_action_mean': sum(r['entropy_nats_mean'] * r['action_count'] for r in rows) / n,
              'entropy_nats_trajectory_mean': mean([r['entropy_nats_mean'] for r in rows]), 'per_k': {}}
    for k in KS:
        records = [r['per_k'][str(k)] for r in rows]
        result['per_k'][str(k)] = {
            'mass_action_mean': sum(x['mass_mean'] * r['action_count'] for x, r in zip(records, rows)) / n,
            'tail_action_mean': sum(x['tail_mean'] * r['action_count'] for x, r in zip(records, rows)) / n,
            'inclusion_action_fraction': sum(x['included_count'] for x in records) / n,
            'tail_equal_trajectory_mean': mean([x['tail_mean'] for x in records]),
            'inclusion_equal_trajectory_mean': mean([x['included_count'] / r['action_count'] for x, r in zip(records, rows)])}
    return result


def run(source, teacher_run, output):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    source_files = ('status.json', 'config.json', 'provenance.json', 'selection.json', 'opd/trajectories.json')
    hashes = {name: sha256(source / name) for name in source_files}
    if read(source / 'status.json').get('status') != 'complete':
        raise ValueError('Source run incomplete')
    config, provenance, selection = (read(source / name) for name in ('config.json', 'provenance.json', 'selection.json'))
    rows = read(source / 'opd/trajectories.json')
    if (not rows or len(rows) != config['max_steps'] or
        [r['step'] for r in rows] != list(range(1, len(rows) + 1)) or
        [r['id'] for r in rows] != selection['train_ids'] or
        len(set(selection['train_ids'])) != len(rows) or
        set(selection['train_ids']) & set(selection['eval_ids'])):
        raise ValueError('Saved training trajectory selection is inconsistent')
    if not provenance['git'].get('commit') or provenance['git'].get('dirty') is not False:
        raise ValueError('Source run lacks clean Git pin')
    teacher_hashes = {name: sha256(teacher_run / name) for name in provenance['teacher_evidence_sha256']}
    adapter_hashes = hash_tree(teacher_run / 'adapter')
    if teacher_hashes != provenance['teacher_evidence_sha256'] or adapter_hashes != provenance['teacher_adapter_files_sha256']:
        raise ValueError('Teacher files differ from original run')
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / 'status.json', {'status': 'incomplete'})
    started = time.monotonic()
    write_json_atomic(output / 'provenance.json', {
        'source_run': str(source.resolve()), 'source_files_sha256': hashes,
        'source_git': provenance['git'], 'dataset_sha256': provenance['dataset_sha256'],
        'teacher_run': str(teacher_run.resolve()), 'teacher_evidence_sha256': teacher_hashes,
        'teacher_adapter_files_sha256': adapter_hashes, 'teacher_id': config['teacher_id'],
        'teacher_revision': config['teacher_revision'], 'student_id': config['student_id'],
        'student_revision': config['student_revision'], 'diagnostic_git': git_state(),
        'argv': sys.orig_argv, 'python': sys.version, 'platform': platform.platform(),
        'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')},
        'selection': 'all saved OPD training trajectories in selection.json order; no evaluation trajectories',
        'replay_teacher_action_log_probability_atol': REPLAY_LOGP_ATOL,
        'probability': 'FP32 full vocabulary softmax, temperature 1; direct sum outside teacher top K',
        'tie_policy': 'descending FP32 teacher logit, ascending token ID for rank and inclusion',
        'denominators': 'mass and entropy: full vocabulary per action; inclusion: sampled action positions; EOS included',
        'interpretation': 'fixed historical student rollouts, teacher-only diagnostic; action observations within trajectories are dependent'})
    progress = {'status': 'incomplete', 'completed_trajectories': 0, 'total_trajectories': len(rows)}
    write_json_atomic(output / 'progress.json', progress)
    tokenizer = AutoTokenizer.from_pretrained(config['teacher_id'], revision=config['teacher_revision'])
    for row in rows:
        validate_trajectory(row, tokenizer.eos_token_id)
    base = load_base(config['teacher_id'], config['teacher_revision'])
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    teacher = PeftModel.from_pretrained(base, teacher_run / 'adapter', is_trainable=False)
    del base
    teacher.eval()
    teacher.requires_grad_(False)
    completed = []
    for row in rows:
        positions = validate_trajectory(row, tokenizer.eos_token_id)
        sequence = torch.tensor([row['prompt_token_ids'] + row['action_token_ids']], device='cuda:0', dtype=torch.long)
        with torch.no_grad():
            logits = teacher(input_ids=sequence, use_cache=False, logits_to_keep=torch.tensor(positions, device=sequence.device)).logits[0]
            stats = teacher_statistics(logits, sequence[0, len(row['prompt_token_ids']):], KS)
        saved = torch.tensor(row['teacher_action_log_probs'], dtype=torch.float32)
        max_delta = check_replay_log_probs(stats['action_log_probability'], saved, REPLAY_LOGP_ATOL)
        item = {'id': row['id'], 'step': row['step'], 'prompt_sha256': row['prompt_sha256'],
                'action_count': len(positions), 'reached_eos': row['reached_eos'],
                'entropy_nats_mean': stats['entropy'].mean().item(),
                'saved_teacher_action_logp_max_abs_difference': max_delta, 'per_k': {}, 'actions': []}
        for k in KS:
            s = stats['per_k'][k]
            item['per_k'][str(k)] = {'mass_mean': s['mass'].mean().item(), 'tail_mean': s['tail'].mean().item(),
                                      'included_count': int(s['included'].sum())}
        for i, (token_id, position) in enumerate(zip(row['action_token_ids'], positions)):
            item['actions'].append({'index': i, 'token_id': token_id, 'logit_position': position,
                'entropy_nats': stats['entropy'][i].item(), 'teacher_rank': int(stats['action_rank'][i]),
                'teacher_action_log_probability': stats['action_log_probability'][i].item(),
                'per_k': {str(k): {'mass': stats['per_k'][k]['mass'][i].item(),
                    'tail': stats['per_k'][k]['tail'][i].item(),
                    'included': bool(stats['per_k'][k]['included'][i])} for k in KS}})
        completed.append(item)
        write_json_atomic(output / 'trajectories.json', completed)
        progress.update(completed_trajectories=len(completed), last_step=row['step'], elapsed_seconds=time.monotonic()-started)
        write_json_atomic(output / 'progress.json', progress)
        del logits, sequence, stats
    if ({name: sha256(source / name) for name in source_files} != hashes or
        {name: sha256(teacher_run / name) for name in teacher_hashes} != teacher_hashes or
        hash_tree(teacher_run / 'adapter') != adapter_hashes):
        raise RuntimeError('Source or teacher artifact changed during replay')
    write_json_atomic(output / 'summary.json', aggregate(completed))
    progress.update(status='complete', elapsed_seconds=time.monotonic()-started)
    write_json_atomic(output / 'progress.json', progress)
    write_json_atomic(output / 'status.json', {'status': 'complete'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'teacher-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.run, args.teacher_run, args.output)
    except Exception as exc:
        if (args.output / 'progress.json').exists():
            write_json_atomic(args.output / 'status.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
