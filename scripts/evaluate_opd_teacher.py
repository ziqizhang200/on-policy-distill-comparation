"""Evaluate a frozen teacher on an already completed paired run's exact cohort.

No generated call is executed. This entry point must not be run until the cohort
and comparison are approved; tests exercise validation without loading models.
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
from transformers import AutoTokenizer, GenerationConfig, set_seed

from scripts.run_opd_pilot import check_tokenizers, hash_tree, load_base, sha256
from scripts.run_sft_baseline import aggregate_scores, evaluate_generation, git_state, paired_joint_counts
from src.baseline_data import prepare_splits, read_rows, score_output
from src.json_io import write_json_atomic

PREDICTION_FILES = {'starting': 'starting_predictions.json', 'opd': 'opd/predictions.json',
                    'continued_sft': 'continued_sft/predictions.json'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate_completed_run(run_dir, data_path, teacher_dir):
    """Validate prior artifacts and teacher identity without selecting new IDs."""
    config = read_json(run_dir / 'config.json')
    selection = read_json(run_dir / 'selection.json')
    provenance = read_json(run_dir / 'provenance.json')
    if read_json(run_dir / 'status.json').get('status') != 'complete':
        raise ValueError('Paired run is not complete')
    if not config.get('match_supervised_tokens') or config.get('eval_split') != 'test':
        raise ValueError('Expected matched-token test run')
    if selection.get('eval_split') != 'test':
        raise ValueError('Selection is not a test cohort')
    ids = selection.get('eval_ids')
    if not isinstance(ids, list) or not ids or len(ids) != config.get('eval_samples') or len(set(ids)) != len(ids):
        raise ValueError('Invalid locked cohort IDs')
    if sha256(data_path) != provenance.get('dataset_sha256'):
        raise ValueError('Dataset hash differs')
    if provenance.get('git', {}).get('dirty') is not False or not provenance.get('git', {}).get('commit'):
        raise ValueError('Paired run lacks clean code provenance')
    teacher_cfg = read_json(teacher_dir / 'config.json')
    for run_key, teacher_key in [('teacher_id', 'model_id'), ('teacher_revision', 'model_revision'),
                                 ('dataset_source', 'dataset_source'), ('dataset_revision', 'dataset_revision'),
                                 ('seed', 'seed'), ('validation_fraction', 'validation_fraction'), ('test_fraction', 'test_fraction')]:
        if config.get(run_key) != teacher_cfg.get(teacher_key):
            raise ValueError('Teacher configuration differs from paired run')
    evidence = provenance.get('teacher_evidence_sha256', {})
    if not {'status.json', 'config.json', 'provenance.json', 'metrics.json', 'train_history.json',
            'split_manifest.json', 'base_predictions.json', 'predictions.json'}.issubset(evidence):
        raise ValueError('Teacher evidence is incomplete')
    for name, digest in evidence.items():
        if sha256(teacher_dir / name) != digest:
            raise ValueError('Teacher evidence hash differs')
    if not provenance.get('teacher_evidence_sha256') or hash_tree(teacher_dir / 'adapter') != provenance.get('teacher_adapter_files_sha256'):
        raise ValueError('Teacher evidence or adapter hash differs')
    records = {key: read_json(run_dir / name) for key, name in PREDICTION_FILES.items()}
    reference = None
    for predictions in records.values():
        if [p['id'] for p in predictions] != ids:
            raise ValueError('Predictions are not aligned to locked IDs')
        targets = [p['target'] for p in predictions]
        if reference is None:
            reference = targets
        if targets != reference:
            raise ValueError('Paired target calls differ')
        for prediction in predictions:
            if score_output(prediction['prediction'], {'calls': prediction['target']}) != prediction['scores']:
                raise ValueError('Stored prediction scores do not reproduce')
    return config, ids, records


def match_locked_samples(test_samples, ids, records):
    by_id = {sample['id']: sample for sample in test_samples}
    if len(by_id) != len(test_samples) or not set(ids).issubset(by_id):
        raise ValueError('Locked IDs do not belong uniquely to reconstructed test split')
    selected = [by_id[i] for i in ids]
    if [s['calls'] for s in selected] != [p['target'] for p in records['starting']]:
        raise ValueError('Reconstructed targets differ from paired predictions')
    return selected


def run(run_dir, data_path, teacher_dir, output, seed):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    code = git_state()
    if code.get('dirty') is not False:
        raise ValueError('Teacher evaluation requires clean committed code')
    config, ids, comparisons = validate_completed_run(run_dir, data_path, teacher_dir)
    data_hash = sha256(data_path)
    rows = read_rows(data_path)
    if sha256(data_path) != data_hash:
        raise ValueError('Dataset changed during read')
    splits, _ = prepare_splits(rows, seed=config['seed'], validation_fraction=config['validation_fraction'], test_fraction=config['test_fraction'])
    samples = match_locked_samples(splits['test'], ids, comparisons)
    student_tok = AutoTokenizer.from_pretrained(config['student_id'], revision=config['student_revision'])
    teacher_tok = AutoTokenizer.from_pretrained(config['teacher_id'], revision=config['teacher_revision'])
    check_tokenizers(student_tok, teacher_tok)
    student_gen = GenerationConfig.from_pretrained(config['student_id'], revision=config['student_revision']).to_dict()
    teacher_gen = GenerationConfig.from_pretrained(config['teacher_id'], revision=config['teacher_revision']).to_dict()
    # Model-specific default generation settings must not silently change the comparison.
    if student_gen != teacher_gen:
        raise ValueError('Teacher and student generation defaults differ')
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / 'status.json', {'status': 'incomplete'})
    write_json_atomic(output / 'progress.json', {'status': 'incomplete', 'stage': 'loading', 'completed': 0, 'total': len(ids)})
    hashes = {name: sha256(run_dir / name) for name in ('config.json', 'selection.json', 'provenance.json', 'status.json', *PREDICTION_FILES.values())}
    write_json_atomic(output / 'provenance.json', {'git': code, 'source_run': str(run_dir.resolve()), 'source_sha256': hashes,
        'dataset_sha256': data_hash, 'teacher_run': str(teacher_dir.resolve()), 'teacher_adapter_sha256': hash_tree(teacher_dir / 'adapter'),
        'generation_defaults': teacher_gen, 'generation_overrides': {'do_sample': False, 'max_new_tokens': config['max_new_tokens'], 'pad_token_id': teacher_tok.eos_token_id},
        'seed': seed, 'python': sys.version, 'platform': platform.platform(), 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'packages': {n: importlib.metadata.version(n) for n in ('torch','transformers','peft','accelerate','bitsandbytes')}})
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = load_base(config['teacher_id'], config['teacher_revision'])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, teacher_dir / 'adapter', is_trainable=False)
    if model.generation_config.to_dict() != teacher_gen:
        raise ValueError('Loaded teacher generation settings differ from pinned defaults')
    model.eval(); model.requires_grad_(False); set_seed(seed)
    def progress(completed, total):
        write_json_atomic(output / 'progress.json', {'status': 'incomplete', 'stage': 'evaluation',
            'completed': completed, 'total': total, 'elapsed_seconds': time.monotonic() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    predictions = evaluate_generation(model, teacher_tok, [(sample, None) for sample in samples], config,
                                      torch.device('cuda:0'), output / 'predictions.jsonl', progress)
    if [p['id'] for p in predictions] != ids:
        raise ValueError('Teacher predictions differ from locked IDs')
    paired = {key: paired_joint_counts(value, predictions) for key, value in comparisons.items()}
    write_json_atomic(output / 'predictions.json', predictions)
    write_json_atomic(output / 'metrics.json', {'count': len(ids), 'metrics': aggregate_scores(predictions), 'paired_teacher_vs': paired,
        'elapsed_seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    write_json_atomic(output / 'progress.json', {'status': 'complete', 'stage': 'complete', 'completed': len(ids), 'total': len(ids)})
    write_json_atomic(output / 'status.json', {'status': 'complete'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'teacher-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be non-negative')
    try:
        run(args.run, args.data, args.teacher_run, args.output, args.seed)
    except Exception as exc:
        if (args.output / 'status.json').exists():
            write_json_atomic(args.output / 'progress.json', {'status': 'failed', 'stage': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            write_json_atomic(args.output / 'status.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
