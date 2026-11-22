"""Run a small paired on-policy distillation and continued-SFT pilot.

Generated tool calls are recorded as inert text and are never executed.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from src.baseline_data import prepare_splits, read_rows, score_output
from src.opd import action_logits, student_teacher_kl
from src.json_io import write_json_atomic
from scripts.run_sft_baseline import (aggregate_scores, encode_sample, evaluate_generation,
                                      git_state, paired_joint_counts, tensor_batch)

write_json = write_json_atomic


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path):
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}


def prompt_ids(tokenizer, sample):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': sample['query']}],
        tools=sample['tools'], tokenize=True, add_generation_prompt=True, enable_thinking=False)


def check_tokenizers(student, teacher):
    if (student.get_vocab() != teacher.get_vocab() or
        student.special_tokens_map != teacher.special_tokens_map or
        student.chat_template != teacher.chat_template or
        student.eos_token_id != teacher.eos_token_id or len(student) != len(teacher)):
        raise ValueError('Student and teacher tokenizer vocabulary, special tokens or template differ')


def load_base(model_id, revision):
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16,
        device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16))


def load_student(config, adapter, trainable):
    base = load_base(config['student_id'], config['student_revision'])
    # Match the original SFT preparation (including FP32 norm casts) on reload.
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    model.config.use_cache = False
    model.eval()  # LoRA dropout disabled in both branches, including during gradients.
    return model


def release(*models):
    # Caller removes its own references before invoking this function.
    gc.collect()
    torch.cuda.empty_cache()


def selected_samples(config, tokenizer, splits, manifest, teacher_manifest):
    rng = random.Random(config['seed'])
    candidates = list(splits['train'])
    rng.shuffle(candidates)
    train, rejected = [], []
    used = (set(manifest['tokenization']['train']['kept_ids']) |
            set(teacher_manifest['tokenization']['train']['kept_ids']))
    for sample in candidates:
        if sample['id'] in used:
            continue
        encoded = encode_sample(tokenizer, sample, config['max_length'])
        prompt = prompt_ids(tokenizer, sample)
        if encoded is None or len(prompt) > config['max_prompt_tokens']:
            rejected.append(sample['id'])
            continue
        train.append((sample, encoded))
        if len(train) == config['max_steps']:
            break
    if len(train) != config['max_steps']:
        raise ValueError('Insufficient training examples within common length filters')
    baseline_excluded = set(manifest['tokenization']['validation']['kept_ids'])
    teacher_excluded = set(teacher_manifest['tokenization']['validation']['kept_ids'])
    candidates = list(splits['validation'])
    random.Random(config['seed'] + 3).shuffle(candidates)
    test, eval_rejected = [], []
    for sample in candidates:
        if sample['id'] in baseline_excluded or sample['id'] in teacher_excluded:
            continue
        if encode_sample(tokenizer, sample, config['max_length']) is None:
            eval_rejected.append(sample['id'])
            continue
        test.append((sample, None))
        if len(test) == config['eval_samples']:
            break
    if len(test) != config['eval_samples']:
        raise ValueError('Insufficient independent validation samples')
    return train, test, rejected, eval_rejected


def validate_teacher_run(path, config, dataset_hash):
    """Fail closed unless the paired 4B validation gate has passed."""
    def read(name):
        return json.loads((path / name).read_text(encoding='utf-8-sig'))
    status, teacher_config, provenance = read('status.json'), read('config.json'), read('provenance.json')
    metrics, history = read('metrics.json'), read('train_history.json')
    if status.get('status') != 'complete':
        raise ValueError('Teacher run is incomplete')
    expected = {'model_id': config['teacher_id'], 'model_revision': config['teacher_revision'],
                'dataset_source': config['dataset_source'], 'dataset_revision': config['dataset_revision'],
                'seed': config['seed'], 'validation_fraction': config['validation_fraction'],
                'test_fraction': config['test_fraction'], 'quantization': 'nf4', 'max_steps': 64,
                'gradient_accumulation_steps': 4, 'generation_eval_split': 'validation',
                'evaluate_base_model': True}
    if any(teacher_config.get(k) != v for k, v in expected.items()):
        raise ValueError('Teacher run configuration does not match required validated 4B run')
    if provenance.get('dataset_sha256') != dataset_hash:
        raise ValueError('Teacher dataset hash differs')
    if provenance.get('git', {}).get('dirty') is not False or not provenance.get('git', {}).get('commit'):
        raise ValueError('Teacher provenance does not identify a clean Git commit')
    if len(history) != 64 or any(row.get('step') != i or not all(
            isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in ('loss', 'gradient_norm'))
            for i, row in enumerate(history, 1)):
        raise ValueError('Teacher training history is incomplete or non-finite')
    if not metrics.get('adapter_reloaded'):
        raise ValueError('Teacher adapter reload is not verified')
    validation = metrics.get('validation', {})
    if not isinstance(validation.get('loss'), (float, int)) or not math.isfinite(validation['loss']):
        raise ValueError('Teacher validation loss is missing or non-finite')
    before, after = read('base_predictions.json'), read('predictions.json')
    ids = read('split_manifest.json')['tokenization']['validation']['kept_ids']
    if len(ids) != 32 or [x['id'] for x in before] != ids or [x['id'] for x in after] != ids:
        raise ValueError('Teacher validation IDs are not aligned')
    paired = paired_joint_counts(before, after)
    if paired != metrics.get('paired_joint') or aggregate_scores(before) != metrics.get('base_metrics') or aggregate_scores(after) != metrics.get('metrics'):
        raise ValueError('Teacher validation metrics do not match predictions')
    if paired['regressed'] > paired['improved']:
        raise ValueError('Teacher failed validation gate: joint match declined')
    if not (path / 'adapter' / 'adapter_config.json').is_file() or not (path / 'adapter' / 'adapter_model.safetensors').is_file():
        raise ValueError('Teacher adapter is missing')
    return {'validation_count': len(ids), 'paired_joint': paired}


def smoke_prefix(model, ids, prompt_length, action_count):
    """Compare selected full-sequence predictions with causal prefix forwards."""
    checks = sorted(set((0, action_count - 1)))
    positions = torch.tensor([prompt_length - 1 + i for i in checks], device=ids.device)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0]
        max_diff = 0.0
        for j, i in enumerate(checks):
            prefix = model(input_ids=ids[:, :prompt_length + i], use_cache=False, logits_to_keep=1).logits[0, -1]
            delta = (full[j].float() - prefix.float()).abs().max().item()
            max_diff = max(max_diff, delta)
            if not torch.allclose(full[j].float(), prefix.float(), rtol=0.03, atol=0.15):
                raise ValueError(f'Prefix alignment smoke failed at action {i}, max difference {delta}')
        return {'checked_action_indices': checks, 'max_absolute_logit_difference': max_diff,
                'rtol': 0.03, 'atol': 0.15}


def run(config, data_path, adapter, baseline, teacher_run, output):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    for key in ('max_steps', 'eval_samples', 'max_length', 'max_prompt_tokens', 'rollout_max_new_tokens', 'max_new_tokens'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['quantization'] != 'nf4' or config['learning_rate'] <= 0:
        raise ValueError('Pilot requires NF4 and a positive learning rate')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'status.json', {'status': 'incomplete'})
    write_json(output / 'config.json', config)
    progress = {'status': 'incomplete', 'stage': 'initializing', 'branch': None, 'step': 0}
    write_json(output / 'progress.json', progress)
    data_hash = sha256(data_path)
    teacher_gate = validate_teacher_run(teacher_run, config, data_hash)
    write_json(output / 'teacher_gate.json', teacher_gate)
    baseline_config = json.loads((baseline / 'config.json').read_text(encoding='utf-8-sig'))
    baseline_provenance = json.loads((baseline / 'provenance.json').read_text(encoding='utf-8-sig'))
    if json.loads((baseline / 'status.json').read_text(encoding='utf-8-sig'))['status'] != 'complete':
        raise ValueError('Baseline artifact is incomplete')
    for key, value in [('model_id', config['student_id']), ('model_revision', config['student_revision']),
                       ('dataset_source', config['dataset_source']), ('dataset_revision', config['dataset_revision']),
                       ('seed', config['seed']), ('validation_fraction', config['validation_fraction']),
                       ('test_fraction', config['test_fraction']), ('max_new_tokens', config['max_new_tokens'])]:
        if baseline_config[key] != value:
            raise ValueError(f'Baseline setting differs: {key}')
    if baseline_provenance['dataset_sha256'] != data_hash or hash_tree(adapter) != hash_tree(baseline / 'adapter'):
        raise ValueError('Dataset or starting adapter differs from baseline artifact')
    write_json(output / 'provenance.json', {'git': git_state(), 'dataset_path': str(data_path.resolve()),
        'dataset_sha256': data_hash, 'adapter_path': str(adapter.resolve()), 'adapter_files_sha256': hash_tree(adapter),
        'teacher_run_path': str(teacher_run.resolve()), 'teacher_evidence_sha256': {n: sha256(teacher_run / n) for n in ('status.json', 'config.json', 'provenance.json', 'metrics.json', 'train_history.json', 'split_manifest.json', 'base_predictions.json', 'predictions.json')}, 'teacher_adapter_files_sha256': hash_tree(teacher_run / 'adapter'),
        'baseline_path': str(baseline.resolve()), 'baseline_files_sha256': {n: sha256(baseline / n) for n in
            ('config.json', 'provenance.json', 'split_manifest.json', 'predictions.jsonl', 'status.json')},
        'argv': sys.orig_argv, 'python': sys.version, 'platform': platform.platform(),
        'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')}})
    set_seed(config['seed'])
    rows = read_rows(data_path)
    if sha256(data_path) != data_hash:
        raise RuntimeError('Dataset changed during read')
    splits, _ = prepare_splits(rows, seed=config['seed'], validation_fraction=config['validation_fraction'], test_fraction=config['test_fraction'])
    teacher_rows = {sample['id']: sample for sample in splits['validation']}
    for name in ('base_predictions.json', 'predictions.json'):
        for prediction in json.loads((teacher_run / name).read_text(encoding='utf-8-sig')):
            sample = teacher_rows.get(prediction['id'])
            if sample is None or prediction['target'] != sample['calls'] or prediction['scores'] != score_output(prediction['prediction'], sample):
                raise ValueError('Teacher validation predictions do not match reconstructed validation data')
    student_tok = AutoTokenizer.from_pretrained(config['student_id'], revision=config['student_revision'])
    teacher_tok = AutoTokenizer.from_pretrained(config['teacher_id'], revision=config['teacher_revision'])
    check_tokenizers(student_tok, teacher_tok)
    manifest = json.loads((baseline / 'split_manifest.json').read_text(encoding='utf-8-sig'))
    teacher_manifest = json.loads((teacher_run / 'split_manifest.json').read_text(encoding='utf-8-sig'))
    train, test, rejected, eval_rejected = selected_samples(config, student_tok, splits, manifest, teacher_manifest)
    for sample, _ in train + test:
        if prompt_ids(student_tok, sample) != prompt_ids(teacher_tok, sample):
            raise ValueError('Rendered student/teacher prompt IDs differ')
    write_json(output / 'selection.json', {'train_ids': [x[0]['id'] for x in train],
        'eval_split': 'validation', 'eval_ids': [x[0]['id'] for x in test],
        'student_prior_validation_excluded_ids': manifest['tokenization']['validation']['kept_ids'],
        'teacher_validation_excluded_ids': teacher_manifest['tokenization']['validation']['kept_ids'],
        'student_prior_train_excluded_ids': manifest['tokenization']['train']['kept_ids'],
        'teacher_train_excluded_ids': teacher_manifest['tokenization']['train']['kept_ids'],
        'rejected_train_ids': rejected, 'rejected_eval_ids': eval_rejected,
        'common_full_sample_max_length': config['max_length'],
        'common_prompt_max_tokens': config['max_prompt_tokens'], 'seed': config['seed']})
    device = torch.device('cuda:0')
    progress.update(stage='starting_student_evaluation', branch='starting_sft', step=0)
    write_json(output / 'progress.json', progress)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = load_student(config, adapter, False)
    prior = []
    write_json(output / 'starting_predictions.json', prior)
    for sample in test:
        prior.extend(evaluate_generation(model, student_tok, [sample], config, device, output / '_current_prediction.jsonl'))
        write_json(output / 'starting_predictions.json', prior)
        progress.update(step=len(prior), elapsed_seconds=time.monotonic()-start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'progress.json', progress)
    (output / '_current_prediction.jsonl').unlink()
    write_json(output / 'starting_metrics.json', {'metrics': aggregate_scores(prior), 'count': len(prior),
        'elapsed_seconds': time.monotonic()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    del model
    release()
    for branch in ('opd', 'continued_sft'):
        progress.update(stage='training', branch=branch, step=0)
        write_json(output / 'progress.json', progress)
        branch_dir = output / branch
        branch_dir.mkdir()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model = load_student(config, adapter, True)
        teacher = None
        if branch == 'opd':
            teacher_base = load_base(config['teacher_id'], config['teacher_revision'])
            teacher_base = prepare_model_for_kbit_training(teacher_base, use_gradient_checkpointing=False)
            teacher = PeftModel.from_pretrained(teacher_base, teacher_run / 'adapter', is_trainable=False)
            del teacher_base
            teacher.eval()
            teacher.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config['learning_rate'])
        target_tokens = 0
        trajectories = []
        write_json(branch_dir / 'trajectories.json', trajectories)
        with (branch_dir / 'trajectories.jsonl').open('w', encoding='utf-8') as log:
            for step, (sample, encoded) in enumerate(train, 1):
                optimizer.zero_grad(set_to_none=True)
                if branch == 'opd':
                    prompt = prompt_ids(student_tok, sample)
                    inputs = torch.tensor([prompt], dtype=torch.long, device=device)
                    model.config.use_cache = True
                    with torch.no_grad():
                        sequence = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                            max_new_tokens=config['rollout_max_new_tokens'], pad_token_id=student_tok.eos_token_id)
                    model.config.use_cache = False
                    actions = sequence[0, len(prompt):]
                    count = len(actions)
                    if not count:
                        raise ValueError('Empty rollout')
                    smoke = None
                    if step == 1:
                        smoke = {'student': smoke_prefix(model, sequence, len(prompt), count),
                                 'teacher': smoke_prefix(teacher, sequence, len(prompt), count)}
                    with torch.no_grad():
                        positions = torch.arange(len(prompt)-1, len(prompt)+count-1, device=device)
                        teacher_logits = teacher(input_ids=sequence, use_cache=False, logits_to_keep=positions).logits[0]
                    student_logits = model(input_ids=sequence, use_cache=False, logits_to_keep=positions).logits[0]
                    loss = student_teacher_kl(student_logits, teacher_logits)
                    student_action_lp = torch.log_softmax(student_logits.detach().float(), -1).gather(1, actions[:, None]).squeeze(1)
                    teacher_action_lp = torch.log_softmax(teacher_logits.float(), -1).gather(1, actions[:, None]).squeeze(1)
                    text = student_tok.decode(actions, skip_special_tokens=True)
                    reached_eos = actions[-1].item() == student_tok.eos_token_id
                    row = {'step': step, 'id': sample['id'], 'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                           'prompt_tokens': len(prompt), 'prompt_token_ids': prompt, 'action_token_ids': actions.tolist(),
                           'logit_positions': list(range(len(prompt)-1, len(prompt)+count-1)),
                           'student_action_log_probs': student_action_lp.tolist(), 'teacher_action_log_probs': teacher_action_lp.tolist(),
                           'rollout_text': text, 'parser_scores': score_output(text, sample),
                           'reached_eos': reached_eos, 'token_limit_reached': count == config['rollout_max_new_tokens'] and not reached_eos,
                           'rollout_max_new_tokens': config['rollout_max_new_tokens'], 'smoke': smoke}
                    target_tokens += count
                else:
                    loss = model(**tensor_batch(encoded, device)).loss
                    count = sum(x != -100 for x in encoded['labels'])
                    target_tokens += count
                    row = {'step': step, 'id': sample['id'], 'supervised_tokens': count}
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss')
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):
                    raise ValueError('Missing or non-finite student gradients')
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(grad_norm.item()) or grad_norm.item() == 0:
                    raise ValueError('Student gradient is zero or non-finite')
                if teacher is not None and any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
                    raise ValueError('Teacher is not isolated')
                optimizer.step()
                row.update(loss=loss.item(), gradient_norm=grad_norm.item(), elapsed_seconds=time.monotonic()-start)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                trajectories.append(row)
                write_json(branch_dir / 'trajectories.json', trajectories)
                progress.update(step=step, loss=row['loss'], gradient_norm=row['gradient_norm'],
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-start)
                write_json(output / 'progress.json', progress)
                if branch == 'opd':
                    del student_logits, teacher_logits, student_action_lp, teacher_action_lp, loss, sequence, actions
            del grads, grad_norm
            if branch == 'continued_sft':
                del loss
            model.save_pretrained(branch_dir / 'adapter')
        train_metrics = {'steps': config['max_steps'], 'target_tokens': target_tokens,
            'elapsed_seconds': time.monotonic()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'model_mode': 'eval with gradients',
            'optimizer': 'AdamW', 'learning_rate': config['learning_rate'], 'teacher_gradients': False if teacher is not None else None}
        write_json(branch_dir / 'training_metrics.json', train_metrics)
        del optimizer, model, teacher
        release()
    # Teacher and training models are gone; reload each saved adapter for evaluation.
    for branch in ('opd', 'continued_sft'):
        progress.update(stage='evaluation', branch=branch, step=0)
        write_json(output / 'progress.json', progress)
        branch_dir = output / branch
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model = load_student(config, branch_dir / 'adapter', False)
        predictions = []
        write_json(branch_dir / 'predictions.json', predictions)
        for sample in test:
            current = evaluate_generation(model, student_tok, [sample], config, device, branch_dir / '_current_prediction.jsonl')
            predictions.extend(current)
            write_json(branch_dir / 'predictions.json', predictions)
            progress.update(step=len(predictions), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-start)
            write_json(output / 'progress.json', progress)
        (branch_dir / '_current_prediction.jsonl').unlink()
        write_json(branch_dir / 'metrics.json', {'metrics': aggregate_scores(predictions),
            'paired_vs_starting_sft': paired_joint_counts(prior, predictions), 'count': len(predictions),
            'elapsed_seconds': time.monotonic()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        del model
        release()
    progress.update(status='complete', stage='complete', branch=None)
    write_json(output / 'progress.json', progress)
    write_json(output / 'status.json', {'status': 'complete'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'data', 'adapter', 'baseline', 'teacher-run', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text(encoding='utf-8-sig')), args.data, args.adapter, args.baseline, args.teacher_run, args.output)
    except Exception as exc:
        if (args.output / 'progress.json').exists():
            progress = json.loads((args.output / 'progress.json').read_text(encoding='utf-8-sig'))
            progress.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            write_json(args.output / 'progress.json', progress)
            write_json(args.output / 'status.json', {'status': 'failed'})
        raise


if __name__ == '__main__':
    main()
