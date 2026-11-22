import json
import math

import pytest
import torch

from src.opd import action_logits, student_teacher_kl
from scripts.run_opd_pilot import validate_teacher_run
from scripts.run_sft_baseline import aggregate_scores, paired_joint_counts


def test_action_positions_include_eos_and_causal_boundary():
    x = torch.arange(1 * 7 * 3).reshape(1, 7, 3)
    assert torch.equal(action_logits(x, 4, 3), x[0, 3:6])
    with pytest.raises(ValueError):
        action_logits(x, 0, 3)
    with pytest.raises(ValueError):
        action_logits(x, 4, 4)
    with pytest.raises(ValueError):
        action_logits(x, 4, 0)


def test_reverse_kl_fp32_gradient_and_teacher_isolation():
    student = torch.tensor([[2., -1.], [-0.5, 1.]], requires_grad=True)
    teacher = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    loss = student_teacher_kl(student, teacher)
    p = student.softmax(-1)
    expected = (p * (student.log_softmax(-1) - teacher.log_softmax(-1))).sum(-1).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
    assert teacher.grad is None
    # The final row represents EOS and must contribute to the gradient.
    assert student.grad[-1].abs().sum() > 0
    assert loss.dtype == torch.float32


def test_kl_rejects_nonfinite_and_mismatched():
    with pytest.raises(ValueError):
        student_teacher_kl(torch.tensor([[float('nan')]]), torch.zeros(1, 1))
    with pytest.raises(ValueError):
        student_teacher_kl(torch.zeros(1, 2), torch.zeros(2, 2))


def test_teacher_gate_rejects_regression_and_incomplete(tmp_path):
    cfg = {'teacher_id': 'teacher', 'teacher_revision': 'rev', 'dataset_source': 'data',
           'dataset_revision': 'v', 'seed': 42, 'validation_fraction': .1, 'test_fraction': .1}
    teacher_cfg = dict(model_id='teacher', model_revision='rev', dataset_source='data', dataset_revision='v',
                       seed=42, validation_fraction=.1, test_fraction=.1, quantization='nf4', max_steps=64,
                       gradient_accumulation_steps=4, generation_eval_split='validation', evaluate_base_model=True)
    def write(name, obj):
        (tmp_path / name).write_text(json.dumps(obj), encoding='utf-8')
    before = [{'id': i, 'target': [], 'scores': {'joint_match': True}} for i in range(32)]
    after = [{'id': i, 'target': [], 'scores': {'joint_match': i != 0}} for i in range(32)]
    write('status.json', {'status': 'complete'})
    write('config.json', teacher_cfg)
    write('provenance.json', {'dataset_sha256': 'hash', 'git': {'commit': 'abc', 'dirty': False}})
    write('train_history.json', [{'step': i, 'loss': 1., 'gradient_norm': 1.} for i in range(1, 65)])
    write('split_manifest.json', {'tokenization': {'validation': {'kept_ids': list(range(32))}}})
    write('base_predictions.json', before)
    write('predictions.json', after)
    write('metrics.json', {'adapter_reloaded': True, 'validation': {'loss': 1.},
                           'base_metrics': aggregate_scores(before), 'metrics': aggregate_scores(after),
                           'paired_joint': paired_joint_counts(before, after)})
    with pytest.raises(ValueError, match='declined'):
        validate_teacher_run(tmp_path, cfg, 'hash')
    write('status.json', {'status': 'incomplete'})
    with pytest.raises(ValueError, match='incomplete'):
        validate_teacher_run(tmp_path, cfg, 'hash')


def test_fixed_shape_causal_smoke_detects_future_leak():
    from types import SimpleNamespace
    from scripts.run_opd_pilot import smoke_prefix
    class Model:
        def __init__(self, leak=False):
            self.leak = leak
        def __call__(self, input_ids, use_cache, logits_to_keep):
            x = input_ids.float()
            if self.leak:
                x = x + x.sum(dim=1, keepdim=True)
            else:
                x = x.cumsum(dim=1)
            logits = torch.stack((x, -x, x * .25, x * .5, x * .75, x * .1, x * .2, x * .3), dim=-1)
            return SimpleNamespace(logits=logits[:, -logits_to_keep:] if isinstance(logits_to_keep, int) else logits[:, logits_to_keep])
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    smoke = smoke_prefix(Model(), ids, 3, 2)
    assert [item['logit_position'] for item in smoke['checks']] == [2, 3]
    assert all(item['fixed_shape_causal_max_absolute_logit_difference'] == 0 for item in smoke['checks'])
    with pytest.raises(ValueError, match='causality'):
        smoke_prefix(Model(leak=True), ids, 3, 2)
