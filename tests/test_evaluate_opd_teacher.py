import json

import pytest

from scripts.evaluate_opd_teacher import match_locked_samples, validate_completed_run
from scripts.run_opd_pilot import hash_tree, sha256
from src.baseline_data import score_output


def test_teacher_uses_only_locked_cohort_and_checks_artifacts(tmp_path):
    run = tmp_path / 'paired'; run.mkdir()
    teacher = tmp_path / 'teacher'; teacher.mkdir()
    (teacher / 'adapter').mkdir()
    (teacher / 'adapter' / 'weights').write_bytes(b'weights')
    data = tmp_path / 'data.json'; data.write_text('[]', encoding='utf8')
    cfg = {'student_id': 'student', 'student_revision': 's', 'teacher_id': 'teacher', 'teacher_revision': 't',
           'dataset_source': 'data', 'dataset_revision': 'v', 'seed': 42, 'validation_fraction': .1,
           'test_fraction': .1, 'eval_split': 'test', 'eval_samples': 2, 'match_supervised_tokens': True}
    teacher_cfg = {'model_id': 'teacher', 'model_revision': 't', 'dataset_source': 'data', 'dataset_revision': 'v',
                   'seed': 42, 'validation_fraction': .1, 'test_fraction': .1}
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
    write(teacher / 'config.json', teacher_cfg)
    evidence_names = ('status.json', 'config.json', 'provenance.json', 'metrics.json', 'train_history.json',
                      'split_manifest.json', 'base_predictions.json', 'predictions.json')
    for name in evidence_names:
        if name != 'config.json':
            write(teacher / name, {})
    write(run / 'config.json', cfg)
    write(run / 'status.json', {'status': 'complete'})
    write(run / 'selection.json', {'eval_split': 'test', 'eval_ids': [12, 10]})
    write(run / 'provenance.json', {'dataset_sha256': sha256(data), 'git': {'commit': 'abc', 'dirty': False},
          'teacher_evidence_sha256': {name: sha256(teacher / name) for name in evidence_names},
          'teacher_adapter_files_sha256': hash_tree(teacher / 'adapter')})
    calls = [{'name': 'f', 'arguments': {}}]
    text = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
    predictions = [{'id': i, 'target': calls, 'prediction': text, 'scores': score_output(text, {'calls': calls})} for i in (12, 10)]
    for name in ('starting_predictions.json', 'opd/predictions.json', 'continued_sft/predictions.json'):
        write(run / name, predictions)
    got_cfg, ids, records = validate_completed_run(run, data, teacher)
    assert got_cfg == cfg and ids == [12, 10]
    selected = match_locked_samples([{'id': 10, 'calls': calls}, {'id': 11, 'calls': []}, {'id': 12, 'calls': calls}], ids, records)
    assert [s['id'] for s in selected] == [12, 10]
    with pytest.raises(ValueError, match='do not belong'):
        match_locked_samples([{'id': 10, 'calls': calls}], ids, records)
    with pytest.raises(ValueError, match='targets differ'):
        match_locked_samples([{'id': 10, 'calls': []}, {'id': 12, 'calls': calls}], ids, records)
    write(run / 'status.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='not complete'):
        validate_completed_run(run, data, teacher)
    write(run / 'status.json', {'status': 'complete'})
    (teacher / 'adapter' / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='adapter hash differs'):
        validate_completed_run(run, data, teacher)
