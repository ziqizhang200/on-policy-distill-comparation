import hashlib
import json

import pytest
import torch

from src.teacher_tail import check_replay_log_probs, teacher_statistics, validate_trajectory


def test_matches_full_softmax_and_tail_monotonic():
    logits = torch.tensor([[2., 1., -2., 0., 0.5], [25., -20., -23., -26., -29.]], dtype=torch.float16)
    actions = torch.tensor([1, 4])
    s = teacher_statistics(logits, actions, (1, 2, 5))
    p = logits.float().softmax(-1)
    assert s['entropy'].dtype == torch.float32
    assert torch.allclose(s['entropy'], -(p * logits.float().log_softmax(-1)).sum(-1))
    for k in (1, 2, 5):
        expected = p.sort(descending=True).values
        assert torch.allclose(s['per_k'][k]['mass'], expected[:, :k].sum(-1))
        assert torch.allclose(s['per_k'][k]['tail'], expected[:, k:].sum(-1))
    assert torch.all(s['per_k'][1]['tail'] >= s['per_k'][2]['tail'])
    assert torch.all(s['per_k'][2]['tail'] >= s['per_k'][5]['tail'])
    assert s['per_k'][1]['tail'][1] > 0  # cancellation in 1 - mass would lose this


def test_ties_use_token_id_and_inclusion_matches_rank():
    s = teacher_statistics(torch.tensor([[2., 2., 2., 0.], [2., 2., 2., 0.]]), torch.tensor([0, 2]), (1, 2, 3))
    assert s['action_rank'].tolist() == [1, 3]
    assert s['per_k'][2]['included'].tolist() == [True, False]
    assert s['per_k'][3]['included'].tolist() == [True, True]


def test_validation_eos_and_shift():
    prompt = [11, 12]
    row = {'prompt_token_ids': prompt, 'action_token_ids': [4, 9], 'prompt_tokens': 2,
           'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
           'logit_positions': [1, 2], 'reached_eos': True}
    assert validate_trajectory(row, 9) == [1, 2]
    with pytest.raises(ValueError):
        validate_trajectory({**row, 'logit_positions': [2, 3]}, 9)
    with pytest.raises(ValueError):
        validate_trajectory({**row, 'action_token_ids': [9, 4], 'reached_eos': False}, 9)
    with pytest.raises(ValueError):
        validate_trajectory({**row, 'prompt_sha256': 'bad'}, 9)


def test_invalid_logits_and_actions():
    with pytest.raises(ValueError):
        teacher_statistics(torch.tensor([[float('nan'), 0.]]), torch.tensor([0]), (1,))
    with pytest.raises(ValueError):
        teacher_statistics(torch.zeros(1, 2), torch.tensor([2]), (1,))
    with pytest.raises(ValueError):
        teacher_statistics(torch.zeros(1, 2), torch.tensor([0]), (3,))


def test_replay_log_probability_check_fails_closed():
    saved = torch.tensor([-1., -2.])
    assert check_replay_log_probs(torch.tensor([-1.01, -2.]), saved, 0.03) == pytest.approx(0.01)
    with pytest.raises(ValueError, match='exceeds atol'):
        check_replay_log_probs(torch.tensor([-1., -2.031]), saved, 0.03)
    with pytest.raises(ValueError, match='Invalid'):
        check_replay_log_probs(torch.tensor([float('nan'), -2.]), saved, 0.03)
    with pytest.raises(ValueError, match='tolerance'):
        check_replay_log_probs(saved, saved, float('nan'))
