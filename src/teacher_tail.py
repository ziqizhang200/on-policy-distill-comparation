"""Teacher distribution diagnostics on fixed autoregressive action positions."""
import hashlib
import json

import torch


def check_replay_log_probs(actual, saved, atol):
    """Return maximum replay drift or fail if any sampled-action log probability drifts."""
    if (actual.ndim != 1 or not actual.numel() or actual.shape != saved.shape or
            not torch.isfinite(actual).all() or not torch.isfinite(saved).all()):
        raise ValueError('Invalid replay or saved teacher action log probabilities')
    if not isinstance(atol, (int, float)) or isinstance(atol, bool) or not 0 < atol < float('inf'):
        raise ValueError('Replay log probability tolerance must be positive and finite')
    delta = (actual.float() - saved.float()).abs().max().item()
    if delta > atol:
        raise ValueError(f'Teacher action log probability replay drift {delta:.9g} exceeds atol {atol:.9g}')
    return delta


def validate_trajectory(row, eos_token_id):
    """Validate saved IDs and return the exact causal logit positions, EOS included."""
    prompt, actions = row['prompt_token_ids'], row['action_token_ids']
    if not prompt or not actions or any(type(x) is not int or x < 0 for x in prompt + actions):
        raise ValueError('Invalid saved token IDs')
    if len(prompt) != row['prompt_tokens'] or hashlib.sha256(json.dumps(prompt).encode()).hexdigest() != row['prompt_sha256']:
        raise ValueError('Saved prompt length or hash differs')
    positions = list(range(len(prompt) - 1, len(prompt) + len(actions) - 1))
    if row['logit_positions'] != positions:
        raise ValueError('Saved causal positions differ')
    if row['reached_eos'] != (actions[-1] == eos_token_id) or eos_token_id in actions[:-1]:
        raise ValueError('Saved EOS indicator or position differs')
    return positions


def teacher_statistics(logits, actions, ks=(8, 32, 128, 512)):
    """FP32 full-vocabulary probabilities; rank ties use ascending token ID.

    logits: [actions, vocabulary]; actions: integer [actions]. Return CPU tensors.
    Mass denominator is full vocabulary; inclusion denominator is sampled actions.
    Tail is summed directly outside Top-K, avoiding cancellation in 1 - mass.
    """
    if logits.ndim != 2 or not logits.numel() or not torch.isfinite(logits).all():
        raise ValueError('Expected finite nonempty [T,V] logits')
    if actions.ndim != 1 or actions.shape[0] != logits.shape[0] or actions.dtype != torch.long:
        raise ValueError('Expected long action IDs [T]')
    vocab = logits.shape[1]
    if not ks or any(type(k) is not int or k < 1 or k > vocab for k in ks) or len(set(ks)) != len(ks):
        raise ValueError('Invalid K values')
    if (actions < 0).any() or (actions >= vocab).any():
        raise ValueError('Action ID outside vocabulary')
    x = logits.detach().float()
    logp = x.log_softmax(-1)
    probs = logp.exp()
    action_logits = x.gather(1, actions[:, None])
    ids = torch.arange(vocab, device=x.device)[None, :]
    ranks = 1 + (x > action_logits).sum(-1) + ((x == action_logits) & (ids < actions[:, None])).sum(-1)
    # Tied tokens share the same probability, so arbitrary topk tie membership
    # changes neither mass nor tail; inclusion uses the deterministic rank above.
    top_indices = x.topk(max(ks), dim=-1, sorted=True).indices
    result = {'entropy': (-(probs * logp).sum(-1)).cpu(), 'action_rank': ranks.cpu(),
              'action_log_probability': logp.gather(1, actions[:, None]).squeeze(1).cpu(), 'per_k': {}}
    for k in ks:
        selected = top_indices[:, :k]
        mass = probs.gather(1, selected).sum(-1)
        tail_probs = probs.clone()
        tail_probs.scatter_(1, selected, 0.)
        result['per_k'][k] = {'mass': mass.cpu(), 'tail': tail_probs.sum(-1).cpu(),
                              'included': (ranks <= k).cpu()}
    return result
