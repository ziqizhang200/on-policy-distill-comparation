"""Position-aligned student-to-teacher KL for a single unpadded rollout."""
import torch
import torch.nn.functional as F


def action_logits(logits, prompt_length, action_count):
    """Return [T,V] logits predicting each generated token, including EOS."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1,L,V]")
    if prompt_length < 1 or action_count < 1 or prompt_length + action_count > logits.shape[1]:
        raise ValueError("invalid prompt or action length")
    return logits[0, prompt_length - 1:prompt_length + action_count - 1]


def student_teacher_kl(student_logits, teacher_logits):
    """Mean per-action KL(student || teacher), computing probabilities in FP32."""
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape or not student_logits.numel():
        raise ValueError("student and teacher logits must have equal nonempty [T,V] shape")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("non-finite logits")
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    loss = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("non-finite KL")
    return loss
