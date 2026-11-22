"""Run a small, reproducible single-turn Qwen tool-calling LoRA baseline."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from src.baseline_data import prepare_splits, read_rows, score_output


def encode_sample(tokenizer, sample, max_length):
    """Encode a conversation with loss confined to the assistant response."""
    messages = [{"role": "user", "content": sample["query"]}]
    kwargs = {"tools": sample["tools"], "tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    calls = [{"type": "function", "function": call} for call in sample["calls"]]
    assistant = {"role": "assistant", "content": "", "tool_calls": calls}
    complete = tokenizer.apply_chat_template(messages + [assistant], **kwargs)
    if complete[:len(prompt)] != prompt:
        raise ValueError("Assistant response does not start with the generation prompt")
    if len(complete) <= len(prompt):
        raise ValueError("Assistant response contains no tokens")
    if tokenizer.eos_token_id not in complete[len(prompt):]:
        raise ValueError("Assistant response is missing the end-of-turn token")
    if len(complete) > max_length:
        return None
    return {"input_ids": complete, "labels": [-100] * len(prompt) + complete[len(prompt):]}


def tensor_batch(encoded, device):
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def prepare_encoded(tokenizer, splits, max_length):
    encoded = {}
    dropped = {}
    for name, samples in splits.items():
        encoded[name] = []
        dropped[name] = []
        for sample in samples:
            item = encode_sample(tokenizer, sample, max_length)
            if item is None:
                dropped[name].append(sample["id"])
            else:
                encoded[name].append((sample, item))
        if not encoded[name]:
            raise ValueError(f"No {name} examples fit within max_length")
    return encoded, dropped


def git_state():
    root = Path(__file__).resolve().parent.parent
    try:
        args = ["git", "-c", f"safe.directory={root.as_posix()}"]
        commit = subprocess.check_output(args + ["rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(args + ["status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(config):
    model_args = {
        "revision": config["model_revision"],
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    if config["quantization"] == "nf4":
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif config["quantization"] != "none":
        raise ValueError("quantization must be nf4 or none")
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], **model_args)
    model.config.use_cache = False
    if config["quantization"] == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["gradient_checkpointing"])
    elif config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return get_peft_model(model, LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["lora_targets"],
        bias="none", task_type="CAUSAL_LM",
    ))


def validate_loss(model, samples, limit, device):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for _, encoded in samples[:limit]:
            count = sum(label != -100 for label in encoded["labels"])
            loss = model(**tensor_batch(encoded, device)).loss.item()
            if not math.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            total += loss * count
            tokens += count
    return {"loss": total / tokens, "samples": min(limit, len(samples)), "target_tokens": tokens}


def run(config, data_path, output):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires a CUDA GPU supporting BF16")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting an earlier run")
    for name in ("max_steps", "gradient_accumulation_steps", "max_validation_samples", "max_eval_samples", "max_new_tokens", "max_length"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "incomplete"})
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    write_json(output / "config.json", config)
    provenance = {
        "git": git_state(), "dataset_path": str(data_path.resolve()),
        "dataset_sha256": dataset_hash, "dataset_source": config["dataset_source"],
        "dataset_revision": config["dataset_revision"], "python": sys.version,
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_runtime": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"]},
    }
    write_json(output / "provenance.json", provenance)
    rows = read_rows(data_path)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != dataset_hash:
        raise RuntimeError("Dataset changed while it was being read")
    splits, manifest = prepare_splits(rows, seed=config["seed"],
                                      validation_fraction=config["validation_fraction"],
                                      test_fraction=config["test_fraction"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    encoded, dropped = prepare_encoded(tokenizer, splits, config["max_length"])
    manifest["kept_ids"] = {name: [sample["id"] for sample, _ in rows] for name, rows in encoded.items()}
    manifest["overlength_ids"] = dropped
    write_json(output / "split_manifest.json", manifest)
    model = load_model(config)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    rng = random.Random(config["seed"])
    order = list(range(len(encoded["train"])))
    rng.shuffle(order)
    position = 0
    start = time.monotonic()
    model.train()
    with (output / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, config["max_steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config["gradient_accumulation_steps"]):
                if position == len(order):
                    rng.shuffle(order)
                    position = 0
                item = encoded["train"][order[position]][1]
                position += 1
                loss = model(**tensor_batch(item, device)).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                losses.append(loss.item())
                (loss / config["gradient_accumulation_steps"]).backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(grad).all() for grad in grads):
                raise ValueError("Missing or non-finite gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "loss": sum(losses) / len(losses), "gradient_norm": grad_norm.item(),
                   "elapsed_seconds": time.monotonic() - start}
            log.write(json.dumps(row) + "\n")
            log.flush()
    validation = validate_loss(model, encoded["validation"], config["max_validation_samples"], device)
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    # Reload the exact saved adapter against the original base before evaluation.
    model = model.unload()
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    rows = encoded["test"][:config["max_eval_samples"]]
    scores = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for sample, _ in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["query"]}], tools=sample["tools"],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                           do_sample=False, max_new_tokens=config["max_new_tokens"],
                                           pad_token_id=tokenizer.eos_token_id)
            generated_tokens = generated[0, prompt.shape[1]:]
            reached_eos = bool(len(generated_tokens) and generated_tokens[-1].item() == tokenizer.eos_token_id)
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            score = score_output(text, sample)
            scores.append(score)
            predictions.write(json.dumps({"id": sample["id"], "prediction": text, "target": sample["calls"],
                                          "generated_tokens": len(generated_tokens), "reached_eos": reached_eos,
                                          "token_limit_reached": len(generated_tokens) == config["max_new_tokens"] and not reached_eos,
                                          "scores": score}, ensure_ascii=False) + "\n")
    metrics = {"count": len(scores), "validation": validation,
               "metrics": {key: {"correct": sum(int(s[key]) for s in scores), "denominator": len(scores),
                                  "rate": sum(int(s[key]) for s in scores) / len(scores)} for key in scores[0]},
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "elapsed_seconds": time.monotonic() - start}
    write_json(output / "metrics.json", metrics)
    write_json(output / "status.json", {"status": "complete"})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
