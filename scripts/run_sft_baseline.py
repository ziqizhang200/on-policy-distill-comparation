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
from datetime import datetime, timezone

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from src.baseline_data import prepare_splits, read_rows, score_output
from src.json_io import write_json_atomic


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


def prepare_encoded(tokenizer, splits, max_length, limits, seed):
    """Select a reproducible bounded set of usable examples from each split."""
    encoded = {}
    audit = {}
    for name, samples in splits.items():
        encoded[name] = []
        candidates = list(samples)
        offset = {"train": 0, "validation": 1, "test": 2}[name]
        random.Random(seed + offset).shuffle(candidates)
        overlength_ids = []
        examined = 0
        for sample in candidates:
            examined += 1
            item = encode_sample(tokenizer, sample, max_length)
            if item is None:
                overlength_ids.append(sample["id"])
            else:
                encoded[name].append((sample, item))
                if len(encoded[name]) == limits[name]:
                    break
        if not encoded[name]:
            raise ValueError(f"No {name} examples fit within max_length")
        audit[name] = {
            "normalized_split_total": len(samples),
            "inspected": examined,
            "kept_ids": [sample["id"] for sample, _ in encoded[name]],
            "overlength_ids": overlength_ids,
            "uninspected": len(samples) - examined,
        }
    return encoded, audit


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
    write_json_atomic(path, value)


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


def evaluate_generation(model, tokenizer, rows, config, device, output_path, progress=None):
    model.eval()
    model.config.use_cache = True
    records = []
    with output_path.open("w", encoding="utf-8") as predictions:
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
            record = {"id": sample["id"], "prediction": text, "target": sample["calls"],
                      "generated_tokens": len(generated_tokens), "reached_eos": reached_eos,
                      "token_limit_reached": len(generated_tokens) == config["max_new_tokens"] and not reached_eos,
                      "scores": score_output(text, sample)}
            records.append(record)
            predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.flush()
            if progress is not None:
                progress(len(records), len(rows))
    return records


def aggregate_scores(records):
    scores = [record["scores"] for record in records]
    return {key: {"correct": sum(int(score[key]) for score in scores), "denominator": len(scores),
                  "rate": sum(int(score[key]) for score in scores) / len(scores)} for key in scores[0]}


def paired_joint_counts(base_records, sft_records):
    if len(base_records) != len(sft_records):
        raise ValueError("Base and SFT predictions must have the same length")
    counts = {"improved": 0, "regressed": 0, "unchanged_correct": 0, "unchanged_incorrect": 0}
    for base, sft in zip(base_records, sft_records):
        if base["id"] != sft["id"] or base["target"] != sft["target"]:
            raise ValueError("Base and SFT predictions are not aligned")
        before = base["scores"]["joint_match"]
        after = sft["scores"]["joint_match"]
        if before and after:
            counts["unchanged_correct"] += 1
        elif before:
            counts["regressed"] += 1
        elif after:
            counts["improved"] += 1
        else:
            counts["unchanged_incorrect"] += 1
    counts["unchanged"] = counts["unchanged_correct"] + counts["unchanged_incorrect"]
    counts["denominator"] = len(base_records)
    return counts


def effective_config(config):
    config = dict(config)
    config.setdefault("evaluate_base_model", False)
    if not isinstance(config["evaluate_base_model"], bool):
        raise ValueError("evaluate_base_model must be a boolean")
    config.setdefault("generation_eval_split", "test")
    if config["generation_eval_split"] not in ("validation", "test"):
        raise ValueError("generation_eval_split must be validation or test")
    if config.get("model_id") == "Qwen/Qwen3-4B" and config["generation_eval_split"] == "validation" and not config["evaluate_base_model"]:
        raise ValueError("4B teacher validation requires evaluate_base_model=true")
    return config


def _run(config, data_path, output):
    config = effective_config(config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires a CUDA GPU supporting BF16")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting an earlier run")
    for name in ("max_steps", "gradient_accumulation_steps", "max_validation_samples", "max_eval_samples", "max_new_tokens", "max_length"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "incomplete"})
    state = {"status": "running", "stage": "initializing", "step": 0,
             "total_steps": config["max_steps"], "evaluation_completed": 0,
             "evaluation_total": config["max_eval_samples"]}
    def progress(stage, **values):
        state.update(values, stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        state["memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
        state["memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
        state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        state["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write_json(output / "progress.json", state)
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    progress("preparing_data")
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    write_json(output / "config.json", config)
    provenance = {
        "git": git_state(), "dataset_path": str(data_path.resolve()),
        "output_path": str(output.resolve()), "working_directory": str(Path.cwd()),
        "command_argv": sys.orig_argv,
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
    limits = {"train": config["max_steps"] * config["gradient_accumulation_steps"],
              "validation": config["max_validation_samples"], "test": config["max_eval_samples"]}
    if config["generation_eval_split"] == "validation":
        limits["validation"] = max(config["max_validation_samples"], config["max_eval_samples"])
        splits = {name: samples for name, samples in splits.items() if name != "test"}
    encoded, audit = prepare_encoded(tokenizer, splits, config["max_length"], limits, config["seed"])
    manifest["tokenization"] = audit
    write_json(output / "split_manifest.json", manifest)
    progress("loading_model")
    model = load_model(config)
    rows = encoded[config["generation_eval_split"]][:config["max_eval_samples"]]
    if config["generation_eval_split"] == "validation" and len(rows) != config["max_eval_samples"]:
        raise ValueError("Not enough usable validation examples for paired generation evaluation")
    base_records = None
    start = time.monotonic()
    if config["evaluate_base_model"]:
        progress("base_evaluation", evaluation_completed=0, evaluation_total=len(rows))
        with torch.random.fork_rng(devices=[0]), model.disable_adapter():
            base_records = evaluate_generation(model, tokenizer, rows, config, device, output / "base_predictions.jsonl",
                lambda count, total: progress("base_evaluation", evaluation_completed=count, evaluation_total=total))
        write_json(output / "base_predictions.json", base_records)
        model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    rng = random.Random(config["seed"])
    order = list(range(len(encoded["train"])))
    rng.shuffle(order)
    position = 0
    model.train()
    history = []
    progress("training")
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
            history.append(row)
            progress("training", step=step, loss=row["loss"], gradient_norm=row["gradient_norm"])
    write_json(output / "train_history.json", history)
    progress("validation_loss")
    validation = validate_loss(model, encoded["validation"], config["max_validation_samples"], device)
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    progress("adapter_saved")
    # Reload the exact saved adapter against the original base before evaluation.
    model = model.unload()
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    progress("sft_evaluation", adapter_reloaded=True)
    progress("sft_evaluation", evaluation_completed=0, evaluation_total=len(rows))
    records = evaluate_generation(model, tokenizer, rows, config, device, output / "predictions.jsonl",
        lambda count, total: progress("sft_evaluation", evaluation_completed=count, evaluation_total=total))
    write_json(output / "predictions.json", records)
    metrics = {"count": len(records), "generation_eval_split": config["generation_eval_split"],
               "adapter_reloaded": True, "validation": validation,
               "metrics": aggregate_scores(records),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "elapsed_seconds": time.monotonic() - start}
    if base_records is not None:
        metrics["base_metrics"] = aggregate_scores(base_records)
        metrics["paired_joint"] = paired_joint_counts(base_records, records)
    write_json(output / "metrics.json", metrics)
    progress("complete", status="complete")
    write_json(output / "status.json", {"status": "complete"})
    return metrics


def run(config, data_path, output):
    try:
        return _run(config, data_path, output)
    except Exception as exc:
        progress_path = output / "progress.json"
        if progress_path.exists():
            try:
                state = json.loads(progress_path.read_text(encoding="utf-8"))
                state.update(status="failed", updated_at=datetime.now(timezone.utc).isoformat(),
                             error={"type": type(exc).__name__, "message": str(exc)})
                write_json(progress_path, state)
            except (OSError, ValueError):
                pass
        raise


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
