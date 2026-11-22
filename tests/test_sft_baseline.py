import unittest
import json
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.json_io import write_json_atomic

from scripts.run_sft_baseline import (aggregate_scores, effective_config, encode_sample, evaluate_generation,
                                      paired_joint_counts, prepare_encoded, tensor_batch, validate_loss)


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("add_generation_prompt"):
            return [1, 2, 3]
        return [1, 2, 3, 4, 5, 99]


SAMPLE = {"id": "one", "query": "query", "tools": [], "calls": []}


class BaselineTests(unittest.TestCase):
    def test_paired_joint_counts_and_alignment(self):
        def prediction(index, correct):
            return {"id": index, "target": [{"name": "call", "arguments": {}}],
                    "scores": {"joint_match": correct}}

        base = [prediction(1, False), prediction(2, True), prediction(3, True), prediction(4, False)]
        sft = [prediction(1, True), prediction(2, False), prediction(3, True), prediction(4, False)]
        self.assertEqual(paired_joint_counts(base, sft), {
            "improved": 1, "regressed": 1, "unchanged_correct": 1,
            "unchanged_incorrect": 1, "unchanged": 2, "denominator": 4,
        })
        self.assertEqual(aggregate_scores(sft)["joint_match"], {"correct": 2, "denominator": 4, "rate": 0.5})
        with self.assertRaisesRegex(ValueError, "same length"):
            paired_joint_counts(base, sft[:-1])
        with self.assertRaisesRegex(ValueError, "not aligned"):
            paired_joint_counts(base, list(reversed(sft)))
        altered = [dict(row) for row in sft]
        altered[0]["target"] = []
        with self.assertRaisesRegex(ValueError, "not aligned"):
            paired_joint_counts(base, altered)

    def test_base_evaluation_default_and_type(self):
        self.assertFalse(effective_config({})["evaluate_base_model"])
        self.assertTrue(effective_config({"evaluate_base_model": True})["evaluate_base_model"])
        with self.assertRaisesRegex(ValueError, "boolean"):
            effective_config({"evaluate_base_model": "false"})

    def test_generation_split_and_teacher_contract(self):
        self.assertEqual(effective_config({})["generation_eval_split"], "test")
        self.assertEqual(effective_config({"generation_eval_split": "validation"})["generation_eval_split"], "validation")
        with self.assertRaises(ValueError):
            effective_config({"generation_eval_split": "train"})
        with self.assertRaises(ValueError):
            effective_config({"model_id": "Qwen/Qwen3-4B", "generation_eval_split": "validation"})
        self.assertTrue(effective_config({"model_id": "Qwen/Qwen3-4B", "generation_eval_split": "validation", "evaluate_base_model": True})["evaluate_base_model"])

    def test_strict_atomic_json_preserves_previous_document(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            write_json_atomic(path, {"status": "running", "step": 1})
            with self.assertRaises(ValueError):
                write_json_atomic(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "running", "step": 1})
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["progress.json"])

    def test_atomic_json_retries_transient_windows_sharing_errors(self):
        import os
        with TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            original_replace = os.replace
            error = PermissionError("sharing violation")
            error.winerror = 5
            attempts = []
            def flaky_replace(source, destination):
                attempts.append(1)
                if len(attempts) <= 2:
                    raise error
                return original_replace(source, destination)
            with patch("src.json_io.os.replace", side_effect=flaky_replace) as replace, \
                    patch("src.json_io.time.sleep") as sleep:
                write_json_atomic(path, {"step": 1})
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"step": 1})

    def test_atomic_json_persistent_windows_error_preserves_prior(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            write_json_atomic(path, {"step": 1})
            error = PermissionError("sharing violation")
            error.winerror = 5
            with patch("src.json_io.os.replace", side_effect=error) as replace, \
                    patch("src.json_io.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    write_json_atomic(path, {"step": 2})
            self.assertEqual(replace.call_count, 10)
            self.assertEqual(sleep.call_count, 9)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"step": 1})
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["progress.json"])

    def test_generation_records_stopping_and_scores(self):
        class GenerationTokenizer:
            eos_token_id = 99

            def apply_chat_template(self, messages, **kwargs):
                self.asserted_tools = kwargs["tools"]
                return torch.tensor([[1, 2]], dtype=torch.long)

            def decode(self, tokens, **kwargs):
                return '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'

        class GenerationModel:
            config = type("Config", (), {"use_cache": False})()

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.arguments = kwargs
                return torch.tensor([[1, 2, 3, 99]], dtype=torch.long)

        sample = {"id": "test", "query": "find", "tools": [],
                  "calls": [{"name": "lookup", "arguments": {}}]}
        model = GenerationModel()
        tokenizer = GenerationTokenizer()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            updates = []
            records = evaluate_generation(model, tokenizer, [(sample, {})], {"max_new_tokens": 4},
                                          torch.device("cpu"), output, lambda count, total: updates.append((count, total)))
            self.assertEqual(updates, [(1, 1)])
            self.assertTrue(output.exists())
        self.assertTrue(records[0]["scores"]["joint_match"])
        self.assertTrue(records[0]["reached_eos"])
        self.assertFalse(records[0]["token_limit_reached"])
        self.assertFalse(model.arguments["do_sample"])

    def test_mask_and_batch(self):
        item = encode_sample(Tokenizer(), SAMPLE, 6)
        self.assertEqual(item, {"input_ids": [1, 2, 3, 4, 5, 99], "labels": [-100, -100, -100, 4, 5, 99]})
        batch = tensor_batch(item, torch.device("cpu"))
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 6))
        self.assertEqual(batch["labels"].dtype, torch.int64)
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 1]])

    def test_overlength_exclusion_and_empty_split(self):
        self.assertIsNone(encode_sample(Tokenizer(), SAMPLE, 5))
        with self.assertRaisesRegex(ValueError, "No train"):
            prepare_encoded(Tokenizer(), {"train": [SAMPLE]}, 5, {"train": 1}, 42)

    def test_bounded_scan_and_seed(self):
        samples = [{**SAMPLE, "id": i} for i in range(10)]

        def encode(_tokenizer, sample, _max_length):
            if sample["id"] % 2 == 0:
                return None
            return {"input_ids": [1, 99], "labels": [-100, 99]}

        with patch("scripts.run_sft_baseline.encode_sample", side_effect=encode):
            encoded, audit = prepare_encoded(Tokenizer(), {"train": samples}, 6, {"train": 2}, 42)
            repeat, repeat_audit = prepare_encoded(Tokenizer(), {"train": samples}, 6, {"train": 2}, 42)
            self.assertEqual(encoded, repeat)
            self.assertEqual(audit, repeat_audit)
            self.assertEqual(len(encoded["train"]), 2)
            self.assertEqual(audit["train"]["inspected"], 2 + len(audit["train"]["overlength_ids"]))
            self.assertEqual(audit["train"]["uninspected"], 10 - audit["train"]["inspected"])
            exhausted, exhausted_audit = prepare_encoded(Tokenizer(), {"train": samples}, 6, {"train": 8}, 42)
            self.assertEqual(len(exhausted["train"]), 5)
            self.assertEqual(exhausted_audit["train"]["uninspected"], 0)

    def test_bad_prefix_or_missing_end_rejected(self):
        tokenizer = Tokenizer()
        with patch.object(tokenizer, "apply_chat_template", side_effect=[[1, 2], [1, 3, 99]]):
            with self.assertRaisesRegex(ValueError, "generation prompt"):
                encode_sample(tokenizer, SAMPLE, 20)
        with patch.object(tokenizer, "apply_chat_template", side_effect=[[1, 2], [1, 2, 3]]):
            with self.assertRaisesRegex(ValueError, "end-of-turn"):
                encode_sample(tokenizer, SAMPLE, 20)

    def test_validation_loss_rejects_nan(self):
        class Model:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                return type("Output", (), {"loss": torch.tensor(float("nan"))})()

        item = encode_sample(Tokenizer(), SAMPLE, 6)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_loss(Model(), [(SAMPLE, item)], 1, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
