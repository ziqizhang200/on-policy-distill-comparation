import unittest
from unittest.mock import patch

import torch

from scripts.run_sft_baseline import encode_sample, prepare_encoded, tensor_batch, validate_loss


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("add_generation_prompt"):
            return [1, 2, 3]
        return [1, 2, 3, 4, 5, 99]


SAMPLE = {"id": "one", "query": "query", "tools": [], "calls": []}


class BaselineTests(unittest.TestCase):
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
