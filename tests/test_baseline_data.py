import json
import tempfile
import unittest
from pathlib import Path

from src.baseline_data import prepare_splits, read_rows, score_output


def row(number, *, query=None, tool=None):
    name = tool or f"tool_{number}"
    return {
        "id": number,
        "query": query or f"Question {number}",
        "tools": json.dumps([{
            "name": name,
            "description": "A function",
            "parameters": {
                "count": {"type": "int", "description": "Count", "required": True},
                "items": {"type": "list", "description": "Items", "required": False},
            },
        }]),
        "answers": json.dumps([{"name": name, "arguments": {"count": number}}]),
    }


class BaselineDataTests(unittest.TestCase):
    def test_read_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            rows = [row(1), row(2)]
            path.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(read_rows(path), rows)
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            self.assertEqual(read_rows(path), rows)

    def test_prepare_deduplicates_groups_and_converts_schema(self):
        rows = [row(1), row(2, query="  QUESTION   1 "), row(3, tool="tool_2"), row(4), row(5), row(6)]
        rows.append(dict(rows[0], id="duplicate"))
        bad = row(7)
        bad["answers"] = '[{"name":"missing","arguments":{}}]'
        rows.append(bad)
        splits, manifest = prepare_splits(rows, seed=17, validation_fraction=0.2, test_fraction=0.2)
        self.assertEqual(manifest["duplicate_rows"], 1)
        self.assertEqual(manifest["invalid_rows"], 1)
        self.assertEqual(manifest["unique_rows"], 6)
        self.assertEqual(prepare_splits(rows, seed=17, validation_fraction=0.2, test_fraction=0.2), (splits, manifest))
        self.assertTrue(all(splits.values()))
        by_id = {sample["id"]: name for name, samples in splits.items() for sample in samples}
        self.assertEqual(by_id[1], by_id[2])
        self.assertEqual(by_id[2], by_id[3])
        sample = next(s for values in splits.values() for s in values if s["id"] == 1)
        params = sample["tools"][0]["function"]["parameters"]
        self.assertEqual(params["properties"]["count"]["type"], "integer")
        self.assertEqual(params["properties"]["items"]["type"], "array")
        self.assertEqual(params["required"], ["count"])

    def test_bad_data_and_split(self):
        with self.assertRaises(ValueError):
            prepare_splits([row(1), row(2)], seed=0, validation_fraction=.2, test_fraction=.2)
        with self.assertRaises(ValueError):
            prepare_splits([row(1), row(2), row(3)], seed=0, validation_fraction=.5, test_fraction=.5)
        duplicated_tool = row(4)
        tool = json.loads(duplicated_tool["tools"])[0]
        duplicated_tool["tools"] = [tool, tool]
        invalid_type = row(5)
        tool = json.loads(invalid_type["tools"])[0]
        tool["parameters"]["count"]["type"] = "imaginary"
        invalid_type["tools"] = [tool]
        _, manifest = prepare_splits([row(1), row(2), row(3), duplicated_tool, invalid_type], seed=0, validation_fraction=.2, test_fraction=.2)
        self.assertEqual(manifest["invalid_rows"], 2)

    def test_strict_ordered_scoring(self):
        sample = {"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]}
        output = ' \n<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>\n<tool_call>{"name":"b","arguments":{}}</tool_call> '
        self.assertTrue(score_output(output, sample)["joint_match"])
        wrong_argument = score_output(output.replace('"x":1', '"x":2'), sample)
        self.assertTrue(wrong_argument["function_name_match"])
        self.assertFalse(wrong_argument["arguments_match"])
        self.assertFalse(score_output(output[:-13], sample)["structure_valid"])
        self.assertFalse(score_output(output + "extra", sample)["structure_valid"])
        self.assertFalse(score_output('<tool_call>{"name":"a","arguments":[]}</tool_call>', sample)["structure_valid"])
        self.assertFalse(score_output('<tool_call>{"name":"a","name":"a","arguments":{}}</tool_call>', sample)["structure_valid"])
        self.assertFalse(score_output('<tool_call>{"name":"a","arguments":{"x":NaN}}</tool_call>', sample)["structure_valid"])
        self.assertFalse(score_output(" \n ", sample)["structure_valid"])
        self.assertTrue(score_output(" ", {"calls": []})["joint_match"])


if __name__ == "__main__":
    unittest.main()
