"""Prepare xLAM-style function-calling data and score generated calls."""

import json
import random
import re
from collections import Counter
from pathlib import Path


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode(value):
    return json.loads(value) if isinstance(value, str) else value


def _validate_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    kind = schema.get("type")
    if kind is not None and kind not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ValueError("invalid JSON Schema type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or any(not isinstance(name, str) for name in properties):
        raise ValueError("schema properties must be an object")
    for child in properties.values():
        _validate_schema(child)
    if "items" in schema:
        _validate_schema(schema["items"])
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(name, str) or name not in properties for name in required):
        raise ValueError("invalid required properties")


def read_rows(path):
    """Read an xLAM JSON array or JSONL file without changing its rows."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("The dataset must contain a JSON array or JSONL objects")
    return value


_TYPES = {
    "str": "string", "string": "string", "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "dict": "object", "object": "object",
}


def _tools(value):
    value = _decode(value)
    if not isinstance(value, list) or not value:
        raise ValueError("tools must be a nonempty array")
    tools = []
    names = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("tool must be an object")
        if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
            raw = raw["function"]
        name = raw.get("name")
        description = raw.get("description", "")
        parameters = raw.get("parameters")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("tool name must be nonempty and unique")
        if not isinstance(description, str) or not isinstance(parameters, dict):
            raise ValueError("invalid tool description or parameters")
        names.add(name)
        if parameters.get("type") == "object" and "properties" in parameters:
            _validate_schema(parameters)
            schema = parameters
        else:
            properties = {}
            required = []
            for key, field in parameters.items():
                if not isinstance(key, str) or not isinstance(field, dict):
                    raise ValueError("invalid parameter definition")
                kind = field.get("type")
                if not isinstance(kind, str) or kind.lower() not in _TYPES:
                    raise ValueError("unsupported parameter type")
                is_required = field.get("required", False)
                if not isinstance(is_required, bool):
                    raise ValueError("parameter required must be boolean")
                properties[key] = {k: v for k, v in field.items() if k != "required"}
                properties[key]["type"] = _TYPES[kind.lower()]
                if is_required:
                    required.append(key)
            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            _validate_schema(schema)
        tools.append({"type": "function", "function": {"name": name, "description": description, "parameters": schema}})
    return tools


def _normalize(row, index):
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    tools = _tools(row.get("tools"))
    names = {tool["function"]["name"] for tool in tools}
    answers = _decode(row.get("answers"))
    if not isinstance(answers, list):
        raise ValueError("answers must be an array")
    calls = []
    for answer in answers:
        if not isinstance(answer, dict) or answer.get("name") not in names or not isinstance(answer.get("arguments"), dict):
            raise ValueError("invalid answer or unknown function")
        calls.append({"name": answer["name"], "arguments": answer["arguments"]})
    sample = {"id": row.get("id", index), "query": query, "tools": tools, "calls": calls}
    _canonical(sample)
    return sample


def prepare_splits(rows, *, seed, validation_fraction, test_fraction):
    """Normalize, deduplicate and split connected query/tool-set groups.

    All rows with the same normalized query or the same complete tool set stay
    together. Group sizes mean actual fractions may differ from requested ones.
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    samples = []
    seen = set()
    invalid = Counter()
    duplicates = 0
    input_rows = 0
    for index, row in enumerate(rows):
        input_rows += 1
        try:
            sample = _normalize(row, index)
            key = _canonical({k: sample[k] for k in ("query", "tools", "calls")})
        except (ValueError, TypeError, KeyError) as exc:
            invalid[str(exc)] += 1
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        samples.append(sample)

    parents = list(range(len(samples)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, sample in enumerate(samples):
        query = " ".join(sample["query"].casefold().split())
        signature = _canonical(sorted(_canonical(tool) for tool in sample["tools"]))
        for key in (("query", query), ("tools", signature)):
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    groups = {}
    for index, sample in enumerate(samples):
        groups.setdefault(find(index), []).append(sample)
    batches = list(groups.values())
    if len(batches) < 3:
        raise ValueError(f"At least three independent query/tool-set groups are required; got {len(batches)}")
    random.Random(seed).shuffle(batches)
    splits = {"train": [], "validation": [], "test": []}
    group_counts = {key: 0 for key in splits}
    targets = {"validation": len(samples) * validation_fraction, "test": len(samples) * test_fraction}
    for index, batch in enumerate(batches):
        if index == 0:
            split = "validation"
        elif index == 1:
            split = "test"
        elif index == 2:
            split = "train"
        else:
            deficits = {key: targets[key] - len(splits[key]) for key in targets}
            split = max(deficits, key=deficits.get)
            if deficits[split] <= 0:
                split = "train"
        splits[split].extend(batch)
        group_counts[split] += 1
    manifest = {
        "input_rows": input_rows, "valid_rows": input_rows - sum(invalid.values()),
        "invalid_rows": sum(invalid.values()), "invalid_reasons": dict(invalid),
        "duplicate_rows": duplicates, "unique_rows": len(samples),
        "group_count": len(batches), "largest_group_size": max(map(len, batches)),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_group_counts": group_counts, "seed": seed,
        "validation_fraction": validation_fraction, "test_fraction": test_fraction,
    }
    return splits, manifest


_CALL = re.compile(r"\s*<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def score_output(text, sample):
    """Score the complete ordered sequence of Qwen tool-call tags exactly."""
    calls = []
    position = 0
    valid = isinstance(text, str)
    if valid:
        while position < len(text) and text[position:].strip():
            match = _CALL.match(text, position)
            if not match:
                valid = False
                break
            try:
                call = json.loads(match.group(1), object_pairs_hook=_strict_object, parse_constant=_reject_constant)
            except (ValueError, TypeError):
                valid = False
                break
            if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not call["name"] or not isinstance(call["arguments"], dict):
                valid = False
                break
            calls.append(call)
            position = match.end()
    expected = sample["calls"]
    if valid and not calls and expected:
        valid = False
    names_match = valid and [call["name"] for call in calls] == [call["name"] for call in expected]
    arguments_match = valid and [_canonical(call["arguments"]) for call in calls] == [_canonical(call["arguments"]) for call in expected]
    return {"structure_valid": valid, "function_name_match": names_match,
            "arguments_match": arguments_match, "joint_match": names_match and arguments_match}
