"""Tests for examples/evaluate_features.py sha256 deduplication."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from thrember import model as thrember_model
from thrember.features import PEFeatureExtractor

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "evaluate_features.py"
_spec = importlib.util.spec_from_file_location("evaluate_features", SCRIPT)
evaluate_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_features)


def _sha(index: int) -> str:
    return hashlib.sha256(f"sample-{index}".encode()).hexdigest()


def _write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _dedup(path_groups, names=("eval",), enabled=True):
    return evaluate_features.dedup_masks(path_groups, list(names), enabled=enabled)


def test_global_dedup_across_files(tmp_path):
    sha_a, sha_b, sha_c = _sha(1), _sha(2), _sha(3)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"sha256": sha_a, "label": 1}, {"sha256": sha_b, "label": 0}])
    _write_jsonl(second, [{"sha256": sha_a, "label": 1}, {"sha256": sha_c, "label": 0}])

    masks, stats = _dedup([[str(first), str(second)]])

    assert masks[0].tolist() == [True, True, False, True]
    assert stats["records"] == 4
    assert stats["kept"] == 3
    assert stats["duplicates_removed"] == 1
    assert stats["groups"]["eval"]["files"] == [str(first), str(second)]


def test_groups_share_seen_and_drop_eval_overlap(tmp_path):
    sha_a, sha_b, sha_c = _sha(1), _sha(2), _sha(3)
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    _write_jsonl(train, [{"sha256": sha_a, "label": 1}, {"sha256": sha_b, "label": 0}])
    _write_jsonl(
        evaluation,
        [{"sha256": sha_b, "label": 0}, {"sha256": sha_c, "label": 1}],
    )

    masks, stats = _dedup([[str(train)], [str(evaluation)]], names=("train", "eval"))

    assert masks[0].tolist() == [True, True]
    assert masks[1].tolist() == [False, True]
    assert stats["overlap_removed"] == 1
    assert stats["groups"]["eval"]["cross_group_duplicates"] == 1


def test_keep_duplicates_skips_scan():
    masks, stats = _dedup([["/nonexistent/features.jsonl"]], enabled=False)

    assert masks is None
    assert stats == {"enabled": False}


@pytest.mark.parametrize(
    "record",
    [
        {"label": 1},
        {"sha256": "a" * 63, "label": 1},
        {"sha256": "z" * 64, "label": 1},
    ],
)
def test_missing_or_invalid_sha_aborts(tmp_path, record):
    path = tmp_path / "features.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(SystemExit) as excinfo:
        _dedup([[str(path)]])

    assert "sha256" in str(excinfo.value)
    assert "--keep-duplicates" in str(excinfo.value)


def test_label_conflict_aborts(tmp_path):
    sha = _sha(1)
    path = tmp_path / "features.jsonl"
    _write_jsonl(
        path, [{"sha256": sha, "label": 0}, {"sha256": sha, "label": 1}]
    )

    with pytest.raises(SystemExit) as excinfo:
        _dedup([[str(path)]])

    message = str(excinfo.value)
    assert "label conflict" in message
    assert "merge_features.py" in message
    assert "--keep-duplicates" in message


def test_unlabeled_copy_replaced_by_labeled_copy(tmp_path):
    sha = _sha(1)
    path = tmp_path / "features.jsonl"
    _write_jsonl(
        path,
        [
            {"sha256": sha, "label": -1},
            {"sha256": sha, "label": 1},
            {"sha256": sha, "label": 1},
        ],
    )

    masks, stats = _dedup([[str(path)]])

    assert masks[0].tolist() == [False, True, False]
    assert stats["unlabeled_replaced"] == 1
    assert stats["duplicates_removed"] == 1
    assert stats["kept"] == 1


def test_sha256_case_is_normalized(tmp_path):
    sha = _sha(1)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"sha256": sha.lower(), "label": 1}])
    _write_jsonl(second, [{"sha256": sha.upper(), "label": 1}])

    masks, stats = _dedup([[str(first), str(second)]])

    assert masks[0].tolist() == [True, False]
    assert stats["duplicates_removed"] == 1


def test_blank_line_aborts(tmp_path):
    path = tmp_path / "features.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="blank line in"):
        _dedup([[str(path)]])


class _SerialPool:
    """In-process stand-in for multiprocessing.Pool to keep tests fast."""

    def __init__(self, *args, **kwargs):
        pass

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)

    def imap_unordered(self, func, iterable):
        yield from self.imap(func, iterable)


@pytest.fixture
def serial_pool(monkeypatch):
    monkeypatch.setattr(thrember_model.multiprocessing, "Pool", _SerialPool)


def _blob(index: int) -> bytes:
    return (
        b"MZ"
        + (b"EMBER2024 evaluate_features dedup test %d " % index)
        + bytes(range(256)) * (index + 1)
    )


def _record(extractor: PEFeatureExtractor, blob: bytes, label: int) -> dict:
    record = extractor.raw_features(blob)
    record["label"] = label
    return record


def _training_args(tmp_path: Path):
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps({"num_iterations": 2, "min_data_in_leaf": 1}),
        encoding="utf-8",
    )
    return ["--train-params-json", str(params)]


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scores(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_train_deduplicates_before_split(tmp_path, serial_pool):
    extractor = PEFeatureExtractor()
    records = [
        _record(extractor, _blob(index), index % 2) for index in range(20)
    ]
    first = tmp_path / "first.jsonl"
    _write_jsonl(first, records)
    second = tmp_path / "second.jsonl"
    _write_jsonl(second, [records[0], records[1]])
    out = tmp_path / "metrics.json"
    scores = tmp_path / "scores.csv"

    args = [
        str(first),
        str(second),
        "--train",
        "--test-size",
        "0.3",
        "--seed",
        "42",
        "--json-out",
        str(out),
        "--scores-out",
        str(scores),
    ] + _training_args(tmp_path)
    assert evaluate_features.main(args) == 0

    summary = _read_summary(out)
    dedup = summary["dedup"]
    assert dedup["enabled"] is True
    assert dedup["records"] == 22
    assert dedup["kept"] == 20
    assert dedup["duplicates_removed"] == 2
    assert summary["training"]["train_count"] + summary["counts"]["evaluated"] == 20

    rows = _read_scores(scores)
    assert len(rows) == summary["counts"]["evaluated"]
    assert len({row["sha256"] for row in rows}) == len(rows)


def test_train_keep_duplicates_counts_all_rows(tmp_path, serial_pool):
    extractor = PEFeatureExtractor()
    records = [
        _record(extractor, _blob(index), index % 2) for index in range(20)
    ]
    first = tmp_path / "first.jsonl"
    _write_jsonl(first, records)
    second = tmp_path / "second.jsonl"
    _write_jsonl(second, [records[0], records[1]])
    out = tmp_path / "metrics.json"

    args = [
        str(first),
        str(second),
        "--train",
        "--keep-duplicates",
        "--json-out",
        str(out),
    ] + _training_args(tmp_path)
    assert evaluate_features.main(args) == 0

    summary = _read_summary(out)
    assert summary["dedup"] == {"enabled": False}
    assert summary["training"]["train_count"] + summary["counts"]["evaluated"] == 22


def test_train_features_drops_eval_overlap(tmp_path, serial_pool):
    extractor = PEFeatureExtractor()
    train_records = [
        _record(extractor, _blob(index), index % 2) for index in range(6)
    ]
    new_records = [
        _record(extractor, _blob(10), 1),
        _record(extractor, _blob(11), 0),
    ]
    evaluation_records = [train_records[0], train_records[1], *new_records]
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(train_file, train_records)
    _write_jsonl(eval_file, evaluation_records)
    out = tmp_path / "metrics.json"
    scores = tmp_path / "scores.csv"

    args = [
        str(eval_file),
        "--train-features",
        str(train_file),
        "--json-out",
        str(out),
        "--scores-out",
        str(scores),
    ] + _training_args(tmp_path)
    assert evaluate_features.main(args) == 0

    summary = _read_summary(out)
    dedup = summary["dedup"]
    assert dedup["groups"]["train"]["kept"] == 6
    assert dedup["groups"]["eval"]["kept"] == 2
    assert dedup["overlap_removed"] == 2
    assert summary["training"]["train_count"] == 6
    assert summary["counts"]["evaluated"] == 2

    rows = _read_scores(scores)
    assert {row["sha256"] for row in rows} == {
        new_records[0]["sha256"],
        new_records[1]["sha256"],
    }
