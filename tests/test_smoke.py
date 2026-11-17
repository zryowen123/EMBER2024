"""Smoke tests for the thrember feature extractor and vectorization pipeline.

These are deliberately small: they exercise the public API on synthetic data so
the suite runs in seconds and needs no downloaded EMBER2024 files.
"""

import hashlib
import json

import numpy as np
import pytest

import thrember
from thrember import model as thrember_model
from thrember.features import PEFeatureExtractor, _iter_signed_datas

EXPECTED_DIM = 2568
EXPECTED_FEATURES = [
    "general",
    "histogram",
    "byteentropy",
    "strings",
    "header",
    "section",
    "imports",
    "exports",
    "datadirectories",
    "richheader",
    "authenticode",
    "pefilewarnings",
]


class _SerialPool:
    """In-process stand-in for multiprocessing.Pool to keep tests fast and deterministic."""

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
    base = b"MZ" + (b"EMBER2024 smoke test sample %d " % index) + bytes(range(256))
    return base * (index + 1)


def _record(extractor: PEFeatureExtractor, blob: bytes, **labels) -> dict:
    record = extractor.raw_features(blob)
    record.update(labels)
    return record


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_feature_extractor_contract():
    extractor = PEFeatureExtractor()
    assert [fe.name for fe in extractor.features] == EXPECTED_FEATURES
    assert extractor.dim == EXPECTED_DIM

    blob = _blob(0)
    raw = extractor.raw_features(blob)
    assert raw["sha256"] == hashlib.sha256(blob).hexdigest()
    assert raw["general"]["is_pe"] == 0  # synthetic blob is not a valid PE

    vector = extractor.process_raw_features(raw)
    assert vector.shape == (EXPECTED_DIM,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()

    np.testing.assert_array_equal(vector, extractor.feature_vector(blob))


def test_iter_signed_datas_across_signify_versions():
    class OldSignify:
        def iter_signed_datas(self):
            return ["old"]

    class NewSignify:
        def iter_embedded_signatures(self):
            return ["new"]

    assert list(_iter_signed_datas(OldSignify())) == ["old"]
    assert list(_iter_signed_datas(NewSignify())) == ["new"]


def test_create_and_read_binary_features(tmp_path, serial_pool):
    extractor = PEFeatureExtractor()
    blobs = [_blob(i) for i in range(6)]
    labels = [i % 2 for i in range(6)]

    _write_jsonl(
        tmp_path / "train.jsonl",
        [_record(extractor, blob, label=label) for blob, label in zip(blobs, labels)],
    )
    _write_jsonl(
        tmp_path / "test.jsonl",
        [_record(extractor, _blob(10), label=0), _record(extractor, _blob(11), label=1)],
    )
    _write_jsonl(
        tmp_path / "challenge.jsonl",
        [_record(extractor, _blob(20), label=1)],
    )

    thrember.create_vectorized_features(tmp_path)

    X_train, y_train = thrember.read_vectorized_features(tmp_path, "train")
    assert X_train.shape == (6, EXPECTED_DIM)
    assert X_train.dtype == np.float32
    assert y_train.tolist() == labels
    np.testing.assert_allclose(X_train[0], extractor.feature_vector(blobs[0]), rtol=1e-6)

    X_test, y_test = thrember.read_vectorized_features(tmp_path, "test")
    assert X_test.shape == (2, EXPECTED_DIM)
    assert y_test.tolist() == [0, 1]

    X_challenge, y_challenge = thrember.read_vectorized_features(tmp_path, "challenge")
    assert X_challenge.shape == (1, EXPECTED_DIM)
    assert y_challenge.tolist() == [1]


def test_create_and_read_multilabel_features(tmp_path, serial_pool):
    extractor = PEFeatureExtractor()
    behaviors = [["ransom", "crypto"], ["crypto"], ["ransom"], ["steal"]]

    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            _record(extractor, _blob(i), label=1, behavior=behavior)
            for i, behavior in enumerate(behaviors)
        ],
    )
    _write_jsonl(
        tmp_path / "test.jsonl",
        [_record(extractor, _blob(10), label=1, behavior=["steal"])],
    )
    _write_jsonl(
        tmp_path / "challenge.jsonl",
        [_record(extractor, _blob(20), label=1)],
    )

    thrember.create_vectorized_features(tmp_path, label_type="behavior", class_min=1)

    X_train, y_train = thrember.read_vectorized_features(tmp_path, "train")
    assert X_train.shape == (4, EXPECTED_DIM)
    assert y_train.shape == (4, 3)  # ransom, crypto, steal
    assert y_train.sum(axis=1).tolist() == [2, 1, 1, 1]
    assert set(np.unique(y_train)).issubset({0, 1})
