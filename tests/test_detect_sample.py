"""Tests for examples/detect_sample.py (no model downloads required)."""

import importlib.util
import types
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "detect_sample.py"
_spec = importlib.util.spec_from_file_location("detect_sample", SCRIPT)
detect_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(detect_sample)

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


class _FakeDataDirectory:
    def __init__(self, virtual_address=0):
        self.VirtualAddress = virtual_address


class _FakePE:
    def __init__(self, machine=0x8664, magic=0x20B, com_rva=0):
        directories = [_FakeDataDirectory() for _ in range(16)]
        directories[14] = _FakeDataDirectory(com_rva)  # COM descriptor
        self.FILE_HEADER = types.SimpleNamespace(Machine=machine)
        self.OPTIONAL_HEADER = types.SimpleNamespace(
            Magic=magic, DATA_DIRECTORY=directories
        )


def test_detect_magic_based_types():
    assert detect_sample.detect_file_type(b"%PDF-1.7\n...") == "PDF"
    assert detect_sample.detect_file_type(b"\x7fELF" + bytes(64)) == "ELF"
    assert detect_sample.detect_file_type(b"not a known format") == "all"


def test_detect_apk():
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("AndroidManifest.xml", "<manifest/>")
        zf.writestr("classes.dex", bytes(16))
    assert detect_sample.detect_file_type(buf.getvalue()) == "APK"


def test_detect_pe_subtypes(monkeypatch):
    def with_pe(fake):
        monkeypatch.setattr(detect_sample.pefile, "PE", lambda data: fake)
        return detect_sample.detect_file_type(b"MZ" + bytes(64))

    assert with_pe(_FakePE(machine=0x8664)) == "Win64"
    assert with_pe(_FakePE(machine=0x14C, magic=0x10B)) == "Win32"
    assert with_pe(_FakePE(machine=0x14C, magic=0x10B, com_rva=0x2000)) == "Dot_Net"
    # PE32+ magic alone routes to Win64 (e.g. ARM64/IA64)
    assert with_pe(_FakePE(machine=0xAA64, magic=0x20B)) == "Win64"


def test_parse_tags():
    assert detect_sample._parse_tags("") == ()
    assert detect_sample._parse_tags("all") == tuple(detect_sample.TAG_MODELS)
    assert detect_sample._parse_tags("packer, behavior") == ("packer", "behavior")
    with pytest.raises(ValueError):
        detect_sample._parse_tags("packer,nope")


def test_empty_file_rejected():
    with pytest.raises(ValueError):
        detect_sample.detect(b"", MODEL_DIR, file_type="all")


@pytest.mark.skipif(
    not (MODEL_DIR / "EMBER2024_all.model").is_file(),
    reason="benchmark models not downloaded into model/",
)
def test_detect_integration():
    result = detect_sample.detect(
        b"MZ" + bytes(range(256)) * 8, MODEL_DIR, file_type="all"
    )
    assert result["model"] == "EMBER2024_all.model"
    assert 0.0 <= result["score"] <= 1.0
    assert result["verdict"] in {"benign", "malicious"}
