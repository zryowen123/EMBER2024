#!/usr/bin/env python3
"""
Detect a single file with the EMBER2024 benchmark models.

The file type is sniffed from its contents, one matching malicious/benign
model is loaded, and features are extracted exactly once. Family identification
and tag prediction are opt-in because they answer different questions and load
extra (sometimes large) models:

    python examples/detect_sample.py sample.bin
    python examples/detect_sample.py sample.bin --json
    python examples/detect_sample.py sample.bin --tags packer,behavior
    python examples/detect_sample.py sample.bin --family

Notes:
- Family and tag outputs are class/tag *indices*: the model files do not embed
  label names, and create_vectorized_features() does not persist its label_map.
- The score is not a calibrated probability; pick --threshold for your
  acceptable false-positive rate.
- Static analysis only: the sample is never executed.
"""

import argparse
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pefile

from thrember.features import PEFeatureExtractor

MODEL_FILES = {
    "all": "EMBER2024_all.model",
    "PE": "EMBER2024_PE.model",
    "Win32": "EMBER2024_Win32.model",
    "Win64": "EMBER2024_Win64.model",
    "Dot_Net": "EMBER2024_Dot_Net.model",
    "APK": "EMBER2024_APK.model",
    "ELF": "EMBER2024_ELF.model",
    "PDF": "EMBER2024_PDF.model",
}

TAG_MODELS = {
    "behavior": "EMBER2024_behavior.model",
    "file_property": "EMBER2024_file_property.model",
    "packer": "EMBER2024_packer.model",
    "exploit": "EMBER2024_exploit.model",
    "group": "EMBER2024_group.model",
}

FAMILY_FILE = "EMBER2024_family.model"


def detect_file_type(data: bytes) -> str:
    """Return one of MODEL_FILES' keys based on the file's contents."""
    if data.startswith(b"%PDF"):
        return "PDF"
    if data.startswith(b"\x7fELF"):
        return "ELF"

    if data.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "AndroidManifest.xml" in zf.namelist():
                    return "APK"
        except (zipfile.BadZipFile, OSError):
            pass

    try:
        pe = pefile.PE(data=data)
    except Exception:
        # pefile raises PEFormatError for most malformed inputs, but other
        # errors are possible on truncated or adversarial files.
        return "all"

    try:
        com_descriptor = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]
        ]
        has_dotnet = com_descriptor.VirtualAddress != 0
    except (AttributeError, IndexError, KeyError):
        has_dotnet = False

    if has_dotnet:
        return "Dot_Net"

    machine = pe.FILE_HEADER.Machine
    magic = pe.OPTIONAL_HEADER.Magic
    # AMD64, ARM64, or any PE32+ image
    if machine in (0x8664, 0xAA64) or magic == 0x20B:
        return "Win64"
    return "Win32"


def _model_path(models_dir: Path, filename: str) -> Path:
    path = models_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"model not found: {path}\n"
            "download it with thrember.download_models() or hf download "
            "joyce8/EMBER2024-benchmark-models --repo-type model"
        )
    return path


def detect(
    data: bytes,
    models_dir: Path,
    *,
    file_type: str = "auto",
    threshold: float = 0.5,
    tag_threshold: float = 0.5,
    family: bool = False,
    family_top_k: int = 5,
    tags: tuple[str, ...] = (),
) -> dict:
    """Run the requested model layers on one sample and return a result dict."""
    if not data:
        raise ValueError("cannot analyze an empty file")

    if file_type == "auto":
        file_type = detect_file_type(data)
    if file_type not in MODEL_FILES:
        raise ValueError(f"unknown file type: {file_type}")

    # Extract features once; every layer below reuses this vector.
    features = np.asarray(PEFeatureExtractor().feature_vector(data), dtype=np.float32)

    model_path = _model_path(models_dir, MODEL_FILES[file_type])
    classifier = lgb.Booster(model_file=str(model_path))
    score = float(classifier.predict([features])[0])

    result = {
        "file_type": file_type,
        "model": model_path.name,
        "score": score,
        "threshold": threshold,
        "verdict": "malicious" if score >= threshold else "benign",
    }

    if family:
        family_model = lgb.Booster(model_file=str(_model_path(models_dir, FAMILY_FILE)))
        probs = np.asarray(family_model.predict([features])[0], dtype=np.float64)
        top = np.argsort(probs)[::-1][: max(1, family_top_k)]
        result["family_top_k"] = [
            {"index": int(i), "prob": float(probs[i])} for i in top
        ]

    if tags:
        result["tags"] = {}
        for task in tags:
            if task not in TAG_MODELS:
                raise ValueError(
                    f"unknown tag model: {task} (choose from {', '.join(TAG_MODELS)})"
                )
            with _model_path(models_dir, TAG_MODELS[task]).open("rb") as f:
                boosters = pickle.load(f)
            probs = np.asarray(
                [float(booster.predict([features])[0]) for booster in boosters]
            )
            active = np.flatnonzero(probs >= tag_threshold)
            active = active[np.argsort(probs[active])[::-1]]
            result["tags"][task] = [
                {"index": int(i), "prob": float(probs[i])} for i in active
            ]

    return result


def _parse_tags(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    if value == "all":
        return tuple(TAG_MODELS)
    tasks = tuple(t.strip() for t in value.split(",") if t.strip())
    unknown = sorted(set(tasks) - set(TAG_MODELS))
    if unknown:
        raise ValueError(
            f"unknown tag model(s): {', '.join(unknown)} "
            f"(choose from {', '.join(TAG_MODELS)} or 'all')"
        )
    return tasks


def _print_human(sample: Path, digest: str, result: dict) -> None:
    print(f"Sample:      {sample}")
    print(f"SHA256:      {digest}")
    print(f"File type:   {result['file_type']}")
    print(f"Model:       {result['model']}")
    print(
        f"Score:       {result['score']:.6f} "
        f"(threshold {result['threshold']:.2f}) -> {result['verdict'].upper()}"
    )
    if "family_top_k" in result:
        pairs = ", ".join(
            f"#{item['index']}={item['prob']:.4f}" for item in result["family_top_k"]
        )
        print(f"Family top:  {pairs or '(none)'}   [indices only]")
    for task, items in result.get("tags", {}).items():
        pairs = ", ".join(f"#{item['index']}={item['prob']:.4f}" for item in items)
        print(f"{task:13s}{pairs or '(none above threshold)'}   [indices only]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect a single file with EMBER2024 benchmark models."
    )
    parser.add_argument("sample", type=Path, help="file to analyze")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "model",
        help="directory containing EMBER2024_*.model (default: <repo>/model)",
    )
    parser.add_argument(
        "--file-type",
        choices=("auto", *MODEL_FILES),
        default="auto",
        help="override content-based file type detection",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="malicious/benign threshold (default: 0.5)")
    parser.add_argument("--tag-threshold", type=float, default=0.5,
                        help="tag activation threshold (default: 0.5)")
    parser.add_argument("--family", action="store_true",
                        help="also run the 2538-class family model")
    parser.add_argument("--family-top-k", type=int, default=5,
                        help="number of family indices to print (default: 5)")
    parser.add_argument("--tags", default="",
                        help="comma-separated tag models to run, or 'all': "
                             + ", ".join(TAG_MODELS))
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit JSON instead of human-readable text")
    args = parser.parse_args(argv)

    try:
        data = args.sample.read_bytes()
    except OSError as exc:
        parser.error(f"cannot read {args.sample}: {exc}")

    try:
        result = detect(
            data,
            args.models_dir,
            file_type=args.file_type,
            threshold=args.threshold,
            tag_threshold=args.tag_threshold,
            family=args.family,
            family_top_k=args.family_top_k,
            tags=_parse_tags(args.tags),
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(data).hexdigest()
    if args.as_json:
        print(json.dumps({"path": str(args.sample), "sha256": digest, **result},
                         indent=2))
    else:
        _print_human(args.sample, digest, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
