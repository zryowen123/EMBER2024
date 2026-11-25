#!/usr/bin/env python3
"""Safely extract EMBER2024 (feature version 3) raw features from files inside
untrusted archives.

Supported archives: zip (ZipCrypto; AES with pyzipper), 7z (py7zr or the
system 7z CLI), tar.gz. A directory of loose benign files can also be supplied
with --benign-dir.

Design notes:
- Nothing from the archive is ever executed. The default 7z backend (py7zr)
  reads samples into memory only. The optional system backend
  (--7z-backend system) extracts only the selected, size-checked members into a
  private temporary directory and removes it afterward (unless
  --keep-7z-extract is given).
- By default only files with a valid PE structure are processed; everything
  else is recorded in the skipped log. Pass --include-non-pe to also extract
  APK/ELF/PDF and other non-PE files, which still receive the useful
  general/byte-histogram/string features (PE-only blocks are zero).
- Samples are parsed by a pool of persistent worker processes. Each worker
  loads the extractor once, so process startup is paid per worker instead of
  per sample. Workers are recycled after --recycle-tasks samples and replaced
  immediately after a timeout or crash, so a malformed PE cannot take down the
  batch. Use --isolate for one fresh process per sample (strictest isolation,
  noticeably slower).
- Every sample is still parsed under file-size/core rlimits, and the
  supervisor enforces a wall-clock timeout by killing and replacing the
  worker.
- Zip members are decompressed and hashed by --read-workers threads, so the
  extraction workers stay busy while the next members are being read.
- The output is a JSONL file whose records are compatible with the EMBER2024
  raw feature format, and therefore with ``thrember.vectorize_subset`` and
  ``thrember.create_vectorized_features``. ``--vectorize`` writes
  ``X_<split>.dat``/``y_<split>.dat`` in the layout read by
  ``thrember.read_vectorized_features``.

Requires this repository (``pip install -e .``); the extractor is the
feature-version-3 ``thrember.PEFeatureExtractor`` (2568 dims), not EMBER v2.

Examples:
    python examples/extract_features.py \\
        --archive malware.zip --output out/malware.jsonl --vectorize
    python examples/extract_features.py \\
        --benign-dir /srv/benign --benign-archive more_benign.7z \\
        --output out/benign.jsonl
    python examples/extract_features.py \\
        --archive samples.7z --7z-backend system --7z-command 7zz \\
        --output out/samples.jsonl
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import queue
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_MAX_FILE_SIZE = 20 * MIB
DEFAULT_MAX_TOTAL_BYTES = 10 * GIB
DEFAULT_MAX_MEMBERS = 100_000
DEFAULT_MAX_COMPRESSION_RATIO = 1000.0
DEFAULT_MAX_RECORD_BYTES = 16 * MIB
DEFAULT_TIMEOUT = 30.0
DEFAULT_MEMORY_PER_SAMPLE = 0  # disabled by default; container memory is authoritative
ZIP_AES_COMPRESS_TYPE = 99
SEVEN_Z_MAGIC = b"7z\xbc\xaf\x27\x1c"
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
GZIP_MAGIC = b"\x1f\x8b"
PE_SIGNATURE = b"PE\0\0"
BATCH_BUDGET_BYTES = 256 * MIB
BATCH_MAX_MEMBERS = 256
PREFETCH_MAX_BYTES = 256 * MIB
DEFAULT_READ_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_RECYCLE_TASKS = 256
DEFAULT_FSYNC_EVERY = 1024
_SYSTEM_7Z_WARNED = False
_ZIP_CLI_WARNED = False
_ZIP_SLOW_WARNED = False


class ArchiveError(Exception):
    """Fatal error for the current archive."""


class ArchiveAbort(ArchiveError):
    """The archive hit a configured limit; stop processing it."""


@dataclass(frozen=True)
class Limits:
    max_file_size: int
    max_total_bytes: int
    max_members: int
    max_compression_ratio: float
    max_record_bytes: int
    timeout: float
    cpu_timeout: float
    memory_per_sample: int


@dataclass
class Member:
    """One archive member.

    Exactly one of ``data``, ``skip_reason`` or ``error_reason`` is meaningful.
    ``sha256`` is filled by readers that already hashed the member (zip reader
    threads) so the supervisor does not hash it a second time.
    """

    name: str
    size: int = 0
    data: Optional[bytes] = None
    skip_reason: Optional[str] = None
    error_reason: Optional[str] = None
    sha256: Optional[str] = None


@dataclass
class Task:
    """One sample queued to a worker. ``proc``/``conn`` live on the Worker."""

    task_id: int
    digest: bytes
    sha256: str
    archive: str
    member: str
    label: int
    deadline: float


@dataclass
class Worker:
    index: int
    proc: Any = None
    conn: Any = None
    tasks_done: int = 0
    task: Optional[Task] = None


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)B?\s*", value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
        "P": 1024 ** 5,
    }[suffix]
    result = int(number * multiplier)
    if result < 0:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}")
    return result


def sanitize_name(name: Any, max_len: int = 512) -> str:
    """Make an archive/member name safe for logs and JSON output.

    The result is never used as a filesystem path.
    """
    text = str(name)
    # Replace lone surrogates and undecodable characters from hostile archives.
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    text = "".join(ch if (ord(ch) >= 0x20 and ord(ch) != 0x7F) else "?" for ch in text)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def is_pe(data: bytes) -> bool:
    """Cheap structural PE check; extensions are deliberately ignored."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return False
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew < 4 or e_lfanew + 4 > len(data):
        return False
    return data[e_lfanew : e_lfanew + 4] == PE_SIGNATURE


def detect_format(path: str, override: str) -> str:
    if override != "auto":
        return override
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        raise ArchiveError(f"cannot open archive: {exc}") from exc
    if head.startswith(ZIP_MAGICS):
        return "zip"
    if head.startswith(SEVEN_Z_MAGIC):
        return "7z"
    if head.startswith(GZIP_MAGIC):
        return "tar.gz"
    lower = path.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".7z"):
        return "7z"
    if lower.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    raise ArchiveError(f"unrecognized archive format: {path}")


def read_capped(handle: Any, max_bytes: int) -> bytes:
    data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ArchiveAbort(f"member exceeds max_file_size ({max_bytes} bytes)")
    return data


def zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _open_zip_archive(path: str, use_aes: bool, password: Optional[bytes]) -> Any:
    if use_aes:
        try:
            import pyzipper  # type: ignore
        except ImportError as exc:
            raise ArchiveError("AES-encrypted zip requires pyzipper") from exc
        archive = pyzipper.AESZipFile(path, "r")
    else:
        archive = zipfile.ZipFile(path, "r")
    if password is not None:
        archive.setpassword(password)
    return archive


def _resolve_executable(command: str) -> Optional[str]:
    if not command:
        return None
    if os.path.sep in command or "/" in command:
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return command
        return None
    return shutil.which(command)


def _resolve_zip_command(command: str) -> Optional[str]:
    found = _resolve_executable(command)
    if found is not None:
        return found
    # p7zip can decrypt ZipCrypto too. Only substitute for the default tool so
    # an explicit --zip-command that is missing fails visibly.
    if os.path.basename(command) == "unzip":
        for fallback in ("7za", "7zz"):
            found = _resolve_executable(fallback)
            if found is not None:
                print(
                    f"WARNING: '{command}' was not found; using '{fallback}' to "
                    "decrypt ZipCrypto members",
                    file=sys.stderr,
                )
                return found
    return None


def _zip_cli_argv(command: str, path: str, member: str, password: str) -> List[str]:
    if "7z" in os.path.basename(command).lower():
        return [command, "e", "-so", "-p" + password, path, member]
    return [command, "-P", password, "-p", path, member]


def _run_zip_cli(
    argv: Sequence[str],
    password: str,
    max_bytes: int,
    timeout: float,
) -> Tuple[Optional[bytes], int, str, bool]:
    """Run a decryption CLI and read its stdout with a hard size cap.

    Returns ``(data, returncode, stderr, overflow)``. ``data`` is ``None`` only
    when the command could not be started. Reading at most ``max_bytes + 1``
    bytes keeps a lying central directory from exhausting memory.
    """
    err_file = tempfile.TemporaryFile()
    try:
        try:
            proc = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=err_file,
            )
        except OSError as exc:
            return None, -1, f"{type(exc).__name__}: {exc}", False
        timer: Optional[threading.Timer] = None
        if timeout > 0:
            timer = threading.Timer(timeout, proc.kill)
            timer.daemon = True
            timer.start()
        overflow = False
        try:
            data = proc.stdout.read(max_bytes + 1) if proc.stdout is not None else b""
            if len(data) > max_bytes:
                overflow = True
                proc.kill()
            returncode = proc.wait()
        except OSError as exc:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait()
            except OSError:
                pass
            return None, -1, f"{type(exc).__name__}: {exc}", False
        finally:
            if timer is not None:
                timer.cancel()
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
        err_file.seek(0)
        stderr = err_file.read(8192).decode("utf-8", "replace")
        if password:
            stderr = stderr.replace(password, "***")
        if returncode < 0:
            stderr = (stderr + " (killed)").strip()
        return data, returncode, stderr.strip()[:500], overflow
    finally:
        err_file.close()


def _zip_member_from_data(name: str, data: bytes) -> Member:
    # Hash here so the work happens in the reader threads, not the supervisor.
    return Member(name, len(data), data=data, sha256=hashlib.sha256(data).hexdigest())


def _read_zip_member_python(
    archive: Any,
    info: zipfile.ZipInfo,
    password: Optional[bytes],
    limits: Limits,
) -> Member:
    name = sanitize_name(info.filename)
    try:
        with archive.open(info, "r", pwd=password) as handle:
            data = read_capped(handle, limits.max_file_size)
    except ArchiveAbort:
        return Member(name, info.file_size, skip_reason="too_large_actual")
    except (RuntimeError, NotImplementedError, zipfile.BadZipFile, OSError, ValueError) as exc:
        return Member(name, info.file_size, error_reason=f"{type(exc).__name__}: {exc}")
    return _zip_member_from_data(name, data)


def _read_zip_member_cli(
    command: str,
    path: str,
    info: zipfile.ZipInfo,
    password: bytes,
    limits: Limits,
    timeout: float,
) -> Optional[Member]:
    """Decrypt one ZipCrypto member with an external CLI (C speed).

    Returns ``None`` when this member cannot be handled by the CLI (odd member
    name or non-UTF-8 password) so the caller can fall back to Python.
    """
    raw_name = info.filename
    if (
        not raw_name
        or raw_name.startswith("-")
        or "\n" in raw_name
        or "\r" in raw_name
        or any(ch in raw_name for ch in "*?[")
    ):
        return None
    try:
        password_str = password.decode("utf-8")
    except UnicodeDecodeError:
        return None
    argv = _zip_cli_argv(command, path, raw_name, password_str)
    data, returncode, stderr, overflow = _run_zip_cli(
        argv, password_str, limits.max_file_size, timeout
    )
    name = sanitize_name(raw_name)
    if data is None:
        return Member(name, info.file_size, error_reason=stderr or "zip command failed")
    if overflow:
        return Member(name, info.file_size, skip_reason="too_large_actual")
    if returncode in (0, 1):
        if len(data) == info.file_size:
            # Exit code 1 is a warning and harmless when the payload is complete.
            return _zip_member_from_data(name, data)
        # Success code but a different payload length: the name may have
        # matched several members or used another encoding. Let Python retry.
        return None
    detail = f"exit {returncode}: got {len(data)} bytes, expected {info.file_size}"
    if stderr:
        detail = f"{detail}: {stderr}"
    return Member(name, info.file_size, error_reason=detail[:500])


class _ZipReader:
    """Reads zip members, using the CLI for ZipCrypto and Python otherwise.

    One instance is used by one thread at a time. The pure-Python archive is
    opened lazily so CLI-only archives never parse it in the workers.
    """

    def __init__(
        self,
        path: str,
        password: Optional[bytes],
        limits: Limits,
        use_aes: bool,
        cli_command: Optional[str],
        cli_timeout: float,
        native_infos: Optional[Dict[int, Any]] = None,
    ) -> None:
        self.path = path
        self.password = password
        self.limits = limits
        self.use_aes = use_aes
        self.cli_command = cli_command
        self.cli_timeout = cli_timeout
        self.native_infos = native_infos
        self.archive: Any = None
        self.fallbacks = 0

    def _get_archive(self) -> Any:
        if self.archive is None:
            self.archive = _open_zip_archive(self.path, self.use_aes, self.password)
        return self.archive

    def _native_info(self, info: zipfile.ZipInfo) -> Any:
        """Map a stdlib ZipInfo to pyzipper's own class when needed.

        Matching by header offset preserves duplicate member names.
        """
        if not self.use_aes or self.native_infos is None:
            return info
        return self.native_infos.get(info.header_offset, info)

    def _is_zipcrypto(self, info: zipfile.ZipInfo) -> bool:
        return bool(
            (info.flag_bits & 0x1)
            and info.compress_type != ZIP_AES_COMPRESS_TYPE
        )

    def read(self, info: zipfile.ZipInfo) -> Member:
        if (
            self.cli_command is not None
            and self.password is not None
            and self._is_zipcrypto(info)
        ):
            member = _read_zip_member_cli(
                self.cli_command,
                self.path,
                info,
                self.password,
                self.limits,
                self.cli_timeout,
            )
            if member is not None:
                return member
            self.fallbacks += 1
        return _read_zip_member_python(
            self._get_archive(), self._native_info(info), self.password, self.limits
        )

    def close(self) -> None:
        if self.archive is not None:
            try:
                self.archive.close()
            except Exception:
                pass
            self.archive = None


def iter_zip(
    path: str,
    password: Optional[bytes],
    limits: Limits,
    progress: Any = None,
    readers: int = 1,
    zip_command: str = "unzip",
    zip_timeout: float = 600.0,
) -> Iterator[Member]:
    global _ZIP_CLI_WARNED, _ZIP_SLOW_WARNED

    try:
        with zipfile.ZipFile(path, "r") as probe:
            infos = probe.infolist()
    except (zipfile.BadZipFile, OSError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"bad zip: {type(exc).__name__}: {exc}") from exc

    has_aes = any(info.compress_type == ZIP_AES_COMPRESS_TYPE for info in infos)
    native_infos: Optional[Dict[int, Any]] = None
    if has_aes:
        try:
            import pyzipper  # type: ignore
        except ImportError as exc:
            raise ArchiveError("AES-encrypted zip requires pyzipper") from exc
        # pyzipper only accepts its own AESZipInfo subclass in open(). Build
        # the mapping once here and share it with every reader thread.
        try:
            with pyzipper.AESZipFile(path, "r") as probe:
                native_infos = {
                    entry.header_offset: entry for entry in probe.infolist()
                }
        except (zipfile.BadZipFile, OSError, UnicodeDecodeError) as exc:
            raise ArchiveError(f"cannot list AES zip: {type(exc).__name__}: {exc}") from exc
    has_zipcrypto = any(
        (info.flag_bits & 0x1) and info.compress_type != ZIP_AES_COMPRESS_TYPE
        for info in infos
    )

    cli_command: Optional[str] = None
    if has_zipcrypto:
        cli_command = _resolve_zip_command(zip_command)
        if cli_command is not None:
            if not _ZIP_CLI_WARNED:
                print(
                    "WARNING: ZipCrypto members are decrypted with the external "
                    f"'{os.path.basename(cli_command)}' command; the archive "
                    "password is passed on its command line and is visible in the "
                    "process list.",
                    file=sys.stderr,
                )
                _ZIP_CLI_WARNED = True
        elif not _ZIP_SLOW_WARNED:
            print(
                "WARNING: this zip uses ZipCrypto (zip -P) encryption and no "
                f"'{zip_command or 'zip command'}' executable was found; "
                "decryption falls back to pure Python (~5 MB/s). Install "
                "Info-ZIP, pass --zip-command, or repack the archive as AES/7z.",
                file=sys.stderr,
            )
            _ZIP_SLOW_WARNED = True

    if len(infos) > limits.max_members:
        raise ArchiveAbort(f"archive has {len(infos)} members (limit {limits.max_members})")
    if progress is not None:
        progress.total = sum(1 for info in infos if not info.is_dir())
        progress.refresh()

    candidates: List[zipfile.ZipInfo] = []
    for info in infos:
        name = sanitize_name(info.filename)
        if info.is_dir():
            continue
        if zip_is_symlink(info):
            yield Member(name, info.file_size, skip_reason="symlink")
            continue
        if info.file_size > limits.max_file_size:
            yield Member(name, info.file_size, skip_reason="too_large")
            continue
        if (
            info.file_size >= MIB
            and info.compress_size > 0
            and info.file_size / info.compress_size > limits.max_compression_ratio
        ):
            yield Member(name, info.file_size, skip_reason="compression_ratio")
            continue
        candidates.append(info)

    if not candidates:
        return

    workers = max(1, min(readers, len(candidates)))
    total = 0

    def account(member: Member) -> Member:
        nonlocal total
        if member.data is not None:
            total += member.size
            if total > limits.max_total_bytes:
                raise ArchiveAbort(
                    f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
                )
        return member

    def make_reader() -> _ZipReader:
        return _ZipReader(
            path,
            password,
            limits,
            has_aes,
            cli_command,
            zip_timeout,
            native_infos,
        )

    def report_fallbacks(count: int) -> None:
        if count:
            print(
                f"WARNING: {count} ZipCrypto member(s) could not be read by the "
                "external command and fell back to pure-Python decryption",
                file=sys.stderr,
            )

    if workers == 1:
        reader = make_reader()
        try:
            for info in candidates:
                yield account(reader.read(info))
        finally:
            reader.close()
            report_fallbacks(reader.fallbacks)
        return

    readers_pool = [make_reader() for _ in range(workers)]
    sessions: "queue.Queue[_ZipReader]" = queue.Queue()
    for reader in readers_pool:
        sessions.put(reader)

    def read_task(info: zipfile.ZipInfo) -> Member:
        reader = sessions.get()
        try:
            return reader.read(info)
        finally:
            sessions.put(reader)

    pool = ThreadPoolExecutor(max_workers=workers)
    pending: "deque[Tuple[Future, int]]" = deque()
    pending_bytes = 0
    source = iter(candidates)
    current = next(source, None)
    try:
        while current is not None or pending:
            while current is not None and len(pending) < workers:
                if pending and pending_bytes + current.file_size > PREFETCH_MAX_BYTES:
                    break
                pending.append((pool.submit(read_task, current), current.file_size))
                pending_bytes += current.file_size
                current = next(source, None)
            future, declared_size = pending.popleft()
            pending_bytes -= declared_size
            yield account(future.result())
    finally:
        pool.shutdown(wait=True)
        fallbacks = 0
        for reader in readers_pool:
            reader.close()
            fallbacks += reader.fallbacks
        report_fallbacks(fallbacks)


def iter_tar_gz(path: str, limits: Limits) -> Iterator[Member]:
    try:
        raw = open(path, "rb")
    except OSError as exc:
        raise ArchiveError(f"cannot open archive: {exc}") from exc
    with raw:
        try:
            archive = tarfile.open(fileobj=raw, mode="r|gz")
        except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
            raise ArchiveError(f"bad tar.gz: {type(exc).__name__}: {exc}") from exc
        with archive:
            count = 0
            total = 0
            declared_total = 0
            try:
                for member in archive:
                    count += 1
                    if count > limits.max_members:
                        raise ArchiveAbort(
                            f"archive has more than {limits.max_members} members"
                        )
                    name = sanitize_name(member.name)
                    if member.isdir():
                        continue
                    if member.issym() or member.islnk():
                        yield Member(name, 0, skip_reason="link")
                        continue
                    if not member.isfile():
                        yield Member(name, 0, skip_reason="not_regular_file")
                        continue
                    if member.size > limits.max_file_size:
                        # In streaming mode we cannot cheaply discard a huge member.
                        raise ArchiveAbort(f"declared member too large: {name}")
                    declared_total += member.size
                    if declared_total > limits.max_total_bytes:
                        raise ArchiveAbort(
                            f"declared archive size exceeds max_total_bytes ({limits.max_total_bytes})"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        yield Member(name, member.size, error_reason="no member data")
                        continue
                    try:
                        data = read_capped(handle, limits.max_file_size)
                    except (tarfile.TarError, OSError, EOFError, zlib.error, ArchiveAbort) as exc:
                        raise ArchiveAbort(f"read error at {name}: {exc}") from exc
                    total += len(data)
                    if total > limits.max_total_bytes:
                        raise ArchiveAbort(
                            f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
                        )
                    yield Member(name, len(data), data=data)
            except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
                raise ArchiveError(f"tar.gz read error: {type(exc).__name__}: {exc}") from exc


def iter_7z_py7zr(
    path: str,
    password: Optional[bytes],
    limits: Limits,
    progress: Any = None,
) -> Iterator[Member]:
    try:
        import py7zr  # type: ignore
        from py7zr.io import BytesIOFactory  # type: ignore
    except ImportError as exc:
        raise ArchiveError("7z archives require py7zr>=1.1,<2") from exc

    password_str: Optional[str] = None
    if password is not None:
        try:
            password_str = password.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArchiveError("7z passwords must be UTF-8") from exc

    try:
        archive = py7zr.SevenZipFile(path, mode="r", password=password_str)
    except Exception as exc:  # py7zr raises several archive/password exceptions
        raise ArchiveError(f"bad 7z: {type(exc).__name__}: {exc}") from exc

    with archive:
        try:
            infos = archive.list()
        except Exception as exc:
            raise ArchiveError(f"cannot list 7z: {type(exc).__name__}: {exc}") from exc
        if len(infos) > limits.max_members:
            raise ArchiveAbort(f"archive has {len(infos)} members (limit {limits.max_members})")
        if progress is not None:
            progress.total = sum(1 for info in infos if not info.is_directory)
            progress.refresh()

        total = 0
        batch: List[Any] = []
        batch_bytes = 0
        seen_names: set[str] = set()

        def flush(current: List[Any]) -> Iterator[Member]:
            nonlocal total
            if not current:
                return
            factory = BytesIOFactory(limits.max_file_size + 1)
            try:
                archive.reset()
                archive.extract(
                    targets=[info.filename for info in current],
                    factory=factory,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for info in current:
                    yield Member(sanitize_name(info.filename), info.uncompressed, error_reason=reason)
                return
            for info in current:
                name = sanitize_name(info.filename)
                try:
                    product = factory.get(info.filename)
                    product.seek(0)
                    data = product.read(limits.max_file_size + 2)
                except KeyError:
                    yield Member(name, info.uncompressed, error_reason="missing extracted member")
                    continue
                if len(data) > limits.max_file_size:
                    yield Member(name, info.uncompressed, skip_reason="too_large")
                    continue
                total += len(data)
                if total > limits.max_total_bytes:
                    raise ArchiveAbort(
                        f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
                    )
                yield Member(name, len(data), data=data)

        for info in infos:
            name = sanitize_name(info.filename)
            if info.is_directory:
                continue
            if info.is_symlink:
                yield Member(name, 0, skip_reason="symlink")
                continue
            if not info.is_file:
                yield Member(name, 0, skip_reason="not_regular_file")
                continue
            if info.filename in seen_names:
                yield Member(name, info.uncompressed, error_reason="duplicate member name")
                continue
            seen_names.add(info.filename)
            if info.uncompressed > limits.max_file_size:
                yield Member(name, info.uncompressed, skip_reason="too_large")
                continue
            if (
                info.uncompressed >= MIB
                and info.compressed
                and info.uncompressed / info.compressed > limits.max_compression_ratio
            ):
                yield Member(name, info.uncompressed, skip_reason="compression_ratio")
                continue
            if batch and (
                len(batch) >= BATCH_MAX_MEMBERS
                or batch_bytes + info.uncompressed > BATCH_BUDGET_BYTES
            ):
                yield from flush(batch)
                batch = []
                batch_bytes = 0
            batch.append(info)
            batch_bytes += info.uncompressed

        yield from flush(batch)


def find_7z_command(command: str) -> str:
    if os.path.sep in command or "/" in command:
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return command
        raise ArchiveError(f"7z command is not executable: {command}")
    candidates = [command]
    for fallback in ("7z", "7zz"):
        if fallback not in candidates:
            candidates.append(fallback)
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
        local = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.isfile(local) and os.access(local, os.X_OK):
            return local
    raise ArchiveError(
        f"{command} not found; install p7zip/7-Zip or use --7z-backend py7zr"
    )


def run_7z_command(
    command: Sequence[str], timeout: float, password: Optional[str]
) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout if timeout > 0 else None,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ArchiveError(f"7z command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ArchiveError(f"7z command timed out after {timeout} seconds") from exc
    stdout = completed.stdout.decode("utf-8", "replace")
    stderr = completed.stderr.decode("utf-8", "replace")
    if password:
        stderr = stderr.replace(password, "***")
    return completed.returncode, stdout, stderr


def parse_7z_slt(output: str) -> List[Dict[str, str]]:
    lines = output.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("----------"):
            start = index + 1
            break
    if start is None:
        return []
    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in lines[start:]:
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key.strip()] = value
    if current:
        entries.append(current)
    return entries


def entry_size(entry: Dict[str, str]) -> int:
    try:
        return int(entry.get("Size", "0").strip() or 0)
    except ValueError:
        return 0


def entry_kind(entry: Dict[str, str]) -> Tuple[bool, bool]:
    attributes = entry.get("Attributes", "")
    parts = attributes.split()
    unix_mode = parts[-1] if parts else ""
    is_dir = (
        attributes.startswith("D")
        or unix_mode.startswith("d")
        or entry.get("Folder") == "+"
    )
    is_symlink = unix_mode.startswith("l")
    return is_dir, is_symlink


def secure_extracted_tree(root: str) -> None:
    """Remove executable bits before reading untrusted extracted files."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames):
            full_path = os.path.join(dirpath, name)
            if os.path.islink(full_path):
                dirnames.remove(name)
                continue
            try:
                os.chmod(full_path, 0o700)
            except OSError:
                pass
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            if os.path.islink(full_path):
                continue
            try:
                os.chmod(full_path, 0o600)
            except OSError:
                pass


def iter_7z_system(
    path: str,
    password: Optional[bytes],
    limits: Limits,
    progress: Any = None,
    command: str = "7za",
    extract_base: Optional[str] = None,
    keep: bool = False,
    timeout: float = 600.0,
) -> Iterator[Member]:
    """Extract with the system 7z CLI into a temporary directory.

    The archive is listed first so member/size/compression limits are enforced
    before anything is written. Only selected members are extracted via a
    ``@listfile``. The temporary directory is removed afterward unless
    ``keep`` is true.
    """
    if not os.path.exists(path):
        raise ArchiveError(f"archive does not exist: {path}")
    seven_zip = find_7z_command(command)

    password_str: Optional[str] = None
    if password is not None:
        try:
            password_str = password.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArchiveError("7z passwords must be UTF-8") from exc

    list_cmd = [seven_zip, "l", "-slt", "-bd", path]
    if password_str:
        list_cmd.append("-p" + password_str)
    returncode, listing, stderr = run_7z_command(list_cmd, timeout, password_str)
    if returncode != 0:
        raise ArchiveError(
            f"7z listing failed (exit {returncode}): {stderr.strip()[:500]}"
        )
    entries = parse_7z_slt(listing)
    if not entries:
        raise ArchiveError("7z listing returned no members")

    non_dir = [entry for entry in entries if not entry_kind(entry)[0]]
    if len(non_dir) > limits.max_members:
        raise ArchiveAbort(f"archive has {len(non_dir)} members (limit {limits.max_members})")
    if progress is not None:
        progress.total = len(non_dir)
        progress.refresh()

    targets: List[str] = []
    target_names: set[str] = set()
    selected_bytes = 0
    for entry in entries:
        name = sanitize_name(entry.get("Path", ""))
        size = entry_size(entry)
        is_dir, is_symlink = entry_kind(entry)
        if is_dir:
            continue
        if is_symlink:
            yield Member(name, size, skip_reason="symlink")
            continue
        raw_name = entry.get("Path", "")
        normalized = raw_name.replace("\\", "/")
        if (
            not raw_name
            or normalized.startswith("/")
            or any(part == ".." for part in normalized.split("/"))
        ):
            yield Member(name, size, error_reason="unsafe archive path")
            continue
        if "\n" in raw_name or "\r" in raw_name:
            yield Member(name, size, error_reason="member name contains newline")
            continue
        if normalized in target_names:
            yield Member(name, size, error_reason="duplicate member name")
            continue
        target_names.add(normalized)
        if entry.get("Encrypted") == "+" and not password_str:
            yield Member(name, size, error_reason="password required for encrypted member")
            continue
        if size > limits.max_file_size:
            yield Member(name, size, skip_reason="too_large")
            continue
        try:
            packed = int(entry.get("Packed Size", "0").strip() or 0)
        except ValueError:
            packed = 0
        if size >= MIB and packed > 0 and size / packed > limits.max_compression_ratio:
            yield Member(name, size, skip_reason="compression_ratio")
            continue
        if selected_bytes + size > limits.max_total_bytes:
            yield Member(name, size, skip_reason="total_budget")
            continue
        targets.append(raw_name)
        selected_bytes += size

    if not targets:
        return

    base = extract_base or tempfile.gettempdir()
    os.makedirs(base, mode=0o700, exist_ok=True)
    extract_dir = tempfile.mkdtemp(prefix="ember_7z_", dir=base)
    list_path: Optional[str] = None
    try:
        fd, list_path = tempfile.mkstemp(prefix="ember_7z_targets_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for target in targets:
                handle.write(target + "\n")

        extract_cmd = [
            seven_zip,
            "x",
            path,
            "@" + list_path,
            "-o" + extract_dir,
            "-y",
            "-bd",
            "-spd",
            "-scsUTF-8",
        ]
        if password_str:
            extract_cmd.append("-p" + password_str)
        returncode, _stdout, stderr = run_7z_command(
            extract_cmd, timeout, password_str
        )
        if returncode not in (0, 1):
            detail = stderr.strip()[:500]
            hint = ""
            if "E_FAIL" in detail:
                hint = (
                    "; E_FAIL is generic: check the password and archive "
                    "integrity, the compression method (try a newer 7zz via "
                    "--7z-command, or --7z-backend py7zr), and free space in "
                    "the extraction directory"
                )
            raise ArchiveError(
                f"7z extraction failed (exit {returncode}, "
                f"{len(targets)} selected members): {detail}{hint}"
            )

        secure_extracted_tree(extract_dir)
        expected = {os.path.normpath(target.replace("\\", "/")) for target in targets}
        found: set[str] = set()
        total = 0
        for dirpath, dirnames, filenames in os.walk(extract_dir, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if not os.path.islink(os.path.join(dirpath, name))
            ]
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                relative = os.path.relpath(full_path, extract_dir)
                normalized = os.path.normpath(relative.replace("\\", "/"))
                found.add(normalized)
                name = sanitize_name(relative)
                if normalized not in expected:
                    yield Member(name, 0, skip_reason="unexpected_extracted_file")
                    continue
                if os.path.islink(full_path):
                    yield Member(name, 0, skip_reason="symlink")
                    continue
                try:
                    info = os.lstat(full_path)
                except OSError as exc:
                    yield Member(name, 0, error_reason=f"{type(exc).__name__}: {exc}")
                    continue
                if not stat.S_ISREG(info.st_mode):
                    yield Member(name, info.st_size, skip_reason="not_regular_file")
                    continue
                if info.st_size > limits.max_file_size:
                    yield Member(name, info.st_size, skip_reason="too_large")
                    continue
                try:
                    with open(full_path, "rb") as handle:
                        data = read_capped(handle, limits.max_file_size)
                except (OSError, ArchiveAbort) as exc:
                    yield Member(
                        name, info.st_size, error_reason=f"{type(exc).__name__}: {exc}"
                    )
                    continue
                total += len(data)
                if total > limits.max_total_bytes:
                    raise ArchiveAbort(
                        f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
                    )
                yield Member(name, len(data), data=data)

        for missing in sorted(expected - found):
            yield Member(sanitize_name(missing), 0, error_reason="not extracted")
    finally:
        if list_path:
            try:
                os.unlink(list_path)
            except OSError:
                pass
        if keep:
            print(
                f"WARNING: kept extracted 7z samples in {extract_dir}",
                file=sys.stderr,
            )
        else:
            shutil.rmtree(extract_dir, ignore_errors=True)


def iter_7z(
    path: str,
    password: Optional[bytes],
    limits: Limits,
    progress: Any = None,
    backend: str = "py7zr",
    command: str = "7za",
    extract_base: Optional[str] = None,
    keep: bool = False,
    timeout: float = 600.0,
) -> Iterator[Member]:
    global _SYSTEM_7Z_WARNED
    if backend == "py7zr":
        yield from iter_7z_py7zr(path, password, limits, progress)
    else:
        if not _SYSTEM_7Z_WARNED:
            print(
                "WARNING: --7z-backend system extracts selected samples to a "
                "temporary directory; use --7z-backend py7zr for in-memory-only "
                "processing.",
                file=sys.stderr,
            )
            _SYSTEM_7Z_WARNED = True
        yield from iter_7z_system(
            path,
            password,
            limits,
            progress,
            command,
            extract_base,
            keep,
            timeout,
        )


def iter_benign_dir(root: str, limits: Limits) -> Iterator[Member]:
    total = 0
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            relative = os.path.relpath(full_path, root)
            name = sanitize_name(relative)
            try:
                info = os.lstat(full_path)
            except OSError as exc:
                yield Member(name, 0, error_reason=f"{type(exc).__name__}: {exc}")
                continue
            if not stat.S_ISREG(info.st_mode):
                yield Member(name, info.st_size, skip_reason="not_regular_file")
                continue
            count += 1
            if count > limits.max_members:
                raise ArchiveAbort(f"directory has more than {limits.max_members} files")
            if info.st_size > limits.max_file_size:
                yield Member(name, info.st_size, skip_reason="too_large")
                continue
            try:
                with open(full_path, "rb") as handle:
                    data = read_capped(handle, limits.max_file_size)
            except (OSError, ArchiveAbort) as exc:
                yield Member(name, info.st_size, error_reason=f"{type(exc).__name__}: {exc}")
                continue
            total += len(data)
            if total > limits.max_total_bytes:
                raise ArchiveAbort(f"directory exceeds max_total_bytes ({limits.max_total_bytes})")
            yield Member(name, len(data), data=data)


class OutputSink:
    def __init__(
        self,
        path: str,
        lock: bool = False,
        sync_every: int = DEFAULT_FSYNC_EVERY,
    ) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, mode=0o700, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self.fd = os.open(path, flags, 0o600)
        self.sync_every = sync_every
        try:
            os.fchmod(self.fd, 0o600)
        except OSError:
            pass
        if lock:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(self.fd)
                self.fd = -1
                raise SystemExit(f"output is locked by another process: {path} ({exc})") from exc
        self.lines = 0

    def _write(self, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            try:
                written = os.write(self.fd, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError(errno.EIO, f"short write to {self.path}")
            view = view[written:]
        self.lines += 1
        if self.sync_every > 0 and self.lines % self.sync_every == 0:
            os.fsync(self.fd)

    def write_line(self, line: str) -> None:
        self._write(line.encode("utf-8") + b"\n")

    def write_line_bytes(self, line: bytes) -> None:
        self._write(line + b"\n")

    def write_obj(self, obj: Dict[str, Any]) -> None:
        try:
            line = json.dumps(obj, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            line = json.dumps(
                {
                    "error": "serialization_failed",
                    "detail": f"{type(exc).__name__}: {exc}"[:300],
                },
                ensure_ascii=True,
            )
        self.write_line(line)

    def close(self) -> None:
        if self.fd >= 0:
            try:
                os.fsync(self.fd)
            finally:
                os.close(self.fd)
                self.fd = -1


class Sinks:
    def __init__(
        self,
        output: str,
        errors: str,
        skipped: str,
        max_record_bytes: int,
        sync_every: int = DEFAULT_FSYNC_EVERY,
    ) -> None:
        self.max_record_bytes = max_record_bytes
        self.output = OutputSink(output, lock=True, sync_every=sync_every)
        self.errors = OutputSink(errors, sync_every=sync_every)
        self.skipped = OutputSink(skipped, sync_every=sync_every)

    def write_line_bytes(self, line: bytes) -> None:
        self.output.write_line_bytes(line)

    def close(self) -> None:
        for sink in (self.output, self.errors, self.skipped):
            sink.close()


def scan_completed(path: str, repair: bool) -> Dict[bytes, Optional[int]]:
    """Return {sha256_bytes: label} for records already in the output file."""
    completed: Dict[bytes, Optional[int]] = {}
    if not os.path.lexists(path):
        return completed
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"output is not a regular file: {path}")
    last_good = 0
    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if repair:
                    os.truncate(path, last_good)
                    print(
                        f"WARNING: truncated incomplete trailing line in {path}",
                        file=sys.stderr,
                    )
                    break
                raise SystemExit(
                    f"{path} ends with an incomplete line; rerun with --repair "
                    "or remove the partial line"
                )
            line_start = last_good
            last_good += len(line)
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if repair:
                    os.truncate(path, line_start)
                    print(
                        f"WARNING: truncated invalid line in {path} ({exc})",
                        file=sys.stderr,
                    )
                    break
                raise SystemExit(
                    f"invalid JSON in {path} at byte {line_start}: {exc}; "
                    "rerun with --repair to truncate it"
                ) from exc
            sha = obj.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                continue
            try:
                digest = bytes.fromhex(sha)
            except ValueError:
                continue
            label = obj.get("label")
            completed[digest] = label if isinstance(label, int) else None
    return completed


def load_password(password_file: Optional[str], password_env: Optional[str]) -> Optional[bytes]:
    if password_file:
        try:
            info = os.stat(password_file)
        except OSError as exc:
            raise SystemExit(f"cannot read password file: {exc}") from exc
        if stat.S_IMODE(info.st_mode) & 0o077:
            print(
                "WARNING: password file is readable by group/other; chmod 600 it",
                file=sys.stderr,
            )
        with open(password_file, "rb") as handle:
            password = handle.read().strip(b"\r\n")
        if not password:
            raise SystemExit("password file is empty")
        return password
    if password_env:
        value = os.environ.get(password_env)
        if value is None:
            raise SystemExit(f"environment variable {password_env} is not set")
        return value.encode("utf-8")
    return None


def set_child_limits(limits: Limits, cpu_timeout: Optional[float] = None) -> None:
    try:
        import resource  # type: ignore
    except ImportError:
        return

    def apply(kind: int, value: int, allow_zero: bool = False) -> None:
        if value < 0 or (value == 0 and not allow_zero):
            return
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass

    timeout = limits.cpu_timeout if cpu_timeout is None else cpu_timeout
    if timeout > 0:
        apply(resource.RLIMIT_CPU, int(timeout) + 1)
    if limits.memory_per_sample > 0:
        apply(resource.RLIMIT_AS, limits.memory_per_sample)
    apply(resource.RLIMIT_FSIZE, 0, allow_zero=True)
    apply(resource.RLIMIT_CORE, 0, allow_zero=True)
    apply(resource.RLIMIT_NOFILE, 64)


def _redirect_stdio_to_devnull() -> None:
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        pass


def worker_main(conn: Any, limits: Limits, cpu_timeout: float) -> None:
    """Long-lived feature extraction worker.

    The extractor is built once and reused for every task. Results are
    serialized here, so the supervisor only writes the finished line. The
    supervisor kills and replaces this process on a timeout or crash.
    """
    try:
        sys.dont_write_bytecode = True
        set_child_limits(limits, cpu_timeout)
        _redirect_stdio_to_devnull()
        extractor: Any = None
        init_error = ""
        try:
            from thrember.features import PEFeatureExtractor

            extractor = PEFeatureExtractor()
        except Exception as exc:  # noqa: BLE001 - report to every task
            init_error = f"{type(exc).__name__}: {exc}"[:500]
        while True:
            task = conn.recv()
            if task is None:
                break
            task_id, bytez, sha256, label, archive, member = task
            if extractor is None:
                conn.send((task_id, "error", "worker_init", init_error))
                continue
            try:
                features = extractor.raw_features(bytez)
            except Exception as exc:  # noqa: BLE001 - one bad sample only
                conn.send((task_id, "error", type(exc).__name__, str(exc)[:512]))
                continue
            record = dict(features)
            record["sha256"] = sha256
            record["label"] = label
            record["source"] = {"archive": archive, "member": member}
            try:
                line = json.dumps(
                    record,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                conn.send(
                    (task_id, "serialize", f"{type(exc).__name__}: {exc}"[:300])
                )
                continue
            # ensure_ascii=True means the character count is the byte count.
            if len(line) > limits.max_record_bytes:
                conn.send((task_id, "too_large", len(line)))
                continue
            conn.send((task_id, "ok", line.encode("ascii")))
    except (EOFError, OSError):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def make_context():  # type: ignore[no-untyped-def]
    import multiprocessing as mp

    try:
        ctx = mp.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(["thrember.features"])
        except Exception:
            pass
        return ctx
    except ValueError:
        return mp.get_context("spawn")


class WorkerPool:
    """Persistent feature-extraction workers with per-sample supervision.

    Each worker loads the extractor once and then serves tasks until it is
    recycled (``recycle_tasks``), killed for exceeding a sample's deadline, or
    dies. Dead or recycled workers are replaced lazily on the next dispatch, so
    the cost is paid once per worker rather than once per sample.

    The supervisor never sends a second task to a busy worker; the number of
    in-flight samples is therefore bounded by the pool size.
    """

    def __init__(
        self,
        ctx: Any,
        size: int,
        limits: Limits,
        recycle_tasks: int,
    ) -> None:
        self.ctx = ctx
        self.limits = limits
        self.recycle_tasks = recycle_tasks
        self.workers = [Worker(index) for index in range(size)]
        self._next_task_id = 0
        # Spawn everything up front so the (relatively slow) interpreter and
        # extractor imports overlap instead of being paid one at a time on the
        # first dispatch to each worker.
        for worker in self.workers:
            self._spawn(worker)

    def new_task_id(self) -> int:
        self._next_task_id += 1
        return self._next_task_id

    def _kill(self, worker: Worker) -> None:
        proc, conn = worker.proc, worker.conn
        worker.proc = None
        worker.conn = None
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.join(1.0)
            except Exception:
                pass
            try:
                if proc.is_alive():
                    proc.kill()
                    proc.join(1.0)
            except Exception:
                pass
        _close_conn(conn)

    def _spawn(self, worker: Worker) -> bool:
        self._kill(worker)
        # A per-task CPU rlimit only makes sense when the process handles one
        # sample (--isolate); otherwise CPU time would accumulate and kill a
        # healthy worker. Wall-clock timeouts still apply in both modes.
        cpu_timeout = self.limits.cpu_timeout if self.recycle_tasks == 1 else 0.0
        recv_conn = send_conn = None
        try:
            # Duplex: the supervisor sends tasks and receives results on its end.
            recv_conn, send_conn = self.ctx.Pipe(duplex=True)
            proc = self.ctx.Process(
                target=worker_main,
                args=(send_conn, self.limits, cpu_timeout),
                daemon=True,
            )
            proc.start()
        except Exception:  # noqa: BLE001 - spawn failure is per-sample
            _close_conn(recv_conn)
            _close_conn(send_conn)
            return False
        _close_conn(send_conn)
        worker.proc = proc
        worker.conn = recv_conn
        worker.tasks_done = 0
        worker.task = None
        return True

    def idle_worker(self) -> Optional[Worker]:
        """Return an idle worker, replacing dead/exhausted ones as needed."""
        for worker in self.workers:
            if worker.task is not None:
                continue
            if worker.proc is not None and not worker.proc.is_alive():
                self._kill(worker)
            if worker.proc is None or (
                self.recycle_tasks > 0 and worker.tasks_done >= self.recycle_tasks
            ):
                if not self._spawn(worker):
                    continue
            return worker
        return None

    def busy(self) -> bool:
        return any(worker.task is not None for worker in self.workers)

    def next_timeout(self) -> Optional[float]:
        deadlines = [
            worker.task.deadline
            for worker in self.workers
            if worker.task is not None
        ]
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - time.monotonic())

    def dispatch(self, worker: Worker, task: Task, data: bytes) -> bool:
        worker.task = task
        try:
            worker.conn.send(
                (
                    task.task_id,
                    data,
                    task.sha256,
                    task.label,
                    task.archive,
                    task.member,
                )
            )
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._kill(worker)
            worker.task = None
            _record_failure(
                task, "crash", f"send failed: {type(exc).__name__}: {exc}"
            )
            return False
        return True

    def wait_one(self, timeout: Optional[float] = None) -> None:
        from multiprocessing.connection import wait as mp_wait

        busy = [worker for worker in self.workers if worker.task is not None]
        if not busy:
            return
        ready = mp_wait([worker.conn for worker in busy], timeout=timeout)
        if not ready:
            now = time.monotonic()
            for worker in busy:
                if worker.task is not None and now >= worker.task.deadline:
                    task = worker.task
                    self._kill(worker)
                    worker.task = None
                    _record_failure(task, "timeout", "per-sample wall-clock timeout")
            return

        by_conn = {worker.conn: worker for worker in busy}
        for conn in ready:
            worker = by_conn.get(conn)
            if worker is not None and worker.task is not None:
                self._complete(worker)

    def _complete(self, worker: Worker) -> None:
        task = worker.task
        assert task is not None
        try:
            result = worker.conn.recv()
        except (EOFError, OSError) as exc:
            self._kill(worker)
            worker.task = None
            _record_failure(task, "crash", f"{type(exc).__name__}: {exc}")
            return

        if (
            not isinstance(result, tuple)
            or len(result) < 2
            or result[0] != task.task_id
        ):
            self._kill(worker)
            worker.task = None
            _record_failure(task, "crash", "invalid result from worker")
            return

        worker.task = None
        worker.tasks_done += 1
        status = result[1]
        if status == "ok" and len(result) >= 3 and isinstance(result[2], bytes):
            _record_success(task, result[2])
        elif status == "too_large":
            size = result[2] if len(result) > 2 else 0
            _record_record_error(
                task,
                f"record is {size} bytes (max {self.limits.max_record_bytes})",
            )
        elif status == "serialize":
            reason = str(result[2]) if len(result) > 2 else "serialization failed"
            _record_record_error(task, reason[:300])
        else:
            reason = ": ".join(str(part) for part in result[2:]) or "worker error"
            _record_failure(task, "parse_error", reason[:1000])

    def drain(self) -> None:
        """Wait for every in-flight sample, enforcing per-sample deadlines."""
        while self.busy():
            self.wait_one(self.next_timeout())

    def abort(self) -> None:
        """Kill in-flight workers without recording results (fatal error path)."""
        for worker in self.workers:
            if worker.task is not None:
                self._kill(worker)
                worker.task = None

    def close(self) -> None:
        for worker in self.workers:
            if worker.proc is not None and worker.task is None:
                try:
                    worker.conn.send(None)
                except Exception:
                    pass
        for worker in self.workers:
            proc = worker.proc
            if proc is not None:
                try:
                    proc.join(1.0)
                except Exception:
                    pass
                try:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(1.0)
                except Exception:
                    pass
            _close_conn(worker.conn)
            worker.proc = None
            worker.conn = None
            worker.task = None


# The following globals are populated in main() so the supervisor callbacks
# can stay small and easy to reason about.
_SINKS: Optional[Sinks] = None
_SEEN: Optional[Dict[bytes, Optional[int]]] = None
_INFLIGHT: Optional[Dict[bytes, int]] = None
_STATS: Optional[Dict[str, int]] = None
_PROGRESS: Any = None


def make_progress(desc: str, enabled: bool):  # type: ignore[no-untyped-def]
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=None,
        desc=sanitize_name(desc, 40),
        unit="file",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
        maxinterval=5.0,
        disable=False,
    )


def _update_progress() -> None:
    if _PROGRESS is None or _STATS is None:
        return
    try:
        _PROGRESS.set_postfix(
            queued=_STATS["queued"],
            done=_STATS["extracted"],
            skip=_STATS["skipped"] + _STATS["skipped_not_pe"],
            dup=_STATS["duplicates"] + _STATS["label_conflicts"],
            err=(
                _STATS["timeouts"]
                + _STATS["crashes"]
                + _STATS["parse_errors"]
                + _STATS["record_errors"]
                + _STATS["archive_errors"]
            ),
        )
    except Exception:
        pass


def _close_conn(conn: Any) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _finish_inflight(item: Task) -> None:
    if _INFLIGHT is not None:
        _INFLIGHT.pop(item.digest, None)


def _record_success(item: Task, line: bytes) -> None:
    assert _SINKS is not None and _SEEN is not None and _STATS is not None
    _SINKS.write_line_bytes(line)
    _finish_inflight(item)
    _SEEN[item.digest] = item.label
    _STATS["extracted"] += 1
    if item.label == 0:
        _STATS["label_0"] += 1
    elif item.label == 1:
        _STATS["label_1"] += 1
    _update_progress()


def _record_record_error(item: Task, detail: str) -> None:
    assert _SINKS is not None and _SEEN is not None and _STATS is not None
    _SINKS.errors.write_obj(
        {
            "sha256": item.sha256,
            "archive": item.archive,
            "member": item.member,
            "error": "record_too_large_or_serialization_failed",
            "max_record_bytes": _SINKS.max_record_bytes,
            "message": detail[:300],
        }
    )
    _finish_inflight(item)
    _SEEN[item.digest] = item.label
    _STATS["record_errors"] += 1
    _update_progress()


def _record_failure(item: Task, kind: str, reason: str) -> None:
    assert _SINKS is not None and _SEEN is not None and _STATS is not None
    _SINKS.errors.write_obj(
        {
            "sha256": item.sha256,
            "archive": item.archive,
            "member": item.member,
            "error": kind,
            "message": reason,
        }
    )
    _finish_inflight(item)
    _SEEN[item.digest] = item.label
    if kind == "timeout":
        _STATS["timeouts"] += 1
    elif kind == "crash":
        _STATS["crashes"] += 1
    else:
        _STATS["parse_errors"] += 1
    _update_progress()


def process_members(
    members: Iterable[Member],
    archive_name: str,
    label: int,
    pool: WorkerPool,
    limits: Limits,
    seen: Dict[bytes, Optional[int]],
    stats: Dict[str, int],
    sinks: Sinks,
    progress: Any = None,
    allow_label_conflict: bool = False,
    require_pe: bool = True,
) -> None:
    global _PROGRESS, _INFLIGHT

    _PROGRESS = progress
    inflight: Dict[bytes, int] = {}
    _INFLIGHT = inflight
    drained = False
    try:
        for member in members:
            stats["members"] += 1
            if progress is not None:
                progress.update(1)
            _update_progress()
            if member.skip_reason:
                sinks.skipped.write_obj(
                    {
                        "archive": archive_name,
                        "member": member.name,
                        "size": member.size,
                        "reason": member.skip_reason,
                    }
                )
                stats["skipped"] += 1
                _update_progress()
                continue
            if member.error_reason or member.data is None:
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "member": member.name,
                        "size": member.size,
                        "error": "member_error",
                        "message": member.error_reason or "no data",
                    }
                )
                stats["parse_errors"] += 1
                _update_progress()
                continue

            data = member.data
            if require_pe and not is_pe(data):
                sinks.skipped.write_obj(
                    {
                        "archive": archive_name,
                        "member": member.name,
                        "size": len(data),
                        "reason": "not_pe",
                    }
                )
                stats["skipped_not_pe"] += 1
                _update_progress()
                continue

            if member.sha256 and len(member.sha256) == 64:
                sha256 = member.sha256
                try:
                    digest = bytes.fromhex(sha256)
                except ValueError:
                    digest = hashlib.sha256(data).digest()
                    sha256 = digest.hex()
            else:
                digest = hashlib.sha256(data).digest()
                sha256 = digest.hex()
            if digest in seen or digest in inflight:
                previous = seen.get(digest) if digest in seen else inflight.get(digest)
                if isinstance(previous, int) and previous != label:
                    message = (
                        f"label conflict: {sha256} is already recorded with label "
                        f"{previous}, but {archive_name}:{member.name} was requested "
                        f"as label {label}; the existing record was kept"
                    )
                    sinks.errors.write_obj(
                        {
                            "sha256": sha256,
                            "archive": archive_name,
                            "member": member.name,
                            "error": "label_conflict",
                            "message": message,
                        }
                    )
                    stats["label_conflicts"] += 1
                    if not allow_label_conflict:
                        raise SystemExit(
                            message
                            + ". Use a fresh output file, or pass "
                            "--allow-label-conflict to keep the first label."
                        )
                    print(f"WARNING: {message}", file=sys.stderr)
                else:
                    stats["duplicates"] += 1
                _update_progress()
                continue

            worker: Optional[Worker] = None
            while True:
                worker = pool.idle_worker()
                if worker is not None:
                    break
                if pool.busy():
                    pool.wait_one(pool.next_timeout())
                    continue
                # Every worker failed to start; this sample is not processed.
                sinks.errors.write_obj(
                    {
                        "sha256": sha256,
                        "archive": archive_name,
                        "member": member.name,
                        "error": "spawn_failed",
                        "message": "no worker process could be started",
                    }
                )
                seen[digest] = label
                stats["crashes"] += 1
                _update_progress()
                break
            if worker is None:
                continue

            task = Task(
                task_id=pool.new_task_id(),
                digest=digest,
                sha256=sha256,
                archive=archive_name,
                member=member.name,
                label=label,
                deadline=time.monotonic() + limits.timeout,
            )
            if pool.dispatch(worker, task, data):
                inflight[digest] = label
                stats["queued"] += 1
                _update_progress()

        pool.drain()
        drained = True
    except ArchiveError:
        # The archive hit a limit after some samples were already queued.
        # Finish those samples before reporting the archive-level error.
        pool.drain()
        drained = True
        raise
    finally:
        if not drained:
            pool.abort()
        _INFLIGHT = None
        _PROGRESS = None


def run(args: argparse.Namespace) -> int:
    global _SINKS, _SEEN, _STATS

    if not args.archive and not args.benign_archive and not args.benign_dir:
        raise SystemExit(
            "provide at least one --archive, --benign-archive, or --benign-dir"
        )

    archive_jobs: List[Tuple[str, int]] = []
    archive_labels: Dict[str, int] = {}

    def add_archive_job(path: str, label: int) -> None:
        key = os.path.abspath(path)
        if key in archive_labels:
            if archive_labels[key] != label:
                raise SystemExit(
                    f"archive specified with conflicting labels: {path} "
                    f"({archive_labels[key]} vs {label})"
                )
            return
        archive_labels[key] = label
        archive_jobs.append((path, label))

    for path in args.archive:
        add_archive_job(path, args.label)
    for path in args.benign_archive:
        add_archive_job(path, 0)

    output = os.path.abspath(args.output)
    errors = os.path.abspath(args.errors) if args.errors else output + ".errors.jsonl"
    skipped = os.path.abspath(args.skipped) if args.skipped else output + ".skipped.jsonl"

    password = load_password(args.password_file, args.password_env)

    if args.resume:
        seen = scan_completed(output, args.repair)
    else:
        seen = {}
        if os.path.exists(output) and os.path.getsize(output) > 0:
            raise SystemExit(
                f"output exists and --no-resume was given: {output}; "
                "remove it or use --resume"
            )

    limits = Limits(
        max_file_size=args.max_file_size,
        max_total_bytes=args.max_total_bytes,
        max_members=args.max_members,
        max_compression_ratio=args.max_compression_ratio,
        max_record_bytes=args.max_record_bytes,
        timeout=args.timeout_per_sample,
        cpu_timeout=args.timeout_per_sample,
        memory_per_sample=args.memory_per_sample,
    )

    sinks = Sinks(
        output, errors, skipped, args.max_record_bytes, sync_every=args.fsync_every
    )
    stats = {
        "archives": 0,
        "members": 0,
        "queued": 0,
        "extracted": 0,
        "skipped": 0,
        "skipped_not_pe": 0,
        "duplicates": 0,
        "timeouts": 0,
        "crashes": 0,
        "parse_errors": 0,
        "record_errors": 0,
        "label_conflicts": 0,
        "label_0": 0,
        "label_1": 0,
        "archive_errors": 0,
    }
    _SINKS = sinks
    _SEEN = seen
    _STATS = stats

    ctx = make_context()
    pool = WorkerPool(ctx, args.workers, limits, args.recycle_tasks)
    try:
        for archive_path, archive_label in archive_jobs:
            stats["archives"] += 1
            archive_name = sanitize_name(os.path.basename(archive_path))
            if not os.path.exists(archive_path):
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "error": "missing_archive",
                        "message": archive_path,
                    }
                )
                stats["archive_errors"] += 1
                continue
            if args.verbose:
                print(
                    f"processing {archive_path} with label {archive_label}",
                    file=sys.stderr,
                )
            bar = make_progress(archive_name, args.progress)
            try:
                fmt = detect_format(archive_path, args.format)
                if fmt == "zip":
                    members: Iterable[Member] = iter_zip(
                        archive_path,
                        password,
                        limits,
                        progress=bar,
                        readers=args.read_workers,
                        zip_command=args.zip_command,
                        zip_timeout=args.zip_timeout,
                    )
                elif fmt == "tar.gz":
                    members = iter_tar_gz(archive_path, limits)
                elif fmt == "7z":
                    members = iter_7z(
                        archive_path,
                        password,
                        limits,
                        progress=bar,
                        backend=args.seven_zip_backend,
                        command=args.seven_zip_command,
                        extract_base=args.seven_zip_extract_dir,
                        keep=args.keep_7z_extract,
                        timeout=args.seven_zip_timeout,
                    )
                else:  # pragma: no cover - argparse restricts this
                    raise ArchiveError(f"unsupported format: {fmt}")
                process_members(
                    members,
                    archive_name,
                    archive_label,
                    pool,
                    limits,
                    seen,
                    stats,
                    sinks,
                    bar,
                    args.allow_label_conflict,
                    require_pe=not args.include_non_pe,
                )
            except ArchiveError as exc:
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
                if args.verbose:
                    print(f"archive error: {archive_name}: {exc}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - untrusted archive parsers
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
                if args.verbose:
                    print(
                        f"unexpected archive error: {archive_name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            finally:
                if bar is not None:
                    bar.close()

        for benign_dir in args.benign_dir:
            stats["archives"] += 1
            name = sanitize_name(os.path.basename(os.path.abspath(benign_dir)))
            if args.verbose:
                print(f"processing {benign_dir} with label 0", file=sys.stderr)
            bar = make_progress(name, args.progress)
            try:
                process_members(
                    iter_benign_dir(benign_dir, limits),
                    name,
                    0,
                    pool,
                    limits,
                    seen,
                    stats,
                    sinks,
                    bar,
                    args.allow_label_conflict,
                    require_pe=not args.include_non_pe,
                )
            except ArchiveError as exc:
                sinks.errors.write_obj(
                    {
                        "archive": name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
            except Exception as exc:  # noqa: BLE001 - untrusted filesystem input
                sinks.errors.write_obj(
                    {
                        "archive": name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
            finally:
                if bar is not None:
                    bar.close()
    finally:
        pool.close()
        sinks.close()

    print(
        "summary: "
        + " ".join(f"{key}={value}" for key, value in stats.items()),
        file=sys.stderr,
    )

    if args.vectorize:
        vectorize_output(args, output)
    return 0


def vectorize_output(args: argparse.Namespace, output: str) -> None:
    import thrember
    from thrember.features import PEFeatureExtractor

    nrows = 0
    with open(output, "rb") as handle:
        for _ in handle:
            nrows += 1
    if nrows == 0:
        print("nothing to vectorize", file=sys.stderr)
        return
    data_dir = args.vectorize_dir or os.path.dirname(output)
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    extractor = PEFeatureExtractor()
    X_path = os.path.join(data_dir, f"X_{args.split}.dat")
    y_path = os.path.join(data_dir, f"y_{args.split}.dat")
    thrember.vectorize_subset(X_path, y_path, [Path(output)], extractor, nrows)
    print(
        f"vectorized {nrows} rows x {extractor.dim} dims -> {X_path}, {y_path}",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely extract EMBER2024 (feature version 3) raw features from files in "
            "zip/7z/tar.gz archives or a benign directory."
        )
    )
    parser.add_argument("--archive", action="append", default=[], help="archive path (repeatable)")
    parser.add_argument(
        "--benign-archive",
        action="append",
        default=[],
        help="archive path labelled 0 (repeatable; overrides --label)",
    )
    parser.add_argument(
        "--benign-dir",
        action="append",
        default=[],
        help="directory of loose files labelled 0 (repeatable)",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=(0, 1),
        default=1,
        help="label applied to --archive entries (default: 1)",
    )
    parser.add_argument(
        "--allow-label-conflict",
        action="store_true",
        help="keep the first label when the same sha256 appears with a different label",
    )
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--errors", help="error JSONL path (default: <output>.errors.jsonl)")
    parser.add_argument("--skipped", help="skipped JSONL path (default: <output>.skipped.jsonl)")
    parser.add_argument(
        "--format",
        choices=("auto", "zip", "7z", "tar.gz"),
        default="auto",
        help="archive format (default: sniff magic bytes)",
    )
    parser.add_argument(
        "--7z-backend",
        dest="seven_zip_backend",
        choices=("py7zr", "system"),
        default="py7zr",
        help="7z backend: py7zr reads in memory (default); system extracts "
        "selected samples with a 7z CLI into a temporary directory",
    )
    parser.add_argument(
        "--7z-command",
        dest="seven_zip_command",
        default="7za",
        help="7z executable for the system backend (default: 7za)",
    )
    parser.add_argument(
        "--7z-extract-dir",
        dest="seven_zip_extract_dir",
        help="base directory for temporary 7z extraction (default: system temp)",
    )
    parser.add_argument(
        "--keep-7z-extract",
        dest="keep_7z_extract",
        action="store_true",
        help="keep extracted 7z files (plaintext samples on disk; debugging only)",
    )
    parser.add_argument(
        "--7z-timeout",
        dest="seven_zip_timeout",
        type=float,
        default=600.0,
        help="timeout per 7z list/extract command in seconds (default: 600)",
    )
    password_group = parser.add_mutually_exclusive_group()
    password_group.add_argument("--password-file", help="file containing the archive password")
    password_group.add_argument("--password-env", help="environment variable with the password")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip sha256 values already present in the output (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="do not scan the output; requires a fresh/empty output file",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="truncate an incomplete/invalid trailing line during resume",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="persistent feature-extraction workers (default: min(4, cpu count))",
    )
    parser.add_argument(
        "--recycle-tasks",
        type=int,
        default=DEFAULT_RECYCLE_TASKS,
        help="replace each worker after this many samples; 0 never recycles "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="use a fresh process for every sample (strict isolation; slower)",
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=DEFAULT_READ_WORKERS,
        help="threads that decompress and hash zip members (default: %(default)s)",
    )
    parser.add_argument(
        "--zip-command",
        default="unzip",
        help="external command used to decrypt ZipCrypto (zip -P) members "
        "(default: unzip; executables named *7z* use 7z syntax; '' disables)",
    )
    parser.add_argument(
        "--zip-timeout",
        type=float,
        default=600.0,
        help="timeout per external zip decryption call in seconds (default: 600)",
    )
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=DEFAULT_FSYNC_EVERY,
        help="fsync the output every N records; 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--max-file-size",
        type=parse_size,
        default=DEFAULT_MAX_FILE_SIZE,
        help="maximum uncompressed size per member (default: 20M)",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=parse_size,
        default=DEFAULT_MAX_TOTAL_BYTES,
        help="maximum bytes read per archive/directory (default: 10G)",
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=DEFAULT_MAX_MEMBERS,
        help="maximum members per archive/directory (default: 100000)",
    )
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
        help="skip members whose declared expansion ratio exceeds this (default: 1000)",
    )
    parser.add_argument(
        "--max-record-bytes",
        type=parse_size,
        default=DEFAULT_MAX_RECORD_BYTES,
        help="maximum serialized feature record size (default: 16M)",
    )
    parser.add_argument(
        "--timeout-per-sample",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="wall-clock seconds per sample (default: 30)",
    )
    parser.add_argument(
        "--memory-per-sample",
        type=parse_size,
        default=DEFAULT_MEMORY_PER_SAMPLE,
        help="optional RLIMIT_AS per child in bytes (0 disables; default: 0)",
    )
    parser.add_argument(
        "--include-non-pe",
        action="store_true",
        help=(
            "also extract APK/ELF/PDF/other non-PE files; they still get the "
            "general/byte/string features, with PE-only blocks zeroed"
        ),
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="after extraction, vectorize with thrember.vectorize_subset (binary label only)",
    )
    parser.add_argument(
        "--vectorize-dir",
        help="directory for X_<split>.dat/y_<split>.dat (default: output directory)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "challenge"),
        default="train",
        help="subset name for vectorized output (default: train)",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="show per-archive progress on stderr (default: on)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable progress output",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-archive errors")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.read_workers < 1:
        parser.error("--read-workers must be >= 1")
    if args.recycle_tasks < 0:
        parser.error("--recycle-tasks must be >= 0")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be >= 0")
    if args.zip_timeout < 0:
        parser.error("--zip-timeout must be >= 0")
    if args.timeout_per_sample <= 0:
        parser.error("--timeout-per-sample must be > 0")
    if args.isolate:
        # One task per process: the per-task CPU rlimit becomes meaningful and
        # no state can leak between samples.
        args.recycle_tasks = 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
