from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any


def resolve_repository_path(
    project_root: str | Path,
    value: str | Path,
    *,
    must_exist: bool = True,
    expected_kind: str | None = None,
) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {value}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    if path.exists() and expected_kind == "file" and not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    if path.exists() and expected_kind == "directory" and not path.is_dir():
        raise ValueError(f"Expected a directory: {path}")
    return path


def repository_relative(project_root: str | Path, path: str | Path) -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path is outside project root: {path}")
    return str(resolved.relative_to(root)).replace("\\", "/")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def replace_json(path: str | Path, value: object, *, newline: str | None = "\n") -> None:
    """Replace mutable JSON output; use newline=None for legacy platform newlines.

    Immutable evidence must use atomic_write_json instead. This writer does not
    fsync; callers requiring durable session state retain their dedicated writer.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline=newline,
    )
    temporary.replace(target)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            yield value


def gzip_jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        for record in records:
            compressed.write(canonical_json_bytes(record) + b"\n")
    return output.getvalue()


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    allow_identical: bool = True,
) -> str:
    target = Path(path)
    digest = sha256_bytes(payload)
    if target.exists():
        if allow_identical and target.is_file() and sha256_file(target) == digest:
            return digest
        raise ValueError(f"Refusing to overwrite conflicting artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise ValueError(f"Conflicting temporary artifact exists: {temporary}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def atomic_write_json(
    path: str | Path,
    value: Mapping[str, object],
    *,
    allow_identical: bool = True,
) -> str:
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return atomic_write_bytes(path, payload, allow_identical=allow_identical)
