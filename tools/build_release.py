"""Build a deterministic, allowlisted source release bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_DIRECTORIES = (
    "src/posterior_memory_harness",
    "tests",
    "examples",
    "schemas",
    "benchmark",
    "tools",
)
INCLUDED_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CALIBRATION_DRIFT_MONITOR_ZH.md",
    "DOGFOOD_AUDIT_ZH.md",
    "IMPROVEMENT_TARGET_ZH.md",
    "LLM_AGENT_AUDIT_ZH.md",
    "RELEASE_NOTES_0.8.0.md",
    "RELEASE_NOTES_0.9.0.md",
    "SQLITE_SCHEMA_MIGRATION_ZH.md",
    "STRICT_INPUT_BEAM_STABILITY_ZH.md",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_SUFFIXES = {
    ".db",
    ".log",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".tmp",
}
MAX_FILE_BYTES = 20 * 1024 * 1024
ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version() -> str:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    return str(payload["project"]["version"])


def is_allowed(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    lowered = path.name.lower()
    if lowered.endswith("-wal") or lowered.endswith("-shm"):
        return False
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(
            f"refusing oversized release file: {relative} "
            f"({path.stat().st_size} bytes)"
        )
    return True


def source_files() -> list[Path]:
    files: set[Path] = set()
    for name in INCLUDED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required release file is missing: {name}")
        files.add(path)
    for name in INCLUDED_DIRECTORIES:
        directory = ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"required release directory is missing: {name}"
            )
        files.update(
            path
            for path in directory.rglob("*")
            if path.is_file() and is_allowed(path)
        )
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def manifest_bytes(rows: list[tuple[str, int, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("sha256", "bytes", "path"))
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_source_zip(
    destination: Path,
    *,
    wheel: Path | None,
) -> Path:
    release_version = version()
    prefix = PurePosixPath(
        f"posterior_memory_harness-{release_version}"
    )
    payloads: list[tuple[str, bytes]] = []
    for path in source_files():
        relative = PurePosixPath(path.relative_to(ROOT).as_posix())
        payloads.append((str(prefix / relative), path.read_bytes()))
    if wheel is not None:
        wheel = wheel.resolve()
        if not wheel.is_file():
            raise FileNotFoundError(wheel)
        if wheel.suffix != ".whl":
            raise ValueError("--wheel must point to a .whl artifact")
        payloads.append(
            (str(prefix / "artifacts" / wheel.name), wheel.read_bytes())
        )
    payloads.sort(key=lambda item: item[0])
    rows = [
        (sha256_bytes(data), len(data), name)
        for name, data in payloads
    ]
    payloads.append(
        (
            str(prefix / "MANIFEST_SHA256.csv"),
            manifest_bytes(rows),
        )
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, data in payloads:
            archive.writestr(zip_info(name), data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("release ZIP CRC verification failed")
        roots = {
            PurePosixPath(name).parts[0]
            for name in archive.namelist()
        }
        if roots != {prefix.name}:
            raise RuntimeError("release ZIP must have exactly one top-level root")
        if any(
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            for name in archive.namelist()
        ):
            raise RuntimeError("release ZIP contains an unsafe path")
    temporary.replace(destination)
    return destination


def write_checksums(directory: Path, release_version: str) -> Path:
    artifacts = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and release_version in path.name
            and path.suffix in {".whl", ".zip"}
        ),
        key=lambda item: item.name,
    )
    checksum_path = directory / f"SHA256SUMS-{release_version}.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in artifacts
        ),
        encoding="utf-8",
        newline="\n",
    )
    return checksum_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=Path,
        default=ROOT / "dist",
    )
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    release_version = version()
    destination = (
        args.dist
        / f"posterior_memory_harness-{release_version}-source.zip"
    )
    archive = build_source_zip(destination, wheel=args.wheel)
    checksums = write_checksums(args.dist, release_version)
    print(archive)
    print(checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
