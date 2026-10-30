#!/usr/bin/env python3
"""Fetch the minimal QAIRT 2.51 SDK subset used by the SM8735/HTP-v73 build.

The official archive is several gigabytes. This script uses HTTP Range requests
through ``remotezip`` and writes only the CMake-required headers, SampleApp
sources, notices, common Android libraries, and HTP v73 runtime libraries.
It does not install or alter any SDK license acceptance state.

Requires: ``python3 -m pip install remotezip``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ARCHIVE_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    "Qualcomm_AI_Runtime_Community/All/2.51.0.260929/"
    "v2.51.0.260929.zip"
)
ARCHIVE_PREFIX = "qairt/2.51.0.260929/"
DIRECTORIES = (
    "include/QNN/",
    "examples/QNN/SampleApp/SampleApp/",
)
FILES = (
    "LICENSE.pdf",
    "NOTICE.txt",
    "QNN_NOTICE.txt",
    "lib/aarch64-android/libQnnHtp.so",
    "lib/aarch64-android/libQnnSystem.so",
    "lib/aarch64-android/libQnnHtpV73Stub.so",
    "lib/hexagon-v73/unsigned/libQnnHtpV73.so",
    "lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so",
)
CHUNK_SIZE = 8 * 1024 * 1024


def _sha256_and_crc32(path: Path) -> tuple[int, str, int]:
    digest = hashlib.sha256()
    crc = 0
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return size, digest.hexdigest(), crc & 0xFFFFFFFF


def _ensure_safe_parent(root: Path, relative: Path) -> Path:
    """Create destination parents without following pre-existing symlinks."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe SDK archive path: {relative}")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise FileExistsError(f"Refusing to follow symlinked SDK parent directory: {current}")
        if current.exists() and not current.is_dir():
            raise NotADirectoryError(f"SDK destination parent is not a directory: {current}")
        current.mkdir(exist_ok=True)
    return current


def _retry_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="QAIRT SDK root directory, e.g. /opt/qairt/2.51.0.260929",
    )
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    parser.add_argument("--read-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.read_attempts < 1:
        parser.error("--read-attempts must be at least 1")

    from remotezip import RemoteZip

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = _retry_session()
    try:
        with RemoteZip(
            args.archive_url,
            session=session,
            timeout=(15, 90),
        ) as archive:
            members = archive.infolist()
            selected = [
                info
                for info in members
                if not info.is_dir()
                and info.filename.startswith(ARCHIVE_PREFIX)
                and (
                    any(
                        info.filename.startswith(ARCHIVE_PREFIX + prefix)
                        for prefix in DIRECTORIES
                    )
                    or info.filename[len(ARCHIVE_PREFIX) :] in FILES
                )
            ]
            present = {info.filename[len(ARCHIVE_PREFIX) :] for info in selected}
            missing = sorted(set(FILES) - present)
            if missing:
                raise RuntimeError(f"Official archive is missing required files: {missing}")

            manifest: list[dict[str, Any]] = []
            total = len(selected)
            total_bytes = 0
            for index, info in enumerate(selected, 1):
                rel = Path(info.filename[len(ARCHIVE_PREFIX) :])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe path in official archive: {info.filename}")
                parent = _ensure_safe_parent(output, rel)
                target = parent / rel.name

                if target.is_symlink():
                    raise FileExistsError(f"Refusing to follow existing SDK symlink: {target}")
                if target.exists():
                    size, digest, crc = _sha256_and_crc32(target)
                    if size != info.file_size or crc != info.CRC:
                        raise FileExistsError(
                            f"Existing SDK file does not match archive member size/CRC; refusing overwrite: {target}"
                        )
                else:
                    last_error: Exception | None = None
                    data: bytes | None = None
                    for attempt in range(1, args.read_attempts + 1):
                        try:
                            data = archive.read(info.filename)
                            last_error = None
                            break
                        except Exception as error:
                            last_error = error
                            if attempt == args.read_attempts:
                                break
                            delay = min(2 ** (attempt - 1), 16)
                            print(
                                f"Range read failed for {rel.as_posix()} "
                                f"({type(error).__name__}); retrying in {delay}s "
                                f"({attempt + 1}/{args.read_attempts})",
                                flush=True,
                            )
                            time.sleep(delay)
                    if last_error is not None or data is None:
                        raise RuntimeError(
                            f"Failed to read archive member after {args.read_attempts} attempts: {info.filename}"
                        ) from last_error
                    if len(data) != info.file_size:
                        raise IOError(
                            f"Short extraction for {info.filename}: "
                            f"{len(data)} != {info.file_size} bytes"
                        )
                    if (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
                        raise IOError(f"Archive CRC mismatch for {info.filename}")
                    digest = hashlib.sha256(data).hexdigest()
                    temporary_file = target.with_name(target.name + ".part")
                    if temporary_file.is_symlink():
                        raise FileExistsError(f"Refusing to follow partial-file symlink: {temporary_file}")
                    temporary_file.unlink(missing_ok=True)
                    temporary_file.write_bytes(data)
                    temporary_file.replace(target)
                    mode = (info.external_attr >> 16) & 0o777
                    os.chmod(target, (mode & 0o755) or 0o644)
                    size = len(data)

                manifest.append(
                    {
                        "path": rel.as_posix(),
                        "size_bytes": size,
                        "sha256": digest,
                    }
                )
                total_bytes += manifest[-1]["size_bytes"]
                if index % 50 == 0 or index == total:
                    print(f"Fetched/verified {index}/{total} files ({total_bytes:,} bytes)", flush=True)

        manifest_path = output.parent / (output.name + ".subset-manifest.json")
        manifest_data = json.dumps(
            {
                "sdk_version": "2.51.0.260929",
                "archive_url": args.archive_url,
                "extraction_method": "HTTP Range / official archive members; resumable size/CRC verification",
                "intended_target": "SM8735 candidate, HTP v73 only",
                "file_count": len(manifest),
                "total_bytes": total_bytes,
                "files": manifest,
            },
            indent=2,
        ) + "\n"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(manifest_data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        session.close()

    print(f"SDK subset ready: {output} ({total_bytes:,} bytes)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
