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
from pathlib import Path
from typing import Any

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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="QAIRT SDK root directory, e.g. /opt/qairt/2.51.0.260929",
    )
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    args = parser.parse_args()

    from remotezip import RemoteZip

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with RemoteZip(args.archive_url) as archive:
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
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(info.filename)
            if len(data) != info.file_size:
                raise IOError(
                    f"Short extraction for {info.filename}: "
                    f"{len(data)} != {info.file_size} bytes"
                )
            digest = sha256(data)
            if target.exists():
                if target.stat().st_size != len(data) or sha256(target.read_bytes()) != digest:
                    raise FileExistsError(
                        f"Refusing to overwrite differing existing SDK file: {target}"
                    )
            else:
                temp = target.with_name(target.name + ".part")
                temp.write_bytes(data)
                temp.replace(target)
                mode = (info.external_attr >> 16) & 0o777
                os.chmod(target, (mode & 0o755) or 0o644)
            manifest.append(
                {
                    "path": rel.as_posix(),
                    "size_bytes": len(data),
                    "sha256": digest,
                }
            )
            total_bytes += len(data)
            if index % 50 == 0 or index == total:
                print(f"Fetched {index}/{total} files ({total_bytes:,} bytes)", flush=True)

    manifest_path = output.parent / (output.name + ".subset-manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "sdk_version": "2.51.0.260929",
                "archive_url": args.archive_url,
                "extraction_method": "HTTP Range / official archive members",
                "intended_target": "SM8735 candidate, HTP v73 only",
                "file_count": len(manifest),
                "total_bytes": total_bytes,
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SDK subset ready: {output} ({total_bytes:,} bytes)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
