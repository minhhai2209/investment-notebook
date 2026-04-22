from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "gh-artifacts"
DEFAULT_ARTIFACT_PREFIX = "core-artifacts-"


def _run_command(args: Sequence[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout


def _load_json_command(args: Sequence[str]) -> Any:
    payload = _run_command(args).strip()
    return json.loads(payload) if payload else {}


def _parse_repo_from_remote_url(url: str) -> str:
    normalized = str(url).strip()
    if not normalized:
        raise ValueError("Git remote URL is empty")

    https_match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$", normalized)
    if https_match:
        owner = https_match.group("owner")
        repo = https_match.group("repo")
        return f"{owner}/{repo}"
    raise ValueError(f"Could not derive GitHub repo from remote URL: {url}")


def _derive_repo_name() -> str:
    remote_url = _run_command(["git", "remote", "get-url", "origin"]).strip()
    return _parse_repo_from_remote_url(remote_url)


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-")
    return token or "artifact"


def _list_artifacts(repo: str) -> list[dict[str, Any]]:
    payload = _load_json_command(["gh", "api", f"repos/{repo}/actions/artifacts?per_page=100"])
    artifacts = payload.get("artifacts") or []
    if not isinstance(artifacts, list):
        raise RuntimeError("GitHub artifact list payload is malformed")
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _select_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    prefix: str,
    branch: str | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw in artifacts:
        name = str(raw.get("name") or "")
        if not name.startswith(prefix):
            continue
        if bool(raw.get("expired")):
            continue
        workflow_run = raw.get("workflow_run") or {}
        if branch and str(workflow_run.get("head_branch") or "") != branch:
            continue
        selected.append(dict(raw))
    selected.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            int(item.get("id") or 0),
        ),
        reverse=True,
    )
    return selected


def _artifact_cache_dir(cache_dir: Path, artifact_id: int) -> Path:
    return cache_dir / "by-artifact" / str(int(artifact_id))


def _artifact_meta_path(cache_dir: Path, artifact_id: int) -> Path:
    return _artifact_cache_dir(cache_dir, artifact_id) / "artifact.json"


def _artifact_files_dir(cache_dir: Path, artifact_id: int) -> Path:
    return _artifact_cache_dir(cache_dir, artifact_id) / "files"


def _load_local_metadata(cache_dir: Path, artifact_id: int) -> dict[str, Any] | None:
    meta_path = _artifact_meta_path(cache_dir, artifact_id)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _artifact_is_cached(cache_dir: Path, artifact: Mapping[str, Any]) -> bool:
    artifact_id = int(artifact["id"])
    metadata = _load_local_metadata(cache_dir, artifact_id)
    files_dir = _artifact_files_dir(cache_dir, artifact_id)
    if metadata is None or not files_dir.exists():
        return False
    if str(metadata.get("digest") or "") != str(artifact.get("digest") or ""):
        return False
    return any(files_dir.iterdir())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _download_artifact(repo: str, cache_dir: Path, artifact: Mapping[str, Any]) -> Path:
    artifact_id = int(artifact["id"])
    run_id = int((artifact.get("workflow_run") or {}).get("id"))
    artifact_name = str(artifact["name"])
    base_dir = _artifact_cache_dir(cache_dir, artifact_id)
    files_dir = _artifact_files_dir(cache_dir, artifact_id)
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"artifact-{artifact_id}-", dir=str(cache_dir)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        download_dir = tmpdir_path / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                "--repo",
                repo,
                "--name",
                artifact_name,
                "--dir",
                str(download_dir),
            ],
            check=True,
        )
        shutil.move(str(download_dir), str(files_dir))

    _write_json(
        _artifact_meta_path(cache_dir, artifact_id),
        {
            "id": artifact_id,
            "name": artifact_name,
            "digest": artifact.get("digest"),
            "size_in_bytes": artifact.get("size_in_bytes"),
            "created_at": artifact.get("created_at"),
            "updated_at": artifact.get("updated_at"),
            "expires_at": artifact.get("expires_at"),
            "repo": repo,
            "workflow_run": artifact.get("workflow_run") or {},
        },
    )
    return files_dir


def _update_latest_pointer(cache_dir: Path, prefix: str, artifact: Mapping[str, Any]) -> None:
    latest_root = cache_dir / "latest"
    latest_root.mkdir(parents=True, exist_ok=True)
    alias = _sanitize_token(prefix.rstrip("-"))
    target_dir = _artifact_files_dir(cache_dir, int(artifact["id"]))
    link_path = latest_root / alias
    meta_path = latest_root / f"{alias}.json"
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    try:
        relative_target = os.path.relpath(target_dir, start=latest_root)
        link_path.symlink_to(relative_target)
    except OSError:
        shutil.copytree(target_dir, link_path)

    _write_json(
        meta_path,
        {
            "prefix": prefix,
            "artifact_id": int(artifact["id"]),
            "artifact_name": artifact.get("name"),
            "digest": artifact.get("digest"),
            "created_at": artifact.get("created_at"),
            "path": str(target_dir),
        },
    )


def _iter_local_metadata(cache_dir: Path) -> list[dict[str, Any]]:
    root = cache_dir / "by-artifact"
    if not root.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for meta_path in root.glob("*/artifact.json"):
        try:
            payloads.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return payloads


def _prune_local_cache(cache_dir: Path, prefix: str, keep: int) -> list[int]:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    payloads = [item for item in _iter_local_metadata(cache_dir) if str(item.get("name") or "").startswith(prefix)]
    payloads.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            int(item.get("id") or 0),
        ),
        reverse=True,
    )
    removed: list[int] = []
    for item in payloads[keep:]:
        artifact_id = int(item["id"])
        cache_path = _artifact_cache_dir(cache_dir, artifact_id)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            removed.append(artifact_id)
    return removed


def _delete_remote_artifacts(repo: str, artifacts: Sequence[Mapping[str, Any]]) -> list[int]:
    deleted: list[int] = []
    for artifact in artifacts:
        artifact_id = int(artifact["id"])
        subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "DELETE",
                f"repos/{repo}/actions/artifacts/{artifact_id}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        deleted.append(artifact_id)
    return deleted


def sync_latest_artifact(
    *,
    repo: str,
    prefix: str,
    branch: str | None,
    cache_dir: Path,
    keep_local: int,
    delete_remote_extra: bool,
    keep_remote: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _select_artifacts(_list_artifacts(repo), prefix=prefix, branch=branch)
    if not artifacts:
        raise RuntimeError(f"No non-expired artifacts found for prefix={prefix!r} branch={branch!r}")

    latest = artifacts[0]
    downloaded = False
    if not _artifact_is_cached(cache_dir, latest):
        _download_artifact(repo, cache_dir, latest)
        downloaded = True

    _update_latest_pointer(cache_dir, prefix, latest)
    removed_local = _prune_local_cache(cache_dir, prefix, keep_local)
    removed_remote: list[int] = []
    if delete_remote_extra:
        removed_remote = _delete_remote_artifacts(repo, artifacts[keep_remote:])

    return {
        "repo": repo,
        "prefix": prefix,
        "branch": branch,
        "downloaded": downloaded,
        "artifact_id": int(latest["id"]),
        "artifact_name": latest.get("name"),
        "digest": latest.get("digest"),
        "created_at": latest.get("created_at"),
        "cache_dir": str(_artifact_cache_dir(cache_dir, int(latest["id"]))),
        "latest_path": str((cache_dir / "latest" / _sanitize_token(prefix.rstrip("-")))),
        "removed_local_artifact_ids": removed_local,
        "removed_remote_artifact_ids": removed_remote,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/cache GitHub Actions artifacts without re-downloading the same digest.")
    parser.add_argument("--repo", help="GitHub repo in OWNER/REPO form. Default: derive from git remote origin.")
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX, help="Only sync artifacts whose names start with this prefix.")
    parser.add_argument("--branch", default="main", help="Only consider artifacts from this branch. Use empty string for any branch.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Local cache directory for downloaded artifacts.")
    parser.add_argument("--keep-local", type=int, default=3, help="Keep this many local cached artifact versions per prefix.")
    parser.add_argument("--delete-remote-extra", action="store_true", help="Delete older matching artifacts from GitHub after syncing.")
    parser.add_argument("--keep-remote", type=int, default=5, help="When deleting remote artifacts, keep this many newest versions.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo = args.repo or _derive_repo_name()
    branch = str(args.branch).strip() or None
    result = sync_latest_artifact(
        repo=repo,
        prefix=str(args.artifact_prefix),
        branch=branch,
        cache_dir=args.cache_dir,
        keep_local=int(args.keep_local),
        delete_remote_extra=bool(args.delete_remote_extra),
        keep_remote=int(args.keep_remote),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
