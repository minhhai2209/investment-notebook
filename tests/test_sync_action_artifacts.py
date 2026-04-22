from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools import sync_action_artifacts


class SyncActionArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_parse_repo_from_remote_url_supports_https_and_ssh(self) -> None:
        self.assertEqual(
            sync_action_artifacts._parse_repo_from_remote_url("https://github.com/minhhai2209/investment-notebook.git"),
            "minhhai2209/investment-notebook",
        )
        self.assertEqual(
            sync_action_artifacts._parse_repo_from_remote_url("git@github.com:minhhai2209/investment-notebook.git"),
            "minhhai2209/investment-notebook",
        )

    def test_select_artifacts_keeps_latest_matching_prefix_and_branch(self) -> None:
        artifacts = [
            {
                "id": 1,
                "name": "core-artifacts-1",
                "expired": False,
                "created_at": "2026-04-21T01:00:00Z",
                "workflow_run": {"head_branch": "main", "id": 101},
            },
            {
                "id": 2,
                "name": "core-artifacts-2",
                "expired": False,
                "created_at": "2026-04-22T01:00:00Z",
                "workflow_run": {"head_branch": "main", "id": 102},
            },
            {
                "id": 3,
                "name": "core-artifacts-3",
                "expired": False,
                "created_at": "2026-04-23T01:00:00Z",
                "workflow_run": {"head_branch": "dev", "id": 103},
            },
            {
                "id": 4,
                "name": "other-artifacts-4",
                "expired": False,
                "created_at": "2026-04-24T01:00:00Z",
                "workflow_run": {"head_branch": "main", "id": 104},
            },
        ]
        selected = sync_action_artifacts._select_artifacts(artifacts, prefix="core-artifacts-", branch="main")
        self.assertEqual([item["id"] for item in selected], [2, 1])

    def test_prune_local_cache_keeps_latest_versions(self) -> None:
        cache_dir = self.base / "cache"
        for artifact_id, created_at in [(11, "2026-04-20T01:00:00Z"), (12, "2026-04-21T01:00:00Z"), (13, "2026-04-22T01:00:00Z")]:
            files_dir = sync_action_artifacts._artifact_files_dir(cache_dir, artifact_id)
            files_dir.mkdir(parents=True, exist_ok=True)
            (files_dir / "candidate_watchlist_core.json").write_text("{}", encoding="utf-8")
            sync_action_artifacts._write_json(
                sync_action_artifacts._artifact_meta_path(cache_dir, artifact_id),
                {
                    "id": artifact_id,
                    "name": f"core-artifacts-{artifact_id}",
                    "created_at": created_at,
                    "digest": f"sha256:{artifact_id}",
                },
            )

        removed = sync_action_artifacts._prune_local_cache(cache_dir, "core-artifacts-", keep=2)
        self.assertEqual(removed, [11])
        self.assertFalse(sync_action_artifacts._artifact_cache_dir(cache_dir, 11).exists())
        self.assertTrue(sync_action_artifacts._artifact_cache_dir(cache_dir, 12).exists())
        self.assertTrue(sync_action_artifacts._artifact_cache_dir(cache_dir, 13).exists())

    def test_artifact_is_cached_requires_matching_digest_and_files(self) -> None:
        cache_dir = self.base / "cache"
        artifact = {"id": 25, "digest": "sha256:abc"}
        files_dir = sync_action_artifacts._artifact_files_dir(cache_dir, 25)
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / "dummy.txt").write_text("ok", encoding="utf-8")
        sync_action_artifacts._write_json(
            sync_action_artifacts._artifact_meta_path(cache_dir, 25),
            {"id": 25, "digest": "sha256:abc"},
        )
        self.assertTrue(sync_action_artifacts._artifact_is_cached(cache_dir, artifact))

        sync_action_artifacts._write_json(
            sync_action_artifacts._artifact_meta_path(cache_dir, 25),
            {"id": 25, "digest": "sha256:def"},
        )
        self.assertFalse(sync_action_artifacts._artifact_is_cached(cache_dir, artifact))

    def test_update_latest_pointer_writes_manifest(self) -> None:
        cache_dir = self.base / "cache"
        files_dir = sync_action_artifacts._artifact_files_dir(cache_dir, 31)
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / "summary.json").write_text("{}", encoding="utf-8")
        artifact = {"id": 31, "name": "core-artifacts-31", "digest": "sha256:xyz", "created_at": "2026-04-22T03:33:15Z"}

        sync_action_artifacts._update_latest_pointer(cache_dir, "core-artifacts-", artifact)

        alias = cache_dir / "latest" / "core-artifacts"
        manifest = json.loads((cache_dir / "latest" / "core-artifacts.json").read_text(encoding="utf-8"))
        self.assertTrue(alias.exists())
        self.assertEqual(manifest["artifact_id"], 31)


if __name__ == "__main__":
    unittest.main()
