# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise cache credential isolation and failure policy without registry access."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CACHE_SCRIPT = Path(__file__).with_name("deps_image_cache.sh")
_KEY = "fake-key-for-cache-auth-test"
_PEER = "peer.example.test/team/cache"
_NGC = "nvcr.io/example/team/cache"

# Docker's external credential backend is the boundary under test. Network and
# image-store operations are stubbed; the real cache script must choose the
# backend, preserve the caller config and enforce its fallback/failure policy.
_DOCKER_STUB = r"""
import base64
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
config_dir = Path(os.environ["DOCKER_CONFIG"])
config_file = config_dir / "config.json"
config = json.loads(config_file.read_text())
auths = config.get("auths", {})
helpers = config.get("credHelpers", {})
helper = helpers.get("nvcr.io", config.get("credsStore", ""))
if not auths and not helpers and not config.get("credsStore"):
    helper = "automatically-discovered-host-store"
failure = os.environ.get("STUB_FAILURE", "")
event = {
    "args": args,
    "config_directory": str(config_dir),
    "ngc_backend": helper or "file",
    "ngc_authenticated": "nvcr.io" in auths,
    "peer_authenticated": "peer.example.test" in auths,
    "peer_helper": config.get("credHelpers", {}).get("peer.example.test"),
}
with open(os.environ["STUB_TRACE"], "a") as output:
    print(json.dumps(event), file=output)

if args[0] == "login":
    password = sys.stdin.read()
    if failure == "login":
        print("unauthorized", file=sys.stderr)
        sys.exit(1)
    if helper:
        # An external helper survives a copied config directory. Record its
        # write independently of that directory without persisting the key.
        Path(os.environ["STUB_EXTERNAL_STORE"]).write_text("credential stored")
    else:
        auths["nvcr.io"] = {
            "auth": base64.b64encode(f"$oauthtoken:{password}".encode()).decode()
        }
        config["auths"] = auths
        config_file.write_text(json.dumps(config))
elif args[0] == "pull":
    if args[-1].startswith("nvcr.io/"):
        authenticated = "nvcr.io" in auths or Path(os.environ["STUB_EXTERNAL_STORE"]).exists()
        if not authenticated or failure == "pull":
            print("denied", file=sys.stderr)
            sys.exit(1)
    elif "peer.example.test" not in auths:
        sys.exit(1)
elif args[:2] == ["image", "inspect"]:
    if failure == "inspect" and args[-1].startswith("nvcr.io/"):
        print("image disappeared", file=sys.stderr)
        sys.exit(1)
    print("linux/arm64")
elif args[0] == "tag":
    if failure == "tag" and args[1].startswith("nvcr.io/"):
        print("image disappeared", file=sys.stderr)
        sys.exit(1)
elif args[0] == "push":
    if failure == "push":
        print("denied: upload permission missing", file=sys.stderr)
        sys.exit(1)
else:
    sys.exit(2)
"""


def _run_cache(
    tmp_path: Path,
    *,
    mode: str = "pull",
    failure: str = "",
    key: str = _KEY,
    config: dict | None = None,
    repositories: str = _NGC,
) -> tuple[subprocess.CompletedProcess, list[dict], str]:
    """Run the actual cache entry point with isolated Docker state."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f"#!{sys.executable}\n{_DOCKER_STUB}", encoding="utf-8")
    docker.chmod(0o755)
    config_dir = tmp_path / "caller-config"
    config_dir.mkdir()
    if config is None:
        # A caller's registry helpers survive copying config.json even when no
        # file credential exists. Keep peer auth and those backend selections.
        config = {
            "auths": {"peer.example.test": {"auth": "cGVlcjpwYXNzd29yZA=="}},
            "credsStore": "host-store",
            "credHelpers": {"nvcr.io": "host-ngc-store", "peer.example.test": "peer-store"},
        }
    original = json.dumps(config)
    config_file = config_dir / "config.json"
    config_file.write_text(original, encoding="utf-8")
    output = tmp_path / "github-output"
    summary = tmp_path / "github-summary"
    output.touch()
    summary.touch()
    trace = tmp_path / "trace.jsonl"
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "DOCKER_CONFIG": str(config_dir),
        "NGC_API_KEY": key,
        "IMAGE_TAG": "isaac-lab:credential-test",
        "DEPS_HASH": "0123456789abcdef",
        "TARGET_PLATFORM": "linux/arm64",
        "CACHE_REPOSITORIES": repositories,
        "CACHE_TIMEOUT": "5",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "STUB_TRACE": str(trace),
        "STUB_EXTERNAL_STORE": str(tmp_path / "external-store"),
        "STUB_FAILURE": failure,
    }
    result = subprocess.run(
        ["bash", str(_CACHE_SCRIPT), mode], env=env, capture_output=True, text=True, check=False, timeout=20
    )
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    assert config_file.read_text() == original
    assert all(not Path(event["config_directory"]).exists() for event in events)
    if key:
        assert key not in result.stdout + result.stderr
    return result, events, output.read_text()


@pytest.mark.parametrize("failure", ["login", "pull", "inspect", "tag"])
def test_ngc_read_failure_preserves_peer_fallback_and_credential_isolation(tmp_path: Path, failure: str) -> None:
    """A private-cache failure retains peer auth and never writes the host helper."""
    repositories = f"{_NGC}\nnvcr.io/example/team/another-cache\n{_PEER}"
    result, events, output = _run_cache(tmp_path, failure=failure, repositories=repositories)
    assert result.returncode == 0, result.stderr
    assert "hit=true" in output
    assert f"source={_PEER}" in output
    assert not (tmp_path / "external-store").exists()
    logins = [event for event in events if event["args"][0] == "login"]
    assert len(logins) == 1
    assert logins[0]["ngc_backend"] == "file"
    assert logins[0]["args"] == ["login", "--username", "$oauthtoken", "--password-stdin", "nvcr.io"]
    peer_pull = next(event for event in events if event["args"][0] == "pull" and _PEER in event["args"][-1])
    assert peer_pull["peer_authenticated"]
    assert peer_pull["peer_helper"] == "peer-store"
    ngc_pulls = [event for event in events if event["args"][0] == "pull" and "nvcr.io/" in event["args"][-1]]
    if failure == "login":
        assert not ngc_pulls
    else:
        assert ngc_pulls and all(event["ngc_authenticated"] for event in ngc_pulls)


@pytest.mark.parametrize("failure", ["login", "push"])
def test_ngc_explicit_publication_failure_is_fatal(tmp_path: Path, failure: str) -> None:
    """An explicit upload request must not report success when authentication fails."""
    result, events, _ = _run_cache(tmp_path, mode="push", failure=failure)
    assert result.returncode != 0
    assert not (tmp_path / "external-store").exists()
    pushes = [event for event in events if event["args"][0] == "push"]
    assert len(pushes) == (0 if failure == "login" else 1)


def test_ngc_preconfigured_auth_is_used_when_no_key_is_supplied(tmp_path: Path) -> None:
    """A caller's file credential can read the cache without a second login."""
    auth = base64.b64encode(b"$oauthtoken:already-configured-test-key").decode()
    result, events, output = _run_cache(tmp_path, key="", config={"auths": {"nvcr.io": {"auth": auth}}})
    assert result.returncode == 0, result.stderr
    assert "hit=true" in output
    assert f"source={_NGC}" in output
    assert not any(event["args"][0] == "login" for event in events)


@pytest.mark.parametrize("peer_auth", [False, True])
def test_ngc_key_restores_private_access_without_changing_anonymous_base_config(
    tmp_path: Path, peer_auth: bool
) -> None:
    """NGC login restores private access after public-base credential removal.

    The empty config also exercises Docker's automatic credential-helper
    discovery, which must not write a credential outside the temporary config.
    """
    auths = {"peer.example.test": {"auth": "cGVlcjpwYXNzd29yZA=="}} if peer_auth else {}
    result, events, output = _run_cache(tmp_path, config={"auths": auths, "credsStore": ""})
    assert result.returncode == 0, result.stderr
    assert "hit=true" in output
    assert f"source={_NGC}" in output
    assert not (tmp_path / "external-store").exists()
    login = next(event for event in events if event["args"][0] == "login")
    pull = next(event for event in events if event["args"][0] == "pull")
    assert not login["ngc_authenticated"]
    assert login["ngc_backend"] == "file"
    assert pull["ngc_authenticated"]
