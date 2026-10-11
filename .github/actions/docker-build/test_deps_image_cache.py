# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in dependency-image transfer contracts against real disposable registries.

Run with ISAACLAB_RUN_DOCKER_CACHE_TESTS=1 on a local Linux Docker daemon after
pulling registry:2. Tiny imported images are never executed; no GPU is required.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).with_name("deps_image_cache.sh")
_REGISTRY_IMAGE = "registry:2"

pytestmark = pytest.mark.skipif(
    os.environ.get("ISAACLAB_RUN_DOCKER_CACHE_TESTS") != "1",
    reason="Set ISAACLAB_RUN_DOCKER_CACHE_TESTS=1 to start disposable local registries.",
)


def _http_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


class RegistryLab:
    """Own the registry containers, tiny images and isolated client configuration."""

    def __init__(self, directory: Path, endpoint: str):
        self.directory = directory
        self.namespace = f"isaaclab-deps-test-{uuid.uuid4().hex}"
        self.deps_hash = uuid.uuid4().hex[:16]
        self.containers: list[str] = []
        self.references: set[str] = set()
        self.image_ids: set[str] = set()
        config = directory / "docker-config"
        config.mkdir()
        (config / "config.json").write_text('{"auths":{}}\n', encoding="utf-8")
        self.environment = os.environ.copy()
        for key in ("NGC_API_KEY", "DOCKER_AUTH_CONFIG", "DOCKER_CONTEXT"):
            self.environment.pop(key, None)
        self.environment.update(DOCKER_CONFIG=str(config), DOCKER_HOST=endpoint)

    def docker(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["docker", *arguments], env=self.environment, text=True, capture_output=True, timeout=60
        )
        if check:
            assert result.returncode == 0, f"docker {arguments}:\n{result.stdout}\n{result.stderr}"
        return result

    def registry(self, readonly: bool = False) -> str:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        name = f"{self.namespace}-{len(self.containers)}"
        # Track before starting so failed startup/readiness still removes this container.
        self.containers.append(name)
        configuration = (
            "version: 0.1\n"
            "storage:\n"
            "  filesystem:\n"
            "    rootdirectory: /var/lib/registry\n"
            "  maintenance:\n"
            "    readonly:\n"
            f"      enabled: {str(readonly).lower()}\n"
            "http:\n"
            f"  addr: 127.0.0.1:{port}\n"
        )
        self.docker(
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            "host",
            "--tmpfs",
            "/var/lib/registry",
            "--env",
            f"REGISTRY_TEST_CONFIG={configuration}",
            "--entrypoint",
            "/bin/sh",
            _REGISTRY_IMAGE,
            "-c",
            'printf "%s\\n" "$REGISTRY_TEST_CONFIG" > /tmp/registry.yml\nexec registry serve /tmp/registry.yml',
        )
        host = f"127.0.0.1:{port}"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if _http_status(f"http://{host}/v2/") == 200:
                    return host
            except urllib.error.URLError:
                pass
            if self.docker("inspect", "--format", "{{.State.Running}}", name).stdout.strip() != "true":
                break
            time.sleep(0.1)
        logs = self.docker("logs", name, check=False)
        pytest.fail(f"Registry did not become ready:\n{logs.stdout}{logs.stderr}")

    def image(self, platform: str = "linux/amd64") -> tuple[str, str]:
        reference = f"{self.namespace}/image:{uuid.uuid4().hex}"
        self.references.add(reference)
        rootfs = self.directory / f"{uuid.uuid4().hex}.tar"
        payload = uuid.uuid4().hex.encode("ascii")
        with tarfile.open(rootfs, "w") as archive:
            entry = tarfile.TarInfo("cache-test-marker")
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
        image_id = self.docker("import", "--platform", platform, str(rootfs), reference).stdout.strip()
        self.image_ids.add(image_id)
        return reference, image_id

    def ref(self, repository: str) -> str:
        reference = f"{repository}:deps-{self.deps_hash}"
        self.references.add(reference)
        return reference

    def seed(self, repository: str, platform: str = "linux/amd64") -> str:
        source, image_id = self.image(platform)
        remote = self.ref(repository)
        self.docker("tag", source, remote)
        self.docker("push", remote)
        # No local tag or image remains: the consumer must obtain the actual registry bytes.
        self.docker("image", "rm", source, remote)
        assert self.docker("image", "inspect", image_id, check=False).returncode != 0
        return image_id

    def cache(
        self, mode: str, repositories: list[str], image_tag: str, platform: str = "linux/amd64"
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        self.references.add(image_tag)
        for repository in repositories:
            self.ref(repository)
        output = self.directory / f"{uuid.uuid4().hex}.output"
        summary = self.directory / f"{uuid.uuid4().hex}.summary"
        environment = self.environment | {
            "DEPS_HASH": self.deps_hash,
            "IMAGE_TAG": image_tag,
            "TARGET_PLATFORM": platform,
            "CACHE_REPOSITORIES": "\n".join(repositories),
            "CACHE_TIMEOUT": "3",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        }
        result = subprocess.run(
            ["bash", str(_SCRIPT), mode], env=environment, text=True, capture_output=True, timeout=60
        )
        values = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
        return result, values

    def image_id(self, reference: str) -> str:
        return self.docker("image", "inspect", "--format", "{{.Id}}", reference).stdout.strip()

    def close(self) -> None:
        failures = []
        for name in reversed(self.containers):
            result = self.docker("container", "rm", "--force", "--volumes", name, check=False)
            if result.returncode and "No such container" not in result.stderr:
                failures.append(result.stderr)
        # Never prune or sweep a shared daemon. Delete only names/IDs created by this lab.
        for reference in sorted(self.references | self.image_ids):
            result = self.docker("image", "rm", reference, check=False)
            if result.returncode and "No such image" not in result.stderr:
                failures.append(result.stderr)
        assert not failures, "Failed to clean owned Docker resources:\n" + "\n".join(failures)


@pytest.fixture
def registry_lab(tmp_path: Path) -> Iterator[RegistryLab]:
    endpoint = os.environ.get("DOCKER_HOST")
    if not endpoint:
        result = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        endpoint = result.stdout.strip()
    if not endpoint.startswith("unix://"):
        pytest.skip("Registry tests require a local Linux daemon reachable over a Unix socket.")
    lab = RegistryLab(tmp_path, endpoint)
    try:
        assert lab.docker("info", "--format", "{{.OSType}}").stdout.strip() == "linux"
        result = lab.docker("image", "inspect", _REGISTRY_IMAGE, check=False)
        assert result.returncode == 0, f"Pull the small test prerequisite first: docker pull {_REGISTRY_IMAGE}"
        yield lab
    finally:
        lab.close()


@pytest.fixture
def unreachable_repository() -> Iterator[str]:
    # A bound, non-listening port fails connections without colliding with another service.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        yield f"127.0.0.1:{reservation.getsockname()[1]}/unreachable"


def test_first_peer_hit_wins(registry_lab: RegistryLab) -> None:
    lab = registry_lab
    first = f"{lab.registry()}/first"
    second = f"{lab.registry()}/second"
    first_id = lab.seed(first)
    lab.seed(second)
    target = f"{lab.namespace}/consumer:ordered"

    result, values = lab.cache("pull", [first, second], target)

    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {"hit": "true", "source": first}
    assert lab.image_id(target) == first_id
    assert lab.docker("image", "inspect", lab.ref(second), check=False).returncode != 0


def test_missing_and_unreachable_peers_fall_through(registry_lab: RegistryLab, unreachable_repository: str) -> None:
    lab = registry_lab
    host = lab.registry()
    fallback = f"{host}/fallback"
    expected_id = lab.seed(fallback)
    target = f"{lab.namespace}/consumer:fallback"

    result, values = lab.cache("pull", [f"{host}/missing", unreachable_repository, fallback], target)

    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {"hit": "true", "source": fallback}
    assert lab.image_id(target) == expected_id


def test_all_misses_leave_no_consumer_alias(registry_lab: RegistryLab, unreachable_repository: str) -> None:
    lab = registry_lab
    target = f"{lab.namespace}/consumer:miss"

    result, values = lab.cache("pull", [f"{lab.registry()}/missing", unreachable_repository], target)

    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {"hit": "false"}
    assert lab.docker("image", "inspect", target, check=False).returncode != 0


def test_wrong_platform_is_skipped_for_a_compatible_image(registry_lab: RegistryLab) -> None:
    lab = registry_lab
    host = lab.registry()
    wrong = f"{host}/arm-only"
    right = f"{host}/amd-only"
    lab.seed(wrong, platform="linux/arm64")
    expected_id = lab.seed(right)
    target = f"{lab.namespace}/consumer:platform"

    result, values = lab.cache("pull", [wrong, right], target)

    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {"hit": "true", "source": right}
    assert lab.image_id(target) == expected_id


def test_wrong_platform_cannot_be_published(registry_lab: RegistryLab) -> None:
    lab = registry_lab
    host = lab.registry()
    repository = f"{host}/wrong-platform"
    source, _ = lab.image(platform="linux/arm64")

    result, _ = lab.cache("push", [repository], source)

    assert result.returncode != 0
    assert _http_status(f"http://{host}/v2/wrong-platform/manifests/deps-{lab.deps_hash}") == 404


def test_publication_can_be_read_by_a_cold_consumer(registry_lab: RegistryLab) -> None:
    lab = registry_lab
    repository = f"{lab.registry()}/published"
    source, expected_id = lab.image()
    result, _ = lab.cache("push", [repository], source)
    assert result.returncode == 0, result.stdout + result.stderr
    lab.docker("image", "rm", source, lab.ref(repository))
    assert lab.docker("image", "inspect", expected_id, check=False).returncode != 0
    target = f"{lab.namespace}/consumer:published"

    result, values = lab.cache("pull", [repository], target)

    assert result.returncode == 0, result.stdout + result.stderr
    assert values == {"hit": "true", "source": repository}
    assert lab.image_id(target) == expected_id


def test_publication_failure_after_one_push_fails_the_operation(registry_lab: RegistryLab) -> None:
    lab = registry_lab
    host = lab.registry()
    first = f"{host}/published"
    readonly = f"{lab.registry(readonly=True)}/readonly"
    source, _ = lab.image()

    result, _ = lab.cache("push", [first, readonly], source)

    assert result.returncode != 0, result.stdout + result.stderr
    assert _http_status(f"http://{host}/v2/published/manifests/deps-{lab.deps_hash}") == 200
    readonly_host = readonly.split("/", 1)[0]
    assert _http_status(f"http://{readonly_host}/v2/readonly/manifests/deps-{lab.deps_hash}") == 404
