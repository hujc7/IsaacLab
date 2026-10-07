# docker-build

Builds or reuses the complete CI dependency image. The image includes the CI pytest
harness. Lookup order is local exact tag, local dependency tag, configured registry
repositories in order, then a build. Local hits avoid every remote image-cache pull.
Base-image authentication and dependency identity retain their existing owners.

## Shared dependency-image inputs

| Input | Behavior |
| --- | --- |
| `deps-cache-repositories` | Newline-separated, untagged repositories tried after local misses. List peers before NGC. Empty disables remote lookup. |
| `deps-cache-publish-repositories` | Newline-separated destinations for explicit publication. Empty disables publication. Requires `verify-test-path`. |
| `deps-cache-timeout` | Seconds allowed for each pull or push; default `600`. A timed-out command has an additional ten-second kill grace. NGC login is bounded separately by the smaller of this value and 30 seconds. |
| `verify-test-path` | Host pytest path used to verify fresh builds and every publication, with `IMAGE_TAG` set. The runner needs `uv`; failure prevents dependency tagging and publication of a fresh build. |
| `platform` | Target `linux/arm64` or `linux/amd64`. Remote images are inspected before reuse or publication to reject a different architecture. |
| `deps-hash` | Optional existing 16-character dependency hash. Normally computed by `_lib/compute-deps-hash`; callers providing it must include the recipe, resolved base digest and target platform. |

The `cache-source` output is `local-exact`, `local-deps`, the selected repository,
or `build`. Registry tags are `deps-<hash>` using the existing dependency identity.
With remote caching configured, digest-pinned consumers defer base authentication
until a fallback build. Mutable base tags still need authenticated manifest resolution.
Producers resolve the current dependency tag instead of trusting an exact local alias.
Treat published tags as immutable in the registry's access policy: the action uses
ordinary Docker tags and does not enforce registry-side immutability. Dependency
images are shared across source commits; task consumers must retain the existing
checkout/mount behavior rather than treating the cache as a complete source artifact.

```yaml
- uses: ./.github/actions/docker-build
  env:
    NGC_API_KEY: ${{ secrets.NGC_API_KEY }}
  with:
    image-tag: isaac-lab-experiment:${{ github.sha }}-arm64
    isaacsim-base-image: ${{ env.ISAACSIM_IMAGE }}
    isaacsim-version: ${{ env.ISAACSIM_VERSION }}
    platform: linux/arm64
    deps-cache-repositories: |
      registry.peer.example:5443/ci/isaac-lab-arm64
      nvcr.io/<org>/<team>/isaac-lab-arm64-cache
    verify-test-path: docker/test/test_image_invariants.py
```

A missing tag, unreachable registry, authentication failure or incompatible image
continues to the next repository, then to the normal build. Invalid input fails
configuration. Requested publication fails on verification, platform, authentication
or push failure. Publication to several repositories is sequential: if a later push
fails, earlier successful pushes remain. A failure is visible to the caller.

This shares full Docker images. It does not establish NGC support for the optional
`cache-from`/`cache-to` BuildKit cache-manifest format.

## Required setup

1. Create an NGC organization/team container repository and set repository variable
   `ARM_CI_NGC_CACHE_REPOSITORY` to its untagged name, for example
   `nvcr.io/<org>/<team>/isaac-lab-arm64-cache`. Configure `NGC_API_KEY` as a GitHub
   Actions secret. Consumers need private-registry read permission on that namespace;
   a producer also needs write permission. Reading the public Sim base does not prove
   either permission. See the [NGC private registry guide](https://docs.nvidia.com/ngc/latest/ngc-private-registry-user-guide.html).
2. Provision one LAN registry or explicitly managed registries on peer hosts. Set
   `ARM_CI_PEER_CACHE_REPOSITORIES` to newline-separated untagged repositories in
   preferred order. Use DNS names or IPv4 addresses, an optional port, and a repository
   path; omit URL schemes, tags and digests. Confirm each runner's Docker daemon can
   route to those endpoints and that firewall/ACL rules allow the registry port.
3. Configure peer registry TLS, trusted CA certificates on the Docker daemon, access
   controls, persistent storage and a retention policy. Install peer read credentials
   in the runner's Docker client configuration or available credential helper before
   invoking the action. Producers need write access. The action preserves peer
   credential configuration while authenticating NGC in a temporary Docker config;
   it does not provision peer credentials or change TLS trust. Follow the
   [Distribution deployment guide](https://distribution.github.io/distribution/about/deploying/).
4. Make an ARM64 self-hosted runner available to the repository running the experiment,
   with Docker/buildx, `uv`, Git LFS, host Python 3 and coreutils `timeout`. The manual
   workflow checks ARM64 and records the GPU/driver with `nvidia-smi`. Runner labels,
   variables and secrets from another repository are not inherited by a fork.
5. Enable Actions and make the manual workflow available on the experiment repository's
   default branch before dispatching a chosen branch. A draft PR alone does not enable
   dispatch of a new workflow. GitHub documents this prerequisite in
   [manually running a workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
   Restrict write credentials and `publish=true` dispatches to trusted producers.

These are direct registry pulls from listed peers. Docker transfers the image's
missing layers; the action does not discover runners, serve their Docker stores,
replicate images automatically or coordinate a P2P swarm. Registry hosts may be
separate from runners. Peer addresses, routes and usable throughput must be verified
on the deployment; this prototype does not configure the runner fleet.

## Manual producer and consumer experiment

The workflow is [arm-image-cache-prototype.yaml](../../workflows/arm-image-cache-prototype.yaml).
Its defaults try peers then NGC, keep publication disabled and leave eviction disabled.
`cache-source=ngc` tries only NGC after local misses. `cache-source=local` performs
only local lookup/build and requires `publish=false`. Set `runner-label` to `arm64`
for the pool or an existing label that selects one ARM runner for repeat measurements.
For `peer-then-ngc`, either registry variable can be absent; at least one must be set.

After the setup above, dispatch a trusted producer once with `publish=true`. The
selected lookup repositories are also its publication destinations. A cache hit can
be verified and republished, so a producer is not guaranteed to build. Then dispatch
consumers with `publish=false`, retaining the exact commit, base pin and platform:

```bash
gh workflow run arm-image-cache-prototype.yaml --repo OWNER/REPO --ref BRANCH \
  -f cache-source=peer-then-ngc -f publish=false -f runner-label=arm64 \
  -f transfer-timeout=600
```

The workflow verifies image invariants for every consumer and verifies any producer
before publishing. It does not run the ARM task suite. Its summary records the chosen
source, transfer seconds, runner, storage driver, GPU/driver and total verified image
setup time. Build and push durations also remain visible in step logs.

Use dedicated disposable Docker stores or otherwise controlled benchmark runners:

1. **Build baseline:** choose `local`, with no exact/dependency image and the base-layer
   state recorded. Measure full build, harness installation and verification.
2. **Producer:** measure build or reuse, verification and upload separately. Producer
   cost is paid when introducing or populating a dependency identity.
3. **Cold consumer:** use a store containing neither the dependency image nor its base
   layers. Compare `ngc` and `peer-then-ngc` on the same host and verify the summary's
   actual source. A peer miss that falls through to NGC is not a peer measurement.
4. **Warm-layer consumer:** prepare the same store with the pinned Sim base, but no
   exact/dependency cache tag, then repeat. Docker pulls reuse existing layer content;
   record downloaded bytes and extraction time as well as wall time.
5. **Local consumer:** repeat on the same runner without removing the dependency tag.
   Expect `local-deps` and no remote transfer. Measure fleet fan-out separately if
   multiple consumers contend for peer registry storage or network bandwidth.

Do not use shared-daemon image/container pruning to manufacture cold-cache results.
The prototype opts into no eviction; leave `evict-stale-cache` disabled for the
experiment. Any deliberate cleanup must name only owned experimental resources.
The local tests below prove transfer/fallback contracts with tiny images; they do not
measure NGC acceptance, TLS/auth deployment, Spark routing or full-image performance.

## CPU-only local registry validation

Use a local Linux Docker daemon reached over a Unix socket. The test requires the
small registry server image; it never pulls or runs an Isaac Sim/Lab image. No GPU,
QEMU, registry credentials or daemon configuration changes are needed.

```bash
docker pull registry:2
ISAACLAB_RUN_DOCKER_CACHE_TESTS=1 uv run --frozen --extra test python -m pytest -q \
  .github/actions/docker-build/test_deps_image_cache.py
```

Without the environment variable, integration cases skip. The tests create unique
loopback-only registry containers with ephemeral storage and tiny imported images,
including ARM64 metadata that is never executed. They check ordered selection,
missing/unreachable fallback, all-miss aliases, platform safety, publication readback
and failure after a partial publication. Cleanup removes only their registered
containers/image references/IDs, including when a test fails. The separately pulled
`registry:2` prerequisite remains available for subsequent runs.
