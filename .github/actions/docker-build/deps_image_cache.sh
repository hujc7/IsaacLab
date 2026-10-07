#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Full dependency-image sharing; the build action owns hashing, verification and fallback builds.
set -euo pipefail

mode="${1:?Expected pull or push}"
case "$mode" in pull|push) ;; *) exit 2 ;; esac
[[ "${DEPS_HASH}" =~ ^[0-9a-f]{16}$ ]] || { echo "::error::Invalid dependency hash"; exit 2; }
[[ "${CACHE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo "::error::Invalid transfer timeout"; exit 2; }
case "${TARGET_PLATFORM}" in linux/amd64|linux/arm64) ;; *) exit 2 ;; esac

# Preserve peer credentials and isolate NGC login from the base-image config.
# setup-docker-config may have removed an NGC credential to read a public Sim base;
# the private dependency cache needs that credential without undoing the fallback.
cache_config="$(mktemp -d)"
trap 'rm -rf "$cache_config"' EXIT
if [ -f "${DOCKER_CONFIG:-${HOME}/.docker}/config.json" ]; then
  cp "${DOCKER_CONFIG:-${HOME}/.docker}/config.json" "${cache_config}/config.json"
else
  printf '{"auths":{}}\n' > "${cache_config}/config.json"
fi
chmod 600 "${cache_config}/config.json"
export DOCKER_CONFIG="${cache_config}"
ngc_login_done=false
ngc_login_failed=false

if [ "$mode" = pull ]; then
  echo "hit=false" >> "$GITHUB_OUTPUT"
fi
while IFS= read -r repository || [ -n "$repository" ]; do
  repository="${repository%$'\r'}"
  [ -n "$repository" ] || continue
  # Require an explicit registry and an untagged repository. IPv4/DNS hosts and ports work.
  if ! [[ "$repository" =~ ^[a-z0-9][a-z0-9.-]*(:[0-9]+)?/[a-z0-9][a-z0-9._/-]*$ ]]; then
    echo "::error::Cache repository must be <registry>/<repository>, without a tag: ${repository}"
    exit 2
  fi
  if [[ "$repository" == nvcr.io/* ]] && [ -n "${NGC_API_KEY:-}" ]; then
    if [ "$ngc_login_failed" = true ]; then
      echo "::warning::Skipping ${repository} after the NGC cache login failed."
      continue
    fi
    if [ "$ngc_login_done" = false ]; then
      # Copying config.json alone still lets docker login write the caller's
      # external credential store. An empty registry-specific helper forces
      # file storage for NGC only, preserving peer helpers and avoiding Docker's
      # automatic default-helper discovery when the config has no auth entries.
      login_timeout="$CACHE_TIMEOUT"
      if [ "$login_timeout" -gt 30 ]; then login_timeout=30; fi
      if python3 - "${cache_config}/config.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    config = json.load(handle)
helpers = config.get("credHelpers") or {}
helpers["nvcr.io"] = ""
config["credHelpers"] = helpers
with open(path, "w") as handle:
    json.dump(config, handle)
PY
      then
        if printf '%s' "$NGC_API_KEY" | timeout --kill-after=10s "${login_timeout}s" \
            docker login --username '$oauthtoken' --password-stdin nvcr.io; then
          ngc_login_done=true
        fi
      fi
      if [ "$ngc_login_done" = false ]; then
        ngc_login_failed=true
        if [ "$mode" = push ]; then
          echo "::error::NGC cache authentication failed; cannot publish ${repository}."
          exit 1
        fi
        echo "::warning::NGC cache login failed; trying the next cache or local build."
        continue
      fi
    fi
  fi
  ref="${repository}:deps-${DEPS_HASH}"
  started=$SECONDS
  echo "Dependency cache ${mode}: ${ref} (${TARGET_PLATFORM})"
  if [ "$mode" = pull ]; then
    if timeout --kill-after=10s "${CACHE_TIMEOUT}s" docker pull --platform "$TARGET_PLATFORM" "$ref"; then
      if ! actual_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$ref")"; then
        echo "::warning::Cannot inspect cache image ${ref}; continuing."
        continue
      fi
      if [ "$actual_platform" != "$TARGET_PLATFORM" ]; then
        echo "::warning::Ignoring cache image for ${actual_platform}; requested ${TARGET_PLATFORM}."
        continue
      fi
      if ! docker tag "$ref" "$IMAGE_TAG"; then
        echo "::warning::Cannot tag cache image ${ref}; continuing."
        continue
      fi
      {
        echo "hit=true"
        echo "source=${repository}"
      } >> "$GITHUB_OUTPUT"
      echo "Dependency cache pull hit: ${repository}, $((SECONDS - started))s" | tee -a "$GITHUB_STEP_SUMMARY"
      exit 0
    fi
    echo "::warning::Cache pull failed or timed out: ${repository}; continuing."
  else
    if ! actual_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$IMAGE_TAG")"; then
      echo "::error::Cannot inspect the image requested for publication."
      exit 1
    fi
    if [ "$actual_platform" != "$TARGET_PLATFORM" ]; then
      echo "::error::Cannot publish ${actual_platform} as ${TARGET_PLATFORM}."
      exit 1
    fi
    docker tag "$IMAGE_TAG" "$ref"
    timeout --kill-after=10s "${CACHE_TIMEOUT}s" docker push "$ref"
  fi
  echo "Dependency cache ${mode}: ${repository}, $((SECONDS - started))s" | tee -a "$GITHUB_STEP_SUMMARY"
done <<< "$CACHE_REPOSITORIES"
