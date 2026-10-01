#!/usr/bin/env bash
# Step 0: snapshot the uncommitted SSM/gate work inside the old container vllm-mamba-sg.
# No docker commit: that container's writable layer is 289 GB and would fill the root disk.
set -euo pipefail
B=/home/adutt/backups/vllm-mamba-sg-$(date +%F); mkdir -p "$B"
docker exec vllm-mamba-sg bash -c 'cd /vllm && git status --short > /tmp/status.txt \
  && git diff > /tmp/uncommitted.patch \
  && git bundle create /tmp/local-commits.bundle 3034c8d38..HEAD \
  && tar czf /tmp/untracked.tgz $(git ls-files --others --exclude-standard) \
  && git rev-parse HEAD > /tmp/head.txt'
for f in status.txt uncommitted.patch local-commits.bundle untracked.tgz head.txt; do
  docker cp vllm-mamba-sg:/tmp/$f "$B/"
done
docker cp vllm-mamba-sg:/vllm "$B/vllm"
( cd "$B" && sha256sum uncommitted.patch local-commits.bundle untracked.tgz > SHA256SUMS )
git -C /home/adutt/masarani/vllm bundle verify "$B/local-commits.bundle"
echo "backup complete in $B"; du -sh "$B"
