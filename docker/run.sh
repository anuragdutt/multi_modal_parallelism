#!/usr/bin/env bash
# Start (or re-attach to) the stage-1 development container.
# Runs as the host user so NFS root-squash does not block reads of the HF cache or writes into the repo.
set -euo pipefail
IMAGE=${IMAGE:-mmp:stage1}
NAME=${NAME:-mmp-dev}
REPO=${REPO:-/home/adutt/multi_modal_parallelism}
HF=${HF:-/home/adutt/.cache/huggingface}
UIDGID="$(id -u):$(id -g)"
mkdir -p "$REPO/.container_home" "$REPO/results/raw" "$REPO/logs"
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=32g --user "$UIDGID" \
    --ulimit memlock=-1 --ulimit stack=67108864 --env-file "$REPO/docker/env.list" \
    -e HOME=/workspace/.container_home -e HF_HOME=/hf -e MMP_IMAGE_TAG="$IMAGE" \
    -v "$REPO":/workspace -v "$HF":/hf -w /workspace "$IMAGE" -c "sleep infinity"
elif ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME"
fi
echo "container $NAME is up (image $IMAGE, user $UIDGID); attach with: docker exec -it $NAME bash"
