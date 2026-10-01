#!/usr/bin/env bash
# Start (or re-attach to) the stage-1 development container.
set -euo pipefail
IMAGE=${IMAGE:-mmp:stage1}
NAME=${NAME:-mmp-dev}
REPO=${REPO:-/home/adutt/multi_modal_parallelism}
HF=${HF:-/home/adutt/.cache/huggingface}
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 --env-file "$REPO/docker/env.list" \
    -v "$REPO":/workspace -v "$HF":/root/.cache/huggingface -w /workspace "$IMAGE" -c "sleep infinity"
elif ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME"
fi
echo "container $NAME is up (image $IMAGE); attach with: docker exec -it $NAME bash"
