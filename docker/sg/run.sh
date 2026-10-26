#!/usr/bin/env bash
# Start (or re-attach to) the SG development container `mmp-sg` on guppy. Never touches `vllm-mamba-sg`.
# Runs as the host user so NFS root-squash does not block reads of the HF cache or writes into the repo.
set -euo pipefail
IMAGE=${IMAGE:-mmp:sg-dev}
NAME=${NAME:-mmp-sg}
PORT=${PORT:-8001}                      # host port for `vllm serve` inside (container port 8000)
REPO=${REPO:-/home/adutt/multi_modal_parallelism}
HF=${HF:-/home/adutt/.cache/huggingface}
UIDGID="$(id -u):$(id -g)"
mkdir -p "$REPO/.container_home" "$REPO/results/raw" "$REPO/logs"
[ -f "$REPO/docker/sg/env.list" ] || cp "$REPO/docker/sg/env.list.example" "$REPO/docker/sg/env.list"
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=32g --user "$UIDGID" \
    --ulimit memlock=-1 --ulimit stack=67108864 --env-file "$REPO/docker/sg/env.list" \
    -p "${PORT}:8000" -e HOME=/workspace/.container_home -e HF_HOME=/hf -e MMP_IMAGE_TAG="$IMAGE" \
    -v "$REPO":/workspace -v "$HF":/hf -w /workspace "$IMAGE" -c "sleep infinity"
elif ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME"
fi
echo "container $NAME is up (image $IMAGE, user $UIDGID, host port $PORT); attach with: docker exec -it $NAME bash"
