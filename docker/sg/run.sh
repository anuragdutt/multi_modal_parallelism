#!/usr/bin/env bash
# Start (or re-attach to) the SG development container `mmp-sg` on guppy. Never touches `vllm-mamba-sg`.
# Runs as the host user so NFS root-squash does not block reads of the HF cache or writes into the repo.
set -euo pipefail
IMAGE=${IMAGE:-mmp:sg-src}
NAME=${NAME:-mmp-sg}
PORT=${PORT:-8001}                      # host port for `vllm serve` inside (container port 8000)
REPO=${REPO:-/home/adutt/multi_modal_parallelism}
HF=${HF:-/home/adutt/.cache/huggingface}
BUILD=${BUILD:-/tmp/mmp-build-$(id -un)}        # host-local disk for the compile (build dir, ccache, cargo)
UIDGID="$(id -u):$(id -g)"
USERNAME="$(id -un)"
mkdir -p "$REPO/.container_home" "$REPO/results/raw" "$REPO/logs" "$BUILD"
[ -f "$REPO/docker/sg/env.list" ] || cp "$REPO/docker/sg/env.list.example" "$REPO/docker/sg/env.list"
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --init --name "$NAME" --gpus all --ipc=host --shm-size=32g --user "$UIDGID" \
    --ulimit memlock=-1 --ulimit stack=67108864 --env-file "$REPO/docker/sg/env.list" \
    -p "${PORT}:8000" -e HOME=/workspace/.container_home -e HF_HOME=/hf -e MMP_IMAGE_TAG="$IMAGE" \
    -e USER="$USERNAME" -e LOGNAME="$USERNAME" \
    -v "$REPO":/workspace -v "$HF":/hf -v "$BUILD":/build -w /workspace "$IMAGE" -c "sleep infinity"
elif ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME"
fi
echo "container $NAME is up (image $IMAGE, user $UIDGID, host port $PORT); attach with: docker exec -it $NAME bash"
docker exec "$NAME" python3 -c "import vllm, os; print('vllm imports from', os.path.dirname(vllm.__file__))" 2>/dev/null \
  || echo "vLLM is not built yet in this checkout: run  docker exec $NAME bash /workspace/docker/sg/build_vllm.sh"
