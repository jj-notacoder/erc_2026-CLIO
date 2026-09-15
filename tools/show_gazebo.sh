#!/usr/bin/env bash
# Host entry point for the verified, visible Gazebo demonstration.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./tools/show_gazebo.sh [--check] [COLUMN COLOUR]

Run a visible simulation; defaults to column 2, red.
COLUMN: 1..5 (the numbered shelf marker)
COLOUR: red, blue, green, yellow

Examples:
  ./tools/show_gazebo.sh 4 blue
  ./tools/show_gazebo.sh --check 4 blue  # validate setup without starting a run

Requires Linux, Docker, Python 3, and an X11 desktop. The first run builds the
Docker image if absent and builds the workspace if needed. Close the Gazebo
window or press Ctrl+C to stop the run. Results remain in results/.

Optional environment:
  ERC_SEED=101                 Reproducible simulator seed
  ERC_CONTAINER_NAME=...      Preferred container name (default erc_friend_demo)
  ERC_IMAGE=...               Image name (default erc-2026:humble-harmonic)
  ERC_USE_PERFORMANCE=0       Skip the optional temporary performance profile
EOF
}

demo_check=0
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ "${1:-}" == --check ]]; then demo_check=1; shift; fi
if (( $# != 0 && $# != 2 )); then usage >&2; exit 2; fi
demo_column=${1:-2}
demo_colour=${2:-red}
demo_colour=${demo_colour,,}
[[ "$demo_column" =~ ^[1-5]$ ]] || { echo 'COLUMN must be 1..5.' >&2; exit 2; }
case "$demo_colour" in red|blue|green|yellow) ;; *) echo 'Invalid COLOUR.' >&2; exit 2;; esac

demo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
demo_image=${ERC_IMAGE:-erc-2026:humble-harmonic}
demo_container=${ERC_CONTAINER_NAME:-erc_friend_demo}
for demo_tool in docker python3; do
  command -v "$demo_tool" >/dev/null || { echo "Missing required command: $demo_tool" >&2; exit 1; }
done
[[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]] || {
  echo 'Run this from your Linux desktop terminal with DISPLAY set and X11 available.' >&2
  exit 1
}
demo_auth=${XAUTHORITY:-}
if [[ -z "$demo_auth" ]]; then
  for demo_candidate in "/run/user/$(id -u)/gdm/Xauthority" "$HOME/.Xauthority"; do
    if [[ -r "$demo_candidate" && -f "$demo_candidate" ]]; then demo_auth=$demo_candidate; break; fi
  done
fi
[[ -n "$demo_auth" && -r "$demo_auth" && -f "$demo_auth" ]] || {
  echo 'Set XAUTHORITY to the readable X11 authorization file for your desktop session.' >&2
  exit 1
}
demo_auth=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$demo_auth")
docker info >/dev/null

# Reuse only a container attached to this checkout and desktop. Never replace
# another checkout's container or kill another running simulation.
compatible_container() {
  docker inspect "$1" | python3 -c '
import json, os, sys
c = json.load(sys.stdin)[0]
root, auth, image = sys.argv[1:]
mounts = {m["Destination"]: m for m in c.get("Mounts", [])}
required = {"/opt/erc_ws": root, "/tmp/.X11-unix": "/tmp/.X11-unix",
            "/tmp/erc-demo.xauthority": auth}
ok = all(dst in mounts and mounts[dst]["Type"] == "bind"
         and os.path.realpath(mounts[dst]["Source"]) == src
         for dst, src in required.items())
ok = ok and mounts.get("/opt/erc_ws", {}).get("RW", False)
ok = ok and c["HostConfig"]["NetworkMode"] == "host" and c["Config"]["Image"] == image
if os.path.isdir("/dev/dri"):
    devices = c["HostConfig"].get("Devices") or []
    ok = ok and (c["HostConfig"].get("Privileged", False)
                 or any(d["PathOnHost"] == "/dev/dri" for d in devices))
sys.exit(0 if ok else 1)
' "$demo_root" "$demo_auth" "$demo_image"
}

if docker container inspect "$demo_container" >/dev/null 2>&1 && ! compatible_container "$demo_container"; then
  demo_suffix=$(python3 -c 'import hashlib,sys; print(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest()[:12])' "$demo_root" "$demo_auth" "$demo_image")
  demo_container="erc_demo_$demo_suffix"
  if docker container inspect "$demo_container" >/dev/null 2>&1 && ! compatible_container "$demo_container"; then
    echo "Container $demo_container has incompatible mounts. Choose another ERC_CONTAINER_NAME." >&2
    exit 1
  fi
fi

demo_domain=$(python3 -c 'import hashlib,sys; print(31 if sys.argv[2] == "erc_friend_demo" else 40 + int(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest()[:8], 16) % 50)' "$demo_root" "$demo_container")
echo "Project: $demo_root"
echo "Container: $demo_container; ROS domain: $demo_domain; target: column $demo_column, $demo_colour"
if (( demo_check )); then
  if docker container inspect "$demo_container" >/dev/null 2>&1; then
    echo 'Desktop and existing container configuration checked. No simulation started.'
  else
    echo 'Desktop checked. A container will be created on the first run. No simulation started.'
  fi
  exit 0
fi

# A second live simulator competes for CPU/GPU and invalidates run timing.
# Inspect only containers built for ERC or mounting the ERC workspace layout.
python3 - "$demo_image" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

ids = subprocess.check_output(['docker', 'ps', '-q'], text=True).split()
containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True)) if ids else []
for container in containers:
    known = container['Config']['Image'] == sys.argv[1]
    for mount in container.get('Mounts', []):
        source = Path(mount['Source'])
        if mount['Destination'] == '/opt/erc_ws':
            known = known or (source / 'src/erc_bringup').is_dir()
        elif mount['Destination'] == '/opt/erc_ws/src':
            known = known or (source / 'erc_bringup').is_dir()
    if not known:
        continue
    active = subprocess.run(
        ['docker', 'exec', container['Id'], 'pgrep', '-f',
         '[r]os2 launch erc_bringup simulation.launch.py'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if active.returncode == 0:
        print(f"A simulation is already running in {container['Name'].lstrip('/')}. Close that run first.", file=sys.stderr)
        sys.exit(1)
PY

if ! docker image inspect "$demo_image" >/dev/null 2>&1; then
  docker build -t "$demo_image" -f "$demo_root/docker/Dockerfile" "$demo_root"
fi
if ! docker container inspect "$demo_container" >/dev/null 2>&1; then
  demo_devices=()
  if [[ -d /dev/dri ]]; then demo_devices+=(--device /dev/dri:/dev/dri); fi
  docker run -d --name "$demo_container" --network host --shm-size 512m \
    "${demo_devices[@]}" \
    -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/erc-demo.xauthority \
    -e QT_X11_NO_MITSHM=1 -e "ROS_DOMAIN_ID=$demo_domain" -e ROS_LOCALHOST_ONLY=1 \
    -e "GZ_PARTITION=$demo_container" \
    -v "$demo_root:/opt/erc_ws:rw" \
    -v "$demo_auth:/tmp/erc-demo.xauthority:ro" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "$demo_image" sleep infinity >/dev/null
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$demo_container")" != true ]]; then
  docker start "$demo_container" >/dev/null
fi

demo_exec=(-i)
if [[ -t 0 && -t 1 ]]; then demo_exec+=(-t); fi
demo_command=(docker exec "${demo_exec[@]}"
  -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/erc-demo.xauthority
  -e "ERC_SHELF_COLUMN=$demo_column" -e "ERC_BOOK_COLOUR=$demo_colour"
  -e "ERC_SEED=${ERC_SEED:-101}" -e "ROS_DOMAIN_ID=$demo_domain" -e ROS_LOCALHOST_ONLY=1
  -e "GZ_PARTITION=$demo_container"
  -e CYCLONEDDS_URI=file:///opt/erc_ws/docker/cyclonedds.xml
  "$demo_container" /opt/erc_ws/docker/entrypoint.sh /opt/erc_ws/tools/run_gazebo_demo.sh)

if [[ "${ERC_USE_PERFORMANCE:-1}" != 0 ]] && command -v powerprofilesctl >/dev/null 2>&1 \
    && powerprofilesctl list 2>/dev/null | python3 -c 'import re,sys; sys.exit(not re.search(r"(?m)^\s*\*?\s*performance:", sys.stdin.read()))'; then
  exec powerprofilesctl launch --profile=performance \
    --reason='ERC Gazebo demonstration' --appid=erc-demo "${demo_command[@]}"
fi
exec "${demo_command[@]}"
